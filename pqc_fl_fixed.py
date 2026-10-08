"""
PQC-FL (fixed): post-quantum secured, hierarchical federated learning on Flower.

What this version does (and what the earlier 50client.py did not):
  * Adaptive signatures: each client signs with ONE scheme -- Falcon-512 if it holds a
    large dataset (>= median size), else Dilithium2. The server verifies using the key it
    registered for that client, never a key supplied in the payload.
  * Hierarchy: clients are assigned to NUM_SUB_SERVERS sub-aggregate servers (each with
    its own Kyber512 key pair). Each sub-server verifies, decrypts and aggregates its
    clients; the global server aggregates the sub-aggregates. (Simulated in-process.)
  * Tamper test: chosen clients flip a byte of their ciphertext AFTER signing in chosen
    rounds; the server must reject those updates. Rejections are logged per round.
  * Real metrics: precision / recall / F1 from confusion counts, plus the majority-class
    baseline accuracy, because the dataset is ~12% positive (an all-zero model scores ~88%).
    Training uses a positive-class weight to counter the imbalance.
  * No plaintext weights leave the client in PQC mode (fit() returns an empty parameter list).
  * No pickle on network data: arrays travel as np.savez (allow_pickle=False), the envelope
    is JSON. Signatures cover client id + round + ciphertext (replay/rebinding protection),
    and are verified BEFORE any decryption or parsing of the encrypted body.
  * No secrets in logs.
  * MODE=plain runs the same pipeline without crypto, so overhead is MEASURED, not assumed.

Usage:
    set PQC_FL_DATA_DIR=C:\\path\\to\\IN_2_clients_partitioned_multi
    python pqc_fl_fixed.py            # MODE=pqc (default)
    set PQC_FL_MODE=plain && python pqc_fl_fixed.py
Results are written to results_<mode>.json.
"""
import base64
import hashlib
import io
import json
import os
import time
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from cryptography.fernet import Fernet
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, random_split

# ----------------------------------------------------------------------------- config
NUM_CLIENTS = int(os.environ.get("PQC_FL_CLIENTS", 50))
NUM_SUB_SERVERS = 3
NUM_ROUNDS = int(os.environ.get("PQC_FL_ROUNDS", 15))
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MODE = os.environ.get("PQC_FL_MODE", "pqc").lower()  # "pqc" or "plain"
DATA_DIR = os.environ.get("PQC_FL_DATA_DIR", "IN_2_clients_partitioned_multi")
SEED = 42

# Tamper simulation (only meaningful in pqc mode)
MALICIOUS_CLIENTS = {7, 23}
MALICIOUS_ROUNDS = {5, 10}

KEM_ALG = "Kyber512"
SIG_LARGE = "Falcon-512"   # smaller signatures -> used by clients with more data
SIG_SMALL = "Dilithium2"

NUMERICAL = ["Income", "Age", "Experience", "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS"]
CATEGORICAL = ["Married/Single", "House_Ownership", "Car_Ownership", "Profession", "CITY", "STATE"]
TARGET = "Risk_Flag"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------- serialization
def arrays_to_bytes(arrays: List[np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, *arrays)
    return buf.getvalue()


def bytes_to_arrays(data: bytes) -> List[np.ndarray]:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return [z[f"arr_{i}"] for i in range(len(z.files))]


def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode()


def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode())


# ----------------------------------------------------------------------------- crypto
def signed_message(client_id: int, rnd: int, kem_ct: bytes, enc: bytes) -> bytes:
    """Bytes covered by the signature: binds client id, round and the whole ciphertext."""
    return (
        f"{client_id}|{rnd}|".encode()
        + len(kem_ct).to_bytes(4, "big") + kem_ct + enc
    )


_ALIASES = {
    "Dilithium2": ["Dilithium2", "ML-DSA-44"],     # NIST renamed Dilithium -> ML-DSA
    "Falcon-512": ["Falcon-512"],
    "Kyber512": ["Kyber512", "ML-KEM-512"],        # NIST renamed Kyber -> ML-KEM
}


def _lib_name(alg: str) -> str:
    """Map our logical algorithm name to whatever this liboqs build calls it."""
    import oqs
    enabled = set(oqs.get_enabled_sig_mechanisms()) | set(oqs.get_enabled_kem_mechanisms())
    for name in _ALIASES.get(alg, [alg]):
        if name in enabled:
            return name
    raise RuntimeError(f"{alg} not available in this liboqs build")


def sig_keygen(alg: str) -> Tuple[bytes, bytes]:
    import oqs
    with oqs.Signature(_lib_name(alg)) as s:
        pub = s.generate_keypair()
        return pub, s.export_secret_key()


def sig_sign(alg: str, secret: bytes, msg: bytes) -> bytes:
    import oqs
    with oqs.Signature(_lib_name(alg), secret_key=secret) as s:
        return s.sign(msg)


def sig_verify(alg: str, pub: bytes, msg: bytes, sig: bytes) -> bool:
    import oqs
    try:
        with oqs.Signature(_lib_name(alg)) as s:
            return bool(s.verify(msg, sig, pub))
    except Exception:
        return False


def kem_keygen() -> Tuple[bytes, bytes]:
    import oqs
    with oqs.KeyEncapsulation(_lib_name(KEM_ALG)) as k:
        pub = k.generate_keypair()
        return pub, k.export_secret_key()


def fernet_from_secret(shared_secret: bytes) -> Fernet:
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(shared_secret).digest()))


def encrypt_weights(weights: List[np.ndarray], sub_pub: bytes) -> Tuple[bytes, bytes]:
    import oqs
    with oqs.KeyEncapsulation(_lib_name(KEM_ALG)) as k:
        kem_ct, shared = k.encap_secret(sub_pub)
    return kem_ct, fernet_from_secret(shared).encrypt(arrays_to_bytes(weights))


def decrypt_weights(kem_ct: bytes, enc: bytes, sub_secret: bytes) -> List[np.ndarray]:
    import oqs
    with oqs.KeyEncapsulation(_lib_name(KEM_ALG), secret_key=sub_secret) as k:
        shared = k.decap_secret(kem_ct)
    return bytes_to_arrays(fernet_from_secret(shared).decrypt(enc))


# ----------------------------------------------------------------------------- data + model
class CreditRiskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, encoders: Dict[str, LabelEncoder]):
        d = df.copy()
        for c in CATEGORICAL:
            le = encoders[c]
            known = set(le.classes_)
            d[c] = d[c].astype(str).map(lambda x: le.transform([x])[0] if x in known else 0)
        for c in NUMERICAL:
            d[c] = pd.to_numeric(d[c], errors="coerce")
            d[c] = d[c].fillna(d[c].mean() if not d[c].isnull().all() else 0.0)
        self.x_num = torch.tensor(d[NUMERICAL].values, dtype=torch.float32)
        self.x_cat = torch.tensor(d[CATEGORICAL].values, dtype=torch.long)
        self.y = torch.tensor(d[TARGET].values.reshape(-1, 1), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x_num[i], self.x_cat[i], self.y[i]


class CreditRiskNet(nn.Module):
    def __init__(self, n_num: int, cardinalities: List[int]):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(c, min(50, (c + 1) // 2)) for c in cardinalities]
        )
        self.bn_num = nn.BatchNorm1d(n_num)
        in_dim = n_num + sum(e.embedding_dim for e in self.embeddings)
        self.fc1 = nn.Linear(in_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x_num, x_cat):
        emb = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat([self.bn_num(x_num)] + emb, dim=1)
        x = self.bn1(torch.relu(self.fc1(x)))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


def load_client_split(client_id: int, encoders):
    df = pd.read_csv(os.path.join(DATA_DIR, f"client_{client_id}.csv"))
    ds = CreditRiskDataset(df, encoders)
    n_train = int(0.8 * len(ds))
    g = torch.Generator().manual_seed(SEED + client_id)
    return random_split(ds, [n_train, len(ds) - n_train], generator=g)


def get_weights(model: nn.Module) -> List[np.ndarray]:
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_weights(model: nn.Module, arrays: List[np.ndarray]) -> None:
    keys = model.state_dict().keys()
    model.load_state_dict({k: torch.tensor(v) for k, v in zip(keys, arrays)})


# ----------------------------------------------------------------------------- client
class FLClient(fl.client.NumPyClient):
    def __init__(self, cid: int, ctx: dict):
        self.cid = cid
        self.ctx = ctx
        self.model = CreditRiskNet(len(NUMERICAL), ctx["cardinalities"]).to(device)
        self.train_ds, self.test_ds = load_client_split(cid, ctx["encoders"])
        self.n_total = len(self.train_ds) + len(self.test_ds)
        self.sub = cid % NUM_SUB_SERVERS
        self.sig_alg = SIG_LARGE if self.n_total >= ctx["size_threshold"] else SIG_SMALL

    def get_parameters(self, config=None):
        return get_weights(self.model)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        pos_w = self.ctx["pos_weight"]
        loader = DataLoader(self.train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        for _ in range(LOCAL_EPOCHS):
            for xn, xc, y in loader:
                xn, xc, y = xn.to(device), xc.to(device), y.to(device)
                opt.zero_grad()
                p = self.model(xn, xc)
                w = 1.0 + (pos_w - 1.0) * y
                loss = (nn.functional.binary_cross_entropy(p, y, reduction="none") * w).mean()
                loss.backward()
                opt.step()

        weights = get_weights(self.model)
        n = len(self.train_ds)
        rnd = int(config.get("server_round", 0))

        if MODE == "plain":
            body = arrays_to_bytes(weights)
            return [], n, {"client_id": self.cid, "payload": body,
                           "payload_bytes": len(body), "crypto_s": 0.0}

        t0 = time.perf_counter()
        kem_ct, enc = encrypt_weights(weights, self.ctx["sub_pubs"][self.sub])
        msg = signed_message(self.cid, rnd, kem_ct, enc)
        sig = sig_sign(self.sig_alg, self.ctx["sig_keys"][self.cid][self.sig_alg]["secret"], msg)
        crypto_s = time.perf_counter() - t0

        # Tamper simulation: modify the ciphertext AFTER it has been signed.
        if self.cid in MALICIOUS_CLIENTS and rnd in MALICIOUS_ROUNDS:
            tampered = bytearray(enc)
            tampered[len(tampered) // 2] ^= 0xFF
            enc = bytes(tampered)

        envelope = json.dumps({"alg": self.sig_alg, "ct": b64e(kem_ct),
                               "enc": b64e(enc), "sig": b64e(sig)}).encode()
        return [], n, {"client_id": self.cid, "payload": envelope,
                       "payload_bytes": len(envelope), "crypto_s": crypto_s}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        self.model.eval()
        tp = fp = fn = tn = 0
        loss_sum = 0.0
        with torch.no_grad():
            for xn, xc, y in DataLoader(self.test_ds, batch_size=512):
                xn, xc, y = xn.to(device), xc.to(device), y.to(device)
                p = self.model(xn, xc)
                loss_sum += nn.functional.binary_cross_entropy(p, y, reduction="sum").item()
                pred = (p > 0.5).float()
                tp += int(((pred == 1) & (y == 1)).sum())
                fp += int(((pred == 1) & (y == 0)).sum())
                fn += int(((pred == 0) & (y == 1)).sum())
                tn += int(((pred == 0) & (y == 0)).sum())
        total = tp + fp + fn + tn
        return loss_sum / max(total, 1), total, {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ----------------------------------------------------------------------------- server strategy
def weighted_average(arrays_list: List[List[np.ndarray]], counts: List[int]) -> List[np.ndarray]:
    total = float(sum(counts))
    out = []
    for layer in range(len(arrays_list[0])):
        acc = sum(a[layer].astype(np.float64) * n for a, n in zip(arrays_list, counts)) / total
        out.append(acc.astype(arrays_list[0][layer].dtype))
    return out


class HierarchicalPQCStrategy(fl.server.strategy.FedAvg):
    def __init__(self, sub_secrets, sig_pubs, history, **kw):
        super().__init__(**kw)
        self.sub_secrets = sub_secrets          # {sub_id: kyber secret key}
        self.sig_pubs = sig_pubs                # {cid: {alg: public key}}
        self.history = history                  # dict collecting per-round results
        self.cur_round = 0

    # -- verify then decrypt one client's update; returns weights or None
    def _open(self, cid: int, rnd: int, payload: bytes):
        sub = cid % NUM_SUB_SERVERS
        try:
            env = json.loads(payload.decode())
            alg = env["alg"]
            if alg not in (SIG_LARGE, SIG_SMALL):
                return None, "unknown-alg"
            kem_ct, enc, sig = b64d(env["ct"]), b64d(env["enc"]), b64d(env["sig"])
        except Exception:
            return None, "malformed"
        pub = self.sig_pubs.get(cid, {}).get(alg)
        if pub is None or not sig_verify(alg, pub, signed_message(cid, rnd, kem_ct, enc), sig):
            return None, "bad-signature"
        try:
            return decrypt_weights(kem_ct, enc, self.sub_secrets[sub]), None
        except Exception:
            return None, "decrypt-failed"

    def configure_fit(self, server_round, parameters, client_manager):
        self.cur_round = server_round
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        t0 = time.perf_counter()
        per_sub: Dict[int, Tuple[List, List]] = {s: ([], []) for s in range(NUM_SUB_SERVERS)}
        rejected, payload_bytes, client_crypto = [], [], []
        for _, fit_res in results:
            m = fit_res.metrics
            cid = int(m["client_id"])
            payload_bytes.append(int(m["payload_bytes"]))
            client_crypto.append(float(m["crypto_s"]))
            if MODE == "plain":
                weights, why = bytes_to_arrays(m["payload"]), None
            else:
                weights, why = self._open(cid, server_round, m["payload"])
            if weights is None:
                rejected.append({"client": cid, "reason": why})
                log(f"[round {server_round}] REJECTED client {cid}: {why}")
                continue
            per_sub[cid % NUM_SUB_SERVERS][0].append(weights)
            per_sub[cid % NUM_SUB_SERVERS][1].append(fit_res.num_examples)

        sub_aggs, sub_counts = [], []
        for s, (ws, ns) in per_sub.items():
            if ws:
                sub_aggs.append(weighted_average(ws, ns))
                sub_counts.append(sum(ns))
        server_s = time.perf_counter() - t0
        self.history["rounds"].append({
            "round": server_round,
            "accepted": sum(len(v[0]) for v in per_sub.values()),
            "rejected": rejected,
            "avg_payload_bytes": float(np.mean(payload_bytes)) if payload_bytes else 0.0,
            "avg_client_crypto_s": float(np.mean(client_crypto)) if client_crypto else 0.0,
            "server_verify_decrypt_aggregate_s": server_s,
        })
        if not sub_aggs:
            return None, {}
        global_w = weighted_average(sub_aggs, sub_counts)   # global server aggregates sub-servers
        return fl.common.ndarrays_to_parameters(global_w), {}

    def aggregate_evaluate(self, server_round, results, failures):
        loss, _ = super().aggregate_evaluate(server_round, results, failures)
        tp = sum(r.metrics["tp"] for _, r in results)
        fp = sum(r.metrics["fp"] for _, r in results)
        fn = sum(r.metrics["fn"] for _, r in results)
        tn = sum(r.metrics["tn"] for _, r in results)
        total = tp + fp + fn + tn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        row = {"round": server_round, "loss": loss, "accuracy": (tp + tn) / max(total, 1),
               "majority_baseline_accuracy": max(tp + fn, tn + fp) / max(total, 1),
               "precision": prec, "recall": rec, "f1": f1}
        self.history["eval"].append(row)
        log(f"[round {server_round}] acc={row['accuracy']:.4f} "
            f"(majority baseline {row['majority_baseline_accuracy']:.4f}) "
            f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
        return loss, {}


# ----------------------------------------------------------------------------- main
def main():
    # global label encoders + stats (done once on the coordinator for the simulation)
    frames, sizes, pos, tot = [], [], 0, 0
    for i in range(NUM_CLIENTS):
        df = pd.read_csv(os.path.join(DATA_DIR, f"client_{i}.csv"))
        frames.append(df[CATEGORICAL])
        sizes.append(len(df))
        pos += int(df[TARGET].sum())
        tot += len(df)
    all_cat = pd.concat(frames, ignore_index=True)
    encoders = {}
    for c in CATEGORICAL:
        le = LabelEncoder()
        le.fit(all_cat[c].astype(str).unique())
        encoders[c] = le
    cardinalities = [len(encoders[c].classes_) for c in CATEGORICAL]
    pos_rate = pos / tot
    log(f"clients={NUM_CLIENTS} rows={tot} positive_rate={pos_rate:.4f} "
        f"-> all-zero baseline accuracy={1 - pos_rate:.4f}")

    ctx = {"encoders": encoders, "cardinalities": cardinalities,
           "size_threshold": float(np.median(sizes)),
           "pos_weight": (1 - pos_rate) / pos_rate}

    sub_secrets, sig_pubs = {}, {}
    if MODE == "pqc":
        ctx["sub_pubs"] = {}
        for s in range(NUM_SUB_SERVERS):
            pub, sec = kem_keygen()
            ctx["sub_pubs"][s], sub_secrets[s] = pub, sec
        ctx["sig_keys"] = {}
        for cid in range(NUM_CLIENTS):
            ctx["sig_keys"][cid], sig_pubs[cid] = {}, {}
            for alg in (SIG_LARGE, SIG_SMALL):
                pub, sec = sig_keygen(alg)
                ctx["sig_keys"][cid][alg] = {"secret": sec}
                sig_pubs[cid][alg] = pub

    history = {"mode": MODE, "config": {
        "clients": NUM_CLIENTS, "sub_servers": NUM_SUB_SERVERS, "rounds": NUM_ROUNDS,
        "local_epochs": LOCAL_EPOCHS, "lr": LEARNING_RATE, "batch_size": BATCH_SIZE,
        "malicious_clients": sorted(MALICIOUS_CLIENTS) if MODE == "pqc" else [],
        "malicious_rounds": sorted(MALICIOUS_ROUNDS) if MODE == "pqc" else []},
        "rounds": [], "eval": []}

    strategy = HierarchicalPQCStrategy(
        sub_secrets=sub_secrets, sig_pubs=sig_pubs, history=history,
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS, min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda r: {"server_round": r},
    )

    def client_fn(cid: str):
        return FLClient(int(cid), ctx).to_client()

    t0 = time.time()
    fl.simulation.start_simulation(
        client_fn=client_fn, num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy, client_resources={"num_cpus": 1},
    )
    history["total_seconds"] = time.time() - t0

    out = f"results_{MODE}.json"
    with open(out, "w") as f:
        json.dump(history, f, indent=2, default=float)
    rej = sum(len(r["rejected"]) for r in history["rounds"])
    log(f"done in {history['total_seconds']:.1f}s; total rejected updates: {rej}; wrote {out}")


if __name__ == "__main__":
    main()

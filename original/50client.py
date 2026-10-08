import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import os
import oqs
from cryptography.fernet import Fernet, InvalidToken
import hashlib
import base64
import pickle
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple, List, Union
import time
import matplotlib.pyplot as plt

# --- Configuration and Global Variables ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define numerical and categorical columns
numerical_cols = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']
categorical_cols = ['Married/Single', 'House_Ownership', 'Car_Ownership', 'Profession', 'CITY', 'STATE']

NUM_CLIENTS = 50
DATA_DIR = r"C:\Users\Pranav\Desktop\PQC_FL\21st\IN_2_clients_partitioned_multi"

global_label_encoders = {} # Stores fitted LabelEncoders for categorical columns
round_accuracy, round_loss, round_std_loss, round_times = [], [], [], []

# --- Custom Logger for Consistent Output ---
class Logger:
    def warning(self, message):
        print(f"[WARNING] {message}")
    def info(self, message):
        print(f"[INFO] {message}")
    def error(self, message):
        print(f"[ERROR] {message}")
    def debug(self, message):
        print(f"[DEBUG] {message}")

logger = Logger()

# --- Serialization/Deserialization Utilities ---
def ndarray_list_to_bytes(arrays: List[np.ndarray]) -> bytes:
    """Converts a list of NumPy arrays to bytes using pickle."""
    return pickle.dumps(arrays)

def bytes_to_ndarray_list(data: bytes) -> List[np.ndarray]:
    """Converts bytes back to a list of NumPy arrays using pickle."""
    return pickle.loads(data)

def print_debug(label: str, data: Union[bytes, List[np.ndarray]]):
    """Prints debug information about data type and size."""
    logger.debug(f"{label}: {type(data)} | Size: {len(data) if hasattr(data, '__len__') else 'N/A'}")

# --- Post-Quantum Cryptography (PQC) Functions ---
def generate_dilithium_key_pair(algorithm: str = "Dilithium2") -> Tuple[bytes, bytes]:
    """Generates a Dilithium key pair and returns raw public/secret keys."""
    with oqs.Signature(algorithm) as sig:
        public_key = sig.generate_keypair()
        secret_key = sig.export_secret_key()
    return public_key, secret_key

def generate_falcon_key_pair(algorithm: str = "Falcon-512") -> Tuple[bytes, bytes]:
    """Generates a Falcon key pair and returns raw public/secret keys."""
    with oqs.Signature(algorithm) as sig:
        public_key = sig.generate_keypair()
        secret_key = sig.export_secret_key()
    return public_key, secret_key

def sign_data(data_bytes: bytes, private_key: bytes, algorithm: str = "Dilithium2") -> bytes:
    """Signs data using the specified OQS signature algorithm."""
    with oqs.Signature(algorithm, secret_key=private_key) as signer:
        signature = signer.sign(data_bytes)
    return signature

def verify_signature(data_bytes: bytes, signature: bytes, public_key: bytes, algorithm: str = "Dilithium2") -> bool:
    """Verifies a signature using the specified OQS signature algorithm."""
    with oqs.Signature(algorithm) as verifier:
        return verifier.verify(data_bytes, signature, public_key)

def encrypt_model(weights: List[np.ndarray], server_pubkey: bytes) -> Dict[str, bytes]:
    """Encrypts model weights using Kyber KEM and Fernet AES."""
    model_bytes = ndarray_list_to_bytes(weights)
    print_debug("Model Bytes (before encrypt)", model_bytes)

    with oqs.KeyEncapsulation("Kyber512") as kem:
        ciphertext, shared_secret = kem.encap_secret(server_pubkey)
        logger.debug(f"Client Encrypt: Shared Secret (hex): {shared_secret.hex()}")
    
    aes_key = hashlib.sha256(shared_secret).digest()
    logger.debug(f"Client Encrypt: AES Key (hex): {aes_key.hex()}")
    fernet_key = base64.urlsafe_b64encode(aes_key)
    logger.debug(f"Client Encrypt: Fernet Key (base64): {fernet_key.decode()}")
    
    encrypted_model = Fernet(fernet_key).encrypt(model_bytes)
    
    print_debug("Encrypted Model", encrypted_model)
    return {"ciphertext": ciphertext, "encrypted_weights": encrypted_model}

def decrypt_model(payload: Dict[str, bytes], server_secret: bytes) -> List[np.ndarray]:
    """Decrypts model weights using Kyber KEM and Fernet AES."""
    try:
        with oqs.KeyEncapsulation("Kyber512", secret_key=server_secret) as kem:
            shared_secret = kem.decap_secret(payload["ciphertext"])
            logger.debug(f"Server Decrypt: Shared Secret (hex): {shared_secret.hex()}")
        
        aes_key = hashlib.sha256(shared_secret).digest()
        logger.debug(f"Server Decrypt: AES Key (hex): {aes_key.hex()}")
        fernet_key = base64.urlsafe_b64encode(aes_key)
        logger.debug(f"Server Decrypt: Fernet Key (base64): {fernet_key.decode()}")
        
        decrypted_bytes = Fernet(fernet_key).decrypt(payload["encrypted_weights"])
        
        print_debug("Decrypted Model Bytes", decrypted_bytes)
        return bytes_to_ndarray_list(decrypted_bytes)
    except InvalidToken:
        logger.error("[ERROR] Fernet decryption failed. InvalidToken. Returning empty weights.")
        return []
    except Exception as e:
        logger.error(f"[ERROR] Decryption failed with unexpected error: {e}. Returning empty weights.")
        return []

# --- Dataset Classes ---
class EmptyDataset(Dataset):
    """A dummy dataset for clients with no data."""
    def __len__(self):
        return 0
    
    def __getitem__(self, idx):
        raise IndexError("This dataset is empty.")

class CreditRiskDataset(Dataset):
    """PyTorch Dataset for Credit Risk data."""
    def __init__(self, df: pd.DataFrame, numerical_cols: List[str], categorical_cols: List[str], global_label_encoders: Dict[str, LabelEncoder], target_col: str = "Risk_Flag"):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.target_col = target_col

        if df.empty:
            self.X_num = torch.tensor([], dtype=torch.float32).to(device)
            self.X_cat = torch.tensor([], dtype=torch.long).to(device)
            self.y = torch.tensor([], dtype=torch.float32).to(device)
        else:
            processed_df = df.copy()
            for col in self.categorical_cols:
                if col in processed_df.columns:
                    if col in global_label_encoders:
                        le = global_label_encoders[col]
                        processed_df[col] = processed_df[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else 0
                        )
                    else:
                        logger.warning(f"No global LabelEncoder for '{col}'. Fitting local encoder.")
                        le = LabelEncoder()
                        processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                else:
                    logger.warning(f"Categorical column '{col}' not in DataFrame.")
            
            cols_to_process_cat = [c for c in self.categorical_cols if c in processed_df.columns]
            if cols_to_process_cat:
                self.X_cat = torch.tensor(processed_df[cols_to_process_cat].values, dtype=torch.long).to(device)
            else:
                self.X_cat = torch.empty((len(processed_df), 0), dtype=torch.long).to(device)

            cols_to_process_num = [c for c in self.numerical_cols if c in processed_df.columns]
            if cols_to_process_num:
                for col in cols_to_process_num:
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mean() if not processed_df[col].isnull().all() else 0.0)
                self.X_num = torch.tensor(processed_df[cols_to_process_num].values, dtype=torch.float32).to(device)
            else:
                logger.warning("No numerical columns found. Creating empty numerical tensor.")
                self.X_num = torch.empty((len(processed_df), 0), dtype=torch.float32).to(device)

            if self.target_col in processed_df.columns:
                self.y = torch.tensor(processed_df[self.target_col].values.reshape(-1, 1), dtype=torch.float32).to(device)
            else:
                raise ValueError(f"Target column '{self.target_col}' not found in DataFrame.")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_num[idx], self.X_cat[idx], self.y[idx]

def load_client_data_internal(client_id: int, numerical_cols: List[str], categorical_cols: List[str], global_label_encoders: Dict[str, LabelEncoder]) -> Tuple[Dataset, Dataset]:
    """Loads and splits data for a specific client."""
    path = os.path.join(DATA_DIR, f"client_{client_id}.csv")
    df = pd.DataFrame()

    if not os.path.exists(path):
        logger.error(f"File not found: {path}. Using dummy data.")
    else:
        try:
            df = pd.read_csv(path)
            if df.empty:
                logger.warning(f"Client {client_id} CSV is empty. Using dummy data.")
        except Exception as e:
            logger.error(f"Failed to load CSV for client {client_id}: {e}. Using dummy data.")
            df = pd.DataFrame()

    if df.empty or len(df.columns) < (len(numerical_cols) + len(categorical_cols) + 1):
        logger.info(f"Generating dummy data for client {client_id}.")
        dummy_data = {col: [0] for col in numerical_cols}
        dummy_data.update({col: [0] for col in categorical_cols})
        dummy_data["Risk_Flag"] = [0]
        df = pd.DataFrame(dummy_data)
        if numerical_cols:
            df[numerical_cols[0]] = 1.0

    dataset = CreditRiskDataset(df, numerical_cols, categorical_cols, global_label_encoders)
    total_size = len(dataset)

    if total_size == 0:
        return EmptyDataset(), EmptyDataset()
    elif total_size < 2:
        return dataset, EmptyDataset()
    else:
        train_size = int(0.8 * total_size)
        test_size = total_size - train_size
        return random_split(dataset, [train_size, test_size])

# --- Neural Network Model ---
class CreditRiskNet(nn.Module):
    """Neural network with embeddings for credit risk prediction."""
    def __init__(self, num_numerical: int, cat_cardinalities: List[int]):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card, min(50, (card + 1) // 2)) for card in cat_cardinalities])
        self.numerical_bn = nn.BatchNorm1d(num_numerical) if num_numerical > 0 else None
        
        embedding_output_size = sum(emb.embedding_dim for emb in self.embeddings)
        input_size = num_numerical + embedding_output_size
        
        if input_size == 0:
            raise ValueError("Model initialized with 0 input features.")
        
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x_emb = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        
        if self.numerical_bn is not None and x_num.size(1) > 0:
            numerical_out = self.numerical_bn(x_num) if self.training and x_num.size(0) > 1 else x_num
        else:
            numerical_out = x_num

        x = torch.cat([numerical_out] + x_emb, dim=1) if x_emb else numerical_out
        
        x = self.fc1(x)
        x = self.relu1(x)
        
        if self.training and x.size(0) == 1:
            pass
        else:
            x = self.bn1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# --- Flower Client Implementation ---
class FLClient(fl.client.NumPyClient):
    """Flower client for credit risk prediction with PQC."""
    def __init__(self, model: nn.Module, server_pubkey: bytes, client_id: int, numerical_cols: List[str], categorical_cols: List[str], global_label_encoders: Dict[str, LabelEncoder], client_dilithium_priv_key: bytes, client_dilithium_pub_key: bytes, client_falcon_priv_key: bytes, client_falcon_pub_key: bytes):
        self.model = model.to(device)
        self.server_pubkey = server_pubkey
        self.client_id = client_id
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.client_dilithium_priv_key = client_dilithium_priv_key
        self.client_dilithium_pub_key = client_dilithium_pub_key
        self.client_falcon_priv_key = client_falcon_priv_key
        self.client_falcon_pub_key = client_falcon_pub_key

        train_ds, test_ds = load_client_data_internal(self.client_id, self.numerical_cols, self.categorical_cols, global_label_encoders)
        
        train_batch_size = 1 if len(train_ds) == 0 else min(len(train_ds), 32)
        test_batch_size = 1 if len(test_ds) == 0 else min(len(test_ds), 32)

        self.train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
        self.test_loader = DataLoader(test_ds, batch_size=test_batch_size)
        
        logger.debug(f"Client {self.client_id} initialized. Train: {len(self.train_loader.dataset)}, Test: {len(self.test_loader.dataset)}")

    def get_parameters(self, config: Dict[str, str] = {}) -> List[np.ndarray]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict({k: torch.tensor(v).to(device) for k, v in state_dict.items()})

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict[str, Union[bytes, bool, int]]]:
        self.set_parameters(parameters)
        self.model.train()

        if len(self.train_loader.dataset) == 0:
            logger.warning(f"Client {self.client_id} has no training data. Skipping.")
            return self.get_parameters(), 0, {"trained_successfully": False, "encrypted_data": pickle.dumps({}), "client_id": self.client_id}

        logger.debug(f"Client {self.client_id} starting training...")
        
        for x_num, x_cat, y in self.train_loader:
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(x_num, x_cat)
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            self.optimizer.step()
        
        logger.debug(f"Client {self.client_id} training finished.")
        weights = self.get_parameters()

        encrypted_payload = encrypt_model(weights, self.server_pubkey)
        data_to_sign_bytes = pickle.dumps(encrypted_payload)
        dilithium_signature = sign_data(data_to_sign_bytes, self.client_dilithium_priv_key, algorithm="Dilithium2")
        falcon_signature = sign_data(data_to_sign_bytes, self.client_falcon_priv_key, algorithm="Falcon-512")
        
        full_payload = {
            "encrypted_data": encrypted_payload,
            "dilithium_signature": dilithium_signature,
            "dilithium_public_key": self.client_dilithium_pub_key,
            "falcon_signature": falcon_signature,
            "falcon_public_key": self.client_falcon_pub_key,
        }
        
        payload_bytes = pickle.dumps(full_payload)

        return self.get_parameters(), len(self.train_loader.dataset), {
            "encrypted_signed_data_with_falcon": payload_bytes,
            "trained_successfully": True,
            "client_id": self.client_id
        }

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict[str, float]]:
        self.set_parameters(parameters)
        self.model.eval()

        loss, correct, total = 0.0, 0, 0
        
        if len(self.test_loader.dataset) == 0:
            logger.warning(f"Client {self.client_id} has no test data.")
            return 0.0, 0, {"accuracy": 0.0, "loss": 0.0}

        try:
            with torch.no_grad():
                for i, (x_num, x_cat, y) in enumerate(self.test_loader):
                    x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                    y_pred = self.model(x_num, x_cat)
                    batch_loss = self.loss_fn(y_pred, y).item()
                    loss += batch_loss * y.size(0)
                    pred = (y_pred > 0.5).float()
                    correct += (pred == y).sum().item()
                    total += y.size(0)
        except Exception as e:
            logger.error(f"CRITICAL ERROR in client {self.client_id} evaluation: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, 0, {"accuracy": 0.0, "loss": 0.0}

        acc = correct / total if total > 0 else 0.0
        final_loss = loss / total if total > 0 else 0.0
        
        logger.debug(f"Client {self.client_id} EVALUATE returning: Loss={final_loss:.4f}, Total={total}, Acc={acc:.4f}")
        return final_loss, total, {"accuracy": acc, "loss": final_loss}

# --- Flower Strategy Implementation (Server-side) ---
class LoggingStrategy(fl.server.strategy.FedAvg):
    """Custom Flower strategy with PQC decryption and verification."""
    def __init__(self, server_secret: bytes, client_dilithium_pub_keys_map: Dict[int, bytes], client_falcon_pub_keys_map: Dict[int, bytes], **kwargs):
        super().__init__(**kwargs)
        self.server_secret = server_secret
        self.client_dilithium_pub_keys_map = client_dilithium_pub_keys_map
        self.client_falcon_pub_keys_map = client_falcon_pub_keys_map

    def aggregate_fit(self, server_round: int, results: List[Tuple], failures: List) -> Tuple[Union[fl.common.Parameters, None], Dict]:
        logger.info(f"Aggregating results for round {server_round}...")
        
        decrypted_weights_list = []
        num_examples_list = []

        for client_proxy, fit_res in results:
            client_id = fit_res.metrics.get("client_id", "UNKNOWN")
            
            try:
                if not fit_res.metrics.get("trained_successfully", False):
                    logger.info(f"Client {client_id} did not train. Skipping.")
                    continue
                
                if "encrypted_signed_data_with_falcon" not in fit_res.metrics:
                    logger.error(f"Client {client_id} missing payload. Skipping.")
                    continue
                
                full_payload = pickle.loads(fit_res.metrics["encrypted_signed_data_with_falcon"])

                encrypted_payload = full_payload.get("encrypted_data")
                dilithium_signature = full_payload.get("dilithium_signature")
                dilithium_pub_key_from_payload = full_payload.get("dilithium_public_key")
                falcon_signature = full_payload.get("falcon_signature")
                falcon_pub_key_from_payload = full_payload.get("falcon_public_key")

                if not all([encrypted_payload, dilithium_signature, dilithium_pub_key_from_payload, falcon_signature, falcon_pub_key_from_payload]):
                    logger.error(f"Incomplete payload from client {client_id}. Skipping.")
                    continue
                
                data_to_verify = pickle.dumps(encrypted_payload)
                
                expected_dilithium_pub_key = self.client_dilithium_pub_keys_map.get(int(client_id))
                is_dilithium_valid = False
                if expected_dilithium_pub_key and expected_dilithium_pub_key == dilithium_pub_key_from_payload:
                    is_dilithium_valid = verify_signature(data_to_verify, dilithium_signature, dilithium_pub_key_from_payload, algorithm="Dilithium2")
                
                expected_falcon_pub_key = self.client_falcon_pub_keys_map.get(int(client_id))
                is_falcon_valid = False
                if expected_falcon_pub_key and expected_falcon_pub_key == falcon_pub_key_from_payload:
                    is_falcon_valid = verify_signature(data_to_verify, falcon_signature, falcon_pub_key_from_payload, algorithm="Falcon-512")

                if not is_dilithium_valid or not is_falcon_valid:
                    logger.error(f"Invalid signature from client {client_id} (D:{is_dilithium_valid}, F:{is_falcon_valid}). Skipping.")
                    continue
                
                logger.debug(f"✅ Signatures verified for client {client_id}")
                decrypted_weights = decrypt_model(encrypted_payload, self.server_secret)
                
                if not decrypted_weights:
                    logger.error(f"Decryption failed for client {client_id}. Skipping.")
                    continue
                
                decrypted_weights_list.append(decrypted_weights)
                num_examples_list.append(fit_res.num_examples)
                logger.debug(f"✅ Decryption succeeded for client {client_id}")

            except Exception as e:
                logger.error(f"❌ Processing failed for client {client_id}: {e}")
                import traceback
                traceback.print_exc()

        if not decrypted_weights_list:
            logger.error("🚨 No valid weights from any client. Returning initial parameters.")
            return super().aggregate_fit(server_round, [], failures)

        total_examples = sum(num_examples_list)
        if total_examples == 0:
            logger.error("Total examples from clients is zero. Cannot aggregate.")
            return super().aggregate_fit(server_round, [], failures)
            
        aggregated_weights = []
        first_client_weights_len = len(decrypted_weights_list[0])
        if not all(len(w) == first_client_weights_len for w in decrypted_weights_list):
            logger.error("Inconsistent weight lengths from clients. Cannot aggregate.")
            return super().aggregate_fit(server_round, [], failures)

        for i in range(first_client_weights_len):
            layer_weights = [client_weights[i] for client_weights in decrypted_weights_list]
            weighted_sum_layer = np.sum([w * n for w, n in zip(layer_weights, num_examples_list)], axis=0)
            aggregated_layer = weighted_sum_layer / total_examples
            aggregated_weights.append(aggregated_layer)

        if not aggregated_weights:
            logger.error("Aggregated weights list empty. Returning initial parameters.")
            return super().aggregate_fit(server_round, [], failures)

        logger.info(f"✅ Aggregated weights computed successfully for round {server_round}")
        return fl.common.ndarrays_to_parameters(aggregated_weights), {}

    def aggregate_evaluate(self, rnd: int, results: List[Tuple], failures: List) -> Tuple[Union[float, None], Dict]:
        if not results:
            logger.warning(f"No evaluation results for round {rnd}.")
            return 0.0, {}

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)
        
        accuracy_list = [r.metrics["accuracy"] for _, r in results if "accuracy" in r.metrics]
        losses_list = [r.metrics["loss"] for _, r in results if "loss" in r.metrics]

        if accuracy_list:
            avg_accuracy = sum(accuracy_list) / len(accuracy_list)
            logger.info(f"Round {rnd} accuracy: {avg_accuracy:.4f}")
            round_accuracy.append(avg_accuracy)
        
        if losses_list:
            avg_loss = sum(losses_list) / len(losses_list)
            std_loss = np.std(losses_list) if len(losses_list) > 1 else 0.0
            logger.info(f"Round {rnd} loss: {avg_loss:.4f} (std: {std_loss:.4f})")
            round_loss.append(avg_loss)
            round_std_loss.append(std_loss)

        return aggregated_loss, aggregated_metrics

def client_fn(cid: str) -> FLClient:
    """Creates a Flower client."""
    client_id = int(cid)
    model = CreditRiskNet(num_numerical=len(numerical_cols), cat_cardinalities=cat_cardinalities)
    return FLClient(
        model, 
        server_pubkey, 
        client_id, 
        numerical_cols, 
        categorical_cols,
        global_label_encoders, 
        client_dilithium_keys[client_id]["private_key"],
        client_dilithium_keys[client_id]["public_key"],
        client_falcon_keys[client_id]["private_key"],
        client_falcon_keys[client_id]["public_key"]
    )

# --- Main Simulation Execution ---
if __name__ == "__main__":
    # --- Step 1: Fit Global Label Encoders ---
    all_data_for_encoders = pd.DataFrame()
    for i in range(NUM_CLIENTS):
        path = os.path.join(DATA_DIR, f"client_{i}.csv")
        if os.path.exists(path):
            try:
                temp_df = pd.read_csv(path)
                all_data_for_encoders = pd.concat([all_data_for_encoders, temp_df], ignore_index=True)
            except Exception as e:
                logger.warning(f"Could not load {path} for encoder fitting: {e}")
    
    if all_data_for_encoders.empty:
        raise ValueError("No data loaded to fit global label encoders. Cannot proceed.")
    
    for col in categorical_cols:
        if col in all_data_for_encoders.columns:
            le = LabelEncoder()
            le.fit(all_data_for_encoders[col].astype(str).unique())
            global_label_encoders[col] = le
            logger.info(f"Global LabelEncoder for '{col}' fitted with {len(le.classes_)} classes.")
        else:
            raise ValueError(f"Categorical column '{col}' not found in global data.")

    cat_cardinalities = [len(global_label_encoders[col].classes_) for col in categorical_cols]

    # --- Step 2: Generate Server and Client PQC Keys ---
    with oqs.KeyEncapsulation("Kyber512") as kem:
        server_pubkey = kem.generate_keypair()
        server_secret_key = kem.export_secret_key()
    logger.info("Server Kyber512 KEM key pair generated.")

    client_dilithium_keys = {}
    client_falcon_keys = {}
    for i in range(NUM_CLIENTS):
        dilithium_pub, dilithium_priv = generate_dilithium_key_pair("Dilithium2")
        falcon_pub, falcon_priv = generate_falcon_key_pair("Falcon-512")
        
        client_dilithium_keys[i] = {"public_key": dilithium_pub, "private_key": dilithium_priv}
        client_falcon_keys[i] = {"public_key": falcon_pub, "private_key": falcon_priv}
    logger.info(f"Generated Dilithium2 and Falcon-512 keys for {NUM_CLIENTS} clients.")

    # --- Step 3: Configure and Start Flower Simulation ---
    # --- THIS IS THE MODIFIED SECTION ---
    start_time = time.time()
    strategy = LoggingStrategy(
        server_secret=server_secret_key,
        client_dilithium_pub_keys_map={k: v["public_key"] for k, v in client_dilithium_keys.items()},
        client_falcon_pub_keys_map={k: v["public_key"] for k, v in client_falcon_keys.items()},
        fraction_fit=1.0, # Use 100% of clients for training
        fraction_evaluate=1.0, # Use 100% of clients for evaluation
        min_fit_clients=NUM_CLIENTS, # Require all clients for training
        min_evaluate_clients=NUM_CLIENTS, # Require all clients for evaluation
        min_available_clients=NUM_CLIENTS, # Wait for all clients to be available
    )
    # --- END MODIFIED SECTION ---
    
    logger.info("Starting Flower simulation...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
        client_resources={"num_cpus": 1},
        ray_init_args={}
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("Flower simulation finished.")

    # --- Step 4: Log and Plot Results ---
    logger.info(f"Total simulation time: {elapsed_time:.2f} seconds")
    num_completed_rounds = len(round_accuracy)
    
    logger.info(f"Aggregation History: {history}")
    logger.info(f"Round Accuracy History: {round_accuracy}")
    logger.info(f"Round Loss History: {round_loss}")

    # Plotting results
    if round_accuracy:
        rounds = range(1, num_completed_rounds + 1)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(rounds, round_accuracy, marker='o', linestyle='-', color='b')
        plt.title('Global Model Accuracy per Round')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.errorbar(rounds, round_loss, yerr=round_std_loss, fmt='-o', color='r', ecolor='lightgray', capsize=4)
        plt.title('Global Model Loss per Round (with Std Dev)')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    else:
        logger.warning("No data to plot accuracy and loss.")
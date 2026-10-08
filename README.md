# PQC-FL: Decentralized Post-Quantum Federated Learning for Banking Systems

> **Status note (October 2026).** The paper describes sub-aggregate servers and adaptive signature selection. The original experiment scripts implement a simpler version: a single aggregator, with every client signing using both Dilithium2 and Falcon-512. `pqc_fl_fixed.py` is a later rework that implements the design the paper describes. Also, the 88% accuracy reported in the paper equals the majority-class rate on this dataset (about 87.7% of records are non-defaults), so it is not evidence of predictive skill.

## Code

| File | What it is |
| --- | --- |
| `original/50client.py` | The original experiment script: 50 clients, Kyber512 + Fernet encryption, Dilithium2 and Falcon-512 signatures, single FedAvg aggregator. Data path is hard-coded for a local machine. |
| `pqc_fl_fixed.py` | 2026 rework: adaptive signature choice, three sub-aggregate servers, a tamper test, precision/recall/F1 reporting, no plaintext weights returned, no pickle on network data, and a `plain` mode for measuring overhead. |
| `requirements.txt` | Python dependencies. |

## Running the rework

```
pip install -r requirements.txt
set PQC_FL_DATA_DIR=path\to\client_partitions   # folder with client_0.csv ... client_49.csv
python pqc_fl_fixed.py                           # PQC mode
set PQC_FL_MODE=plain && python pqc_fl_fixed.py  # same pipeline, no crypto
```

Each run writes `results_<mode>.json` with per-round accuracy, precision, recall, F1, rejected updates, payload size and timing. The dataset is not included. Each client file needs the columns Income, Age, Experience, CURRENT_JOB_YRS, CURRENT_HOUSE_YRS, Married/Single, House_Ownership, Car_Ownership, Profession, CITY, STATE and Risk_Flag.

A smoke test of the rework (12 clients, 6 rounds, not the paper's setup) showed PQC payloads 81% larger than plain, server time per round 2.1 times higher, and about 5 ms of client crypto per round. These are not the paper's reported figures.

---

# Paper summary

The summary below is the paper's, unchanged.

## Overview
In today’s digital banking environment, protecting sensitive consumer data is paramount as institutions increasingly rely on AI-driven models for credit risk assessment. Classical cryptographic methods such as RSA and ECC are vulnerable to quantum attacks via Shor’s algorithm, which can break these schemes in polynomial time.  

To address this, we propose **PQC-FL**, a federated learning framework enhanced with **decentralized post-quantum cryptography**. Our system integrates a hybrid cryptographic scheme combining:

- **Kyber** for quantum-safe key exchange  
- **Dilithium** and **Falcon** for quantum-safe digital signatures  
- **AES** for efficient symmetric encryption  

These primitives align with **NIST’s PQC standards**, ensuring confidentiality, integrity, and authentication in a future-proof manner.

---

## Architecture
Unlike typical federated learning setups that rely on a single central server, PQC-FL employs a **decentralized three-server architecture**:
- **Clients** train local models on non-IID customer data.  
- **Sub-aggregate servers** receive encrypted updates, verify signatures, and perform local aggregation.  
- **Global aggregation server** combines results from sub-servers to update the global model.  

This design eliminates single points of failure, enhances fault tolerance, and improves trustworthiness.

---

## Methodology
- **Confidentiality**: Hybrid encryption using Kyber512 KEM + AES symmetric encryption.  
- **Integrity & Authenticity**: Adaptive digital signatures with Dilithium2 and Falcon-512.  
- **Attack Simulation**: MITM poisoning attacks tested to validate tamper detection.  
- **Implementation**: Built using Flower (FL framework), PyTorch (model training), and liboqs (PQC algorithms).  

---

## Results
Our experiments on a non-IID Indian credit risk dataset (50 clients, 15–20 rounds) produced the following outcomes:

- **Model Accuracy**:  
  - PQC-FL achieved **88% accuracy**, identical to classical and non-crypto baselines.  
  - **0% degradation** in learning utility with PQC integration.

- **Security Effectiveness**:  
  - PQC-FL detected and rejected **100% of malicious updates** in simulated MITM attacks.  
  - Classical and non-crypto systems failed to detect tampering.

- **Performance Overheads**:  
  - **Compute overhead**: PQC ~10⁻² s per round vs classical ~10⁻³ s.  
  - **Communication cost**: PQC ~0.14 MB/client vs 0.10 MB baseline (~40% increase).  
  - **Signature sizes**: Dilithium2 (2420 bytes), Falcon-512 (666 bytes), ECDSA-P256 (~72 bytes).  

---

## Key Takeaways
- **Quantum-resilient security** without sacrificing model accuracy.  
- **Absolute tamper detection** with PQC signatures.  
- **Manageable trade-offs**: ~10× compute overhead and ~40% communication increase.  
- **Futureproof design**: Regulation-compliant, privacy-preserving, and robust against single-server failures.

---

## Future Work
- Optimize PQC performance with hardware acceleration.  
- Integrate Differential Privacy or Homomorphic Encryption for stronger guarantees.  
- Explore Byzantine fault-tolerant aggregation.  
- Scale to larger models (e.g., LLMs) and domains like healthcare.  

---

## Citation
If you use this work, please cite:  
**Pranav Manoj P, Chandana Lakya, Dr. P Kavitha.**  
*Future-Proofing Financial AI: A Decentralized Approach to Quantum-Resistant Federated Learning.* PES University, Bengaluru.

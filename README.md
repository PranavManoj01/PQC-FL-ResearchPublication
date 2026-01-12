# PQC-FL: Decentralized Post-Quantum Federated Learning for Banking Systems

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

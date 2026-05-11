# FedMIA: Privacy and Membership Inference Attack in Federated Learning

**Description**: 
Federated Learning (FL) [2] has emerged as a promising paradigm for training machine learning models on decentralised data. Rather than pooling raw data on a central server, FL clients train locally and upload only model updates (gradients or parameters). While this approach preserves data locality, it does not guarantee privacy—the shared updates can leak sensitive membership information.
Membership Inference Attacks (MIAs) seek to determine whether a particular data sample (x, y) was used to train a target client’s local model. In the FL setting, MIAs are typically performed by a semi-honest server. it faithfully runs the protocol but inspects all uploads to infer membership.

**Course**: 
**Framework**: [Flower (flwr)](https://flower.ai) — gRPC mode


---
## 📄 Data Set Link

 **https://www.cs.toronto.edu/~kriz/cifar.html**

###  Attacks

1. Baseline Attacks 
2. Blackbox-Loss — raw loss threshold (Yeom et al. 2018)
3. Avg-Cosine — temporal average of gradient cosine similarity (Li et al. 2022)
4. FedMIA-I (proposed) — uses model loss as measurement
5. FedMIA-II (proposed) — uses gradient cosine similarity as measurement
6. Grad-Cosine
7. Grad-Diff
8. Loss-Series 
9. Grad-Norm 


## 🎯 Category Metrics (Cat. 1 – Privacy & Inference)

- Attack Success Rate (TPR @ FPR = 0.1%)
- Membership Inference AUC
- Reconstruction MSE (for generative tasks)

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/CS6206-Wireless-Networks-Term-Paper
cd CS6206-Wireless-Networks-Term-Paper
```

### 2. Install Dependencies
```bash
pip install flwr==1.8.0 torch torchvision scikit-learn matplotlib seaborn pyyaml scipy grpcio
```
> ⚠️ Do **NOT** use `flwr[simulation]` — it requires Ray which doesn't work on Windows.  
> Plain `flwr` uses gRPC directly, which works everywhere.

### 3. Run the Experiment (Recommended — one command)

```bash
# FedAvg baseline
python src/run_all.py --config configs/fedavg_baseline.yaml

# FedMIA attack on CIFAR-100
python src/run_all.py --config configs/fedmia_cifar100.yaml

# FedMIA attack on MNIST
python src/run_all.py --config configs/fedmia_mnist.yaml
```

`run_all.py` automatically:
- Starts the Flower gRPC **server** in a background thread
- Spawns one **client subprocess per client** on localhost
- Collects metrics, saves plots, runs the FedMIA attack after training

---

## 🔧 Manual Mode (Advanced)

You can also run server and clients separately in different terminals:

**Terminal 1 — Server:**
```bash
python src/server.py --config configs/fedmia_cifar100.yaml
```

**Terminals 2–11 — One per client (10 clients total):**
```bash
python src/client.py --cid 0 --config configs/fedmia_cifar100.yaml
python src/client.py --cid 1 --config configs/fedmia_cifar100.yaml
# ... up to cid 9
```

---

## 📊 How Flower gRPC Works (Architecture)

```
┌────────────────────────────────────────────────────────┐
│                  Server (server.py)                     │
│  fl.server.start_server(address="0.0.0.0:8080")        │
│  Strategy: FedAvgWithMIA                               │
│    - aggregate_fit()  → FedAvg aggregation              │
│    - captures gradient deltas per client per round      │
│    - evaluates global model centrally after each round  │
│    - runs FedMIA attack after all rounds complete       │
└───────────────────┬────────────────────────────────────┘
                    │  gRPC (localhost:8080)
        ┌───────────┼───────────┐
        ▼           ▼           ▼
  client.py     client.py   client.py
  (cid=0)       (cid=1)     (cid=N)
  NumPyClient   NumPyClient NumPyClient
  fit()         fit()       fit()
  evaluate()    evaluate()  evaluate()
```

---

## 📈 Outputs (saved to `results/`)

| File | Description |
|------|-------------|
| `*_accuracy.png` | Global accuracy vs. communication rounds |
| `*_loss.png` | Global loss vs. communication rounds |
| `*_mia_probe.png` | FedMIA probe score vs. rounds |
| `*_member_dist.png` | Member vs. non-member score distribution |
| `*_metrics.csv` | Per-round metrics (acc, loss, MIA score, comm cost) |
| `*_summary.csv` | Mandatory summary table (method, dataset, AUC, TPR@FPR) |

---

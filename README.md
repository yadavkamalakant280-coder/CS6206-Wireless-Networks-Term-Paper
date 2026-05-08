# FedMIA: Privacy and Membership Inference Attack in Federated Learning

**Category**: Cat. 1 – Privacy & Inference  
**Course**: Federated Learning Term Paper | Jan–April 2026 
**Framework**: [Flower (flwr)](https://flower.ai) — gRPC mode (no Ray, works on Windows)

---

## 📄 Reference Papers

1. **FedMIA**: *An Effective Membership Inference Attack Exploiting "All for One" Principle in Federated Learning* — Zhu et al., CVPR 2025  
   - GitHub: https://github.com/Liar-Mask/FedMIA

---

## 🎯 Category Metrics (Cat. 1 – Privacy & Inference)

- Attack Success Rate (TPR @ FPR = 0.1%)
- Membership Inference AUC
- Reconstruction MSE (for generative tasks)

---

## 📁 Repository Structure

```
fedmia_repo/
├── README.md
├── requirements.txt
├── configs/
│   ├── fedavg_baseline.yaml       ← universal FedAvg baseline
│   ├── fedmia_cifar10.yaml
│   ├── fedmia_cifar100.yaml
│   └── fedmia_mnist.yaml
├── src/
│   ├── run_all.py                 ← ✅ SINGLE COMMAND launcher (start here)
│   ├── server.py                  ← Flower gRPC server + FedMIA attack
│   ├── client.py                  ← Flower gRPC NumPyClient
│   ├── model.py                   ← SimpleCNN, ResNet18, AlexNet
│   ├── data.py                    ← Dirichlet Non-IID partitioning
│   ├── utils.py                   ← seed, logging, 300 DPI plots
│   └── attack/
│       ├── __init__.py
│       ├── fedmia.py              ← FedMIA 3-step algorithm (Eq.7–12)
│       └── metrics.py             ← AUC, TPR@FPR, Attack Success Rate, MSE
├── results/                       ← auto-populated CSV + PNG outputs
└── report/
    └── (place final PDF here)
```

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



## 📌 Notes

- Seeds fixed: `random.seed(42)`, `numpy.seed(42)`, `torch.manual_seed(42)`
- Framework: Flower (flwr) v1.8.0 — gRPC mode
- Python: 3.9+ (3.10 recommended)
- Deep Learning: PyTorch >= 2.0
"# CS6206-Wireless-Networks-Term-Paper" 
"# CS6206-Wireless-Networks-Term-Paper" 
"# CS6206-Wireless-Networks-Term-Paper" 
"# CS6206-Wireless-Networks-Term-Paper" 

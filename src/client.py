"""
src/client.py
Flower gRPC client — works on Windows/Linux/Mac with plain `pip install flwr`.
NO Ray, NO fl.simulation.

Run one process per client:
    python src/client.py --cid 0 --config configs/fedavg_baseline.yaml
    python src/client.py --cid 1 --config configs/fedavg_baseline.yaml
    ...

Or use run_all.py which launches everything automatically.
"""

import argparse
import os
import sys
import copy
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
import flwr as fl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import get_model
from data import build_partitions, get_client_loaders
from utils import set_seed, load_config


# ──────────────────────────────────────────────────────────────────────────────
# Flower NumPyClient
# ──────────────────────────────────────────────────────────────────────────────

class FedMIAClient(fl.client.NumPyClient):
    """
    Standard Flower NumPyClient.
    - get_parameters : send current weights to server
    - set_parameters : load weights received from server
    - fit            : local training (FedAvg style)
    - evaluate       : local evaluation on training data
    """

    def __init__(
        self,
        cid: int,
        train_dataset,
        indices: List[int],
        cfg: dict,
        device: torch.device,
    ):
        self.cid = cid
        self.cfg = cfg
        self.device = device

        self.train_loader = get_client_loaders(
            train_dataset,
            indices,
            batch_size=cfg["federated"]["local_batch_size"],
            shuffle=True,
        )

        self.model = get_model(
            cfg["model"]["architecture"],
            cfg["data"]["num_classes"],
            cfg["data"]["dataset"],
        ).to(device)

        self.criterion = nn.CrossEntropyLoss()

    # ── Parameter helpers ──────────────────────────────────────────────────

    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    # ── Fit (local training) ───────────────────────────────────────────────

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:

        self.set_parameters(parameters)

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg["optimizer"]["lr"],
            momentum=self.cfg["optimizer"]["momentum"],
        )

        local_epochs = self.cfg["federated"]["local_epochs"]
        self.model.train()
        total_loss, num_samples = 0.0, 0

        for _ in range(local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
                num_samples += x.size(0)

        avg_loss = total_loss / max(num_samples, 1)
        print(f"  [Client {self.cid}] local epochs done | loss={avg_loss:.4f} | samples={num_samples}")

        return self.get_parameters(config={}), num_samples, {"loss": avg_loss}

    # ── Evaluate ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:

        self.set_parameters(parameters)
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)
                total_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += x.size(0)

        accuracy = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)
        return avg_loss, total, {"accuracy": accuracy}


# ──────────────────────────────────────────────────────────────────────────────
# Entry point — connect to server via gRPC
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid",    type=int,   required=True,  help="Client ID (0-indexed)")
    parser.add_argument("--config", type=str,   required=True,  help="Path to YAML config")
    parser.add_argument("--server", type=str,   default="127.0.0.1:8080", help="Server address (host:port)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"] + args.cid)   # different seed per client
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build partitions (all clients share same deterministic partitioning)
    train_ds, _, partitions = build_partitions(cfg)
    indices = partitions[args.cid]

    client = FedMIAClient(
        cid=args.cid,
        train_dataset=train_ds,
        indices=indices,
        cfg=cfg,
        device=device,
    )

    print(f"[Client {args.cid}] Connecting to server at {args.server} ...")
    # Suppress deprecation noise in logs
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    fl.client.start_client(
        server_address=args.server,
        client=client.to_client(),
        grpc_max_message_length=536_870_912,   # 512 MB — must match server
    )


if __name__ == "__main__":
    main()

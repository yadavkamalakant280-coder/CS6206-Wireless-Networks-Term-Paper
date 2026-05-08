"""
src/server.py
Flower gRPC server — works on Windows/Linux/Mac with plain `pip install flwr`.
NO Ray, NO fl.simulation.

Usage (run this FIRST, then launch clients):
    python src/server.py --config configs/fedavg_baseline.yaml

Or use run_all.py which launches everything automatically in subprocesses.
"""

import argparse
import os
import sys
import copy
import math
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    Metrics,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import get_model
from data import build_partitions, get_test_loader
from utils import (
    set_seed, load_config, MetricLogger,
    plot_accuracy_vs_rounds, plot_loss_vs_rounds,
    plot_attack_metric_vs_rounds, plot_member_nonmember_dist,
    save_summary_table,
)
from attack.fedmia import FedMIA


# ──────────────────────────────────────────────────────────────────────────────
# Weighted-average metric aggregation
# ──────────────────────────────────────────────────────────────────────────────

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total = sum(n for n, _ in metrics)
    acc   = sum(n * m.get("accuracy", 0.0) for n, m in metrics) / max(total, 1)
    loss  = sum(n * m.get("loss",     0.0) for n, m in metrics) / max(total, 1)
    return {"accuracy": acc, "loss": loss}


# ──────────────────────────────────────────────────────────────────────────────
# Custom FedAvg strategy that:
#   1. Captures per-client gradient deltas each round  (for FedMIA)
#   2. Evaluates global model centrally after each round
#   3. Logs metrics to CSV + plots
# ──────────────────────────────────────────────────────────────────────────────

class FedAvgWithMIA(FedAvg):

    def __init__(
        self,
        cfg: dict,
        global_model: nn.Module,
        test_loader,
        criterion: nn.Module,
        device: torch.device,
        attacker: Optional[FedMIA],
        member_samples: list,
        nonmember_samples: list,
        logger: MetricLogger,
        save_dir: str,
        exp_name: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cfg               = cfg
        self.global_model      = global_model
        self.test_loader       = test_loader
        self.criterion         = criterion
        self.device            = device
        self.attacker          = attacker
        self.member_samples    = member_samples
        self.nonmember_samples = nonmember_samples
        self.logger            = logger
        self.save_dir          = save_dir
        self.exp_name          = exp_name

        # Tracking
        self.round_nums:         List[int]   = []
        self.acc_history:        List[float] = []
        self.loss_history:       List[float] = []
        self.mia_score_history:  List[float] = []
        self.all_round_updates:  List[Dict]  = []
        self.convergence_round:  Optional[int] = None
        self.total_comm_mb:      float = 0.0
        self._model_mb = sum(p.numel() for p in global_model.parameters()) * 4 / 1e6

        # Probe sample for per-round lightweight MIA score
        self._probe = member_samples[0] if member_samples else None

    # ── Central evaluation after every round ─────────────────────────────

    def _eval_global(self, parameters: Parameters) -> Tuple[float, float]:
        """Load aggregated params into global model and evaluate on test set."""
        ndarrays = parameters_to_ndarrays(parameters)
        params_dict = zip(self.global_model.state_dict().keys(), ndarrays)
        state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.global_model.load_state_dict(state_dict, strict=True)
        self.global_model.eval()

        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out  = self.global_model(x)
                loss = self.criterion(out, y)
                total_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += x.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct   / max(total, 1) * 100.0
        return avg_loss, accuracy

    # ── Override aggregate_fit ────────────────────────────────────────────

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ):
        # Capture per-client flat gradient tensors for FedMIA
        round_updates: Dict[int, torch.Tensor] = {}
        for proxy, fit_res in results:
            cid    = proxy.cid
            arrays = parameters_to_ndarrays(fit_res.parameters)
            flat   = torch.cat([torch.tensor(a.copy()).flatten() for a in arrays])
            round_updates[cid] = flat

        self.all_round_updates.append(round_updates)
        k = len(results)
        self.total_comm_mb += self._model_mb * k * 2   # upload + download

        # Standard FedAvg aggregation
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_params is None:
            return aggregated_params, aggregated_metrics

        # ── Central evaluation ─────────────────────────────────────────────
        val_loss, val_acc = self._eval_global(aggregated_params)

        if self.convergence_round is None and val_acc >= 80.0:
            self.convergence_round = server_round

        self.round_nums.append(server_round)
        self.acc_history.append(val_acc)
        self.loss_history.append(val_loss)

        metrics_row: Dict = {
            "val_acc":      round(val_acc,  4),
            "val_loss":     round(val_loss, 4),
            "comm_cost_mb": round(self.total_comm_mb, 2),
        }

        # ── Per-round lightweight MIA probe ───────────────────────────────
        if self.attacker is not None and self._probe is not None:
            score_t = self.attacker.update_round(
                round_t=server_round,
                client_updates=round_updates,
                global_model=self.global_model,
                target_sample=self._probe,
                criterion=self.criterion,
                device=self.device,
            )
            self.mia_score_history.append(score_t)
            metrics_row["mia_probe_score"] = round(score_t, 4)

        self.logger.log(server_round, metrics_row)

        plot_every = self.cfg["logging"]["plot_every"]
        num_rounds = self.cfg["federated"]["num_rounds"]
        if server_round % plot_every == 0 or server_round == 1 or server_round == num_rounds:
            mia_str = (f" | MIA={metrics_row.get('mia_probe_score', 0):.3f}"
                       if self.attacker else "")
            print(
                f"  Round {server_round:>4}/{num_rounds} | "
                f"Acc={val_acc:6.2f}% | Loss={val_loss:.4f} | "
                f"Comm={self.total_comm_mb:7.1f} MB{mia_str}"
            )

        return aggregated_params, aggregated_metrics

    # ── Finalise: full MIA evaluation + plots ─────────────────────────────

    def finalise(self):
        exp  = self.exp_name
        sdir = self.save_dir

        plot_accuracy_vs_rounds(
            self.round_nums, {"FedAvg": self.acc_history},
            save_path=os.path.join(sdir, f"{exp}_accuracy.png"),
        )
        plot_loss_vs_rounds(
            self.round_nums, {"FedAvg": self.loss_history},
            save_path=os.path.join(sdir, f"{exp}_loss.png"),
        )
        if self.mia_score_history:
            plot_attack_metric_vs_rounds(
                self.round_nums, {"FedMIA probe score": self.mia_score_history},
                save_path=os.path.join(sdir, f"{exp}_mia_probe.png"),
            )

        attack_results: Dict = {}
        if self.attacker is not None and self.all_round_updates:
            print("\n[Attack] Running full FedMIA batch evaluation …")
            attack_results = self.attacker.evaluate_batch(
                member_samples    = self.member_samples,
                nonmember_samples = self.nonmember_samples,
                all_round_updates = self.all_round_updates,
                global_model      = self.global_model,
                criterion         = self.criterion,
                device            = self.device,
            )
            print("[Attack] Final Metrics:")
            for k, v in attack_results.items():
                print(f"  {k}: {v:.4f}")

            plot_member_nonmember_dist(
                self.attacker.member_scores,
                self.attacker.nonmember_scores,
                save_path=os.path.join(sdir, f"{exp}_member_dist.png"),
                method_name="FedMIA",
            )

        # Mandatory summary table
        fpr_key = f"TPR@FPR={self.cfg['attack'].get('fpr_target', 0.001)}"
        final_acc  = self.acc_history[-1]  if self.acc_history  else 0.0
        final_loss = self.loss_history[-1] if self.loss_history else 0.0

        summary_rows = [{
            "Method":            "FedMIA" if self.attacker else "FedAvg",
            "Dataset":           self.cfg["data"]["dataset"],
            "#Clients":          self.cfg["federated"]["num_clients"],
            "#Rounds":           self.cfg["federated"]["num_rounds"],
            "Test Accuracy (%)": f"{final_acc:.2f}",
            "Convergence Round": self.convergence_round if self.convergence_round else "N/A",
            "Comm. Cost (MB)":   f"{self.total_comm_mb:.2f}",
            "AUC":               f"{attack_results.get('AUC', 0):.4f}" if attack_results else "N/A",
            "TPR@FPR=0.1%":      f"{attack_results.get(fpr_key, 0):.4f}" if attack_results else "N/A",
            "AttackSuccessRate": f"{attack_results.get('AttackSuccessRate', 0):.4f}" if attack_results else "N/A",
        }]
        save_summary_table(summary_rows, os.path.join(sdir, f"{exp}_summary.csv"))

        print(f"\n{'='*60}")
        print(f" Final Test Accuracy : {final_acc:.2f}%")
        print(f" Final Test Loss     : {final_loss:.4f}")
        print(f" Total Comm Cost     : {self.total_comm_mb:.2f} MB")
        if self.convergence_round:
            print(f" Converged at round  : {self.convergence_round}")
        print(f"{'='*60}")
        print(f"[Done] Results saved to: {os.path.abspath(sdir)}/")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fedavg_baseline.yaml")
    parser.add_argument("--server", type=str, default="0.0.0.0:8080",
                        help="Address to bind the gRPC server on")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = cfg["experiment"]["name"]
    save_dir = cfg["logging"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f" Experiment  : {exp_name}")
    print(f" Device      : {device}")
    print(f" Dataset     : {cfg['data']['dataset']}")
    print(f" Architecture: {cfg['model']['architecture']}")
    print(f" Clients     : {cfg['federated']['num_clients']}")
    print(f" Rounds      : {cfg['federated']['num_rounds']}")
    print("=" * 60)

    logger = MetricLogger(save_dir, exp_name)

    # ── Data (server needs test set + partitions for attack) ──────────────
    train_ds, test_ds, partitions = build_partitions(cfg)
    test_loader = get_test_loader(test_ds)
    criterion   = nn.CrossEntropyLoss()
    num_clients = cfg["federated"]["num_clients"]

    # ── Global model ──────────────────────────────────────────────────────
    global_model = get_model(
        cfg["model"]["architecture"],
        cfg["data"]["num_classes"],
        cfg["data"]["dataset"],
    ).to(device)
    init_params = ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in global_model.state_dict().items()]
    )

    # ── Attack setup ──────────────────────────────────────────────────────
    attack_enabled = cfg["attack"].get("enabled", False)
    attacker: Optional[FedMIA] = None
    member_samples, nonmember_samples = [], []

    if attack_enabled:
        target_cid = cfg["attack"]["target_client"]
        attacker = FedMIA(
            target_client_id = target_cid,
            measurement      = cfg["attack"]["measurement"],
            sigma_filter     = cfg["attack"]["sigma_filter"],
            threshold        = cfg["attack"]["threshold"],
            fpr_target       = cfg["attack"]["fpr_target"],
        )
        member_samples = [
            (train_ds[i][0], train_ds[i][1])
            for i in partitions[target_cid][:100]
        ]
        other_cid = (target_cid + 1) % num_clients
        nonmember_samples = [
            (train_ds[i][0], train_ds[i][1])
            for i in partitions[other_cid][:100]
        ]
        print(f"[Attack] FedMIA enabled | target_client={target_cid} | "
              f"measurement={cfg['attack']['measurement']}")

    # ── Strategy ──────────────────────────────────────────────────────────
    frac      = cfg["federated"]["client_fraction"]
    min_fit   = max(2, math.ceil(num_clients * frac))

    strategy = FedAvgWithMIA(
        cfg               = cfg,
        global_model      = global_model,
        test_loader       = test_loader,
        criterion         = criterion,
        device            = device,
        attacker          = attacker,
        member_samples    = member_samples,
        nonmember_samples = nonmember_samples,
        logger            = logger,
        save_dir          = save_dir,
        exp_name          = exp_name,
        # FedAvg base args
        fraction_fit                    = frac,
        fraction_evaluate               = frac,
        min_fit_clients                 = min_fit,
        min_evaluate_clients            = min_fit,
        min_available_clients           = num_clients,
        evaluate_metrics_aggregation_fn = weighted_average,
        fit_metrics_aggregation_fn      = weighted_average,
        initial_parameters              = init_params,
    )

    # ── Start Flower gRPC server ───────────────────────────────────────────
    # [::]:PORT works on Windows AND Linux; 0.0.0.0:PORT can fail on Windows
    port = args.server.split(":")[-1]
    bind_addr = f"[::]:{port}"
    print(f"\n[Server] Listening on {bind_addr}")
    print(f"[Server] Waiting for {num_clients} clients to connect …\n")

    fl.server.start_server(
        server_address=bind_addr,
        config=fl.server.ServerConfig(num_rounds=cfg["federated"]["num_rounds"]),
        strategy=strategy,
        grpc_max_message_length=536_870_912,   # 512 MB
    )
    # ── Post-training: plots + full MIA evaluation ────────────────────────
    strategy.finalise()


if __name__ == "__main__":
    main()

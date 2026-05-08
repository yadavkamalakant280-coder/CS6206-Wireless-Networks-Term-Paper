"""
src/utils.py
Utility functions: seed fixing, metric logging, plotting.
"""

import os
import random
import csv
import yaml
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility (required by course instructions)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# CSV metric logger
# ──────────────────────────────────────────────────────────────────────────────

class MetricLogger:
    """Logs per-round metrics to a CSV file."""

    def __init__(self, save_dir: str, experiment_name: str):
        os.makedirs(save_dir, exist_ok=True)
        self.path = os.path.join(save_dir, f"{experiment_name}_metrics.csv")
        self.rows: List[Dict] = []
        self.headers_written = False

    def log(self, round_num: int, metrics: Dict[str, Any]):
        row = {"round": round_num, **metrics}
        self.rows.append(row)
        # Write immediately so we don't lose data on crash
        mode = "a" if self.headers_written else "w"
        with open(self.path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not self.headers_written:
                writer.writeheader()
                self.headers_written = True
            writer.writerow(row)

    def get_series(self, key: str) -> List[float]:
        return [r[key] for r in self.rows if key in r]


# ──────────────────────────────────────────────────────────────────────────────
# Plotting (300 DPI, font ≥ 11pt, as required)
# ──────────────────────────────────────────────────────────────────────────────

PLOT_STYLE = {
    "figure.dpi": 300,
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 2.0,
}


def plot_accuracy_vs_rounds(
    rounds: List[int],
    series: Dict[str, List[float]],
    save_path: str,
    title: str = "Global Accuracy vs. Communication Rounds",
    ylabel: str = "Test Accuracy (%)",
):
    """Plot multiple accuracy curves on a single figure."""
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, values in series.items():
            ax.plot(rounds[: len(values)], values, label=label)
        ax.set_xlabel("Communication Round")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, format=Path(save_path).suffix.lstrip("."))
        plt.close()
    print(f"[Plot] Saved → {save_path}")


def plot_loss_vs_rounds(
    rounds: List[int],
    series: Dict[str, List[float]],
    save_path: str,
    title: str = "Global Loss vs. Communication Rounds",
):
    plot_accuracy_vs_rounds(rounds, series, save_path, title=title, ylabel="Loss")


def plot_attack_metric_vs_rounds(
    rounds: List[int],
    auc_series: Dict[str, List[float]],
    save_path: str,
    title: str = "Membership Inference AUC vs. Communication Rounds",
):
    plot_accuracy_vs_rounds(
        rounds, auc_series, save_path, title=title, ylabel="MIA AUC"
    )


def plot_iid_vs_noniid(
    results: Dict[str, Dict[str, float]],
    save_path: str,
    metric: str = "AUC",
):
    """
    Bar chart comparing IID vs Non-IID for multiple methods.
    results = { "IID": {"FedMIA": 0.89, ...}, "α=0.1": {...}, ... }
    """
    with plt.rc_context(PLOT_STYLE):
        partitions = list(results.keys())
        methods = list(next(iter(results.values())).keys())
        x = np.arange(len(methods))
        width = 0.8 / len(partitions)

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, part in enumerate(partitions):
            vals = [results[part].get(m, 0.0) for m in methods]
            ax.bar(x + i * width, vals, width, label=part)

        ax.set_xticks(x + width * (len(partitions) - 1) / 2)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"IID vs Non-IID — {metric}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    print(f"[Plot] Saved → {save_path}")


def plot_member_nonmember_dist(
    member_scores: List[float],
    nonmember_scores: List[float],
    save_path: str,
    method_name: str = "FedMIA",
):
    """Histogram of member vs non-member score distributions (like Figure 1 in paper)."""
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(member_scores, bins=50, alpha=0.6, label="Member", color="#2196F3", density=True)
        ax.hist(nonmember_scores, bins=50, alpha=0.6, label="Non-member", color="#FF5722", density=True)
        mu_diff = np.mean(member_scores) - np.mean(nonmember_scores)
        ax.set_title(f"{method_name} | µ_mem − µ_non = {mu_diff:.3f}")
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    print(f"[Plot] Saved → {save_path}")


def save_summary_table(rows: List[Dict], save_path: str):
    """Save the mandatory results table as CSV."""
    if not rows:
        return
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Table] Saved → {save_path}")

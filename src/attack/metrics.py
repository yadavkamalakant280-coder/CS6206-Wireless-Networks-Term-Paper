"""
src/attack/metrics.py

Category 1 — Privacy & Inference evaluation metrics:
  - Membership Inference AUC
  - TPR @ FPR = 0.1%  (Attack Success Rate)
  - Reconstruction MSE (for generative tasks)
"""

import numpy as np
from typing import List, Optional
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auc(labels: List[int], scores: List[float]) -> float:
    """
    Compute Area Under the ROC Curve.
    labels: 1 = member, 0 = non-member.
    scores: higher score → more likely member.
    """
    if len(set(labels)) < 2:
        return 0.5  # degenerate case
    return float(roc_auc_score(labels, scores))


def compute_tpr_at_fpr(
    labels: List[int],
    scores: List[float],
    fpr: float = 0.001,
) -> float:
    """
    TPR @ given FPR level (default 0.1% = 0.001).
    This is the primary metric used in the FedMIA paper.
    """
    if len(set(labels)) < 2:
        return 0.0

    fprs, tprs, thresholds = roc_curve(labels, scores, pos_label=1)

    # Find the largest FPR that is ≤ the target FPR
    valid = np.where(fprs <= fpr)[0]
    if len(valid) == 0:
        return 0.0
    return float(tprs[valid[-1]])


def compute_attack_success_rate(
    labels: List[int],
    scores: List[float],
    threshold: float = 0.5,
) -> float:
    """
    Fraction of samples correctly classified as member/non-member
    at a fixed decision threshold.
    """
    preds = [1 if s > threshold else 0 for s in scores]
    correct = sum(p == l for p, l in zip(preds, labels))
    return correct / len(labels) if labels else 0.0


def compute_reconstruction_mse(
    original_images: np.ndarray,
    reconstructed_images: np.ndarray,
) -> float:
    """
    Mean Squared Error between original and reconstructed images.
    Used in generative FL tasks (e.g. latent diffusion model experiments).

    Parameters
    ----------
    original_images : np.ndarray, shape (N, C, H, W), values in [0, 1]
    reconstructed_images : np.ndarray, same shape
    """
    assert original_images.shape == reconstructed_images.shape, (
        f"Shape mismatch: {original_images.shape} vs {reconstructed_images.shape}"
    )
    mse = float(np.mean((original_images - reconstructed_images) ** 2))
    return mse


def compute_all_metrics(
    labels: List[int],
    scores: List[float],
    threshold: float = 0.5,
    fpr_target: float = 0.001,
    original_images: Optional[np.ndarray] = None,
    reconstructed_images: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute all Cat. 1 metrics in one call.
    Returns a flat dict ready for logging.
    """
    results = {
        "AUC": compute_auc(labels, scores),
        f"TPR@FPR={fpr_target}": compute_tpr_at_fpr(labels, scores, fpr=fpr_target),
        "AttackSuccessRate": compute_attack_success_rate(labels, scores, threshold=threshold),
    }
    if original_images is not None and reconstructed_images is not None:
        results["ReconstructionMSE"] = compute_reconstruction_mse(
            original_images, reconstructed_images
        )
    return results

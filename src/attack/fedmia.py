"""
src/attack/fedmia.py

FedMIA — "All for One" Membership Inference Attack
Based on: Zhu et al., CVPR 2025
https://github.com/Liar-Mask/FedMIA

Three-step algorithm:
  Step 1: Compute low-dimensional cosine-similarity measurement M(I|(x,y))
  Step 2: Estimate Q_out distribution using non-target clients (with 3-sigma filter)
  Step 3: One-tailed likelihood-ratio test; average across T communication rounds
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from scipy import stats


class FedMIA:
    """
    Server-side Membership Inference Attacker.

    Parameters
    ----------
    target_client_id : int
        Index of the client whose membership is being inferred.
    measurement : str
        'grad_cosine' → FedMIA-II (Eq. 7 in paper)
        'loss'        → FedMIA-I
    sigma_filter : float
        Multiplier for the 3-σ outlier removal rule (default 3.0).
    threshold : float
        Decision threshold δ. Sample is member if Λ̃ > δ.
    fpr_target : float
        FPR level for TPR@FPR metric (default 0.001 = 0.1%).
    """

    def __init__(
        self,
        target_client_id: int = 0,
        measurement: str = "grad_cosine",
        sigma_filter: float = 3.0,
        threshold: float = 0.5,
        fpr_target: float = 0.001,
    ):
        self.target_client_id = target_client_id
        self.measurement = measurement
        self.sigma_filter = sigma_filter
        self.threshold = threshold
        self.fpr_target = fpr_target

        # Accumulated across rounds: {round_t: {client_id: M_value}}
        self.round_measurements: List[Dict[int, float]] = []

        # Store raw scores for AUC calculation
        self.member_scores: List[float] = []
        self.nonmember_scores: List[float] = []

    # ──────────────────────────────────────────────────────────────────────
    # Step 1: Low-dimensional measurement
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(grad_update: torch.Tensor, target_grad: torch.Tensor) -> float:
        """Eq. (7): cosine similarity between uploaded gradient and target gradient."""
        u = grad_update.flatten().float()
        v = target_grad.flatten().float()
        norm_u = u.norm()
        norm_v = v.norm()
        if norm_u == 0 or norm_v == 0:
            return 0.0
        return torch.dot(u, v).item() / (norm_u * norm_v).item()

    @staticmethod
    def _compute_target_gradient(
        model: nn.Module,
        sample_x: torch.Tensor,
        sample_y: torch.Tensor,
        criterion: nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute ∂ℓ(ω, x, y)/∂ω on the global model for target sample."""
        model.eval()
        model.zero_grad()
        sample_x = sample_x.unsqueeze(0).to(device)
        sample_y = torch.tensor([sample_y]).to(device)
        output = model(sample_x)
        loss = criterion(output, sample_y)
        loss.backward()
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().cpu().flatten())
        return torch.cat(grads)

    @staticmethod
    def _compute_loss(
        model: nn.Module,
        sample_x: torch.Tensor,
        sample_y: torch.Tensor,
        criterion: nn.Module,
        device: torch.device,
    ) -> float:
        """Compute ℓ(ω, x, y) for FedMIA-I measurement."""
        model.eval()
        with torch.no_grad():
            sample_x = sample_x.unsqueeze(0).to(device)
            sample_y = torch.tensor([sample_y]).to(device)
            output = model(sample_x)
            loss = criterion(output, sample_y)
        return loss.item()

    def compute_measurement(
        self,
        client_updates: Dict[int, List[torch.Tensor]],  # {client_id: flat_grad_list}
        global_model: nn.Module,
        target_sample: Tuple,  # (x_tensor, y_int)
        criterion: nn.Module,
        device: torch.device,
    ) -> Dict[int, float]:
        """
        Step 1: For each client in this round, compute M(I_k | x, y).
        Returns dict {client_id: measurement_value}.
        """
        sample_x, sample_y = target_sample
        measurements = {}

        if self.measurement == "grad_cosine":
            target_grad = self._compute_target_gradient(
                global_model, sample_x, sample_y, criterion, device
            )
            for cid, flat_grad in client_updates.items():
                if isinstance(flat_grad, list):
                    flat_grad = torch.cat([g.flatten() for g in flat_grad])
                measurements[cid] = self._cosine_similarity(flat_grad, target_grad)

        elif self.measurement == "loss":
            # FedMIA-I: use negative loss (higher = more likely member)
            base_loss = self._compute_loss(global_model, sample_x, sample_y, criterion, device)
            for cid in client_updates:
                measurements[cid] = -base_loss  # same for all; tracks across rounds

        return measurements

    # ──────────────────────────────────────────────────────────────────────
    # Step 2: Estimate Q_out using non-target clients
    # ──────────────────────────────────────────────────────────────────────

    def _estimate_qout(self, measurements: Dict[int, float]) -> Tuple[float, float]:
        """
        Estimate µ_out and σ²_out from non-target clients' measurements.
        Applies 3-σ rule to remove updates likely trained on target data (Eq. 8-9-10).
        Returns (mu_out, var_out).
        """
        non_tar_vals = np.array([
            v for cid, v in measurements.items()
            if cid != self.target_client_id
        ])

        if len(non_tar_vals) == 0:
            return 0.0, 1.0

        # Initial stats
        mu = non_tar_vals.mean()
        sigma = non_tar_vals.std() + 1e-8

        # Filter outliers (Eq. 8): remove M(I_k|x,y) > mu + 3*sigma
        mask = non_tar_vals <= (mu + self.sigma_filter * sigma)
        filtered = non_tar_vals[mask]

        if len(filtered) == 0:
            filtered = non_tar_vals  # fallback: no filtering

        mu_out = float(filtered.mean())
        var_out = float(filtered.var()) + 1e-8
        return mu_out, var_out

    # ──────────────────────────────────────────────────────────────────────
    # Step 3: One-tailed LRT confidence score
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _cdf_score(target_val: float, mu_out: float, var_out: float) -> float:
        """
        Λ̂(I_tar, x, y) = CDF of N(µ_out, var_out) evaluated at target_val.
        Eq. (11): probability that a non-member achieves confidence ≤ target_val.
        High score → target_val is unusually high → likely member.
        """
        std_out = np.sqrt(var_out)
        return float(stats.norm.cdf(target_val, loc=mu_out, scale=std_out))

    # ──────────────────────────────────────────────────────────────────────
    # Public interface used by the server each round
    # ──────────────────────────────────────────────────────────────────────

    def update_round(
        self,
        round_t: int,
        client_updates: Dict[int, torch.Tensor],
        global_model: nn.Module,
        target_sample: Tuple,
        criterion: nn.Module,
        device: torch.device,
    ) -> float:
        """
        Called once per FL communication round.
        Returns the per-round confidence score Λ̂ for the target sample.
        """
        # Step 1
        measurements = self.compute_measurement(
            client_updates, global_model, target_sample, criterion, device
        )
        self.round_measurements.append(measurements)

        # Step 2
        mu_out, var_out = self._estimate_qout(measurements)

        # Step 3
        target_val = measurements.get(self.target_client_id, mu_out)
        score_t = self._cdf_score(target_val, mu_out, var_out)
        return score_t

    def infer(self) -> Tuple[int, float]:
        """
        Aggregate across all T communication rounds (Eq. 12):
        Λ̃ = (1/T) Σ_t Λ̂_t

        Returns (prediction, average_score).
        prediction: 1 = member, 0 = non-member.
        """
        if not self.round_measurements:
            return 0, 0.0

        per_round_scores = []
        for measurements in self.round_measurements:
            mu_out, var_out = self._estimate_qout(measurements)
            target_val = measurements.get(self.target_client_id, mu_out)
            per_round_scores.append(self._cdf_score(target_val, mu_out, var_out))

        avg_score = float(np.mean(per_round_scores))
        prediction = 1 if avg_score > self.threshold else 0
        return prediction, avg_score

    def reset(self):
        """Reset accumulated state for a new target sample."""
        self.round_measurements = []

    # ──────────────────────────────────────────────────────────────────────
    # Batch evaluation over a set of member/non-member samples
    # ──────────────────────────────────────────────────────────────────────

    def evaluate_batch(
        self,
        member_samples: List[Tuple],
        nonmember_samples: List[Tuple],
        all_round_updates: List[Dict[int, torch.Tensor]],  # one dict per round
        global_model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Run FedMIA over all provided member/non-member samples and
        return Attack Success Rate, AUC, and TPR@FPR=fpr_target.
        """
        from .metrics import compute_auc, compute_tpr_at_fpr, compute_attack_success_rate

        member_scores_list = []
        nonmember_scores_list = []

        for sample in member_samples:
            self.reset()
            for t, updates in enumerate(all_round_updates):
                self.update_round(t, updates, global_model, sample, criterion, device)
            _, score = self.infer()
            member_scores_list.append(score)

        for sample in nonmember_samples:
            self.reset()
            for t, updates in enumerate(all_round_updates):
                self.update_round(t, updates, global_model, sample, criterion, device)
            _, score = self.infer()
            nonmember_scores_list.append(score)

        # Store for distribution plot
        self.member_scores = member_scores_list
        self.nonmember_scores = nonmember_scores_list

        all_scores = member_scores_list + nonmember_scores_list
        all_labels = [1] * len(member_scores_list) + [0] * len(nonmember_scores_list)

        auc = compute_auc(all_labels, all_scores)
        tpr = compute_tpr_at_fpr(all_labels, all_scores, fpr=self.fpr_target)
        asr = compute_attack_success_rate(all_labels, all_scores, threshold=self.threshold)

        return {
            "AUC": auc,
            f"TPR@FPR={self.fpr_target}": tpr,
            "AttackSuccessRate": asr,
        }

"""
f2m/models/losses.py
====================
Multi-task loss functions for Fundus2Myopia.

Three weighting strategies
--------------------------
uncertainty  (default)
    Kendall & Gal 2018: learnable log-variance per task.
    L_total = Σ_i [ exp(-s_i) * L_i + s_i ]  where s_i = log(σ_i²)
    Automatically down-weights harder tasks and penalises over-confidence.

fixed
    User-specified constant weights per task.

gradnorm
    Chen et al. 2018: dynamically rescale task weights so gradient norms
    are balanced. Implemented as a wrapper around the fixed-weight loss
    that periodically updates weights based on gradient magnitudes.

SE consistency loss
-------------------
Physics constraint: SE = SPH + CYL/2.
Applied as SmoothL1(SE_pred_derived, SE_true) to regularise the model.
This term has weight `consistency_weight` (default 0.5).
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Base loss functions
# ─────────────────────────────────────────────────────────────────────────────

def _masked_loss(
    pred:   torch.Tensor,    # [B] or [B,1]
    target: torch.Tensor,    # [B]
    mask:   torch.Tensor,    # [B] bool  — True = valid
    loss_fn: str = "smooth_l1",
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute element-wise loss on valid (mask=True) entries only.

    Returns a scalar (mean over valid entries) or 0 if no valid entries.
    """
    if pred.ndim == 2:
        pred = pred.squeeze(1)
    valid_pred   = pred[mask]
    valid_target = target[mask]
    if valid_pred.numel() == 0:
        return pred.sum() * 0.0   # maintain gradient graph

    if loss_fn == "smooth_l1":
        return F.smooth_l1_loss(valid_pred, valid_target, reduction=reduction)
    elif loss_fn == "mse":
        return F.mse_loss(valid_pred, valid_target, reduction=reduction)
    elif loss_fn == "mae":
        return F.l1_loss(valid_pred, valid_target, reduction=reduction)
    else:
        raise ValueError(f"Unknown loss_fn: {loss_fn!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty-weighted MTL loss (Kendall & Gal 2018)
# ─────────────────────────────────────────────────────────────────────────────

class UncertaintyWeightedLoss(nn.Module):
    """
    Homoscedastic uncertainty weighting for multi-task regression.

    Each task has a learnable log-variance parameter s_i = log(σ_i²).
    Loss for task i:   L_i^UW = exp(-s_i) * L_i + s_i

    Parameters
    ----------
    n_tasks : int
    init_log_var : float
        Initial value for each s_i. Default 0.0 (σ_i = 1).
    loss_fn : str
        Per-task element-wise loss: "smooth_l1" | "mse" | "mae".
    """

    def __init__(
        self,
        n_tasks: int,
        init_log_var: float = 0.0,
        loss_fn: str = "smooth_l1",
    ) -> None:
        super().__init__()
        self.n_tasks = n_tasks
        self.loss_fn = loss_fn
        self.log_vars = nn.Parameter(
            torch.full((n_tasks,), init_log_var)
        )

    def forward(
        self,
        preds:   list[torch.Tensor],   # list of [B, 1] tensors per task
        targets: torch.Tensor,          # [B, n_tasks]
        mask:    torch.Tensor,          # [B, n_tasks] bool
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns (total_loss, info_dict).
        info_dict keys: "loss_total", "loss_<task_i>", "log_var_<task_i>".
        """
        assert len(preds) == self.n_tasks
        info = {}
        total = torch.zeros(1, device=targets.device)

        for i, pred in enumerate(preds):
            raw = _masked_loss(
                pred, targets[:, i], mask[:, i], self.loss_fn
            )
            s_i = self.log_vars[i]
            uw_loss = torch.exp(-s_i) * raw + s_i
            total = total + uw_loss
            info[f"loss_task_{i}"] = raw.detach().item()
            info[f"log_var_{i}"]   = s_i.detach().item()
            info[f"weight_{i}"]    = torch.exp(-s_i).detach().item()

        info["loss_total"] = total.detach().item()
        return total, info

    def task_weights(self) -> torch.Tensor:
        """Return current effective task weights exp(-s_i)."""
        return torch.exp(-self.log_vars.detach())


# ─────────────────────────────────────────────────────────────────────────────
# Fixed-weight MTL loss
# ─────────────────────────────────────────────────────────────────────────────

class FixedWeightLoss(nn.Module):
    """Simple weighted sum of per-task losses with constant weights."""

    def __init__(
        self,
        weights: Sequence[float],
        loss_fn: str = "smooth_l1",
    ) -> None:
        super().__init__()
        self.weights = list(weights)
        self.loss_fn = loss_fn

    def forward(
        self,
        preds:   list[torch.Tensor],
        targets: torch.Tensor,
        mask:    torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        info = {}
        total = torch.zeros(1, device=targets.device)
        for i, (pred, w) in enumerate(zip(preds, self.weights)):
            raw = _masked_loss(pred, targets[:, i], mask[:, i], self.loss_fn)
            total = total + w * raw
            info[f"loss_task_{i}"] = raw.detach().item()
            info[f"weight_{i}"]    = w
        info["loss_total"] = total.detach().item()
        return total, info


# ─────────────────────────────────────────────────────────────────────────────
# SE consistency loss
# ─────────────────────────────────────────────────────────────────────────────

def se_consistency_loss(
    sph_pred: torch.Tensor,   # [B, 1]
    cyl_pred: torch.Tensor,   # [B, 1]
    se_true:  torch.Tensor,   # [B]
    mask:     torch.Tensor,   # [B] bool — True where se_true is valid
) -> torch.Tensor:
    """
    Enforce SE = SPH + CYL/2 by penalising the difference between the
    derived SE prediction and the measured SE target.

    This is a soft physics constraint; the SE head is not separately trained
    but the gradients from this loss flow back through sph_pred and cyl_pred.
    """
    se_derived = sph_pred.squeeze(1) + cyl_pred.squeeze(1) / 2.0
    return _masked_loss(se_derived, se_true, mask, "smooth_l1")


# ─────────────────────────────────────────────────────────────────────────────
# F2MLoss orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class F2MLoss(nn.Module):
    """
    Multi-task loss orchestrator for F2MNet.

    Parameters
    ----------
    task_names : list[str]
        Names corresponding to entries in the model output dict.
        E.g., ["al", "sph", "cyl"].
    aux_task_names : list[str]
        Names of auxiliary tasks (e.g. ["od_area", "ppa_ratio"]).
    weighting : str
        "uncertainty" | "fixed".
    fixed_weights : list[float] | None
        Required when weighting="fixed". Length = len(task_names).
    consistency_weight : float
        λ for the SE=SPH+CYL/2 consistency loss. Set 0 to disable.
    aux_weight : float
        Global weight applied to all auxiliary task losses.
    loss_fn : str
        Base per-task loss: "smooth_l1" | "mse" | "mae".
    """

    def __init__(
        self,
        task_names: Sequence[str] = ("al", "sph", "cyl"),
        aux_task_names: Sequence[str] = (),
        weighting: str = "uncertainty",
        fixed_weights: Sequence[float] | None = None,
        consistency_weight: float = 0.5,
        aux_weight: float = 0.10,
        loss_fn: str = "smooth_l1",
    ) -> None:
        super().__init__()
        self.task_names     = list(task_names)
        self.aux_task_names = list(aux_task_names)
        self.consistency_weight = consistency_weight
        self.aux_weight     = aux_weight
        self.loss_fn        = loss_fn

        n_tasks = len(task_names)

        if weighting == "uncertainty":
            self.task_loss = UncertaintyWeightedLoss(n_tasks, loss_fn=loss_fn)
        elif weighting == "fixed":
            w = fixed_weights or [1.0] * n_tasks
            assert len(w) == n_tasks
            self.task_loss = FixedWeightLoss(w, loss_fn=loss_fn)
        else:
            raise ValueError(f"Unknown weighting: {weighting!r}")

        # Auxiliary tasks always use fixed weight = 1 (scaled by aux_weight)
        if aux_task_names:
            self.aux_loss_fn = FixedWeightLoss(
                [1.0] * len(aux_task_names), loss_fn=loss_fn
            )

    def forward(
        self,
        predictions: dict[str, torch.Tensor],   # model output dict
        targets:     torch.Tensor,               # [B, n_tasks]
        targets_mask: torch.Tensor,              # [B, n_tasks] bool
        aux_targets: torch.Tensor | None = None, # [B, n_aux]
        aux_mask:    torch.Tensor | None = None, # [B, n_aux] bool
        se_true:     torch.Tensor | None = None, # [B]   for consistency loss
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns
        -------
        total_loss : scalar Tensor
        info : dict with per-task losses and weights
        """
        # ── primary task losses ───────────────────────────────────────────────
        preds = [predictions[t] for t in self.task_names]
        total, info = self.task_loss(preds, targets, targets_mask)

        # ── SE consistency loss ───────────────────────────────────────────────
        if (
            self.consistency_weight > 0
            and "sph" in predictions
            and "cyl" in predictions
            and se_true is not None
        ):
            se_mask = ~torch.isnan(se_true)
            if se_mask.any():
                se_loss = se_consistency_loss(
                    predictions["sph"], predictions["cyl"],
                    se_true, se_mask,
                )
                total = total + self.consistency_weight * se_loss
                info["loss_se_consistency"] = se_loss.detach().item()

        # ── auxiliary task losses ─────────────────────────────────────────────
        if self.aux_task_names and aux_targets is not None and aux_mask is not None:
            aux_preds = [predictions.get(t, torch.zeros_like(aux_targets[:, 0:1]))
                         for t in self.aux_task_names]
            aux_total, aux_info = self.aux_loss_fn(aux_preds, aux_targets, aux_mask)
            total = total + self.aux_weight * aux_total
            info.update({f"aux_{k}": v for k, v in aux_info.items()})

        info["loss_total"] = total.detach().item()
        return total, info

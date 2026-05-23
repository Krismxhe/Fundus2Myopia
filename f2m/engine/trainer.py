"""
f2m/engine/trainer.py
=====================
Training loop for the Fundus2Myopia multi-task regression model.

Design choices
--------------
* Mixed-precision (torch.amp) when a CUDA device is available.
* Gradient clipping (max_norm=1.0) applied before every optimiser step.
* Differential learning rates: backbone_lr (lower) vs head_lr (higher).
* Validation at the end of every epoch; early stopping on val MAE of the
  primary task (axial length by default).
* Checkpoint saved when val primary-task MAE improves.
* All metrics logged via Python logging at INFO level; structured info dict
  available for external experiment loggers (W&B, MLflow, etc.).

Usage
-----
From scripts/train.py — not intended to be invoked directly.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from f2m.models.f2mnet import F2MNet
from f2m.models.losses import F2MLoss
from f2m.utils.metrics import evaluate_all_tasks

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _concat_preds(
    buffer: dict[str, list],
    out: dict[str, torch.Tensor],
    task_names: Sequence[str],
) -> None:
    """Append model output tensors to per-task lists (CPU numpy)."""
    for t in task_names:
        if t in out:
            buffer[t].append(out[t].squeeze(1).detach().cpu().numpy())


def _concat_targets(
    buffer: dict[str, list],
    batch: dict,
    task_names: Sequence[str],
    target_stats: dict | None,
) -> None:
    """Append (un-normalised) ground-truth values to per-task lists."""
    for i, t in enumerate(task_names):
        vals = batch["targets"][:, i].cpu().numpy()
        if target_stats and t in target_stats:
            mu, sigma = target_stats[t]["mean"], target_stats[t]["std"]
            vals = vals * sigma + mu
        buffer[t].append(vals)


def _concat_masks(
    buffer: dict[str, list],
    batch: dict,
    task_names: Sequence[str],
) -> None:
    for i, t in enumerate(task_names):
        buffer[t].append(batch["targets_mask"][:, i].cpu().numpy())


# ─────────────────────────────────────────────────────────────────────────────
# Main trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Full training + validation loop for F2MNet.

    Parameters
    ----------
    model : F2MNet
    criterion : F2MLoss
    optimiser : torch.optim.Optimizer
    scheduler : lr scheduler (step after each epoch) | None
    device : torch.device
    task_names : list of task keys in model output
    target_stats : dict {task: {mean: float, std: float}} — for un-normalising
        during eval metric computation
    primary_task : str
        Task whose val MAE drives early stopping / checkpoint saving.
    grad_clip : float
        Max gradient norm.  Default 1.0.
    amp : bool
        Use automatic mixed precision (only on CUDA).  Default True.
    checkpoint_dir : Path | None
        Where to save best-model checkpoints.
    patience : int
        Early stopping patience (epochs).  0 = disabled.
    """

    def __init__(
        self,
        model: F2MNet,
        criterion: F2MLoss,
        optimiser: torch.optim.Optimizer,
        scheduler=None,
        device: torch.device | None = None,
        task_names: Sequence[str] = ("al", "sph", "cyl"),
        target_stats: dict | None = None,
        primary_task: str = "al",
        grad_clip: float = 1.0,
        amp: bool = True,
        checkpoint_dir: Path | None = None,
        patience: int = 20,
    ) -> None:
        self.model        = model
        self.criterion    = criterion
        self.optimiser    = optimiser
        self.scheduler    = scheduler
        self.device       = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.task_names   = list(task_names)
        self.target_stats = target_stats
        self.primary_task = primary_task
        self.grad_clip    = grad_clip
        self.amp          = amp and self.device.type == "cuda"
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.patience     = patience

        self._scaler = GradScaler(enabled=self.amp)
        self._best_val_mae: float = float("inf")
        self._patience_counter: int = 0

        self.model.to(self.device)
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
    ) -> list[dict]:
        """
        Train for `n_epochs` epochs.

        Returns
        -------
        history : list of per-epoch metric dicts
        """
        history: list[dict] = []

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_info = self._run_epoch(train_loader, train=True)
            val_info   = self._run_epoch(val_loader,   train=False)
            elapsed    = time.time() - t0

            if self.scheduler is not None:
                self.scheduler.step()

            row = {
                "epoch":        epoch,
                "train_loss":   train_info["loss_total"],
                "val_loss":     val_info["loss_total"],
                "elapsed_s":    elapsed,
                **{f"train_{k}": v for k, v in train_info.get("metrics", {}).items()},
                **{f"val_{k}":   v for k, v in val_info.get("metrics", {}).items()},
            }
            history.append(row)

            # ── primary-task val MAE ─────────────────────────────────────────
            primary_key = f"val_per_task_{self.primary_task}_mae"
            val_primary_mae = row.get(primary_key, float("inf"))

            self._log_epoch(epoch, n_epochs, row, val_primary_mae)
            # Record the best MAE *before* checkpointing updates it so the
            # patience comparison can tell whether this epoch improved.
            prev_best = self._best_val_mae
            self._maybe_checkpoint(epoch, val_primary_mae)

            if self.patience > 0:
                if val_primary_mae < prev_best:   # improved vs previous best
                    self._patience_counter = 0
                else:
                    self._patience_counter += 1
                if self._patience_counter >= self.patience:
                    logger.info(
                        "Early stopping at epoch %d — no improvement in val %s MAE "
                        "for %d epochs.",
                        epoch, self.primary_task, self.patience,
                    )
                    break

        return history

    def evaluate(self, loader: DataLoader) -> dict:
        """Run a full evaluation pass and return all metrics."""
        info = self._run_epoch(loader, train=False)
        return info

    # ── private ───────────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict:
        self.model.train(train)
        total_loss = 0.0
        n_batches  = 0

        # Accumulate for metrics
        preds_buf:   dict[str, list] = {t: [] for t in self.task_names}
        targets_buf: dict[str, list] = {t: [] for t in self.task_names}
        masks_buf:   dict[str, list] = {t: [] for t in self.task_names}
        se_buf:      list[np.ndarray] = []

        with torch.set_grad_enabled(train):
            for batch in loader:
                batch  = _to_device(batch, self.device)
                image  = batch["image"]
                age    = batch["age"]
                gender = batch.get("gender")   # [B] int64; None → ConditionEncoder uses zeros

                with autocast(enabled=self.amp):
                    out = self.model(image, age, gender=gender)
                    loss, info = self.criterion(
                        predictions=out,
                        targets=batch["targets"],
                        targets_mask=batch["targets_mask"],
                        se_true=batch.get("se_true"),
                    )

                if train:
                    self.optimiser.zero_grad()
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(self.optimiser)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                    self._scaler.step(self.optimiser)
                    self._scaler.update()

                total_loss += info["loss_total"]
                n_batches  += 1

                # Accumulate for metrics (un-normalise preds on-the-fly)
                _concat_preds(preds_buf, out, self.task_names)
                _concat_targets(targets_buf, batch, self.task_names,
                                self.target_stats)
                _concat_masks(masks_buf, batch, self.task_names)
                if "se_true" in batch and batch["se_true"] is not None:
                    se_buf.append(batch["se_true"].cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)

        # Flatten accumulated arrays
        preds   = {t: np.concatenate(preds_buf[t])   for t in self.task_names}
        targets = {t: np.concatenate(targets_buf[t]) for t in self.task_names}
        masks   = {t: np.concatenate(masks_buf[t])   for t in self.task_names}

        # Un-normalise predictions
        if self.target_stats:
            for t in self.task_names:
                if t in self.target_stats:
                    mu    = self.target_stats[t]["mean"]
                    sigma = self.target_stats[t]["std"]
                    preds[t] = preds[t] * sigma + mu

        se_true = np.concatenate(se_buf) if se_buf else None

        metrics_raw = evaluate_all_tasks(preds, targets, masks, se_true=se_true)

        # Flatten nested metrics dict for easy logging
        flat_metrics: dict[str, float] = {}
        for task, m in metrics_raw.get("per_task", {}).items():
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    flat_metrics[f"per_task_{task}_{k}"] = float(v)
        for k, v in metrics_raw.get("cross_task", {}).items():
            if isinstance(v, (int, float)):
                flat_metrics[f"cross_task_{k}"] = float(v)

        return {
            "loss_total": avg_loss,
            "metrics":    flat_metrics,
            "metrics_nested": metrics_raw,
        }

    def _log_epoch(
        self,
        epoch: int,
        n_epochs: int,
        row: dict,
        val_primary_mae: float,
    ) -> None:
        lr = self.optimiser.param_groups[0]["lr"]
        logger.info(
            "Epoch %d/%d | train_loss=%.4f  val_loss=%.4f  "
            "val_%s_MAE=%.4f  lr=%.2e  elapsed=%.0fs",
            epoch, n_epochs,
            row["train_loss"], row["val_loss"],
            self.primary_task, val_primary_mae,
            lr, row["elapsed_s"],
        )

    def _maybe_checkpoint(self, epoch: int, val_primary_mae: float) -> None:
        if val_primary_mae < self._best_val_mae:
            self._best_val_mae = val_primary_mae
            if self.checkpoint_dir:
                ckpt_path = self.checkpoint_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch":          epoch,
                        "model_state":    self.model.state_dict(),
                        "optimiser_state": self.optimiser.state_dict(),
                        "val_primary_mae": val_primary_mae,
                        "task_names":     self.task_names,
                    },
                    ckpt_path,
                )
                logger.info(
                    "  ✓  New best val %s MAE = %.4f — checkpoint saved to %s",
                    self.primary_task, val_primary_mae, ckpt_path,
                )

    def load_checkpoint(self, path: str | Path) -> int:
        """
        Load model weights from a saved checkpoint.

        Returns
        -------
        epoch : int   epoch at which the checkpoint was saved
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        if "optimiser_state" in ckpt:
            self.optimiser.load_state_dict(ckpt["optimiser_state"])
        epoch = ckpt.get("epoch", -1)
        mae   = ckpt.get("val_primary_mae", float("nan"))
        logger.info(
            "Loaded checkpoint from epoch %d (val %s MAE = %.4f) ← %s",
            epoch, self.primary_task, mae, path,
        )
        return epoch

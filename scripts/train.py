#!/usr/bin/env python3
"""
scripts/train.py
================
Training entry point for the Fundus2Myopia (F2M) multi-task regression model.

Reads a YAML config file (default: configs/base.yaml) and an optional
override config (configs/demo.yaml for quick smoke tests), then:

  1. Loads the manifest and performs subject-level stratified splitting.
  2. Builds F2MDataset / DataLoader instances.
  3. Instantiates F2MNet with the requested backbone.
  4. Configures the multi-task loss (uncertainty weighting by default).
  5. Runs training via the Trainer engine, with checkpointing.
  6. Evaluates on the held-out test set after training completes.

Usage
-----
# Full training (base config)
python scripts/train.py

# Demo / smoke-test run
python scripts/train.py --config configs/demo.yaml

# Override single keys (dot-notation into the YAML)
python scripts/train.py --config configs/demo.yaml \
    --set model.backbone efficientnet_b3 \
    --set training.n_epochs 10

# Resume from a checkpoint
python scripts/train.py --resume /path/to/best_model.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

# ── path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from f2m.data.dataset   import F2MDataset
from f2m.data.splits    import subject_level_split
from f2m.data.transforms import build_transform
from f2m.engine.trainer  import Trainer
from f2m.models.f2mnet   import F2MNet
from f2m.models.losses   import F2MLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ─────────────────────────────────────────────────────────────────────────────
# Config utilities
# ─────────────────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` (override wins)."""
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _set_nested(d: dict, key_path: str, value: str) -> None:
    """Set d[a][b][c] = value from key_path='a.b.c'."""
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    # Try to coerce: int, float, bool, else str
    for t in (int, float):
        try:
            value = t(value)
            break
        except (ValueError, TypeError):
            pass
    if isinstance(value, str) and value.lower() in ("true", "false"):
        value = value.lower() == "true"
    d[keys[-1]] = value


def load_config(config_path: Path, overrides: list[tuple[str, str]]) -> dict:
    base_path = PROJECT_ROOT / "configs" / "base.yaml"
    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    if config_path and config_path != base_path:
        with open(config_path) as f:
            override_cfg = yaml.safe_load(f)
        cfg = _deep_merge(cfg, override_cfg or {})

    for key_path, value in overrides:
        _set_nested(cfg, key_path, value)

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F2M training script")
    p.add_argument("--config", type=Path,
                   default=PROJECT_ROOT / "configs" / "base.yaml",
                   help="Path to YAML config (merged on top of base.yaml)")
    p.add_argument("--set", nargs=2, action="append", metavar=("KEY", "VALUE"),
                   default=[], dest="overrides",
                   help="Override a config key (dot-notation). Repeatable.")
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to checkpoint to resume training from")
    p.add_argument("--no_pretrained", action="store_true",
                   help="Skip ImageNet/MAE weight download (random init)")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training; run test-set evaluation only (requires --resume)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Config shortcuts ──────────────────────────────────────────────────────
    data_cfg  = cfg["data"]
    split_cfg = cfg["split"]
    model_cfg = cfg["model"]
    loss_cfg  = cfg["loss"]
    train_cfg = cfg["training"]
    path_cfg  = cfg.get("paths", {})

    task_names = model_cfg.get("task_names", ["al", "sph", "cyl"])
    target_cols = data_cfg.get("target_cols", task_names)
    input_size  = data_cfg.get("input_size", 300)

    # ── Manifest & split ──────────────────────────────────────────────────────
    manifest_path = Path(data_cfg["manifest_path"])
    img_root      = Path(data_cfg["img_root"])
    img_col       = data_cfg.get("img_col", "img_filename")

    logger.info("Loading manifest: %s", manifest_path)
    manifest = pd.read_csv(manifest_path, low_memory=False)
    logger.info("Manifest shape: %s", manifest.shape)

    train_df, val_df, test_df = subject_level_split(
        manifest,
        train_frac=split_cfg.get("train_frac", 0.70),
        val_frac=split_cfg.get("val_frac", 0.15),
        test_frac=split_cfg.get("test_frac", 0.15),
        stratify_col=split_cfg.get("stratify_col", "myopia_grade"),
        subject_col=split_cfg.get("subject_col", "subject_id"),
        random_state=split_cfg.get("random_state", 42),
    )
    logger.info("Split: train=%d  val=%d  test=%d",
                len(train_df), len(val_df), len(test_df))

    # ── Datasets ──────────────────────────────────────────────────────────────
    cache_images = data_cfg.get("cache_images", False)

    ds_train = F2MDataset(
        manifest_path=train_df,
        img_root=img_root,
        img_col=img_col,
        split="train",
        target_cols=target_cols,
        transform=build_transform("train", input_size=input_size),
        laterality_flip=data_cfg.get("laterality_flip", True),
        cache_images=cache_images,
    )
    age_stats    = ds_train.get_age_stats()
    target_stats = ds_train.get_target_stats()
    logger.info("Train dataset: %d samples", len(ds_train))

    ds_val = F2MDataset(
        manifest_path=val_df,
        img_root=img_root,
        img_col=img_col,
        split="val",
        target_cols=target_cols,
        transform=build_transform("val", input_size=input_size),
        age_stats=age_stats,
        target_stats=target_stats,
        laterality_flip=data_cfg.get("laterality_flip", True),
        cache_images=cache_images,
    )
    ds_test = F2MDataset(
        manifest_path=test_df,
        img_root=img_root,
        img_col=img_col,
        split="test",
        target_cols=target_cols,
        transform=build_transform("test", input_size=input_size),
        age_stats=age_stats,
        target_stats=target_stats,
        laterality_flip=data_cfg.get("laterality_flip", True),
        cache_images=cache_images,
    )

    num_workers = train_cfg.get("num_workers", 4)
    batch_size  = train_cfg.get("batch_size", 32)

    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    dl_test = torch.utils.data.DataLoader(
        ds_test,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    pretrained = not args.no_pretrained
    model = F2MNet(
        backbone_name=model_cfg.get("backbone", "efficientnet_b3"),
        task_names=task_names,
        pretrained=pretrained,
        neck_hidden=model_cfg.get("neck_hidden", 512),
        neck_out=model_cfg.get("neck_out", 256),
        head_hidden=model_cfg.get("head_hidden", 128),
        dropout_neck=model_cfg.get("dropout_neck", 0.30),
        dropout_head=model_cfg.get("dropout_head", 0.10),
        freeze_stages=model_cfg.get("freeze_stages", 0),
    )
    n_params = model.num_parameters()
    logger.info("Model: %s  |  trainable params: %s",
                model_cfg.get("backbone"), f"{n_params:,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = F2MLoss(
        task_names=task_names,
        weighting=loss_cfg.get("weighting", "uncertainty"),
        fixed_weights=loss_cfg.get("fixed_weights"),
        consistency_weight=loss_cfg.get("consistency_weight", 0.5),
        loss_fn=loss_cfg.get("loss_fn", "smooth_l1"),
    )

    # ── Optimiser ─────────────────────────────────────────────────────────────
    param_groups = model.param_groups(
        backbone_lr=train_cfg.get("backbone_lr", 1e-5),
        head_lr=train_cfg.get("head_lr", 1e-4),
    )
    optimiser = torch.optim.AdamW(
        param_groups,
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    # ── LR Scheduler ──────────────────────────────────────────────────────────
    scheduler_name = train_cfg.get("scheduler", "cosine")
    n_epochs = train_cfg.get("n_epochs", 100)

    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            T_max=n_epochs,
            eta_min=train_cfg.get("eta_min", 1e-7),
        )
    elif scheduler_name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimiser,
            max_lr=[train_cfg.get("backbone_lr", 1e-5), train_cfg.get("head_lr", 1e-4)],
            steps_per_epoch=len(dl_train),
            epochs=n_epochs,
        )
    else:
        scheduler = None

    # ── Trainer ───────────────────────────────────────────────────────────────
    checkpoint_dir = Path(path_cfg.get("checkpoint_dir", "checkpoints"))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimiser=optimiser,
        scheduler=scheduler,
        device=device,
        task_names=task_names,
        target_stats=target_stats,
        primary_task=train_cfg.get("primary_task", "al"),
        grad_clip=train_cfg.get("grad_clip", 1.0),
        amp=train_cfg.get("amp", True),
        checkpoint_dir=checkpoint_dir,
        patience=train_cfg.get("patience", 20),
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ── Training ──────────────────────────────────────────────────────────────
    if not args.eval_only:
        logger.info("\n%s", "=" * 60)
        logger.info("Starting training for %d epochs …", n_epochs)
        logger.info("Backbone: %s  |  Tasks: %s  |  Batch: %d",
                    model_cfg.get("backbone"), task_names, batch_size)
        logger.info("=" * 60)

        history = trainer.fit(dl_train, dl_val, n_epochs=n_epochs)

        # Save training history
        import json
        hist_path = checkpoint_dir / "history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2, default=str)
        logger.info("Training history saved to %s", hist_path)

    # ── Test evaluation ───────────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("Test-set evaluation …")
    best_ckpt = checkpoint_dir / "best_model.pt"
    if best_ckpt.exists():
        trainer.load_checkpoint(best_ckpt)

    test_info = trainer.evaluate(dl_test)
    metrics   = test_info.get("metrics_nested", {})

    logger.info("Test results (right eye, post-cycloplegic):")
    for task, m in metrics.get("per_task", {}).items():
        if isinstance(m, dict) and "mae" in m:
            logger.info(
                "  %-6s  MAE=%.3f  RMSE=%.3f  R²=%.3f  r=%.3f",
                task, m["mae"], m["rmse"], m["r2"], m["pearson_r"],
            )

    cross = metrics.get("cross_task", {})
    if "se_consistency_mae" in cross:
        logger.info("  SE consistency MAE = %.3f D", cross["se_consistency_mae"])
    if "myopia_grade" in cross:
        mg = cross["myopia_grade"]
        logger.info(
            "  Myopia grade: accuracy=%.3f  macro_F1=%.3f",
            mg.get("accuracy", float("nan")),
            mg.get("macro_f1",  float("nan")),
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

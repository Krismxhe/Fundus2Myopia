#!/usr/bin/env python3
"""
scripts/demo_test.py
====================
Smoke test for the F2M framework using the 50-subject demo dataset.

Verifies
--------
1. Dataset loads correctly (all 100 rows, no NaN targets).
2. Dataloaders produce correct batch shapes.
3. Model forward pass returns expected output dict.
4. Loss computation does not raise errors.
5. Metrics utilities produce valid numbers.
6. GradCAM produces maps of correct shape.

Usage
-----
From the project root:
    python scripts/demo_test.py

Or with explicit paths:
    python scripts/demo_test.py \
        --manifest /path/to/demo/manifest.csv \
        --img_root /path/to/demo/images
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from f2m.data.dataset   import F2MDataset
from f2m.data.splits    import subject_level_split
from f2m.data.transforms import build_transform
from f2m.models.f2mnet  import F2MNet
from f2m.models.losses   import F2MLoss
from f2m.utils.metrics   import evaluate_all_tasks
from f2m.analysis.gradcam import MultiHeadGradCAM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo_test")

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MANIFEST = (
    Path("/Volumes/Lenovo/001_myopia/data/fundus2myopia/demo/manifest.csv")
)
DEFAULT_IMG_ROOT = (
    Path("/Volumes/Lenovo/001_myopia/data/fundus2myopia/demo/images")
)
TARGET_COLS = ["al", "asph", "acyl"]
INPUT_SIZE  = 224   # smaller for fast CPU test (300 in real training)


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F2M demo smoke test")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--img_root", type=Path, default=DEFAULT_IMG_ROOT)
    p.add_argument("--backbone", type=str, default="resnet50",
                   help="Backbone to test (resnet50 | efficientnet_b3 | …)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--input_size", type=int, default=INPUT_SIZE)
    p.add_argument("--no_pretrained", action="store_true",
                   help="Skip ImageNet weight download (use random init)")
    p.add_argument("--gradcam", action="store_true",
                   help="Also test GradCAM for the first test sample")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def _check(condition: bool, msg: str) -> None:
    if condition:
        logger.info("  ✓  %s", msg)
    else:
        logger.error("  ✗  FAILED: %s", msg)
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    pretrained = not args.no_pretrained
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── 1. Load manifest ──────────────────────────────────────────────────────
    logger.info("\n── Step 1: Manifest loading ──────────────────────────────")
    import pandas as pd
    manifest = pd.read_csv(args.manifest, low_memory=False)
    logger.info("Manifest shape: %s", manifest.shape)
    _check(len(manifest) == 100, f"Expected 100 rows, got {len(manifest)}")
    _check(
        all(c in manifest.columns for c in TARGET_COLS + ["age"]),
        f"Required columns {TARGET_COLS + ['age']} present"
    )

    # ── 2. Split ──────────────────────────────────────────────────────────────
    logger.info("\n── Step 2: Subject-level split ───────────────────────────")
    train_df, val_df, test_df = subject_level_split(
        manifest,
        train_frac=0.70, val_frac=0.15, test_frac=0.15,
        stratify_col="myopia_grade",
        subject_col="subject_id",
        random_state=42,
    )
    total = len(train_df) + len(val_df) + len(test_df)
    _check(total == 100, f"All rows accounted for: {total}")
    _check(len(train_df) > 0, "Non-empty train split")
    _check(len(val_df)   > 0, "Non-empty val split")
    _check(len(test_df)  > 0, "Non-empty test split")
    logger.info(
        "Split sizes (rows): train=%d  val=%d  test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    # ── 3. Datasets ───────────────────────────────────────────────────────────
    logger.info("\n── Step 3: Dataset creation ──────────────────────────────")
    ds_train = F2MDataset(
        manifest_path=train_df,
        img_root=args.img_root,
        img_col="img_filename_demo",
        split="train",
        target_cols=TARGET_COLS,
        transform=build_transform("train", input_size=args.input_size),
        laterality_flip=True,
        cache_images=True,
    )
    age_stats    = ds_train.get_age_stats()
    target_stats = ds_train.get_target_stats()

    ds_val  = F2MDataset(
        manifest_path=val_df,
        img_root=args.img_root,
        img_col="img_filename_demo",
        split="val",
        target_cols=TARGET_COLS,
        transform=build_transform("val", input_size=args.input_size),
        age_stats=age_stats,
        target_stats=target_stats,
        laterality_flip=True,
        cache_images=True,
    )
    ds_test = F2MDataset(
        manifest_path=test_df,
        img_root=args.img_root,
        img_col="img_filename_demo",
        split="test",
        target_cols=TARGET_COLS,
        transform=build_transform("test", input_size=args.input_size),
        age_stats=age_stats,
        target_stats=target_stats,
        laterality_flip=True,
        cache_images=True,
    )
    _check(len(ds_train) > 0, "Train dataset non-empty")
    _check(len(ds_val)   > 0, "Val dataset non-empty")
    _check(len(ds_test)  > 0, "Test dataset non-empty")

    # ── 4. Inspect a batch ────────────────────────────────────────────────────
    logger.info("\n── Step 4: Batch shape inspection ───────────────────────")
    sample = ds_train[0]
    _check("image"        in sample, "Key 'image' in sample")
    _check("age"          in sample, "Key 'age' in sample")
    _check("gender"       in sample, "Key 'gender' in sample")
    _check("targets"      in sample, "Key 'targets' in sample")
    _check("targets_mask" in sample, "Key 'targets_mask' in sample")
    _check(sample["image"].shape[0] == 3, "Image has 3 channels")
    _check(sample["targets"].shape[0] == len(TARGET_COLS),
           f"targets shape = [{len(TARGET_COLS)}]")
    _check(not sample["targets"].isnan().any(), "No NaN in targets")
    _check(sample["gender"].dtype == torch.long,   "gender is int64")
    _check(sample["gender"].item() in (0, 1, 2),   "gender ∈ {0=unk, 1=M, 2=F}")
    logger.info("  sample_id=%s  eye=%d  gender=%d  myopia_grade=%d",
                sample["sample_id"], sample["eye"].item(),
                sample["gender"].item(), sample["myopia_grade"].item())

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=False)
    batch = next(iter(dl_train))
    B = batch["image"].shape[0]
    _check(batch["image"].shape == (B, 3, args.input_size, args.input_size),
           f"Batch image shape: {batch['image'].shape}")
    _check(batch["targets"].shape == (B, len(TARGET_COLS)),
           f"Batch targets shape: {batch['targets'].shape}")
    _check(batch["targets_mask"].dtype == torch.bool, "targets_mask is bool")
    _check(batch["gender"].shape == (B,),             f"Batch gender shape: {batch['gender'].shape}")
    _check(batch["gender"].dtype == torch.long,       "Batch gender is int64")

    # ── 5. Model forward ──────────────────────────────────────────────────────
    logger.info("\n── Step 5: Model instantiation and forward pass ──────────")
    model = F2MNet(
        backbone_name=args.backbone,
        task_names=["al", "sph", "cyl"],
        pretrained=pretrained,
        neck_hidden=256,
        neck_out=128,
        head_hidden=64,
        dropout_neck=0.2,
    ).to(device)

    n_params = model.num_parameters()
    logger.info("Model: %s  |  trainable params: %s", args.backbone,
                f"{n_params:,}")

    image  = batch["image"].to(device)
    age    = batch["age"].to(device)
    gender = batch["gender"].to(device)
    with torch.no_grad():
        out = model(image, age, gender=gender)

    _check("al"  in out, "Output contains 'al'")
    _check("sph" in out, "Output contains 'sph'")
    _check("cyl" in out, "Output contains 'cyl'")
    _check("se"  in out, "Output contains 'se' (derived)")
    _check(out["al"].shape  == (B, 1), f"al output shape: {out['al'].shape}")
    _check(out["sph"].shape == (B, 1), f"sph output shape: {out['sph'].shape}")
    _check(not any(v.isnan().any() for v in out.values() if torch.is_tensor(v)),
           "No NaN in model outputs")

    # Verify SE derivation
    se_check = (out["sph"] + out["cyl"] / 2.0 - out["se"]).abs().max().item()
    _check(se_check < 1e-5, f"SE = SPH + CYL/2 (max diff={se_check:.2e})")

    # ── 6. Loss computation ───────────────────────────────────────────────────
    logger.info("\n── Step 6: Loss computation ──────────────────────────────")
    criterion = F2MLoss(
        task_names=["al", "sph", "cyl"],
        weighting="fixed",
        fixed_weights=[1.0, 1.0, 1.0],
        consistency_weight=0.3,
    )
    targets_b = batch["targets"].to(device)
    mask_b    = batch["targets_mask"].to(device)
    se_true_b = batch["se_true"].to(device)

    total_loss, info = criterion(
        predictions=out,
        targets=targets_b,
        targets_mask=mask_b,
        se_true=se_true_b,
    )
    _check(torch.isfinite(total_loss), f"Loss is finite: {total_loss.item():.4f}")
    _check("loss_total" in info, "info contains 'loss_total'")
    logger.info("  Loss breakdown: %s",
                {k: f"{v:.4f}" for k, v in info.items()})

    # ── 7. Backward pass ─────────────────────────────────────────────────────
    logger.info("\n── Step 7: Backward pass ─────────────────────────────────")
    model.train()
    out_train = model(image, age, gender=gender)
    loss_train, _ = criterion(out_train, targets_b, mask_b, se_true=se_true_b)
    loss_train.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    _check(len(grads) > 0, f"Gradients computed for {len(grads)} params")
    model.zero_grad()
    model.eval()

    # ── 8. Metrics ────────────────────────────────────────────────────────────
    logger.info("\n── Step 8: Metrics computation ───────────────────────────")
    # Dummy arrays for smoke test (5 samples)
    rng = np.random.RandomState(0)
    n_fake = 20
    preds_np   = {"al": rng.normal(24, 1, n_fake), "sph": rng.normal(-2, 2, n_fake),
                  "cyl": rng.normal(-1, 0.5, n_fake)}
    targets_np = {"al": preds_np["al"] + rng.normal(0, 0.3, n_fake),
                  "sph": preds_np["sph"] + rng.normal(0, 0.5, n_fake),
                  "cyl": preds_np["cyl"] + rng.normal(0, 0.2, n_fake)}
    masks_np   = {k: np.ones(n_fake, dtype=bool) for k in preds_np}
    se_true_np = targets_np["sph"] + targets_np["cyl"] / 2.0

    eval_results = evaluate_all_tasks(
        preds_np, targets_np, masks_np,
        se_true=se_true_np,
    )
    _check("per_task" in eval_results,   "eval_results has 'per_task'")
    _check("cross_task" in eval_results, "eval_results has 'cross_task'")
    _check("al"  in eval_results["per_task"], "AL metrics present")
    _check("sph" in eval_results["per_task"], "SPH metrics present")
    _check("cyl" in eval_results["per_task"], "CYL metrics present")
    for task in ["al", "sph", "cyl"]:
        m = eval_results["per_task"][task]
        logger.info("  %s: MAE=%.3f  RMSE=%.3f  R²=%.3f  r=%.3f",
                    task, m["mae"], m["rmse"], m["r2"], m["pearson_r"])

    # ── 9. GradCAM ────────────────────────────────────────────────────────────
    if args.gradcam:
        logger.info("\n── Step 9: GradCAM ───────────────────────────────────────")
        model.eval()
        sample_img    = ds_test[0]["image"].unsqueeze(0).to(device)
        sample_age    = ds_test[0]["age"].unsqueeze(0).to(device)
        sample_gender = ds_test[0]["gender"].unsqueeze(0).to(device)

        cam_engine = MultiHeadGradCAM(model)
        cam_maps = cam_engine.compute(sample_img, sample_age, gender=sample_gender)
        for task, cam in cam_maps.items():
            _check(cam.shape == (args.input_size, args.input_size),
                   f"GradCAM map shape for {task}: {cam.shape}")
            _check(cam.min() >= 0 and cam.max() <= 1.0,
                   f"GradCAM values in [0,1] for {task}")
            logger.info("  %s CAM: shape=%s  max=%.4f", task, cam.shape, cam.max())
        cam_engine.remove_hooks()

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("ALL CHECKS PASSED — F2M framework is operational.")
    logger.info("Backbone: %s  |  Device: %s  |  Demo: 50 subjects / 100 images",
                args.backbone, device)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

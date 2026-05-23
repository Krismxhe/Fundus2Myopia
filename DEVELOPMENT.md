# F2M Development Notes

## Environment Setup

```bash
pip install -r requirements.txt
```

## Running Scripts — macOS Conda Environment

On macOS with a conda Python environment, **two environment variables must be set**
to prevent segfaults caused by OpenMP library conflicts between scikit-learn/joblib
(which uses `libomp.dylib`) and PyTorch (which uses `libiomp5.dylib`):

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```

### Smoke test (demo dataset)

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/demo_test.py --no_pretrained
```

### Training

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/train.py --config configs/demo.yaml --no_pretrained
```

### Full training

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/train.py
```

## Root Cause of macOS OpenMP Conflicts

The macOS conda environment has two competing OpenMP runtimes:
- **Intel MKL OpenMP (`libiomp5.dylib`)**: shipped with PyTorch
- **LLVM OpenMP (`libomp.dylib`)**: shipped with scikit-learn / joblib / scipy

When both are loaded in the same process, macOS's dynamic linker may cause
them to conflict. The `KMP_DUPLICATE_LIB_OK=TRUE` flag suppresses the
fatal-error-on-duplicate behaviour. The `OMP_NUM_THREADS=1` flag prevents
both runtimes from spawning threads, eliminating the crash entirely.

**On Linux**, these variables are not needed. Standard `python scripts/train.py` works.

## Project Structure

```
project/Fundus2Myopia/
├── configs/
│   ├── base.yaml          # Full training configuration
│   └── demo.yaml          # Demo / smoke-test overrides
├── f2m/
│   ├── data/
│   │   ├── dataset.py     # F2MDataset (per-eye, normalised targets)
│   │   ├── splits.py      # Subject-level stratified split
│   │   └── transforms.py  # PIL-native fundus augmentation pipeline
│   ├── models/
│   │   ├── backbone.py    # Backbone registry (EfficientNet, ResNet, RETFound)
│   │   ├── f2mnet.py      # F2MNet (multi-task FiLM-conditioned network)
│   │   └── losses.py      # Uncertainty-weighted MTL loss + SE consistency
│   ├── utils/
│   │   └── metrics.py     # MAE, RMSE, R², Bland-Altman, myopia grade
│   ├── analysis/
│   │   └── gradcam.py     # Multi-head GradCAM + ViT AttentionRollout
│   └── engine/
│       └── trainer.py     # Training loop with AMP, grad clip, checkpoint
├── scripts/
│   ├── demo_test.py       # 8-step smoke test (Steps 1–8 all pass)
│   └── train.py           # Training entry point
└── requirements.txt
```

## Verified Functionality (demo_test.py output)

All 8 checks pass on the 50-subject demo dataset (100 images, bilateral):

| Step | Check | Status |
|---|---|---|
| 1 | Manifest loading (100 rows, required columns) | ✓ |
| 2 | Subject-level split (train=68, val=16, test=16) | ✓ |
| 3 | Dataset creation (all 3 splits non-empty) | ✓ |
| 4 | Batch shapes ([4,3,224,224] images, [4,3] targets, bool mask) | ✓ |
| 5 | F2MNet forward pass (al/sph/cyl/se outputs, SE=SPH+CYL/2) | ✓ |
| 6 | Loss computation (finite, loss_total in info dict) | ✓ |
| 7 | Backward pass (191 params have gradients) | ✓ |
| 8 | Metrics computation (MAE/RMSE/R²/Pearson for all tasks) | ✓ |

# Fundus2Myopia (F2M)

**Multi-task deep learning framework** — predicting myopia parameters from colour fundus photographs and age.

---

## Overview

Fundus2Myopia (F2M) is a research framework for quantitative myopia assessment. Given a colour fundus photograph (CFP) and the patient's age, the model simultaneously predicts three clinically relevant parameters:

| Output | Description | Unit |
|---|---|---|
| `al`  | Axial Length | mm |
| `sph` | Cycloplegic Sphere | D |
| `cyl` | Cycloplegic Cylinder (minus-cylinder convention) | D |
| `se`  | Spherical Equivalent = SPH + CYL/2 (derived, no gradient) | D |

**Research objectives**
- Quantify the relationship between fundus appearance and AL / SER components (scientific discovery)
- Validate the predictive capacity of CFP for cycloplegic refraction and axial biometry
- Generate per-task GradCAM activation maps to localise retinal features driving each prediction

---

## Architecture

```
Image [B, 3, H, W]    +    Age [B]
          │                     │
      Backbone              AgeEncoder
    (CNN / ViT)             MLP(1 → 64)
          │                     │
    FiLM blocks  ←──── age_feat [B, 64]
    (stages 3+4)   age-conditioned feature modulation
          │
   Spatial map [B, C, h, w]
          │
   Shared Neck  (MLP: C → 512 → 256, BN, GELU, Dropout)
          │
   shared_feat [B, 256]
    ┌─────┼─────┐
   AL   SPH   CYL     (optional: OD area, PPA ratio — auxiliary tasks)
  head  head  head
    └─────┼─────┘
     SE = SPH + CYL/2   (physics-derived; detached from computation graph)
```

**Key design choices**

- **FiLM age conditioning** — Feature-wise Linear Modulation (Perez et al., 2018) injected at backbone stages 3 and 4: `out = (1 + γ(age)) × features + β(age)`
- **Uncertainty-weighted MTL loss** — Kendall & Gal (2018) learnable log-variance weights automatically balance AL (mm scale) against SPH/CYL (dioptre scale); augmented with an SE physics consistency term
- **Subject-level stratified splitting** — the Tibet cohort is longitudinal (same child across up to 5 waves); all rows for one subject are guaranteed to land in the same partition, preventing data leakage
- **Per-task GradCAM** — each task head is backpropagated in isolation (`retain_graph=True`), producing task-specific retinal activation maps

---

## Repository Structure

```
Fundus2Myopia/
├── configs/
│   ├── base.yaml          # Full training configuration
│   └── demo.yaml          # 50-subject demo dataset overrides
│
├── f2m/
│   ├── data/
│   │   ├── dataset.py     # F2MDataset — per-eye rows, z-score normalisation, image caching
│   │   ├── splits.py      # Subject-level stratified train / val / test split
│   │   └── transforms.py  # Fundus-specific PIL-native augmentation pipeline
│   │
│   ├── models/
│   │   ├── backbone.py    # Backbone registry (EfficientNet / ResNet / RETFound / timm)
│   │   ├── f2mnet.py      # F2MNet — FiLM conditioning + SharedNeck + multi-head regression
│   │   └── losses.py      # Uncertainty-weighted MTL loss + SE consistency constraint
│   │
│   ├── utils/
│   │   └── metrics.py     # MAE / RMSE / R² / Pearson / Bland-Altman / myopia grading
│   │
│   ├── analysis/
│   │   └── gradcam.py     # MultiHeadGradCAM + ViT AttentionRollout
│   │
│   └── engine/
│       └── trainer.py     # Training loop — AMP, gradient clipping, early stopping, checkpointing
│
├── scripts/
│   ├── demo_test.py       # 8-step smoke test (all checks pass ✓)
│   └── train.py           # Training entry point — YAML config + --set overrides + --resume
│
├── requirements.txt
├── DEVELOPMENT.md         # macOS environment notes
└── README.md
```

---

## Supported Backbones

| Name | Feature dim | Input size | Pre-training | FiLM |
|---|---|---|---|---|
| `efficientnet_b3` | 1,536 | 300 × 300 | ImageNet | ✓ |
| `efficientnet_b4` | 1,792 | 380 × 380 | ImageNet | ✓ |
| `efficientnet_b7` | 2,560 | 600 × 600 | ImageNet | ✓ |
| `resnet50`        | 2,048 | 224 × 224 | ImageNet | ✓ |
| `resnet101`       | 2,048 | 224 × 224 | ImageNet | ✓ |
| `retfound`        | 1,024 | 224 × 224 | MAE on 1.6 M fundus images | ✓ (Attention Rollout) |
| `timm/<name>`     | varies | varies | timm defaults | auto-detected |

> **RETFound** (Zhou et al., 2023) is a ViT-L/16 foundation model pre-trained on 1.6 million retinal images. Weights are downloaded automatically from HuggingFace (`rmaphoh/RETFound_cfp`). ViT backbones use Attention Rollout in place of GradCAM.

**Adding a new backbone** — implement a `builder(pretrained: bool) → nn.Module` that returns a spatial feature map `[B, C, h, w]` (CNN) or a CLS-token vector `[B, D]` (ViT), then register it in `BACKBONE_REGISTRY` inside `f2m/models/backbone.py`.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Smoke test — demo dataset (50 subjects / 100 images)

```bash
# macOS conda environments require two environment variables (see DEVELOPMENT.md)
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
    python scripts/demo_test.py --no_pretrained

# Linux / CUDA — no extra variables needed
python scripts/demo_test.py --no_pretrained
```

Expected output (all 8 steps pass):

```
── Step 1: Manifest loading        ✓  Expected 100 rows, got 100
── Step 2: Subject-level split     ✓  train=68 / val=16 / test=16
── Step 3: Dataset creation        ✓  all splits non-empty; 100/100 images cached
── Step 4: Batch shape inspection  ✓  [4,3,224,224] images / [4,3] targets / bool mask
── Step 5: Model forward pass      ✓  al / sph / cyl / se outputs; SE = SPH+CYL/2 (Δ=0)
── Step 6: Loss computation        ✓  finite; SE consistency loss included
── Step 7: Backward pass           ✓  gradients computed for 191 params
── Step 8: Metrics                 ✓  al MAE=0.200 / sph MAE=0.428 / cyl MAE=0.224
ALL CHECKS PASSED — F2M framework is operational.
```

### 3. Demo training (3 epochs, random weights)

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
    python scripts/train.py --config configs/demo.yaml --no_pretrained
```

### 4. Full training

Edit `data.manifest_path` and `data.img_root` in `configs/base.yaml`, then:

```bash
python scripts/train.py

# Override individual config keys via the command line (dot-notation)
python scripts/train.py \
    --set data.manifest_path /path/to/manifest.csv \
    --set data.img_root      /path/to/images \
    --set model.backbone     efficientnet_b3 \
    --set training.n_epochs  100
```

### 5. Resume training / test-set evaluation

```bash
# Resume from a saved checkpoint
python scripts/train.py --resume outputs/checkpoints/best_model.pt

# Run test-set evaluation only (no training)
python scripts/train.py --resume outputs/checkpoints/best_model.pt --eval_only
```

---

## Data Format

### Manifest CSV

`F2MDataset` reads a flat CSV where **each row represents one eye**:

| Column | Description | Required |
|---|---|---|
| `img_filename_demo` | Image filename relative to `img_root` | ✓ |
| `eye` | Laterality: `R` or `L` | ✓ |
| `subject_id` | Subject identifier — used for subject-level splitting | ✓ |
| `al` | Axial length (mm) | ✓ (primary target) |
| `asph` | Cycloplegic sphere (D) | ✓ |
| `acyl` | Cycloplegic cylinder (D, minus-cylinder convention) | ✓ |
| `age` | Age at examination (years, continuous) | ✓ |
| `myopia_grade` | Grade 0–3; used for stratified sampling | ✓ |
| `se` | Spherical equivalent = asph + acyl/2 (D) | optional |
| `cohort` | Cohort label for `cohort_filter` | optional |
| `wave` | Examination wave for longitudinal datasets | optional |

**Myopia grading thresholds** (cycloplegic SE):

| Grade | Label | SE criterion |
|---|---|---|
| 0 | Non-myopic | SE > −0.50 D |
| 1 | Mild | −3.00 < SE ≤ −0.50 D |
| 2 | Moderate | −6.00 < SE ≤ −3.00 D |
| 3 | High | SE ≤ −6.00 D |

### Image requirements

- **Formats**: JPEG, PNG (converted to RGB on load)
- **Resolution**: ≥ 1,500 × 1,500 px recommended (standard 45° CFP)
- **Laterality**: left-eye images are horizontally flipped to right-eye orientation before augmentation when `laterality_flip=True` (default)

---

## Augmentation Pipeline

### Training (`split="train"`)

```
Resize(input_size × 1.15, BICUBIC)
→ SafeRotation(±30°, PIL-native border fill)
→ RandomHorizontalFlip(p=0.5)
→ ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02)
→ CenterCrop(input_size)
→ CircularCrop(margin=0.97)      ← mask non-fundus corners (pure PIL, no NumPy)
→ ToTensor + Normalize(ImageNet µ/σ)
→ RandomErasing(p=0.1)
```

### Inference (`split="val"` / `"test"`)

```
Resize(input_size, BICUBIC) → CenterCrop(input_size)
→ CircularCrop → ToTensor + Normalize
```

`CircularCrop` fills the four corners of the image (outside the circular fundus disc) with the ImageNet channel mean, ensuring that the model never learns from the camera vignette. It is implemented entirely with `PIL.ImageDraw` to avoid a macOS segfault caused by mixing `np.array(pil_image)` with `T.ToTensor()` under Pillow ≥ 10 / NumPy 1.21.

---

## Loss Functions

### Primary task loss — uncertainty weighting (Kendall & Gal, 2018)

Each task $i$ has a learnable log-variance parameter $s_i = \log \sigma_i^2$:

$$\mathcal{L}_i^{\mathrm{UW}} = e^{-s_i} \cdot \mathcal{L}_i^{\mathrm{base}} + s_i$$

$$\mathcal{L}_{\mathrm{total}} = \sum_i \mathcal{L}_i^{\mathrm{UW}} + \lambda_{\mathrm{se}} \cdot \mathcal{L}_{\mathrm{se}}$$

The learnable weights automatically down-weight tasks with higher intrinsic noise and balance the scale mismatch between AL (millimetre) and SPH/CYL (dioptre).

### SE physics consistency loss

$$\mathcal{L}_{\mathrm{se}} = \mathrm{SmoothL1}\!\left(\hat{\mathrm{SPH}} + \hat{\mathrm{CYL}}/2,\; \mathrm{SE}_{\mathrm{true}}\right)$$

A soft constraint that encourages the SPH and CYL predictions to satisfy the clinical identity SE = SPH + CYL/2. Gradients flow back through both heads.

### Base loss

Configurable per run: `smooth_l1` (default) · `mse` · `mae`. Computed only on valid samples where `targets_mask = True` (handles missing measurements).

---

## Evaluation Metrics

`f2m.utils.metrics.evaluate_all_tasks()` returns a nested dict with all metrics:

### Per-task regression

| Metric | Description |
|---|---|
| MAE | Mean absolute error — primary clinical reporting metric |
| RMSE | Root mean square error |
| R² | Coefficient of determination |
| Pearson *r* | Correlation coefficient and two-tailed *p*-value |
| Bland-Altman | Mean difference, SD of differences, 95% limits of agreement (with CI), proportional bias test |

### Cross-task

| Metric | Description |
|---|---|
| SE consistency MAE | `mean(|SE_true − (SPH_pred + CYL_pred/2)|)` |
| Myopia grade accuracy | 4-class accuracy from thresholding predicted SE |
| Macro F1 | Macro-averaged F1 across grades 0–3 |
| SER residual analysis | Linear regression of `(SE_pred − SE_true)` on `SE_true` — slope, intercept, R² |

---

## Interpretability: GradCAM

```python
from f2m.analysis.gradcam import MultiHeadGradCAM

cam_engine = MultiHeadGradCAM(model)            # auto-detects target layer
cam_maps   = cam_engine.compute(image, age)     # {task: ndarray [H, W] in [0, 1]}

# 4-panel figure: original + AL cam + SPH cam + CYL cam
fig = cam_engine.visualize_all_tasks(
    image, age,
    save_path="outputs/gradcam/subject_01.png",
)
cam_engine.remove_hooks()
```

Each per-task map is computed by backpropagating **only through that task's head** (with `retain_graph=True`), isolating the task-specific gradient signal despite the shared backbone:

```
α_c = GAP(∂ score_task / ∂ A_c)          # global average-pooled gradients
CAM = ReLU(Σ_c  α_c · A_c)               # [h, w] — upsample → min-max normalise
```

- **CNN backbones** (EfficientNet, ResNet): standard Gradient-weighted Class Activation Maps
- **ViT backbones** (RETFound): Attention Rollout (Abnar & Zuidema, 2020) — aggregates self-attention weights across transformer layers

---

## Configuration

All hyperparameters are controlled via YAML files. `scripts/train.py` merges `configs/base.yaml` with a run-specific override file:

```bash
# Use a specific config file
python scripts/train.py --config configs/demo.yaml

# Override any key with dot-notation (repeatable)
python scripts/train.py \
    --set model.backbone          retfound   \
    --set training.n_epochs       80         \
    --set loss.consistency_weight 0.5        \
    --set loss.weighting          uncertainty

# Offline environment — skip pre-trained weight download
python scripts/train.py --no_pretrained
```

Key sections in `configs/base.yaml`:

| Section | Controls |
|---|---|
| `data` | Manifest path, image root, target columns, caching |
| `split` | Train / val / test fractions, stratification, random seed |
| `model` | Backbone, neck/head dimensions, dropout, FiLM stages |
| `loss` | Weighting strategy, SE consistency weight, base loss function |
| `training` | Epochs, batch size, learning rates, scheduler, gradient clipping |
| `augmentation` | Input size, rotation, jitter, circular crop, random erasing |
| `evaluation` | Myopia grade thresholds, Bland-Altman confidence level |
| `paths` | Output, checkpoint, log, GradCAM directories |

---

## Python API

### Dataset

```python
from f2m.data.dataset import F2MDataset
from f2m.data.splits  import subject_level_split
from f2m.data.transforms import build_transform
import pandas as pd

manifest = pd.read_csv("manifest.csv")
train_df, val_df, test_df = subject_level_split(
    manifest,
    train_frac=0.70, val_frac=0.15, test_frac=0.15,
    stratify_col="myopia_grade", subject_col="subject_id",
    random_state=42,
)

ds_train = F2MDataset(
    manifest_path=train_df,
    img_root="/path/to/images",
    img_col="img_filename",
    split="train",
    target_cols=["al", "asph", "acyl"],
    transform=build_transform("train", input_size=300),
    laterality_flip=True,
    cache_images=False,          # set True for small datasets (≤ ~500 images)
)

# Pass these to val/test datasets so all splits share the same normalisation
age_stats    = ds_train.get_age_stats()
target_stats = ds_train.get_target_stats()
```

### Model

```python
from f2m.models.f2mnet import F2MNet
import torch

model = F2MNet(
    backbone_name="efficientnet_b3",   # or "resnet50", "retfound", "timm/convnext_base"
    task_names=["al", "sph", "cyl"],
    pretrained=True,
    neck_hidden=512,
    neck_out=256,
    head_hidden=128,
    dropout_neck=0.30,
    dropout_head=0.10,
)

image = torch.randn(4, 3, 300, 300)
age   = torch.tensor([8.5, 10.2, 9.1, 11.3])

with torch.no_grad():
    out = model(image, age)
# out = {"al": [4,1], "sph": [4,1], "cyl": [4,1], "se": [4,1], "shared_feat": [4,256]}
```

### Loss

```python
from f2m.models.losses import F2MLoss

criterion = F2MLoss(
    task_names=["al", "sph", "cyl"],
    weighting="uncertainty",       # Kendall & Gal 2018
    consistency_weight=0.5,        # SE physics constraint weight
)

total_loss, info = criterion(
    predictions=out,
    targets=targets,               # [B, 3] float tensor
    targets_mask=mask,             # [B, 3] bool tensor
    se_true=se_true,               # [B] ground-truth SE
)
# info: {"loss_total", "loss_task_0", "log_var_0", "loss_se_consistency", …}
```

### Evaluation

```python
from f2m.utils.metrics import evaluate_all_tasks
import numpy as np

results = evaluate_all_tasks(
    predictions={"al": al_pred, "sph": sph_pred, "cyl": cyl_pred},
    targets    ={"al": al_true, "sph": sph_true, "cyl": cyl_true},
    masks      ={"al": mask_al, "sph": mask_sph, "cyl": mask_cyl},
    se_true=se_true,
)

print(results["per_task"]["al"]["mae"])              # AL MAE
print(results["per_task"]["se"]["bland_altman"])     # SE Bland-Altman dict
print(results["cross_task"]["myopia_grade"]["accuracy"])   # Grade accuracy
print(results["cross_task"]["ser_residual"]["slope"])      # SER residual slope
```

---

## Clinical Datasets

This framework is developed alongside a three-centre, five-cohort governed dataset. See `/data/myopia_prediction/README.md` for full TRIPOD+AI 2024 cohort characteristics.

| Cohort | Site | Design | Subjects | Fundus images |
|---|---|---|---|---|
| Tibet BSL | Lhasa, Tibet AR | Longitudinal — 5 waves (2019–2024) | 2,187 | ✓ all waves |
| Beijing-120 | Beijing | Cross-sectional | 120 | — |
| Beijing-1387 | Beijing | Cross-sectional | 1,387 | — |
| ZZ cyc_eye | Zhengzhou | Cross-sectional | 18 | ✓ |
| ZZ kerui | Zhengzhou (Kerui Eye Hospital) | Cross-sectional | 91 | ✓ |

The **demo dataset** (`/data/fundus2myopia/demo/`, 50 subjects × 2 eyes = 100 images) provides bilateral CFPs paired with age, cycloplegic sphere/cylinder, and axial length annotations drawn from the above cohorts via stratified sampling.

---

## Requirements

| Package | Version |
|---|---|
| torch | ≥ 2.2 |
| torchvision | ≥ 0.17 |
| timm | ≥ 0.9.12 |
| numpy | ≥ 1.24 |
| pandas | ≥ 2.0 |
| Pillow | ≥ 10.0 |
| scipy | ≥ 1.11 |
| scikit-learn | ≥ 1.3 |
| PyYAML | ≥ 6.0 |
| huggingface_hub | ≥ 0.20 |
| matplotlib | ≥ 3.7 |
| einops | ≥ 0.7 (ViT helpers) |
| tqdm | ≥ 4.65 |

> **macOS users**: set `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` before running any script to avoid segfaults caused by OpenMP conflicts between PyTorch and scikit-learn. See [DEVELOPMENT.md](DEVELOPMENT.md) for details.

---

## References

- Kendall, A. & Gal, Y. (2018). *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.* CVPR.
- Perez, E. et al. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI.
- Selvaraju, R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
- Abnar, S. & Zuidema, W. (2020). *Quantifying Attention Flow in Transformers.* ACL.
- Zhou, Y. et al. (2023). *A Foundation Model for Generalizable Disease Detection from Retinal Images.* Nature.
- Collins, G. et al. (2024). *TRIPOD+AI Statement: Updated Guidance for Reporting Clinical Prediction Models that Use Regression or Machine Learning Methods.* BMJ.

---

## Citation

```bibtex
@misc{f2m2026,
  title   = {Fundus2Myopia: A Multi-Task Deep Learning Framework for Myopia Parameter Prediction from Fundus Photographs},
  author  = {},
  year    = {2026},
}
```

If you use the RETFound backbone, please also cite the original work:

```bibtex
@article{zhou2023retfound,
  title   = {A Foundation Model for Generalizable Disease Detection from Retinal Images},
  author  = {Zhou, Yukun and others},
  journal = {Nature},
  year    = {2023},
}
```

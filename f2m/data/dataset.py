"""
f2m/data/dataset.py
===================
Standard PyTorch Dataset for Fundus2Myopia (F2M).

Supports
--------
* Single-eye per-row manifests produced by ManifestBuilder or the demo
  manifest at data/fundus2myopia/demo/manifest.csv.
* Arbitrary regression target columns (AL, SPH, CYL, SE, K1 …).
* Optional auxiliary targets (OD area, PPA ratio).
* Laterality normalisation: L-eye images flipped → right-eye space.
* Robust image loading: corrupt images return a zero tensor + all-False mask.
* In-RAM caching for small datasets (demo: 100 images ≈ 100 MB).
* Z-score normalisation of targets (statistics computed from this split or
  passed explicitly so val / test use train-split statistics).

Usage
-----
>>> from f2m.data.dataset import F2MDataset
>>> from f2m.data.transforms import build_transform
>>> ds = F2MDataset(
...     manifest_path="/path/to/demo/manifest.csv",
...     img_root="/path/to/demo/images",
...     img_col="img_filename_demo",
...     split="train",
...     target_cols=["al", "asph", "acyl"],
...     transform=build_transform("train", input_size=300),
... )
>>> batch = ds[0]
>>> batch["image"].shape, batch["targets"], batch["targets_mask"]
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from f2m.data.transforms import build_transform

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

# Canonical target metadata: physical units and reasonable clip ranges.
TASK_META: dict[str, dict] = {
    "al":   {"unit": "mm", "clip": (15.0,  35.0)},
    "asph": {"unit": "D",  "clip": (-25.0,  15.0)},
    "acyl": {"unit": "D",  "clip": (-15.0,   0.0)},
    "se":   {"unit": "D",  "clip": (-25.0,  15.0)},
    "acd":  {"unit": "mm", "clip": (  1.0,   6.0)},
    "k1":   {"unit": "D",  "clip": ( 35.0,  55.0)},
    "k2":   {"unit": "D",  "clip": ( 35.0,  55.0)},
    "cct":  {"unit": "μm", "clip": (400.0, 700.0)},
    "lt":   {"unit": "mm", "clip": (  2.0,   7.0)},
    "prse": {"unit": "D",  "clip": (-25.0,  15.0)},
    "od_area":          {"unit": "px²", "clip": (0.0, 1e8)},
    "ppa_area":         {"unit": "px²", "clip": (0.0, 1e8)},
    "ppa_od_area_ratio":{"unit": "",    "clip": (0.0,  50.0)},
}

# Column in the manifest that holds the image filename / path
DEFAULT_IMG_COL = "img_filename_demo"


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_image(path: Path) -> Image.Image | None:
    """
    Open an image file, converting to RGB.

    Returns None if the file is missing, truncated, or not a valid image.
    Emits a logger warning rather than raising an exception so that a single
    bad image does not abort training.
    """
    if not path.exists():
        logger.warning("Image not found: %s", path)
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = Image.open(path)
            img.verify()           # catches truncated JPEG
        img = Image.open(path).convert("RGB")
        return img
    except (UnidentifiedImageError, OSError, Exception) as exc:
        logger.warning("Cannot open image %s: %s", path, exc)
        return None


def _compute_stats(series: pd.Series) -> dict[str, float]:
    """
    Return {"mean": float, "std": float} for a numeric series, ignoring NaN.

    Edge cases
    ----------
    * n == 0 : mean=0.0, std=1.0  (no data — identity transform)
    * n == 1 : std is undefined with ddof=1 (pandas returns NaN), so we
               fall back to std=1.0 to avoid NaN propagation into _zscore.
    """
    vals = series.dropna()
    n    = len(vals)
    mean = float(vals.mean()) if n > 0 else 0.0
    # std() with ddof=1 returns NaN for n=1; NaN+1e-8 is still NaN inside
    # _zscore, which would poison every normalised value for that column.
    std  = float(vals.std()) if n >= 2 else 1.0
    return {"mean": mean, "std": std}


def _zscore(value: float | np.ndarray, mean: float, std: float) -> float | np.ndarray:
    return (value - mean) / (std + 1e-8)


def _inv_zscore(value: float | np.ndarray, mean: float, std: float) -> float | np.ndarray:
    return value * (std + 1e-8) + mean


# ── main dataset class ────────────────────────────────────────────────────────

class F2MDataset(Dataset):
    """
    Per-eye image + scalar label dataset for Fundus2Myopia.

    Parameters
    ----------
    manifest_path : str | Path | pd.DataFrame
        Path to (or pre-loaded) per-eye manifest CSV.
        Required columns: img_col, age, eye, subject_id + all target_cols.
    img_root : str | Path | None
        Directory that contains the image files named in `img_col`.
        If None, `img_col` must contain absolute paths.
    img_col : str
        Name of the manifest column holding the image filename or absolute path.
        Default: "img_filename_demo".
    split : str
        "train" | "val" | "test" | "demo" — used only if transform is None
        (auto-builds a suitable transform).
    target_cols : sequence of str
        Regression target columns. Default: ("al", "asph", "acyl").
        The model predicts these; SE is always derived post-hoc.
    aux_target_cols : sequence of str
        Auxiliary regression targets (e.g. od_area, ppa_od_area_ratio).
        Rows missing these targets receive mask=False (excluded from aux loss).
    transform : Callable | None
        torchvision-compatible image transform. If None, auto-built from `split`.
    input_size : int
        Passed to the auto-built transform; ignored if transform is provided.
    age_stats : dict | None
        {"mean": float, "std": float}. If None, computed from this split.
        Always pass train-split statistics explicitly for val / test.
    target_stats : dict | None
        {col: {"mean": float, "std": float}}. If None, computed from this split.
        Pass train statistics for val / test.
    normalize_targets : bool
        Whether to z-score the targets. Default False.
        If True, the model predicts in normalised space; remember to invert
        at evaluation time using `target_stats`.
    laterality_flip : bool
        Horizontally flip left-eye images so all images appear in right-eye
        (nasal-to-right) orientation. Default True.
    cohort_filter : str | list[str] | None
        If set, only rows matching the `cohort` column are included.
    require_all_targets : bool
        If True, rows where ANY target_col is NaN are dropped. Default True.
        Set False to allow partial labels (handled by targets_mask).
    cache_images : bool
        Load all images into RAM at init. Safe for demo (≤ 200 images);
        disable for full dataset. Default False.

    Returns (per __getitem__)
    -------------------------
    {
        "image"        : torch.Tensor [3, H, W]  float32
        "age"          : torch.Tensor []          float32  (z-scored)
        "age_missing"  : bool                     (True if age was NaN → imputed)
        "eye"          : torch.Tensor []          int64 (0=L, 1=R)
        "targets"      : torch.Tensor [n_tasks]   float32
        "targets_mask" : torch.Tensor [n_tasks]   bool  (False → NaN target)
        "aux_targets"  : torch.Tensor [n_aux]     float32 (0.0 where missing)
        "aux_mask"     : torch.Tensor [n_aux]     bool
        "se_true"      : torch.Tensor []          float32 (asph + acyl/2 if both present)
        "myopia_grade" : torch.Tensor []          int64   (-1 if unknown)
        "sample_id"    : str
        "subject_id"   : int  (-1 if absent)
        "wave"         : int  (-1 if absent / NaN)
        "cohort"       : str
    }
    """

    TARGET_COLS_DEFAULT  = ("al", "asph", "acyl")
    AUX_TARGET_COLS_DEFAULT = ()

    def __init__(
        self,
        manifest_path: str | Path | pd.DataFrame,
        img_root: str | Path | None = None,
        img_col: str = DEFAULT_IMG_COL,
        split: str = "train",
        target_cols: Sequence[str] = TARGET_COLS_DEFAULT,
        aux_target_cols: Sequence[str] = AUX_TARGET_COLS_DEFAULT,
        transform: Callable | None = None,
        input_size: int = 300,
        age_stats: dict | None = None,
        target_stats: dict | None = None,
        normalize_targets: bool = False,
        laterality_flip: bool = True,
        cohort_filter: str | list[str] | None = None,
        require_all_targets: bool = True,
        cache_images: bool = False,
    ) -> None:
        self.img_root = Path(img_root) if img_root is not None else None
        self.img_col  = img_col
        self.split    = split
        self.input_size      = input_size          # used for corrupt-image fallback shape
        self.target_cols     = list(target_cols)
        self.aux_target_cols = list(aux_target_cols)
        self.laterality_flip = laterality_flip
        self.normalize_targets = normalize_targets
        self.cache_images = cache_images

        # ── load manifest ────────────────────────────────────────────────────
        if isinstance(manifest_path, pd.DataFrame):
            df = manifest_path.copy()
        else:
            df = pd.read_csv(manifest_path, low_memory=False)

        # ensure img_col exists
        if img_col not in df.columns:
            raise ValueError(
                f"img_col='{img_col}' not found in manifest. "
                f"Available: {list(df.columns)}"
            )

        # ── cohort filter ────────────────────────────────────────────────────
        if cohort_filter is not None:
            if isinstance(cohort_filter, str):
                cohort_filter = [cohort_filter]
            if "cohort" in df.columns:
                df = df[df["cohort"].isin(cohort_filter)].copy()

        # ── drop rows missing primary targets ────────────────────────────────
        missing_tgt_cols = [c for c in self.target_cols if c not in df.columns]
        if missing_tgt_cols:
            raise ValueError(
                f"Target columns not in manifest: {missing_tgt_cols}"
            )
        if require_all_targets:
            before = len(df)
            df = df.dropna(subset=self.target_cols).copy()
            dropped = before - len(df)
            if dropped:
                logger.info(
                    "Dropped %d rows with NaN in required target cols %s",
                    dropped, self.target_cols,
                )

        if len(df) == 0:
            raise ValueError(
                "No valid rows remain after filtering. "
                "Check manifest_path, target_cols, and cohort_filter."
            )

        # ── clamp targets to physical ranges ─────────────────────────────────
        for col in self.target_cols + self.aux_target_cols:
            if col in df.columns and col in TASK_META:
                lo, hi = TASK_META[col]["clip"]
                n_out = df[col].notna().sum() - df[col].clip(lo, hi).notna().sum()
                df[col] = df[col].clip(lo, hi)

        self.df = df.reset_index(drop=True)

        # ── age statistics ───────────────────────────────────────────────────
        if "age" not in self.df.columns:
            logger.warning("'age' column not found; will use age=0.0 for all rows.")
            self.df["age"] = 0.0
        if age_stats is None:
            age_stats = _compute_stats(self.df["age"])
            logger.debug("Computed age stats from split: %s", age_stats)
        self.age_stats: dict = age_stats

        # ── target statistics ────────────────────────────────────────────────
        if target_stats is None:
            target_stats = {
                col: _compute_stats(self.df[col])
                for col in self.target_cols
                if col in self.df.columns
            }
        self.target_stats: dict = target_stats

        # ── transform ────────────────────────────────────────────────────────
        self.transform = transform or build_transform(
            split=split, input_size=input_size
        )

        # ── image cache ──────────────────────────────────────────────────────
        self._image_cache: dict[int, Image.Image | None] = {}
        if self.cache_images:
            logger.info("Caching %d images in RAM …", len(self.df))
            for idx in range(len(self.df)):
                self._image_cache[idx] = self._open_image(idx)
            n_ok = sum(v is not None for v in self._image_cache.values())
            logger.info("Cached %d/%d images successfully.", n_ok, len(self.df))

    # ── public API ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # ── image ─────────────────────────────────────────────────────────────
        if self.cache_images:
            img = self._image_cache.get(idx)
        else:
            img = self._open_image(idx)

        img_ok = img is not None
        if img is None:
            # Corrupt / missing image: return zero tensor, mask all targets False.
            # Shape must match the valid-image transform output; use self.input_size
            # (stored at __init__) rather than a hardcoded 300 to avoid collation
            # crashes when input_size != 300.
            img_tensor = torch.zeros(3, self.input_size, self.input_size)
        else:
            # laterality flip: left-eye → right-eye orientation
            if self.laterality_flip and str(row.get("eye", "R")).upper() == "L":
                import torchvision.transforms.functional as TF2
                img = TF2.hflip(img)
            img_tensor = self.transform(img)          # [3, H, W]

        # ── age ───────────────────────────────────────────────────────────────
        age_raw = row.get("age", np.nan)
        age_missing = pd.isna(age_raw)
        if age_missing:
            age_raw = self.age_stats["mean"]           # impute with train mean
        age_val = float(age_raw)
        age_val = max(0.0, min(age_val, 25.0))         # clip [0, 25] yr
        age_norm = _zscore(age_val, self.age_stats["mean"], self.age_stats["std"])
        age_tensor = torch.tensor(age_norm, dtype=torch.float32)

        # ── gender ────────────────────────────────────────────────────────────
        # Convention (matching governed datasets): 0=unknown, 1=male, 2=female.
        # NaN / missing / out-of-range values → 0 (unknown), which the
        # ConditionEncoder maps to a zero embedding (no gender signal).
        gender_raw = row.get("gender", np.nan)
        if pd.isna(gender_raw):
            gender_val = 0
        else:
            gender_val = int(gender_raw)
            if gender_val not in (1, 2):
                gender_val = 0   # clamp unexpected codes to "unknown"
        gender_tensor = torch.tensor(gender_val, dtype=torch.long)

        # ── eye laterality ────────────────────────────────────────────────────
        eye_str = str(row.get("eye", "R")).upper()
        eye_val = 1 if eye_str == "R" else 0

        # ── primary targets ───────────────────────────────────────────────────
        targets = []
        targets_mask = []
        for col in self.target_cols:
            raw = row.get(col, np.nan)
            is_valid = (not pd.isna(raw)) and img_ok
            targets_mask.append(is_valid)
            if is_valid:
                v = float(raw)
                if self.normalize_targets and col in self.target_stats:
                    v = _zscore(v, **self.target_stats[col])
                targets.append(v)
            else:
                targets.append(0.0)  # placeholder; masked out by targets_mask

        targets_tensor      = torch.tensor(targets,      dtype=torch.float32)
        targets_mask_tensor = torch.tensor(targets_mask, dtype=torch.bool)

        # ── auxiliary targets ─────────────────────────────────────────────────
        aux_targets = []
        aux_mask    = []
        for col in self.aux_target_cols:
            raw = row.get(col, np.nan) if col in self.df.columns else np.nan
            is_valid = (not pd.isna(raw)) and img_ok
            aux_mask.append(is_valid)
            aux_targets.append(float(raw) if is_valid else 0.0)
        aux_targets_tensor = torch.tensor(aux_targets, dtype=torch.float32)
        aux_mask_tensor    = torch.tensor(aux_mask,    dtype=torch.bool)

        # ── derived SE ────────────────────────────────────────────────────────
        asph_raw = row.get("asph", np.nan)
        acyl_raw = row.get("acyl", np.nan)
        if not pd.isna(asph_raw) and not pd.isna(acyl_raw):
            se_true = float(asph_raw) + float(acyl_raw) / 2.0
        else:
            se_raw = row.get("se", np.nan)
            se_true = float(se_raw) if not pd.isna(se_raw) else float("nan")
        se_tensor = torch.tensor(se_true, dtype=torch.float32)

        # ── metadata scalars ─────────────────────────────────────────────────
        grade = row.get("myopia_grade", np.nan)
        grade_int = int(grade) if not pd.isna(grade) else -1

        subject_id = row.get("subject_id", np.nan)
        subject_int = int(subject_id) if not pd.isna(subject_id) else -1

        wave = row.get("wave", np.nan)
        wave_int = int(wave) if not pd.isna(wave) else -1

        return {
            "image":         img_tensor,
            "age":           age_tensor,
            "age_missing":   age_missing,
            "gender":        gender_tensor,
            "eye":           torch.tensor(eye_val, dtype=torch.long),
            "targets":       targets_tensor,
            "targets_mask":  targets_mask_tensor,
            "aux_targets":   aux_targets_tensor,
            "aux_mask":      aux_mask_tensor,
            "se_true":       se_tensor,
            "myopia_grade":  torch.tensor(grade_int,   dtype=torch.long),
            "sample_id":     str(row.get("sample_id", idx)),
            "subject_id":    subject_int,
            "wave":          wave_int,
            "cohort":        str(row.get("cohort", "")),
        }

    # ── statistics / helpers ──────────────────────────────────────────────────

    def get_age_stats(self) -> dict[str, float]:
        """Return {"mean": float, "std": float} computed on this split."""
        return dict(self.age_stats)

    def get_target_stats(self) -> dict[str, dict[str, float]]:
        """Return per-task {"al": {"mean", "std"}, …} for inverse normalisation."""
        return {k: dict(v) for k, v in self.target_stats.items()}

    def inverse_transform_targets(
        self,
        targets: np.ndarray,                  # [N, n_tasks]
    ) -> np.ndarray:
        """Inverse z-score normalisation. Identity if normalize_targets=False."""
        if not self.normalize_targets:
            return targets
        out = targets.copy()
        for i, col in enumerate(self.target_cols):
            if col in self.target_stats:
                out[:, i] = _inv_zscore(targets[:, i], **self.target_stats[col])
        return out

    def subject_ids(self) -> np.ndarray:
        """Return array of subject_id values aligned with dataset rows."""
        col = "subject_id" if "subject_id" in self.df.columns else None
        if col is None:
            return np.arange(len(self.df))
        return self.df[col].fillna(-1).values.astype(int)

    def myopia_grades(self) -> np.ndarray:
        """Return array of myopia_grade values, -1 where unknown."""
        col = "myopia_grade"
        if col not in self.df.columns:
            return np.full(len(self.df), -1, dtype=int)
        return self.df[col].fillna(-1).values.astype(int)

    # ── private helpers ───────────────────────────────────────────────────────

    def _resolve_image_path(self, idx: int) -> Path:
        row = self.df.iloc[idx]
        fname = str(row[self.img_col])
        if self.img_root is not None:
            return self.img_root / fname
        # fallback: treat fname as absolute path
        return Path(fname)

    def _open_image(self, idx: int) -> Image.Image | None:
        path = self._resolve_image_path(idx)
        return _load_image(path)

    def __repr__(self) -> str:
        return (
            f"F2MDataset(split={self.split!r}, n={len(self)}, "
            f"targets={self.target_cols}, "
            f"img_root={str(self.img_root)!r})"
        )

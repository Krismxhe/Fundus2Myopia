"""
f2m/data/splits.py
==================
Subject-level stratified train / val / test splitting.

Why subject-level?
------------------
Tibet BSL is longitudinal: the same child appears in waves 2020, 2023, 2024.
If we split by image row, the same subject's retina appears in both train and
test, inflating performance metrics (data leakage). All rows belonging to one
subject_id must land in the same fold.

The demo dataset (50 subjects, single wave) does not have this issue but uses
the same splitting logic for consistency.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


def subject_level_split(
    manifest: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    test_frac:  float = 0.15,
    stratify_col: str = "myopia_grade",
    subject_col:  str = "subject_id",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a per-eye manifest into train / val / test at the subject level.

    All eye-level rows sharing the same `subject_col` value land in the same
    partition. Stratification is applied at the subject level using the modal
    value of `stratify_col` across that subject's rows.

    Parameters
    ----------
    manifest : per-eye manifest DataFrame (one row per image).
    train_frac, val_frac, test_frac : must sum to 1.0.
    stratify_col : column used for stratification; typically "myopia_grade".
        Subjects with NaN in this column are assigned to the most frequent
        stratum label.
    subject_col : column identifying the unique subject. Defaults "subject_id".
    random_state : reproducibility seed.

    Returns
    -------
    train_df, val_df, test_df : three DataFrames with the same columns as
    `manifest`, containing only the rows assigned to each partition.

    Notes
    -----
    * For each subject, the baseline (worst-grade) wave is used for
      stratification when multiple waves exist, ensuring grade balance.
    * Cohort balance (Tibet vs ZZ) is preserved naturally if cohort-level
      grade distributions match; add cohort as a secondary stratification
      column if needed.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "train_frac + val_frac + test_frac must equal 1.0"

    if subject_col not in manifest.columns:
        logger.warning(
            "'%s' column not found; falling back to row-level split.",
            subject_col,
        )
        return _row_level_split(manifest, train_frac, val_frac, test_frac,
                                stratify_col, random_state)

    # ── build subject-level summary ────────────────────────────────────────────
    subj_df = _build_subject_summary(manifest, subject_col, stratify_col)

    n_subjects = len(subj_df)
    logger.info("Total unique subjects: %d", n_subjects)

    # ── handle NaN strata ─────────────────────────────────────────────────────
    if subj_df["stratum"].isna().any():
        mode_result = subj_df["stratum"].mode()
        if len(mode_result) == 0:
            # All subjects have NaN stratum (stratify_col absent or all-NaN).
            # Fall back to a single dummy stratum so StratifiedShuffleSplit
            # still works; the split will be random rather than stratified.
            logger.warning(
                "All subjects have NaN in stratify column '%s'; "
                "stratification is disabled — split will be random.",
                subj_df.columns[1] if len(subj_df.columns) > 1 else "stratum",
            )
            subj_df["stratum"] = 0
        else:
            subj_df["stratum"] = subj_df["stratum"].fillna(mode_result[0])

    strata = subj_df["stratum"].values.astype(int)

    # ── first split: isolate test set ─────────────────────────────────────────
    test_size = test_frac
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    idx_trainval, idx_test = next(sss1.split(subj_df, strata))

    # ── second split: train vs val within trainval ────────────────────────────
    val_size_within = val_frac / (train_frac + val_frac)
    strata_trainval = strata[idx_trainval]
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size_within, random_state=random_state + 1
    )
    idx_rel_train, idx_rel_val = next(
        sss2.split(idx_trainval, strata_trainval)
    )
    idx_train = idx_trainval[idx_rel_train]
    idx_val   = idx_trainval[idx_rel_val]

    # ── map subject indices back to row indices ────────────────────────────────
    train_subjects = set(subj_df.iloc[idx_train][subject_col].tolist())
    val_subjects   = set(subj_df.iloc[idx_val][subject_col].tolist())
    test_subjects  = set(subj_df.iloc[idx_test][subject_col].tolist())

    train_df = manifest[manifest[subject_col].isin(train_subjects)].copy()
    val_df   = manifest[manifest[subject_col].isin(val_subjects)].copy()
    test_df  = manifest[manifest[subject_col].isin(test_subjects)].copy()

    # ── add split column ──────────────────────────────────────────────────────
    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        df["split"] = name

    _log_split_stats(train_df, val_df, test_df, stratify_col)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_subject_summary(
    manifest: pd.DataFrame,
    subject_col: str,
    stratify_col: str,
) -> pd.DataFrame:
    """
    Return one row per unique subject.

    Stratification is by the MAXIMUM severity (highest grade / worst refraction)
    observed across all waves, ensuring subjects with any high-grade visit are
    distributed to each split.
    """
    if stratify_col in manifest.columns:
        # Use the most severe grade observed for that subject
        grade_agg = (
            manifest.groupby(subject_col)[stratify_col]
            .max()                           # worst grade across waves
            .reset_index()
            .rename(columns={stratify_col: "stratum"})
        )
    else:
        unique_ids = manifest[subject_col].unique()
        grade_agg = pd.DataFrame({
            subject_col: unique_ids,
            "stratum": np.zeros(len(unique_ids), dtype=int),
        })
    return grade_agg


def _row_level_split(
    manifest: pd.DataFrame,
    train_frac: float,
    val_frac:   float,
    test_frac:  float,
    stratify_col: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fallback row-level split (used when subject_col is absent)."""
    strata = manifest[stratify_col].fillna(0).values.astype(int) \
             if stratify_col in manifest.columns \
             else np.zeros(len(manifest), dtype=int)

    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_frac, random_state=random_state
    )
    idx_tv, idx_test = next(sss1.split(manifest, strata))

    val_size_within = val_frac / (train_frac + val_frac)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size_within, random_state=random_state + 1
    )
    idx_tr, idx_val = next(sss2.split(idx_tv, strata[idx_tv]))

    train_df = manifest.iloc[idx_tv[idx_tr]].copy()
    val_df   = manifest.iloc[idx_tv[idx_val]].copy()
    test_df  = manifest.iloc[idx_test].copy()
    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        df["split"] = name
    _log_split_stats(train_df, val_df, test_df, stratify_col)
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _log_split_stats(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    stratify_col: str,
) -> None:
    total = len(train_df) + len(val_df) + len(test_df)
    logger.info(
        "Split sizes (rows): train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        len(train_df), 100 * len(train_df) / total,
        len(val_df),   100 * len(val_df)   / total,
        len(test_df),  100 * len(test_df)  / total,
    )
    if stratify_col in train_df.columns:
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            dist = df[stratify_col].value_counts(normalize=True).sort_index()
            logger.debug("%s grade dist: %s", name, dist.to_dict())

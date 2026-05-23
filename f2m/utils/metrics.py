"""
f2m/utils/metrics.py
====================
Regression and agreement metrics for Fundus2Myopia evaluation.

Per-task metrics
----------------
MAE, RMSE, R², Pearson r (with p-value)

Cross-task metrics
------------------
SE consistency: |SE_pred - (SPH_pred + CYL_pred/2)|
SER residual decomposition

Agreement analysis
------------------
Bland-Altman: mean difference, SD of differences, 95% limits of agreement,
proportional bias test (regression of difference on mean, p-value)

Myopia grade
------------
SE → grade thresholding, accuracy, per-class report
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from scipy import stats


# ── type aliases ─────────────────────────────────────────────────────────────
Array = np.ndarray


# ─────────────────────────────────────────────────────────────────────────────
# Basic regression metrics
# ─────────────────────────────────────────────────────────────────────────────

def mae(y_true: Array, y_pred: Array) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: Array, y_pred: Array) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: Array, y_pred: Array) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def pearson(y_true: Array, y_pred: Array) -> tuple[float, float]:
    """Return (r, p_value)."""
    if len(y_true) < 3:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_val, p_val = stats.pearsonr(y_true, y_pred)
    return float(r_val), float(p_val)


def regression_metrics(
    y_true: Array,
    y_pred: Array,
    *,
    task_name: str = "",
) -> dict:
    """
    Compute MAE, RMSE, R², Pearson r for a single task.

    Parameters
    ----------
    y_true, y_pred : 1-D float arrays (after applying valid mask).
    task_name : optional label for the returned dict keys.

    Returns
    -------
    dict with keys: n, mae, rmse, r2, pearson_r, pearson_p
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # remove NaN pairs
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[valid], y_pred[valid]
    n = len(y_true)

    if n == 0:
        return {k: float("nan") for k in ["n","mae","rmse","r2","pearson_r","pearson_p"]}

    r_val, p_val = pearson(y_true, y_pred)
    return {
        "n":          n,
        "mae":        mae(y_true, y_pred),
        "rmse":       rmse(y_true, y_pred),
        "r2":         r2(y_true, y_pred),
        "pearson_r":  r_val,
        "pearson_p":  p_val,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bland-Altman analysis
# ─────────────────────────────────────────────────────────────────────────────

def bland_altman(
    reference: Array,
    measurement: Array,
    confidence: float = 0.95,
) -> dict:
    """
    Bland-Altman agreement analysis.

    Parameters
    ----------
    reference : true / reference values (clinical measurement)
    measurement : predicted / test values
    confidence : confidence level for limits of agreement CI. Default 0.95.

    Returns
    -------
    dict with keys:
      mean_diff      : mean(measurement - reference)
      sd_diff        : SD of differences
      loa_upper      : mean_diff + 1.96 * sd_diff
      loa_lower      : mean_diff - 1.96 * sd_diff
      loa_upper_ci   : (lo, hi) confidence interval on loa_upper
      loa_lower_ci   : (lo, hi) confidence interval on loa_lower
      proportional_bias_r : Pearson r between diff and mean (proportional bias)
      proportional_bias_p : p-value
      n
    """
    ref  = np.asarray(reference, dtype=float)
    meas = np.asarray(measurement, dtype=float)
    valid = ~(np.isnan(ref) | np.isnan(meas))
    ref, meas = ref[valid], meas[valid]
    n = len(ref)
    if n == 0:
        return {k: float("nan") for k in [
            "mean_diff","sd_diff","loa_upper","loa_lower","n",
            "proportional_bias_r","proportional_bias_p"]}

    diff = meas - ref
    mean_d = float(np.mean(diff))
    sd_d   = float(np.std(diff, ddof=1))
    loa_u  = mean_d + 1.96 * sd_d
    loa_l  = mean_d - 1.96 * sd_d

    # CI on limits of agreement (Bland & Altman 1999)
    z = stats.norm.ppf((1 + confidence) / 2)
    se_loa = np.sqrt(3 * sd_d ** 2 / n)
    loa_u_ci = (loa_u - z * se_loa, loa_u + z * se_loa)
    loa_l_ci = (loa_l - z * se_loa, loa_l + z * se_loa)

    means = (ref + meas) / 2.0
    r_bias, p_bias = pearson(means, diff)

    return {
        "n":                     n,
        "mean_diff":             mean_d,
        "sd_diff":               sd_d,
        "loa_upper":             loa_u,
        "loa_lower":             loa_l,
        "loa_upper_ci":          loa_u_ci,
        "loa_lower_ci":          loa_l_ci,
        "proportional_bias_r":   r_bias,
        "proportional_bias_p":   p_bias,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Myopia grade from SE
# ─────────────────────────────────────────────────────────────────────────────

def se_to_grade(
    se: Array,
    thresholds: Sequence[float] = (-0.5, -3.0, -6.0),
) -> Array:
    """
    Convert spherical equivalent (SE) to myopia grade.

    Grades:
        0 = non-myopic  (SE > thresholds[0], default > -0.5 D)
        1 = mild        (thresholds[1] < SE ≤ thresholds[0])
        2 = moderate    (thresholds[2] < SE ≤ thresholds[1])
        3 = high        (SE ≤ thresholds[2])
    """
    se = np.asarray(se, dtype=float)
    t0, t1, t2 = thresholds
    grade = np.full_like(se, 3, dtype=int)
    grade[se > t0] = 0
    grade[(se <= t0) & (se > t1)] = 1
    grade[(se <= t1) & (se > t2)] = 2
    return grade


def myopia_grade_metrics(
    se_true: Array,
    se_pred: Array,
    grade_true: Array | None = None,
    thresholds: Sequence[float] = (-0.5, -3.0, -6.0),
) -> dict:
    """
    Compute myopia grade accuracy from predicted SE.

    Parameters
    ----------
    se_true  : ground-truth SE (D)
    se_pred  : predicted SE (D)
    grade_true : if provided, used instead of deriving grade from se_true
    thresholds : SE breakpoints for grade 0/1/2/3

    Returns
    -------
    dict with: accuracy, macro_f1, per_class_accuracy, confusion_matrix_flat
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix,
        classification_report,
    )

    valid = ~(np.isnan(se_true) | np.isnan(se_pred))
    se_t = np.asarray(se_true)[valid]
    se_p = np.asarray(se_pred)[valid]

    g_true = grade_true[valid] if grade_true is not None \
             else se_to_grade(se_t, thresholds)
    g_pred = se_to_grade(se_p, thresholds)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc    = accuracy_score(g_true, g_pred)
        f1_mac = f1_score(g_true, g_pred, average="macro", zero_division=0)
        cm     = confusion_matrix(g_true, g_pred, labels=[0, 1, 2, 3])
        report = classification_report(
            g_true, g_pred, labels=[0, 1, 2, 3],
            target_names=["non-myopic", "mild", "moderate", "high"],
            zero_division=0, output_dict=True,
        )

    return {
        "n":                  int(valid.sum()),
        "accuracy":           float(acc),
        "macro_f1":           float(f1_mac),
        "confusion_matrix":   cm.tolist(),
        "classification_report": report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SER residual decomposition
# ─────────────────────────────────────────────────────────────────────────────

def ser_residual_analysis(
    se_true: Array,
    se_pred: Array,
) -> dict:
    """
    Analyse prediction residuals as a function of true SE.

    Fits linear regression: residual = a * SE_true + b

    Parameters
    ----------
    se_true : array of measured SE values
    se_pred : array of predicted SE values

    Returns
    -------
    dict with: slope, intercept, r2, pearson_r, pearson_p,
               mean_residual, sd_residual
    """
    valid = ~(np.isnan(se_true) | np.isnan(se_pred))
    yt = np.asarray(se_true)[valid]
    yp = np.asarray(se_pred)[valid]
    if len(yt) < 3:
        return {k: float("nan") for k in [
            "slope","intercept","r2","pearson_r","pearson_p",
            "mean_residual","sd_residual"]}

    residual = yp - yt
    slope, intercept, r_val, p_val, _ = stats.linregress(yt, residual)
    return {
        "slope":         float(slope),
        "intercept":     float(intercept),
        "r2":            float(r_val ** 2),
        "pearson_r":     float(r_val),
        "pearson_p":     float(p_val),
        "mean_residual": float(np.mean(residual)),
        "sd_residual":   float(np.std(residual, ddof=1)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Unified task evaluator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_tasks(
    predictions: dict[str, Array],   # {"al": [N], "sph": [N], "cyl": [N]}
    targets:     dict[str, Array],
    masks:       dict[str, Array],   # bool arrays, True = valid
    se_true:     Array | None = None,
    grade_true:  Array | None = None,
    se_thresholds: Sequence[float] = (-0.5, -3.0, -6.0),
) -> dict:
    """
    Compute all evaluation metrics for a full eval run.

    Parameters
    ----------
    predictions : model output values (un-normalised, in original units)
    targets     : ground-truth values (un-normalised)
    masks       : per-task valid-entry boolean arrays
    se_true     : ground-truth SE values (D), if available separately
    grade_true  : ground-truth myopia grade integers, if available
    se_thresholds : breakpoints for myopia grade classification

    Returns
    -------
    Nested dict:
    {
      "per_task": {
        "al":  {n, mae, rmse, r2, pearson_r, pearson_p, bland_altman: {...}},
        "sph": {...},
        "cyl": {...},
        "se":  {...},      # derived SE from sph + cyl/2
      },
      "cross_task": {
        "se_consistency_mae": float,
        "myopia_grade": {accuracy, macro_f1, confusion_matrix, ...},
        "ser_residual": {slope, intercept, r2, ...},
      },
    }
    """
    out: dict = {"per_task": {}, "cross_task": {}}

    # ── per-task regression metrics ───────────────────────────────────────────
    for task in predictions:
        m  = masks.get(task, np.ones(len(predictions[task]), dtype=bool))
        yt = np.asarray(targets.get(task, np.full_like(predictions[task], np.nan)))
        yp = np.asarray(predictions[task])
        # Apply mask
        yt_valid = yt[m]
        yp_valid = yp[m]
        reg = regression_metrics(yt_valid, yp_valid, task_name=task)
        ba  = bland_altman(yt_valid, yp_valid)
        out["per_task"][task] = {**reg, "bland_altman": ba}

    # ── derived SE metrics ────────────────────────────────────────────────────
    if "sph" in predictions and "cyl" in predictions:
        se_pred = predictions["sph"] + predictions["cyl"] / 2.0
        if se_true is not None:
            sv = ~np.isnan(se_true)
            reg_se = regression_metrics(se_true[sv], se_pred[sv], task_name="se")
            ba_se  = bland_altman(se_true[sv], se_pred[sv])
            out["per_task"]["se"] = {**reg_se, "bland_altman": ba_se}

            # SE consistency
            m_sph = masks.get("sph", np.ones(len(se_pred), dtype=bool))
            m_cyl = masks.get("cyl", np.ones(len(se_pred), dtype=bool))
            valid_both = m_sph & m_cyl & sv
            if valid_both.any():
                out["cross_task"]["se_consistency_mae"] = mae(
                    se_true[valid_both], se_pred[valid_both]
                )

            # SER residual decomposition
            out["cross_task"]["ser_residual"] = ser_residual_analysis(
                se_true[sv], se_pred[sv]
            )

            # Myopia grade classification
            out["cross_task"]["myopia_grade"] = myopia_grade_metrics(
                se_true[sv], se_pred[sv],
                grade_true=(grade_true[sv] if grade_true is not None else None),
                thresholds=se_thresholds,
            )

    return out

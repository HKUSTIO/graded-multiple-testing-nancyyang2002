from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t


def _two_sample_t_pvalue(y: np.ndarray, z: np.ndarray) -> float:
    treated = y[z == 1]
    control = y[z == 0]
    n1 = treated.shape[0]
    n0 = control.shape[0]
    s1 = float(np.var(treated, ddof=1))
    s0 = float(np.var(control, ddof=1))
    se = float(np.sqrt(s1 / n1 + s0 / n0))
    diff = float(np.mean(treated) - np.mean(control))
    if se == 0.0:
        return 1.0
    t_stat = diff / se
    df_num = (s1 / n1 + s0 / n0) ** 2
    df_den = ((s1 / n1) ** 2) / (n1 - 1) + ((s0 / n0) ** 2) / (n0 - 1)
    if df_den == 0.0:
        return 1.0
    df = df_num / df_den
    return float(2.0 * t.sf(np.abs(t_stat), df=df))


def simulate_null_pvalues(config: dict[str, Any]) -> pd.DataFrame:
    """
    Generate p-values under the complete null for L simulations.
    Return columns: sim_id, hypothesis_id, p_value.
    """
    rng = np.random.default_rng(int(config["seed_null"]))
    n = int(config["N"])
    m = int(config["M"])
    l = int(config["L"])
    p_treat = float(config["p_treat"])

    rows: list[dict[str, float | int]] = []
    for sim_id in range(l):
        z = (rng.random(n) < p_treat).astype(int)
        for hypothesis_id in range(m):
            y = rng.normal(loc=0.0, scale=1.0, size=n)
            p_value = _two_sample_t_pvalue(y=y, z=z)
            rows.append(
                {
                    "sim_id": sim_id,
                    "hypothesis_id": hypothesis_id,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def simulate_mixed_pvalues(config: dict[str, Any]) -> pd.DataFrame:
    """
    Generate p-values under mixed true and false null hypotheses for L simulations.
    Return columns: sim_id, hypothesis_id, p_value, is_true_null.
    """
    rng = np.random.default_rng(int(config["seed_mixed"]))
    n = int(config["N"])
    m = int(config["M"])
    m0 = int(config["M0"])
    l = int(config["L"])
    p_treat = float(config["p_treat"])
    tau_alt = float(config["tau_alternative"])

    rows: list[dict[str, float | int | bool]] = []
    for sim_id in range(l):
        z = (rng.random(n) < p_treat).astype(int)
        for hypothesis_id in range(m):
            is_true_null = hypothesis_id >= (m - m0)
            effect = 0.0 if is_true_null else tau_alt
            y = rng.normal(loc=0.0, scale=1.0, size=n) + effect * z
            p_value = _two_sample_t_pvalue(y=y, z=z)
            rows.append(
                {
                    "sim_id": sim_id,
                    "hypothesis_id": hypothesis_id,
                    "p_value": p_value,
                    "is_true_null": is_true_null,
                }
            )
    return pd.DataFrame(rows)


def bonferroni_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Bonferroni correction.
    """
    m = p_values.shape[0]
    return p_values <= (alpha / m)


def holm_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Holm step-down correction.
    """
    m = p_values.shape[0]
    sort_idx = np.argsort(p_values)
    p_sorted = p_values[sort_idx]
    
    thresholds = alpha / (m - np.arange(m))
    is_rejected_sorted = p_sorted <= thresholds
    
    # Step-down: find first False, all subsequent are False
    first_false = np.where(~is_rejected_sorted)[0]
    if first_false.size > 0:
        is_rejected_sorted[first_false[0]:] = False
        
    rejections = np.zeros(m, dtype=bool)
    rejections[sort_idx] = is_rejected_sorted
    return rejections


def benjamini_hochberg_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Benjamini-Hochberg correction.
    """
    m = p_values.shape[0]
    sort_idx = np.argsort(p_values)
    p_sorted = p_values[sort_idx]
    
    thresholds = (np.arange(1, m + 1) / m) * alpha
    is_below_threshold = p_sorted <= thresholds
    
    # Find the largest rank k satisfying the threshold
    passing_ranks = np.where(is_below_threshold)[0]
    if passing_ranks.size == 0:
        return np.zeros(m, dtype=bool)
        
    k_max = passing_ranks[-1]
    is_rejected_sorted = np.zeros(m, dtype=bool)
    is_rejected_sorted[: k_max + 1] = True
    
    rejections = np.zeros(m, dtype=bool)
    rejections[sort_idx] = is_rejected_sorted
    return rejections


def benjamini_yekutieli_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Benjamini-Yekutieli correction.
    """
    m = p_values.shape[0]
    harmonic_sum = np.sum(1.0 / np.arange(1, m + 1))
    alpha_corrected = alpha / harmonic_sum
    return benjamini_hochberg_rejections(p_values, alpha_corrected)


def compute_fwer(rejections_null: np.ndarray) -> float:
    """
    Return family-wise error rate from a [L, M] rejection matrix under the complete null.
    """
    # Proportion of rows with at least one True
    any_rejection = np.any(rejections_null, axis=1)
    return float(np.mean(any_rejection))


def compute_fdr(rejections: np.ndarray, is_true_null: np.ndarray) -> float:
    """
    Return FDR for one simulation: false discoveries among all discoveries.
    Use 0.0 when there are no rejections.
    """
    total_rejections = np.sum(rejections)
    if total_rejections == 0:
        return 0.0
    false_rejections = np.sum(rejections & is_true_null)
    return float(false_rejections / total_rejections)


def compute_power(rejections: np.ndarray, is_true_null: np.ndarray) -> float:
    """
    Return power for one simulation: true rejections among false null hypotheses.
    """
    is_false_null = ~is_true_null
    total_false_nulls = np.sum(is_false_null)
    if total_false_nulls == 0:
        return 1.0  # Or 0.0 depending on convention, but mixed sims usually have false nulls
    true_rejections = np.sum(rejections & is_false_null)
    return float(true_rejections / total_false_nulls)


def summarize_multiple_testing(
    null_pvalues: pd.DataFrame,
    mixed_pvalues: pd.DataFrame,
    alpha: float,
) -> dict[str, float]:
    """
    Return summary metrics:
      fwer_uncorrected, fwer_bonferroni, fwer_holm,
      fdr_uncorrected, fdr_bh, fdr_by,
      power_uncorrected, power_bh, power_by.
    """
    # 1. Complete null FWER
    l_null = null_pvalues["sim_id"].nunique()
    m_null = null_pvalues["hypothesis_id"].nunique()
    
    # Reshape p-values into [L, M]
    p_null_matrix = null_pvalues.sort_values(["sim_id", "hypothesis_id"])["p_value"].values.reshape(l_null, m_null)
    
    rej_uncorr = p_null_matrix <= alpha
    rej_bonf = np.array([bonferroni_rejections(row, alpha) for row in p_null_matrix])
    rej_holm = np.array([holm_rejections(row, alpha) for row in p_null_matrix])
    
    fwer_uncorr = compute_fwer(rej_uncorr)
    fwer_bonf = compute_fwer(rej_bonf)
    fwer_holm = compute_fwer(rej_holm)
    
    # 2. Mixed simulations FDR and Power
    l_mixed = mixed_pvalues["sim_id"].nunique()
    
    fdr_uncorr_list = []
    fdr_bh_list = []
    fdr_by_list = []
    
    pow_uncorr_list = []
    pow_bh_list = []
    pow_by_list = []
    
    for _, group in mixed_pvalues.groupby("sim_id"):
        group = group.sort_values("hypothesis_id")
        p_vals = group["p_value"].values
        is_null = group["is_true_null"].values
        
        # Uncorrected
        rej = p_vals <= alpha
        fdr_uncorr_list.append(compute_fdr(rej, is_null))
        pow_uncorr_list.append(compute_power(rej, is_null))
        
        # BH
        rej = benjamini_hochberg_rejections(p_vals, alpha)
        fdr_bh_list.append(compute_fdr(rej, is_null))
        pow_bh_list.append(compute_power(rej, is_null))
        
        # BY
        rej = benjamini_yekutieli_rejections(p_vals, alpha)
        fdr_by_list.append(compute_fdr(rej, is_null))
        pow_by_list.append(compute_power(rej, is_null))
        
    return {
        "fwer_uncorrected": fwer_uncorr,
        "fwer_bonferroni": fwer_bonf,
        "fwer_holm": fwer_holm,
        "fdr_uncorrected": float(np.mean(fdr_uncorr_list)),
        "fdr_bh": float(np.mean(fdr_bh_list)),
        "fdr_by": float(np.mean(fdr_by_list)),
        "power_uncorrected": float(np.mean(pow_uncorr_list)),
        "power_bh": float(np.mean(pow_bh_list)),
        "power_by": float(np.mean(pow_by_list)),
    }

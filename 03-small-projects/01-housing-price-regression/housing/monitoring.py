import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_WARN, PSI_ALERT = 0.1, 0.25


def psi(reference, current, bins=10):
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    ref_pct = np.clip(ref_pct, 1e-4, None)
    cur_pct = np.clip(cur_pct, 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in reference.columns:
        p = psi(reference[col].to_numpy(), current[col].to_numpy())
        ks = ks_2samp(reference[col], current[col])
        status = "alert" if p >= PSI_ALERT else "warn" if p >= PSI_WARN else "ok"
        rows.append({
            "feature": col, "psi": p,
            "ks_stat": ks.statistic, "ks_p": ks.pvalue, "status": status,
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False)

def performance_report(
    y_true, y_pred, lower, upper, baseline_rmse,
    tolerance=1.2, target_coverage=0.90, coverage_margin=0.05,
):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
    return {
        "n": len(y_true),
        "rmse": rmse,
        "rmse_ratio": rmse / baseline_rmse,
        "bias": float(np.mean(y_true - y_pred)),
        "coverage": coverage,
        "rmse_alert": rmse > tolerance * baseline_rmse,
        "coverage_alert": coverage < target_coverage - coverage_margin,
    }

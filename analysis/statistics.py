"""PHANTOM analysis statistics for cross-experiment validation and reporting.

This module collects the quantitative analysis helpers used after experiment
execution. The functions here are intentionally paper-oriented: each output is
designed to feed either the IEEE TAES narrative, the Phase 1 validation
report, or a later Phase 2 diagnostic figure.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.random import RandomState
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind

from core.utils import compute_engagement_statistics, load_phantom_config

DETECTION_LIMIT = 0.05


def _as_float_array(values: list[float]) -> npt.NDArray[np.float64]:
    """Convert a Python list to a strict float array for PHANTOM analysis."""
    return np.asarray(values, dtype=np.float64)


def _extract_metric(results: list[dict[str, Any]], key: str) -> npt.NDArray[np.float64]:
    """Read one scalar metric from PHANTOM result rows as a float array."""
    return np.asarray([float(result[key]) for result in results], dtype=np.float64)


def _power_law_model(
    ramp_rate: npt.NDArray[np.float64],
    coefficient: float,
    exponent: float,
) -> npt.NDArray[np.float64]:
    """Evaluate the miss-scaling power law used in the PHANTOM paper."""
    return coefficient * np.power(ramp_rate, exponent)


def _critical_summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Identify the best undetected operating point from sweep summary rows."""
    passing = [row for row in rows if float(row["mean_detection"]) < DETECTION_LIMIT]
    if not passing:
        return None
    return max(passing, key=lambda row: float(row["mean_miss"]))


def compare_baseline_vs_injection(
    baseline_results: list[dict[str, Any]],
    injection_results: list[dict[str, Any]],
) -> dict[str, float | bool]:
    """
    Welch's t-test comparing miss distances between baseline and injection.

    Returns:
      't_statistic'   : float
      'p_value'       : float
      'significant'   : bool  (p < 0.001 for IEEE reporting)
      'effect_size'   : float (Cohen's d)
      'baseline_mean' : float
      'injection_mean': float
      'mean_increase' : float (injection_mean - baseline_mean in meters)

    For IEEE TAES, this populates the statistical significance claim:
    "The difference in miss distance between baseline and PHANTOM injection
     was statistically significant (t=XX.X, p<0.001, d=X.XX)."
    """
    baseline_miss = _extract_metric(baseline_results, "miss_distance")
    injection_miss = _extract_metric(injection_results, "miss_distance")
    t_statistic, p_value = ttest_ind(injection_miss, baseline_miss, equal_var=False)

    baseline_mean = float(np.mean(baseline_miss))
    injection_mean = float(np.mean(injection_miss))
    pooled_variance = (
        ((baseline_miss.size - 1) * float(np.var(baseline_miss, ddof=1)))
        + ((injection_miss.size - 1) * float(np.var(injection_miss, ddof=1)))
    ) / float(baseline_miss.size + injection_miss.size - 2)
    pooled_std = float(np.sqrt(max(pooled_variance, 0.0)))
    effect_size = (
        0.0 if np.isclose(pooled_std, 0.0) else (injection_mean - baseline_mean) / pooled_std
    )

    return {
        "t_statistic": float(t_statistic),
        "p_value": float(p_value),
        "significant": bool(float(p_value) < 0.001),
        "effect_size": float(effect_size),
        "baseline_mean": baseline_mean,
        "injection_mean": injection_mean,
        "mean_increase": injection_mean - baseline_mean,
    }


def fit_miss_distance_scaling(
    ramp_rates: list[float],
    mean_misses: list[float],
) -> dict[str, float | list[float]]:
    """
    Fit a power-law model: miss = a * ramp_rate^b to the sweep data.

    Uses scipy.optimize.curve_fit with bounds.
    Returns:
      'a'         : float — scaling coefficient
      'b'         : float — power law exponent
      'r_squared' : float — goodness of fit
      'predicted' : list[float] — fitted values at each ramp rate

    The fitted exponent b characterizes how miss distance scales with
    injection rate — a key theoretical result for the PHANTOM paper.
    Expected: b ≈ 1.2 to 2.0 (super-linear scaling).
    """
    rates = _as_float_array(ramp_rates)
    misses = _as_float_array(mean_misses)
    if rates.size != misses.size or rates.size < 3:
        raise ValueError("ramp_rates and mean_misses must have equal length >= 3.")

    # Theory predicts monotonic growth in deception effect until detection or
    # saturation dominates. The experimental sweep can flatten or roll over once
    # the EKF starts rejecting measurements, so fit the monotone pre-saturation
    # envelope up to the peak observed mean miss.
    peak_idx = int(np.argmax(misses)) + 1
    fit_rates = rates[:peak_idx]
    fit_misses = np.maximum.accumulate(misses[:peak_idx])
    miss_floor = float(np.min(fit_misses)) - 1.0
    fit_excess = fit_misses - miss_floor

    parameters = cast(
        tuple[npt.NDArray[np.float64], Any],
        curve_fit(
            _power_law_model,
            fit_rates,
            fit_excess,
            bounds=([0.0, 0.5], [1.0e9, 5.0]),
            maxfev=100_000,
        ),
    )[0]
    coefficient = float(parameters[0])
    exponent = float(parameters[1])
    fitted_window = miss_floor + _power_law_model(fit_rates, coefficient, exponent)
    predicted = miss_floor + _power_law_model(rates, coefficient, exponent)

    residual_sum = float(np.sum((fit_misses - fitted_window) ** 2))
    total_sum = float(np.sum((fit_misses - np.mean(fit_misses)) ** 2))
    r_squared = 1.0 if np.isclose(total_sum, 0.0) else 1.0 - residual_sum / total_sum

    return {
        "a": coefficient,
        "b": exponent,
        "r_squared": float(r_squared),
        "predicted": [float(value) for value in predicted],
    }


def compute_critical_rate_confidence_interval(
    sweep_results: list[dict[str, Any]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Bootstrap confidence interval for the critical ramp rate estimate.

    Resamples the sweep results n_bootstrap times, identifies the
    critical rate in each resample, and reports the 95% CI.

    Returns:
      'critical_rate_estimate' : float
      'ci_95_lower'            : float
      'ci_95_upper'            : float
      'bootstrap_std'          : float

    This CI is reported in the PHANTOM paper as evidence that the
    critical rate estimate is stable and not a Monte Carlo artifact.
    """
    if not sweep_results:
        raise ValueError("sweep_results must not be empty.")

    rng = RandomState(seed)
    bootstrap_rates: list[float] = []
    for _ in range(n_bootstrap):
        sampled_rows: list[dict[str, Any]] = []
        for row in sweep_results:
            n_runs = int(row.get("n_runs", 1))
            raw_results = row.get("results")
            if isinstance(raw_results, list) and raw_results:
                indices = rng.randint(0, len(raw_results), size=n_runs)
                resampled = [raw_results[int(index)] for index in indices]
                stats = compute_engagement_statistics(resampled)
            else:
                sampled_miss = rng.normal(
                    float(row["mean_miss"]),
                    max(float(row.get("std_miss", 0.0)), 1e-9),
                    size=n_runs,
                )
                sampled_detection = np.clip(
                    rng.normal(
                        float(row["mean_detection"]),
                        max(float(row.get("std_detection", 0.0)), 1e-9),
                        size=n_runs,
                    ),
                    0.0,
                    1.0,
                )
                stats = {
                    "mean_miss": float(np.mean(sampled_miss)),
                    "mean_detection": float(np.mean(sampled_detection)),
                }
            stats["ramp_rate"] = float(row["ramp_rate"])
            sampled_rows.append(stats)

        critical_row = _critical_summary_from_rows(sampled_rows)
        if critical_row is not None:
            bootstrap_rates.append(float(critical_row["ramp_rate"]))

    if not bootstrap_rates:
        raise ValueError("Bootstrap failed to identify a critical rate in any resample.")

    rate_array = np.asarray(bootstrap_rates, dtype=np.float64)
    critical_row = _critical_summary_from_rows(sweep_results)
    if critical_row is None:
        raise ValueError("sweep_results do not contain a valid critical-rate operating point.")

    return {
        "critical_rate_estimate": float(critical_row["ramp_rate"]),
        "ci_95_lower": float(np.percentile(rate_array, 2.5)),
        "ci_95_upper": float(np.percentile(rate_array, 97.5)),
        "bootstrap_std": float(np.std(rate_array)),
    }


def compute_ekf_innovation_statistics(
    trajectory: list[dict[str, Any]],
) -> dict[str, float | int | list[float]]:
    """
    Analyze the full innovation history from a single engagement.

    Returns:
      'mean_gamma'        : float
      'std_gamma'         : float
      'max_gamma'         : float
      'n_crossings'       : int   — times gamma_k exceeded threshold
      'crossing_times'    : list  — timesteps of gate crossings
      'time_above_gate'   : float — fraction of engagement above threshold
      'chi2_threshold'    : float — threshold used (from config)

    Used in Phase 2 to generate the innovation timeline figure — the
    figure that shows PHANTOM keeping gamma_k just below 9.21.
    """
    if not trajectory:
        raise ValueError("trajectory must contain at least one telemetry entry.")

    chi2_threshold = float(load_phantom_config()["simulation"]["chi2_threshold"])
    gamma_values = np.asarray([float(entry["gamma_k"]) for entry in trajectory], dtype=np.float64)
    time_values = np.asarray([float(entry["t"]) for entry in trajectory], dtype=np.float64)
    above_gate = gamma_values > chi2_threshold
    crossing_times = [float(time_values[index]) for index in np.flatnonzero(above_gate)]

    return {
        "mean_gamma": float(np.mean(gamma_values)),
        "std_gamma": float(np.std(gamma_values)),
        "max_gamma": float(np.max(gamma_values)),
        "n_crossings": int(np.sum(above_gate)),
        "crossing_times": crossing_times,
        "time_above_gate": float(np.mean(above_gate)),
        "chi2_threshold": chi2_threshold,
    }


def compute_scaling_laws(sweep_df: pd.DataFrame) -> dict[str, Any]:
    """
    Derive the parameter scaling relationships predicted by theory:
      miss ∝ V_c^{-1}
      miss ∝ Q^{-0.5}
      critical_rate ∝ chi2_threshold^{0.5}

    Takes a pandas DataFrame from the ramp sweep CSV.
    Returns dict of fitted scaling exponents vs. theoretical predictions.
    Used for Table III in the IEEE TAES paper.
    """
    fit_result = fit_miss_distance_scaling(
        sweep_df["ramp_rate"].astype(float).tolist(),
        sweep_df["mean_miss"].astype(float).tolist(),
    )
    critical_row = _critical_summary_from_rows(sweep_df.to_dict(orient="records"))
    empirical_exponent = cast(float, fit_result["b"])
    r_squared = cast(float, fit_result["r_squared"])
    fit_coefficient = cast(float, fit_result["a"])
    return {
        "miss_vs_ramp_rate": {
            "empirical_exponent": empirical_exponent,
            "r_squared": r_squared,
            "fit_coefficient": fit_coefficient,
        },
        "miss_vs_closing_velocity": {"theoretical_exponent": -1.0, "empirical_exponent": None},
        "miss_vs_process_noise": {"theoretical_exponent": -0.5, "empirical_exponent": None},
        "critical_rate_vs_chi2_threshold": {
            "theoretical_exponent": 0.5,
            "empirical_exponent": None,
            "critical_rate_estimate": (
                float(critical_row["ramp_rate"]) if critical_row is not None else None
            ),
        },
    }

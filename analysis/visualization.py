"""PHANTOM publication-figure generation utilities.

These functions generate the camera-ready IEEE figures used in the Phase 1
validation report and the draft TAES manuscript. They deliberately avoid
exploratory styling and instead enforce one publication-quality visual
language across all experiments.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from core.utils import save_ieee_figure, setup_ieee_figure

BLUE = "#2E86AB"
RED = "#E84855"
GREEN = "#28A745"
ORANGE = "#FF8C00"


def _stable_figure_alias(path_str: str) -> str:
    """Create the stable non-timestamped figure alias used by the report."""
    saved_path = Path(path_str)
    stem_parts = saved_path.stem.split("_")
    alias_name = "_".join(stem_parts[:-2]) if len(stem_parts) > 2 else saved_path.stem
    alias_path = saved_path.with_name(f"{alias_name}{saved_path.suffix}")
    shutil.copy2(saved_path, alias_path)
    return str(alias_path)


def _maybe_set_title(axis: Any, title: str, save: bool) -> None:
    """Show titles only during internal review, not IEEE export."""
    if not save:
        axis.set_title(title)


def _extract_positions(
    trajectory: list[dict[str, Any]],
    key: str,
) -> npt.NDArray[np.float64]:
    """Extract 2D trajectory vectors from PHANTOM telemetry."""
    return np.asarray([entry[key] for entry in trajectory], dtype=np.float64)


def plot_baseline_miss_histogram(
    results: list[dict[str, Any]],
    save: bool = True,
) -> tuple[Any, Any]:
    """
    Figure 1 — Baseline miss distance distribution.
    IEEE single-column: 3.5 × 2.8 in, 300 DPI.
    (Enhances the figure already generated in Experiment 1.)
    Save: figures/phantom_fig01_baseline_miss_distribution.png
    """
    miss_distances = np.asarray(
        [float(result["miss_distance"]) for result in results], dtype=np.float64
    )
    fig, ax = setup_ieee_figure(width_inches=3.5, height_inches=2.8)
    ax.hist(miss_distances, bins=20, color=BLUE, edgecolor="black", alpha=0.8)
    ax.axvline(5.0, color=GREEN, linestyle="--", linewidth=1.2, label="5 m gate")
    ax.set_xlabel("Miss Distance (m)")
    ax.set_ylabel("Frequency")
    _maybe_set_title(ax, "Baseline Miss Distribution", save)
    ax.legend(loc="upper right")

    if save:
        saved_path = save_ieee_figure(fig, "fig01_baseline_miss_distribution")
        _stable_figure_alias(saved_path)
    return fig, ax


def plot_ramp_rate_optimization(
    sweep_results: list[dict[str, Any]],
    critical_rate: float,
    save: bool = True,
) -> tuple[Any, Any]:
    """
    Figure 5 — Two-panel critical rate figure (PRIMARY PAPER FIGURE).
    Left: miss distance vs ramp rate with 200m target line.
    Right: detection rate vs ramp rate with 5% threshold line.
    Vertical dashed line at critical rate on both panels.
    IEEE double-column: 7.0 × 3.0 in, 300 DPI.
    Save: figures/phantom_fig05_critical_rate_combined.png
    """
    fig, axis = setup_ieee_figure(width_inches=7.0, height_inches=3.0)
    fig.clf()
    left_ax, right_ax = fig.subplots(1, 2, sharex=True)

    rates = np.asarray([float(row["ramp_rate"]) for row in sweep_results], dtype=np.float64)
    mean_miss = np.asarray([float(row["mean_miss"]) for row in sweep_results], dtype=np.float64)
    std_miss = np.asarray([float(row["std_miss"]) for row in sweep_results], dtype=np.float64)
    detection = (
        np.asarray([float(row["mean_detection"]) for row in sweep_results], dtype=np.float64)
        * 100.0
    )

    left_ax.errorbar(
        rates, mean_miss, yerr=std_miss, color=BLUE, marker="o", capsize=3, label="Mean ± std"
    )
    left_ax.axhline(200.0, color=GREEN, linestyle="--", linewidth=1.2, label="200 m target")
    left_ax.axvline(critical_rate, color=RED, linestyle="--", linewidth=1.2, label="Critical rate")
    left_ax.set_ylabel("Mean Miss Distance (m)")
    _maybe_set_title(left_ax, "Critical Ramp-Rate Trade Space", save)
    left_ax.legend(loc="best")

    right_ax.plot(rates, detection, color=RED, marker="o", label="Detection rate")
    right_ax.axhline(5.0, color=ORANGE, linestyle="--", linewidth=1.2, label="5% threshold")
    right_ax.axvline(critical_rate, color=RED, linestyle="--", linewidth=1.2)
    right_ax.set_ylabel("Detection Rate (%)")
    right_ax.legend(loc="best")

    fig.supxlabel("Ramp Rate (rad/s)")
    fig.tight_layout()
    if save:
        saved_path = save_ieee_figure(fig, "fig05_critical_rate_combined")
        _stable_figure_alias(saved_path)
    return fig, (left_ax, right_ax)


def plot_innovation_timeline(
    trajectory: list[dict[str, Any]],
    chi2_threshold: float = 9.21,
    save: bool = True,
) -> tuple[Any, Any]:
    """
    Figure 6 — EKF innovation gamma_k over engagement time.
    Shows gamma_k (blue line) vs chi2_threshold (red dashed).
    Shaded regions where measurement was rejected (if any).
    Annotates: peak gamma, mean gamma, rejection count.
    IEEE single-column: 3.5 × 2.8 in, 300 DPI.
    Save: figures/phantom_fig06_innovation_timeline.png

    This figure is the visual proof of PHANTOM's core mechanism —
    the missile's filter never detects the deception because gamma_k
    stays below the red line throughout the engagement.
    """
    times = np.asarray([float(entry["t"]) for entry in trajectory], dtype=np.float64)
    gamma = np.asarray([float(entry["gamma_k"]) for entry in trajectory], dtype=np.float64)
    accepted = np.asarray([bool(entry["measurement_accepted"]) for entry in trajectory], dtype=bool)

    fig, ax = setup_ieee_figure(width_inches=3.5, height_inches=2.8)
    ax.plot(times, gamma, color=BLUE, linewidth=1.5, label=r"$\gamma_k$")
    ax.axhline(chi2_threshold, color=RED, linestyle="--", linewidth=1.2, label=r"$\chi^2$ gate")
    for time_value, is_accepted in zip(times, accepted, strict=True):
        if not is_accepted:
            ax.axvspan(time_value, time_value, color=ORANGE, alpha=0.25, linewidth=3.0)

    annotation = (
        f"Peak: {float(np.max(gamma)):.2f}\n"
        f"Mean: {float(np.mean(gamma)):.2f}\n"
        f"Rejects: {int(np.sum(~accepted))}"
    )
    ax.text(
        0.97,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9},
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Innovation Statistic $\gamma_k$")
    _maybe_set_title(ax, "EKF Innovation Timeline", save)
    ax.legend(loc="upper left")

    if save:
        saved_path = save_ieee_figure(fig, "fig06_innovation_timeline")
        _stable_figure_alias(saved_path)
    return fig, ax


def plot_trajectory_comparison(
    baseline_traj: list[dict[str, Any]],
    injection_traj: list[dict[str, Any]],
    save: bool = True,
) -> tuple[Any, Any]:
    """
    Figure 7 — Side-by-side trajectory comparison.
    Left panel: baseline (direct hit), Right panel: PHANTOM injection.
    Shows true missile path, true target path, EKF estimate path.
    Marks intercept point (baseline) vs. miss vector (injection).
    IEEE double-column: 7.0 × 3.5 in, 300 DPI.
    Save: figures/phantom_fig07_trajectory_comparison.png

    This is the intuitive paper figure — the reader immediately sees
    the missile hitting on the left and missing on the right.
    """
    fig, axis = setup_ieee_figure(width_inches=7.0, height_inches=3.5)
    fig.clf()
    left_ax, right_ax = fig.subplots(1, 2, sharex=False, sharey=True)

    for ax, trajectory, label in (
        (left_ax, baseline_traj, "Baseline"),
        (right_ax, injection_traj, "PHANTOM"),
    ):
        missile = _extract_positions(trajectory, "missile_pos")
        target = _extract_positions(trajectory, "target_pos")
        estimate = _extract_positions(trajectory, "ekf_estimate")
        ax.plot(missile[:, 0], missile[:, 1], color=BLUE, label="Missile")
        ax.plot(target[:, 0], target[:, 1], color=RED, label="Target")
        ax.plot(estimate[:, 0], estimate[:, 1], color=GREEN, linestyle="--", label="EKF Estimate")
        _maybe_set_title(ax, label, save)
        ax.set_xlabel("Downrange (m)")
        if ax is left_ax:
            ax.set_ylabel("Cross-range (m)")

    baseline_final = baseline_traj[-1]
    left_ax.plot(
        float(baseline_final["missile_pos"][0]),
        float(baseline_final["missile_pos"][1]),
        marker="x",
        color="black",
        linestyle="None",
        label="Intercept",
    )
    injection_final = injection_traj[-1]
    miss_start = np.asarray(injection_final["missile_pos"], dtype=np.float64)
    miss_end = np.asarray(injection_final["target_pos"], dtype=np.float64)
    right_ax.plot(
        [miss_start[0], miss_end[0]],
        [miss_start[1], miss_end[1]],
        color=ORANGE,
        linestyle=":",
        linewidth=1.5,
        label="Miss vector",
    )
    left_ax.legend(loc="best")
    right_ax.legend(loc="best")
    fig.tight_layout()

    if save:
        saved_path = save_ieee_figure(fig, "fig07_trajectory_comparison")
        _stable_figure_alias(saved_path)
    return fig, (left_ax, right_ax)


def plot_estimation_error_timeline(
    trajectory: list[dict[str, Any]],
    save: bool = True,
) -> tuple[Any, Any]:
    """
    Figure 8 — EKF position estimation error over time.
    Shows |true_target_pos - ekf_estimate| growing under PHANTOM injection.
    Baseline overlay shows near-zero error without injection.
    IEEE single-column: 3.5 × 2.8 in, 300 DPI.
    Save: figures/phantom_fig08_estimation_error.png

    This figure shows WHY the missile misses — the EKF believes it
    knows where the target is, but PHANTOM has steadily corrupted that
    belief, and the error grows until the missile flies past the phantom.
    """
    times = np.asarray([float(entry["t"]) for entry in trajectory], dtype=np.float64)
    error = np.asarray([float(entry["estimation_error"]) for entry in trajectory], dtype=np.float64)
    baseline_error = np.asarray(
        [float(entry.get("baseline_estimation_error", np.nan)) for entry in trajectory],
        dtype=np.float64,
    )

    fig, ax = setup_ieee_figure(width_inches=3.5, height_inches=2.8)
    ax.plot(times, error, color=BLUE, linewidth=1.5, label="PHANTOM injection")
    if np.isfinite(baseline_error).any():
        ax.plot(times, baseline_error, color=GREEN, linestyle="--", linewidth=1.2, label="Baseline")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Estimation Error (m)")
    _maybe_set_title(ax, "EKF Estimation Error", save)
    ax.legend(loc="upper left")

    if save:
        saved_path = save_ieee_figure(fig, "fig08_estimation_error")
        _stable_figure_alias(saved_path)
    return fig, ax


def plot_miss_distance_scaling(
    sweep_results: list[dict[str, Any]],
    fit_result: dict[str, Any],
    save: bool = True,
) -> tuple[Any, Any]:
    """
    Figure 9 — Power-law fit to miss distance vs ramp rate.
    Data points with error bars + fitted curve overlay.
    Annotation showing fitted exponent b and R².
    IEEE single-column: 3.5 × 2.8 in, 300 DPI.
    Save: figures/phantom_fig09_miss_distance_scaling.png
    """
    rates = np.asarray([float(row["ramp_rate"]) for row in sweep_results], dtype=np.float64)
    mean_miss = np.asarray([float(row["mean_miss"]) for row in sweep_results], dtype=np.float64)
    std_miss = np.asarray([float(row["std_miss"]) for row in sweep_results], dtype=np.float64)
    predicted = np.asarray(fit_result["predicted"], dtype=np.float64)

    fig, ax = setup_ieee_figure(width_inches=3.5, height_inches=2.8)
    ax.errorbar(
        rates, mean_miss, yerr=std_miss, color=BLUE, marker="o", capsize=3, label="Sweep data"
    )
    ax.plot(rates, predicted, color=RED, linewidth=1.5, label="Power-law fit")
    annotation = f"b = {float(fit_result['b']):.2f}\n$R^2$ = {float(fit_result['r_squared']):.3f}"
    ax.text(
        0.97,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9},
    )
    ax.set_xlabel("Ramp Rate (rad/s)")
    ax.set_ylabel("Mean Miss Distance (m)")
    _maybe_set_title(ax, "Miss-Distance Scaling", save)
    ax.legend(loc="best")

    if save:
        saved_path = save_ieee_figure(fig, "fig09_miss_distance_scaling")
        _stable_figure_alias(saved_path)
    return fig, ax

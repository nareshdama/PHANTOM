"""
PHANTOM Experiment 1 — Baseline Validation
==========================================
Validates that the PhantomSimulator correctly models a direct missile
intercept when no false signal injection is applied. This experiment
serves two purposes: (1) confirming simulation fidelity against
known PN guidance behavior, and (2) establishing the null-hypothesis
baseline against which all PHANTOM injection experiments are compared.

For IEEE TAES publication, results from this experiment populate
Table I and support the claim that the simulator correctly reproduces
published PN guidance miss distance statistics.

Usage:
    python experiments/baseline_validation.py --seed 42 --runs 100
    python experiments/baseline_validation.py --runs 500 --output custom_name

Phase 1 Gate Criteria (from phantom_config.yaml):
    Mean miss distance < 5.0 m
    95th percentile   < 10.0 m
    Detection rate    == 0.0%
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numpy.random import RandomState
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.injection import NoInjection
from core.simulator import PhantomSimulator
from core.utils import (
    compute_engagement_statistics,
    get_default_initial_conditions,
    load_phantom_config,
    randomize_initial_conditions,
    save_ieee_figure,
    save_results_csv,
    save_results_hdf5,
    setup_ieee_figure,
)

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_output_dir(path_str: str) -> Path:
    """Resolve an experiment output directory relative to the PHANTOM root."""
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _stable_figure_alias(path_str: str) -> str:
    """Create a stable non-timestamped alias alongside PHANTOM figure outputs."""
    saved_path = Path(path_str)
    stem_parts = saved_path.stem.split("_")
    alias_name = "_".join(stem_parts[:-2]) if len(stem_parts) > 2 else saved_path.stem
    alias_path = saved_path.with_name(f"{alias_name}{saved_path.suffix}")
    shutil.copy2(saved_path, alias_path)
    return str(alias_path)


def _format_gate_line(label: str, passed: bool, value_text: str) -> str:
    """Format a terminal gate-status line for the PHANTOM report table."""
    marker = "✅" if passed else "❌"
    return f"  {marker} {label:<28} [{value_text}]"


def _project_to_baseline_collision_course(
    config: dict[str, Any],
    initial_conditions: dict[str, npt.NDArray[np.float64]],
) -> dict[str, npt.NDArray[np.float64]]:
    """Project randomized samples onto the direct-hit baseline geometry.

    Experiment 1 validates the simulator's no-injection null hypothesis:
    when the engagement starts on a head-on collision course, the EKF and PN
    loop should preserve that intercept rather than inventing a miss. The
    Monte Carlo still varies both platform speeds, but keeps the cross-range
    geometry at zero and preserves the canonical intercept duration from the
    config baseline. This isolates physics fidelity from off-axis capture
    effects that belong in later experiments.
    """
    baseline = get_default_initial_conditions(config)
    missile_pos = np.asarray(initial_conditions["missile_pos"], dtype=np.float64)
    missile_speed = float(np.linalg.norm(initial_conditions["missile_vel"]))
    target_speed = float(np.linalg.norm(initial_conditions["target_vel"]))
    baseline_separation = float(np.linalg.norm(baseline["target_pos"] - baseline["missile_pos"]))
    baseline_closing_speed = float(
        np.linalg.norm(baseline["missile_vel"]) + np.linalg.norm(baseline["target_vel"])
    )
    intercept_time = baseline_separation / baseline_closing_speed
    separation = (missile_speed + target_speed) * intercept_time

    return {
        "missile_pos": missile_pos.copy(),
        "missile_vel": np.array([missile_speed, 0.0], dtype=np.float64),
        "target_pos": missile_pos + np.array([separation, 0.0], dtype=np.float64),
        "target_vel": np.array([-target_speed, 0.0], dtype=np.float64),
    }


def _print_summary(
    runs: int,
    seed: int,
    config: dict[str, Any],
    stats: dict[str, float | int],
    gate_passed: bool,
    csv_path: str,
    hdf5_path: str,
    figure_paths: list[str],
) -> None:
    """Render the experiment summary table in the PHANTOM terminal format."""
    sim_cfg = config["simulation"]
    val_cfg = config["validation"]
    mean_gate = float(val_cfg["baseline_miss_threshold"])
    p95_gate = float(val_cfg["baseline_p95_threshold"])
    detection_gate = float(val_cfg["baseline_detection_rate"])

    print("╔══════════════════════════════════════════════════════╗")
    print("║         PHANTOM — Experiment 1: Baseline Validation ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print("Configuration:")
    print(f"  Runs:           {runs}")
    print(f"  Seed:           {seed}")
    print(f"  Nav ratio:      {sim_cfg['nav_ratio']}")
    print(f"  Chi² threshold: {sim_cfg['chi2_threshold']}")
    print("  Profile:        NoInjection")
    print()
    print("Results:")
    print(
        f"  Mean miss distance:    {float(stats['mean_miss']):.2f}"
        f" ± {float(stats['std_miss']):.2f} m"
    )
    print(f"  Median miss distance:  {float(stats['median_miss']):.2f} m")
    print(f"  95th percentile:       {float(stats['p95_miss']):.2f} m")
    print(
        f"  Min / Max:             {float(stats['min_miss']):.2f}"
        f" / {float(stats['max_miss']):.2f} m"
    )
    print(f"  Detection rate:        {float(stats['mean_detection']) * 100.0:.2f}%")
    print(f"  Success rate:          {float(stats['success_rate']) * 100.0:.1f}%")
    print()
    print("Validation Gate:")
    print(
        _format_gate_line(
            f"Mean miss < {mean_gate:.1f} m",
            float(stats["mean_miss"]) < mean_gate,
            f"{float(stats['mean_miss']):.2f} m",
        )
    )
    print(
        _format_gate_line(
            f"95th percentile < {p95_gate:.1f} m",
            float(stats["p95_miss"]) < p95_gate,
            f"{float(stats['p95_miss']):.2f} m",
        )
    )
    print(
        _format_gate_line(
            f"Detection rate == {detection_gate * 100.0:.1f}%",
            float(stats["mean_detection"]) == detection_gate,
            f"{float(stats['mean_detection']) * 100.0:.2f}%",
        )
    )
    print()
    if gate_passed:
        print("  ✅ PHANTOM Experiment 1 PASSED — Baseline validated")
    else:
        print("  ❌ PHANTOM Experiment 1 FAILED — Baseline gate not satisfied")
    print(f"  📊 Results: {csv_path}")
    print(f"  📊 HDF5:    {hdf5_path}")
    for figure_path in figure_paths:
        print(f"  🖼  Figure:  {figure_path}")


def _save_miss_histogram(
    miss_distances: list[float],
    stats: dict[str, float | int],
    runs: int,
    threshold_m: float,
    figures_dir: str,
) -> str:
    """Create the baseline miss-distance histogram for IEEE publication."""
    fig, ax = setup_ieee_figure(width_inches=3.5, height_inches=2.8)
    ax.hist(miss_distances, bins=20, edgecolor="black", alpha=0.7, color="#2E86AB")
    ax.axvline(threshold_m, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Miss Distance (m)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Baseline Miss Distance Distribution (N={runs}, No Injection)")

    # The 95th percentile is the key metric for IEEE TAES Table I.
    # It characterizes worst-case performance across the engagement envelope
    # rather than average-case, which is more informative for safety analysis.
    annotation = (
        f"Mean ± Std: {float(stats['mean_miss']):.2f} ± {float(stats['std_miss']):.2f} m\n"
        f"95th pct: {float(stats['p95_miss']):.2f} m"
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

    saved_path = save_ieee_figure(fig, "fig01_baseline_miss_distribution", output_dir=figures_dir)
    plt.close(fig)
    return _stable_figure_alias(saved_path)


def _save_sample_trajectory(
    representative_result: dict[str, Any],
    figures_dir: str,
) -> str:
    """Plot one representative no-injection engagement trajectory."""
    fig, ax = setup_ieee_figure(width_inches=3.5, height_inches=2.8)
    trajectory = representative_result["trajectory"]

    missile_positions = np.asarray([entry["missile_pos"] for entry in trajectory], dtype=np.float64)
    target_positions = np.asarray([entry["target_pos"] for entry in trajectory], dtype=np.float64)
    ekf_positions = np.asarray([entry["ekf_estimate"] for entry in trajectory], dtype=np.float64)

    ax.plot(missile_positions[:, 0], missile_positions[:, 1], color="blue", label="Missile")
    ax.plot(target_positions[:, 0], target_positions[:, 1], color="red", label="Target")
    ax.plot(
        ekf_positions[:, 0],
        ekf_positions[:, 1],
        color="green",
        linestyle="--",
        label="EKF Estimate",
    )
    ax.plot(
        missile_positions[-1, 0],
        missile_positions[-1, 1],
        marker="x",
        color="black",
        markersize=7,
        linestyle="None",
        label="Termination",
    )
    ax.set_xlabel("Downrange Position (m)")
    ax.set_ylabel("Cross-range Position (m)")
    ax.set_title("Representative Engagement Trajectory (No Injection)")
    ax.legend(loc="upper right")

    saved_path = save_ieee_figure(fig, "fig02_baseline_trajectory", output_dir=figures_dir)
    plt.close(fig)
    return _stable_figure_alias(saved_path)


def run_baseline_validation(
    seed: int = 42,
    runs: int | None = None,
    output: str = "baseline",
    results_dir: str = "data/results",
    figures_dir: str = "figures",
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run PHANTOM Experiment 1 and return its statistics, artifacts, and gate status."""
    config = load_phantom_config()
    runs = int(config["monte_carlo"]["baseline_runs"] if runs is None else runs)
    validation_cfg = config["validation"]
    profile = NoInjection()
    results: list[dict[str, Any]] = []

    LOGGER.info("Starting baseline validation with %d runs and base seed %d", runs, seed)

    iterator = tqdm(
        range(runs),
        total=runs,
        desc=f"Running {runs} engagements",
        disable=not show_progress,
    )
    for run_id in iterator:
        run_seed = seed + run_id

        # Each run uses seed = base_seed + run_id so that:
        # (a) the full experiment is reproducible from a single --seed argument
        # (b) each run has independent random noise (not the same sequence)
        # (c) any individual run can be reproduced by passing its exact seed
        #     to PhantomSimulator directly — critical for debugging outliers
        rng = RandomState(run_seed)
        initial_conditions = _project_to_baseline_collision_course(
            config,
            randomize_initial_conditions(config, rng),
        )
        simulator = PhantomSimulator(config, seed=run_seed)
        result = simulator.run_engagement(profile, initial_conditions)
        result["seed"] = run_seed
        results.append(result)

    stats = compute_engagement_statistics(results)
    gate_passed = (
        float(stats["mean_miss"]) < float(validation_cfg["baseline_miss_threshold"])
        and float(stats["p95_miss"]) < float(validation_cfg["baseline_p95_threshold"])
        and float(stats["mean_detection"]) == float(validation_cfg["baseline_detection_rate"])
    )

    output_stem = f"{output}_seed{seed}"
    csv_path = save_results_csv(results, output_stem, output_dir=results_dir)
    hdf5_path = save_results_hdf5(results, output_stem, output_dir=results_dir)

    representative_result = min(
        results,
        key=lambda item: abs(float(item["miss_distance"]) - float(stats["median_miss"])),
    )

    # Figures are generated here rather than in analysis/ because Experiment 1
    # is a self-contained validation artifact. The analysis/ module generates
    # cross-experiment comparison figures that require all experiments complete.
    figure_paths = [
        _save_miss_histogram(
            [float(result["miss_distance"]) for result in results],
            stats,
            runs,
            float(validation_cfg["baseline_miss_threshold"]),
            figures_dir,
        ),
        _save_sample_trajectory(representative_result, figures_dir),
    ]

    return {
        "config": config,
        "runs": runs,
        "seed": seed,
        "results": results,
        "stats": stats,
        "gate_passed": gate_passed,
        "csv_path": csv_path,
        "hdf5_path": hdf5_path,
        "figure_paths": figure_paths,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for PHANTOM Experiment 1."""
    parser = argparse.ArgumentParser(description="Run PHANTOM baseline validation.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--runs", type=int, default=None, help="Number of Monte Carlo runs.")
    parser.add_argument(
        "--output",
        type=str,
        default="baseline",
        help="Output filename stem, without phantom_ prefix or timestamp.",
    )
    return parser.parse_args()


def main() -> int:
    """Execute PHANTOM Experiment 1 from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    summary = run_baseline_validation(seed=args.seed, runs=args.runs, output=args.output)
    _print_summary(
        runs=summary["runs"],
        seed=summary["seed"],
        config=summary["config"],
        stats=summary["stats"],
        gate_passed=summary["gate_passed"],
        csv_path=summary["csv_path"],
        hdf5_path=summary["hdf5_path"],
        figure_paths=summary["figure_paths"],
    )
    return 0 if summary["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

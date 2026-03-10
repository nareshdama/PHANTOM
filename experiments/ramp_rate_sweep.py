"""
PHANTOM Experiment 3 — Critical Ramp Rate Identification
========================================================
Identifies the maximum false LOS angular rate that can be injected
into the EKF bearing measurement while maintaining kinematic
plausibility — i.e., keeping the chi-squared innovation statistic
below the detection threshold of 9.21 in at least 95% of engagements.

This is the central experiment of the PHANTOM Phase 2 research program.
The identified critical rate and corresponding miss distance constitute
the primary quantitative contribution of the IEEE TAES paper.

Hypothesis (from Phase 1 mathematical analysis):
    H1: A critical ramp rate exists in [0.025, 0.040] rad/s that
        achieves mean miss distance > 200m with detection rate < 5%.

Expected result (from analytical bound derivation):
    Critical rate ≈ 0.032 rad/s
    Mean miss     ≈ 200 ± 27 m
    Detection     ≈ 2–3%

Usage:
    python experiments/ramp_rate_sweep.py --seed 42 --runs 200
    python experiments/ramp_rate_sweep.py --runs 1000 --workers 16
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.injection import LinearRampInjection
from core.simulator import PhantomSimulator
from core.utils import (
    check_monte_carlo_convergence,
    compute_engagement_statistics,
    get_default_initial_conditions,
    load_phantom_config,
    randomize_initial_conditions,
    save_ieee_figure,
    save_results_csv,
    setup_ieee_figure,
)

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _stable_figure_alias(path_str: str) -> str:
    """Create a stable non-timestamped alias alongside PHANTOM figure outputs."""
    saved_path = Path(path_str)
    stem_parts = saved_path.stem.split("_")
    alias_name = "_".join(stem_parts[:-2]) if len(stem_parts) > 2 else saved_path.stem
    alias_path = saved_path.with_name(f"{alias_name}{saved_path.suffix}")
    shutil.copy2(saved_path, alias_path)
    return str(alias_path)


def _project_to_sweep_collision_course(
    config: dict[str, Any],
    initial_conditions: dict[str, npt.NDArray[np.float64]],
) -> dict[str, npt.NDArray[np.float64]]:
    """Project randomized samples onto the canonical PHANTOM collision-course geometry.

    Experiment 3 studies how injected LOS ramps perturb a physically correct
    baseline intercept. Speeds still vary per Monte Carlo sample, but the
    geometry is kept head-on with the canonical intercept duration so the
    measured miss distance is attributable to PHANTOM injection rather than
    off-axis capture transients.
    """
    baseline = get_default_initial_conditions(config)
    missile_speed = float(np.linalg.norm(initial_conditions["missile_vel"]))
    target_speed = float(np.linalg.norm(initial_conditions["target_vel"]))
    baseline_separation = float(np.linalg.norm(baseline["target_pos"] - baseline["missile_pos"]))
    baseline_closing_speed = float(
        np.linalg.norm(baseline["missile_vel"]) + np.linalg.norm(baseline["target_vel"])
    )
    intercept_time = baseline_separation / baseline_closing_speed
    separation = (missile_speed + target_speed) * intercept_time
    missile_pos = np.asarray(initial_conditions["missile_pos"], dtype=np.float64)
    return {
        "missile_pos": missile_pos.copy(),
        "missile_vel": np.array([missile_speed, 0.0], dtype=np.float64),
        "target_pos": missile_pos + np.array([separation, 0.0], dtype=np.float64),
        "target_vel": np.array([-target_speed, 0.0], dtype=np.float64),
    }


def evaluate_single_run(args: tuple[float, int, dict[str, Any]]) -> dict[str, Any]:
    """
    Execute one complete PHANTOM engagement for a given ramp rate.

    This function is intentionally defined at module level (not as a
    class method) to allow pickling by multiprocessing.Pool. Each call
    creates a fresh PhantomSimulator instance to prevent any shared
    state between parallel workers — essential for reproducible
    Monte Carlo results.
    """
    ramp_rate, seed, config = args
    rng = np.random.RandomState(seed)
    sampled_initial_conditions = randomize_initial_conditions(config, rng)
    initial_conditions = _project_to_sweep_collision_course(config, sampled_initial_conditions)
    sim_cfg = config["injection"]
    simulator = PhantomSimulator(config, seed=seed)
    result = simulator.run_engagement(
        LinearRampInjection(
            t_start=float(sim_cfg["ramp_tstart"]),
            t_end=float(sim_cfg["ramp_tend"]),
            ramp_rate=ramp_rate,
        ),
        initial_conditions,
    )
    result["ramp_rate"] = ramp_rate
    result["seed"] = seed
    return result


def evaluate_ramp_rate(
    ramp_rate: float,
    n_runs: int,
    base_seed: int,
    config: dict[str, Any],
    n_workers: int,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    Run N parallel Monte Carlo engagements for one ramp rate.

    Returns aggregated statistics dict with ramp_rate included.
    """
    run_args: list[tuple[float, int, dict[str, Any]]] = []
    for run_id in range(n_runs):
        # Seed construction: base_seed + run_id * 1000 + int(ramp_rate * 10000)
        # The run_id offset (×1000) ensures different noise sequences across runs.
        # The ramp_rate offset ensures different sequences across rates at the
        # same run_id — preventing spurious correlations in the sweep results.
        seed = base_seed + run_id * 1000 + int(ramp_rate * 10000)
        run_args.append((ramp_rate, seed, config))

    with Pool(processes=n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(evaluate_single_run, run_args),
                total=n_runs,
                desc=f"  Rate {ramp_rate:.3f} rad/s",
                leave=False,
                disable=not show_progress,
            )
        )

    stats: dict[str, Any] = compute_engagement_statistics(results)
    convergence_cfg = config["monte_carlo"]
    converged = check_monte_carlo_convergence(
        [float(result["miss_distance"]) for result in results],
        window=int(convergence_cfg["convergence_window"]),
        tolerance=float(convergence_cfg["convergence_tolerance"]),
    )
    stats["ramp_rate"] = ramp_rate
    stats["converged"] = converged
    stats["results"] = results
    return stats


def _identify_critical_rate(
    rate_summaries: list[dict[str, Any]],
    detection_limit: float,
) -> dict[str, Any] | None:
    """Return the highest-value undetected operating point from the ramp sweep."""
    # The critical rate search filters for detection_rate < 0.05 first,
    # then maximizes miss distance among passing rates. This ordering is
    # intentional: detectability is a hard constraint (the missile must not
    # know it is being deceived), while miss distance is the objective.
    # A rate that achieves 500m miss but 90% detection has no operational value.
    passing = [
        summary for summary in rate_summaries if float(summary["mean_detection"]) < detection_limit
    ]
    if not passing:
        return None
    return max(passing, key=lambda summary: float(summary["mean_miss"]))


def _save_miss_figure(
    rate_summaries: list[dict[str, Any]],
    critical_summary: dict[str, Any],
    target_miss: float,
    figures_dir: str,
) -> str:
    """Create Figure 3: miss distance versus ramp rate."""
    rates = np.asarray(
        [float(summary["ramp_rate"]) for summary in rate_summaries], dtype=np.float64
    )
    mean_miss = np.asarray(
        [float(summary["mean_miss"]) for summary in rate_summaries], dtype=np.float64
    )
    std_miss = np.asarray(
        [float(summary["std_miss"]) for summary in rate_summaries], dtype=np.float64
    )

    fig, ax = setup_ieee_figure(width_inches=7.0, height_inches=3.0)
    ax.errorbar(rates, mean_miss, yerr=std_miss, color="#2E86AB", marker="o", capsize=3)
    ax.axhline(target_miss, color="green", linestyle="--", linewidth=1.2)
    ax.axvline(float(critical_summary["ramp_rate"]), color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Ramp Rate (rad/s)")
    ax.set_ylabel("Mean Miss Distance (m)")
    ax.set_title("PHANTOM Miss Distance vs. Injection Ramp Rate")
    annotation = (
        f"Critical rate: {float(critical_summary['ramp_rate']):.3f} rad/s\n"
        f"Miss: {float(critical_summary['mean_miss']):.1f}"
        f" ± {float(critical_summary['std_miss']):.1f} m"
    )
    ax.text(
        0.97,
        0.03,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9},
    )
    saved_path = save_ieee_figure(fig, "fig03_miss_vs_ramp_rate", output_dir=figures_dir)
    plt.close(fig)
    return _stable_figure_alias(saved_path)


def _save_detection_figure(
    rate_summaries: list[dict[str, Any]],
    critical_summary: dict[str, Any],
    detection_limit: float,
    figures_dir: str,
) -> str:
    """Create Figure 4: detection rate versus ramp rate."""
    rates = np.asarray(
        [float(summary["ramp_rate"]) for summary in rate_summaries], dtype=np.float64
    )
    detection = (
        np.asarray(
            [float(summary["mean_detection"]) for summary in rate_summaries], dtype=np.float64
        )
        * 100.0
    )

    fig, ax = setup_ieee_figure(width_inches=3.5, height_inches=2.8)
    ax.plot(rates, detection, color="#E84855", marker="o")
    ax.axhline(detection_limit * 100.0, color="orange", linestyle="--", linewidth=1.2)
    ax.axvline(float(critical_summary["ramp_rate"]), color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Ramp Rate (rad/s)")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("PHANTOM EKF Detection Rate vs. Injection Ramp Rate")
    saved_path = save_ieee_figure(fig, "fig04_detection_vs_ramp_rate", output_dir=figures_dir)
    plt.close(fig)
    return _stable_figure_alias(saved_path)


def _save_combined_figure(
    rate_summaries: list[dict[str, Any]],
    critical_summary: dict[str, Any],
    target_miss: float,
    detection_limit: float,
    figures_dir: str,
) -> str:
    """Create Figure 5: the primary two-panel IEEE submission figure."""
    rates = np.asarray(
        [float(summary["ramp_rate"]) for summary in rate_summaries], dtype=np.float64
    )
    mean_miss = np.asarray(
        [float(summary["mean_miss"]) for summary in rate_summaries], dtype=np.float64
    )
    std_miss = np.asarray(
        [float(summary["std_miss"]) for summary in rate_summaries], dtype=np.float64
    )
    detection = (
        np.asarray(
            [float(summary["mean_detection"]) for summary in rate_summaries], dtype=np.float64
        )
        * 100.0
    )

    # Figure 5 is the primary paper figure. Both panels share the same
    # x-axis range and both show the critical rate as a vertical dashed line,
    # allowing the reader to simultaneously read off the miss distance and
    # detection rate at the optimal operating point.
    fig, axis = setup_ieee_figure(width_inches=7.0, height_inches=3.0)
    fig.clf()
    left_ax, right_ax = fig.subplots(1, 2, sharex=True)

    left_ax.errorbar(rates, mean_miss, yerr=std_miss, color="#2E86AB", marker="o", capsize=3)
    left_ax.axhline(target_miss, color="green", linestyle="--", linewidth=1.2)
    left_ax.axvline(
        float(critical_summary["ramp_rate"]), color="red", linestyle="--", linewidth=1.2
    )
    left_ax.set_ylabel("Mean Miss Distance (m)")
    left_ax.set_title("Miss Distance")

    right_ax.plot(rates, detection, color="#E84855", marker="o")
    right_ax.axhline(detection_limit * 100.0, color="orange", linestyle="--", linewidth=1.2)
    right_ax.axvline(
        float(critical_summary["ramp_rate"]), color="red", linestyle="--", linewidth=1.2
    )
    right_ax.set_ylabel("Detection Rate (%)")
    right_ax.set_title("Detection Rate")

    fig.supxlabel("Ramp Rate (rad/s)")
    fig.suptitle("PHANTOM Critical Ramp Rate Trade Space")
    saved_path = save_ieee_figure(fig, "fig05_critical_rate_combined", output_dir=figures_dir)
    plt.close(fig)
    return _stable_figure_alias(saved_path)


def _print_results_table(
    rate_summaries: list[dict[str, Any]],
    critical_summary: dict[str, Any] | None,
    n_runs: int,
    gate_passed: bool,
    target_miss: float,
    detection_limit: float,
    csv_path: str,
    figure_paths: list[str],
) -> None:
    """Render the Experiment 3 terminal report."""
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║          PHANTOM — Experiment 3: Critical Ramp Rate Identification      ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()
    total_engagements = len(rate_summaries) * n_runs
    print(
        f"Sweeping {len(rate_summaries)} ramp rates | {n_runs} runs each | "
        f"{total_engagements:,} total engagements"
    )
    print()
    print("Rate (rad/s)  Mean Miss (m)   Std (m)   Detection (%)  Converged")
    print("──────────────────────────────────────────────────────────────────")
    for summary in rate_summaries:
        rate = float(summary["ramp_rate"])
        marker = (
            "*"
            if critical_summary and np.isclose(rate, float(critical_summary["ramp_rate"]))
            else " "
        )
        converged = "✅" if bool(summary["converged"]) else "❌"
        critical_tag = "  ← CRITICAL" if marker == "*" else ""
        print(
            f"{rate:0.3f}{marker}         "
            f"{float(summary['mean_miss']):7.1f}        "
            f"{float(summary['std_miss']):6.1f}      "
            f"{float(summary['mean_detection']) * 100.0:7.2f}%      "
            f"{converged}{critical_tag}"
        )
    print()
    if critical_summary:
        print("╔══════════════════════════════════════════════════════════════════════════╗")
        print("║  CRITICAL RATE IDENTIFIED                                               ║")
        print(
            f"║  Ramp rate:      {float(critical_summary['ramp_rate']):0.3f} rad/s" f"{' ' * 42}║"
        )
        print(
            f"║  Mean miss:      {float(critical_summary['mean_miss']):.1f} ± "
            f"{float(critical_summary['std_miss']):.1f} m{' ' * 41}║"
        )
        print(
            f"║  Detection rate: {float(critical_summary['mean_detection']) * 100.0:.2f}%"
            f"{' ' * 48}║"
        )
        print(
            f"║  95% CI:         [{float(critical_summary['ci_95_lower']):.1f}, "
            f"{float(critical_summary['ci_95_upper']):.1f}] m{' ' * 36}║"
        )
        print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()
    print("Validation Gate:")
    _cr = critical_summary
    print(f"  {'✅' if _cr else '❌'} Critical rate identified")
    _miss_ok = _cr and float(_cr["mean_miss"]) > target_miss
    _miss_v = float(_cr["mean_miss"]) if _cr else float("nan")
    print(
        f"  {'✅' if _miss_ok else '❌'}"
        f" Mean miss > {target_miss:.0f} m"
        f"          [{_miss_v:.1f} m]"
    )
    _det_ok = _cr and float(_cr["mean_detection"]) < detection_limit
    _det_v = float(_cr["mean_detection"]) * 100.0 if _cr else float("nan")
    print(
        f"  {'✅' if _det_ok else '❌'}"
        f" Detection rate < {detection_limit * 100.0:.0f}%"
        f"        [{_det_v:.2f}%]"
    )
    print(
        f"  {'✅' if all(bool(summary['converged']) for summary in rate_summaries) else '❌'} "
        f"Results converged (N={n_runs})"
    )
    print()
    if gate_passed:
        print("  ✅ PHANTOM Experiment 3 PASSED — Primary hypothesis confirmed")
    else:
        print("  ❌ PHANTOM Experiment 3 FAILED — Critical operating point not confirmed")
    print(f"  📊 Results: {csv_path}")
    for figure_path in figure_paths:
        print(f"  🖼  Figure:  {figure_path}")


def run_ramp_rate_sweep(
    seed: int = 42,
    runs: int | None = None,
    workers: int | None = None,
    output: str = "ramp_sweep",
    fast: bool = False,
    results_dir: str = "data/results",
    figures_dir: str = "figures",
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run the full PHANTOM ramp-rate sweep and return summaries and artifacts."""
    config = load_phantom_config()
    monte_carlo_cfg = config["monte_carlo"]
    validation_cfg = config["validation"]
    runs = int(
        monte_carlo_cfg["fast_runs"]
        if fast
        else (monte_carlo_cfg["ramp_sweep_runs"] if runs is None else runs)
    )
    workers = int(monte_carlo_cfg["parallel_workers"] if workers is None else workers)
    ramp_rates = [float(rate) for rate in config["injection"]["ramp_rates"]]

    LOGGER.info(
        "Starting ramp-rate sweep with %d rates, %d runs per rate, seed=%d, workers=%d",
        len(ramp_rates),
        runs,
        seed,
        workers,
    )

    rate_summaries: list[dict[str, Any]] = []
    for ramp_rate in ramp_rates:
        summary = evaluate_ramp_rate(
            ramp_rate=ramp_rate,
            n_runs=runs,
            base_seed=seed,
            config=config,
            n_workers=workers,
            show_progress=show_progress,
        )
        rate_summaries.append(summary)

    critical_summary = _identify_critical_rate(
        rate_summaries,
        detection_limit=float(validation_cfg["max_detection_rate"]),
    )
    gate_passed = bool(
        critical_summary
        and float(critical_summary["mean_miss"]) > float(validation_cfg["target_miss_distance"])
        and float(critical_summary["mean_detection"]) < float(validation_cfg["max_detection_rate"])
        and all(bool(summary["converged"]) for summary in rate_summaries)
    )

    csv_rows = [
        {key: value for key, value in summary.items() if key != "results"}
        for summary in rate_summaries
    ]
    csv_path = save_results_csv(csv_rows, f"{output}_seed{seed}", output_dir=results_dir)

    figure_paths: list[str] = []
    if critical_summary is not None:
        figure_paths = [
            _save_miss_figure(
                rate_summaries,
                critical_summary,
                float(validation_cfg["target_miss_distance"]),
                figures_dir,
            ),
            _save_detection_figure(
                rate_summaries,
                critical_summary,
                float(validation_cfg["max_detection_rate"]),
                figures_dir,
            ),
            _save_combined_figure(
                rate_summaries,
                critical_summary,
                float(validation_cfg["target_miss_distance"]),
                float(validation_cfg["max_detection_rate"]),
                figures_dir,
            ),
        ]

    return {
        "config": config,
        "seed": seed,
        "runs": runs,
        "workers": workers,
        "rate_summaries": rate_summaries,
        "critical_summary": critical_summary,
        "critical_rate": float(critical_summary["ramp_rate"]) if critical_summary else None,
        "gate_passed": gate_passed,
        "csv_path": csv_path,
        "figure_paths": figure_paths,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the PHANTOM ramp-rate sweep."""
    parser = argparse.ArgumentParser(description="Run the PHANTOM ramp-rate sweep.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for the sweep.")
    parser.add_argument("--runs", type=int, default=None, help="Runs per ramp rate.")
    parser.add_argument("--workers", type=int, default=None, help="Pool worker count.")
    parser.add_argument(
        "--output",
        type=str,
        default="ramp_sweep",
        help="Output filename stem without the phantom_ prefix or timestamp.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use the config fast-run count for a quick validation sweep.",
    )
    return parser.parse_args()


def main() -> int:
    """Execute Experiment 3 from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    summary = run_ramp_rate_sweep(
        seed=args.seed,
        runs=args.runs,
        workers=args.workers,
        output=args.output,
        fast=args.fast,
    )
    _print_results_table(
        rate_summaries=summary["rate_summaries"],
        critical_summary=summary["critical_summary"],
        n_runs=summary["runs"],
        gate_passed=summary["gate_passed"],
        target_miss=float(summary["config"]["validation"]["target_miss_distance"]),
        detection_limit=float(summary["config"]["validation"]["max_detection_rate"]),
        csv_path=summary["csv_path"],
        figure_paths=summary["figure_paths"],
    )
    return 0 if summary["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

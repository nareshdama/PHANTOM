"""Generate all 9 publication figures for the PHANTOM IEEE TAES paper.

Outputs are saved as .jpg in docs/ so the LaTeX draft can reference them
directly. Each figure uses the IEEE style: serif font, 10pt, 300 DPI,
colorblind-safe palette.

Usage:
    python docs/generate_paper_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from core.injection import LinearRampInjection, NoInjection
from core.simulator import PhantomSimulator
from core.utils import (
    get_default_initial_conditions,
    load_phantom_config,
    randomize_initial_conditions,
)

DOCS_DIR = PROJECT_ROOT / "docs"
BLUE = "#2E86AB"
RED = "#E84855"
GREEN = "#28A745"
ORANGE = "#FF8C00"
DPI = 300


def _ieee_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": DPI,
    })


def _save(fig: plt.Figure, name: str) -> None:
    path = DOCS_DIR / f"{name}.jpg"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", format="jpg")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _project_ic(config, sampled_ic):
    baseline = get_default_initial_conditions(config)
    ms = float(np.linalg.norm(sampled_ic["missile_vel"]))
    ts = float(np.linalg.norm(sampled_ic["target_vel"]))
    sep = float(np.linalg.norm(baseline["target_pos"] - baseline["missile_pos"]))
    cs = float(np.linalg.norm(baseline["missile_vel"]) + np.linalg.norm(baseline["target_vel"]))
    new_sep = (ms + ts) * (sep / cs)
    mp = np.asarray(sampled_ic["missile_pos"], dtype=np.float64)
    return {
        "missile_pos": mp.copy(),
        "missile_vel": np.array([ms, 0.0]),
        "target_pos": mp + np.array([new_sep, 0.0]),
        "target_vel": np.array([-ts, 0.0]),
    }


def run_baseline_mc(config, n=100, base_seed=42):
    results = []
    for i in range(n):
        seed = base_seed + i * 1000
        rng = np.random.RandomState(seed)
        ic = randomize_initial_conditions(config, rng)
        ic = _project_ic(config, ic)
        sim = PhantomSimulator(config, seed=seed)
        result = sim.run_engagement(NoInjection(), ic)
        results.append(result)
    return results


def run_single_engagement(config, ramp_rate, seed=42, use_default_ic=False):
    if use_default_ic:
        ic = get_default_initial_conditions(config)
    else:
        rng = np.random.RandomState(seed)
        ic = randomize_initial_conditions(config, rng)
        ic = _project_ic(config, ic)
    sim = PhantomSimulator(config, seed=seed)
    inj = LinearRampInjection(
        t_start=float(config["injection"]["ramp_tstart"]),
        t_end=float(config["injection"]["ramp_tend"]),
        ramp_rate=ramp_rate,
    )
    return sim.run_engagement(inj, ic)


def run_baseline_single(config, seed=42):
    rng = np.random.RandomState(seed)
    ic = randomize_initial_conditions(config, rng)
    ic = _project_ic(config, ic)
    sim = PhantomSimulator(config, seed=seed)
    return sim.run_engagement(NoInjection(), ic)


# ── Figure 1: Baseline miss-distance histogram ──────────────────────────────

def fig01_baseline_histogram(config, baseline_results):
    print("Fig 1: Baseline miss histogram")
    misses = np.array([r["miss_distance"] for r in baseline_results])
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.hist(misses, bins=20, color=BLUE, edgecolor="black", alpha=0.8)
    ax.axvline(5.0, color=RED, linestyle="--", linewidth=1.2, label="5 m gate")
    ax.set_xlabel("Miss Distance (m)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "phantom_fig01_baseline_miss_distribution")


# ── Figure 2: Baseline trajectory ───────────────────────────────────────────

def fig02_baseline_trajectory(config):
    print("Fig 2: Baseline trajectory")
    result = run_baseline_single(config, seed=42)
    traj = result["trajectory"]
    mp = np.array([e["missile_pos"] for e in traj])
    tp = np.array([e["target_pos"] for e in traj])
    ep = np.array([e["ekf_estimate"] for e in traj])

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(mp[:, 0], mp[:, 1], color=BLUE, label="Missile")
    ax.plot(tp[:, 0], tp[:, 1], color=RED, label="Target")
    ax.plot(ep[:, 0], ep[:, 1], color=GREEN, linestyle="--", label="EKF estimate")
    ax.plot(mp[-1, 0], mp[-1, 1], "kx", markersize=8, label="Intercept")
    ax.set_xlabel("Downrange (m)")
    ax.set_ylabel("Cross-range (m)")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    _save(fig, "phantom_fig02_baseline_trajectory")


# ── Figure 3: Miss distance vs ramp rate ────────────────────────────────────

def fig03_miss_vs_rate(sweep_df):
    print("Fig 3: Miss vs ramp rate")
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.errorbar(
        sweep_df["ramp_rate"], sweep_df["mean_miss"], yerr=sweep_df["std_miss"],
        color=BLUE, marker="o", capsize=3, markersize=4,
    )
    ax.axhline(200.0, color=GREEN, linestyle="--", linewidth=1.2, label="200 m target")
    ax.axvline(0.035, color=RED, linestyle="--", linewidth=1.0, alpha=0.7, label="Critical rate")
    ax.set_xlabel("Ramp Rate (rad/s)")
    ax.set_ylabel("Mean Miss Distance (m)")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    _save(fig, "phantom_fig03_miss_vs_ramp_rate")


# ── Figure 4: Detection rate vs ramp rate ───────────────────────────────────

def fig04_detection_vs_rate(sweep_df):
    print("Fig 4: Detection vs ramp rate")
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(
        sweep_df["ramp_rate"], sweep_df["mean_detection"] * 100,
        color=RED, marker="o", markersize=4,
    )
    ax.axhline(5.0, color=ORANGE, linestyle="--", linewidth=1.2, label="5% threshold")
    ax.axvline(0.035, color=RED, linestyle="--", linewidth=1.0, alpha=0.7, label="Critical rate")
    ax.set_xlabel("Ramp Rate (rad/s)")
    ax.set_ylabel("Detection Rate (%)")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    _save(fig, "phantom_fig04_detection_vs_ramp_rate")


# ── Figure 5: Two-panel critical rate (primary) ────────────────────────────

def fig05_critical_rate_combined(sweep_df):
    print("Fig 5: Critical rate combined (primary)")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True)
    rates = sweep_df["ramp_rate"]
    ax1.errorbar(
        rates, sweep_df["mean_miss"], yerr=sweep_df["std_miss"],
        color=BLUE, marker="o", capsize=3, markersize=4, label=r"Mean $\pm$ std",
    )
    ax1.axhline(200.0, color=GREEN, linestyle="--", linewidth=1.2, label="200 m target")
    ax1.axvline(0.035, color=RED, linestyle="--", linewidth=1.0)
    ax1.set_ylabel("Mean Miss Distance (m)")
    ax1.legend(loc="best", fontsize=7)

    ax2.plot(rates, sweep_df["mean_detection"] * 100, color=RED, marker="o", markersize=4)
    ax2.axhline(5.0, color=ORANGE, linestyle="--", linewidth=1.2, label="5% threshold")
    ax2.axvline(0.035, color=RED, linestyle="--", linewidth=1.0)
    ax2.set_ylabel("Detection Rate (%)")
    ax2.legend(loc="best", fontsize=7)

    fig.supxlabel("Ramp Rate (rad/s)")
    fig.tight_layout()
    _save(fig, "phantom_fig05_critical_rate_combined")


# ── Figure 6: Innovation timeline ───────────────────────────────────────────

def fig06_innovation_timeline(config):
    print("Fig 6: Innovation timeline")
    result = run_single_engagement(config, ramp_rate=0.035, seed=3042)
    traj = result["trajectory"]
    times = np.array([e["t"] for e in traj])
    gammas = np.array([e["gamma_k"] for e in traj])
    accepted = np.array([e["measurement_accepted"] for e in traj])

    t_start = float(config["injection"]["ramp_tstart"])
    t_end = float(config["injection"]["ramp_tend"])
    window_mask = (times >= t_start) & (times <= t_end)
    window_times = times[window_mask]
    window_gammas = gammas[window_mask]
    window_accepted = accepted[window_mask]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(window_times, window_gammas, color=BLUE, linewidth=1.0, label=r"$\gamma_k$")
    ax.axhline(9.21, color=RED, linestyle="--", linewidth=1.2, label=r"$\chi^2$ gate = 9.21")

    for t_val, acc in zip(window_times, window_accepted):
        if not acc:
            ax.axvspan(t_val - 0.01, t_val + 0.01, color=ORANGE, alpha=0.3, linewidth=0)

    mean_g = float(np.mean(window_gammas))
    max_g = float(np.max(window_gammas))
    n_rej = int(np.sum(~window_accepted))
    ax.text(
        0.97, 0.97,
        f"Mean: {mean_g:.2f}\nPeak: {max_g:.2f}\nRejects: {n_rej}",
        transform=ax.transAxes, ha="right", va="top", fontsize=7,
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.9),
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Innovation Statistic $\gamma_k$")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    _save(fig, "phantom_fig06_innovation_timeline")


# ── Figure 7: Trajectory comparison (baseline vs injection) ────────────────

def fig07_trajectory_comparison(config):
    print("Fig 7: Trajectory comparison")
    baseline = run_baseline_single(config, seed=42)
    injection = run_single_engagement(config, ramp_rate=0.035, seed=3042)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=True)

    for ax, result, title in [(ax1, baseline, "Baseline"), (ax2, injection, "PHANTOM")]:
        traj = result["trajectory"]
        mp = np.array([e["missile_pos"] for e in traj])
        tp = np.array([e["target_pos"] for e in traj])
        ep = np.array([e["ekf_estimate"] for e in traj])
        ax.plot(mp[:, 0], mp[:, 1], color=BLUE, label="Missile")
        ax.plot(tp[:, 0], tp[:, 1], color=RED, label="Target")
        ax.plot(ep[:, 0], ep[:, 1], color=GREEN, linestyle="--", label="EKF estimate")
        ax.set_xlabel("Downrange (m)")
        ax.set_title(title, fontsize=10)

    ax1.set_ylabel("Cross-range (m)")
    ax1.plot(
        baseline["trajectory"][-1]["missile_pos"][0],
        baseline["trajectory"][-1]["missile_pos"][1],
        "kx", markersize=8, label="Intercept",
    )
    inj_final = injection["trajectory"][-1]
    ax2.plot(
        [inj_final["missile_pos"][0], inj_final["target_pos"][0]],
        [inj_final["missile_pos"][1], inj_final["target_pos"][1]],
        color=ORANGE, linestyle=":", linewidth=1.5, label="Miss vector",
    )
    ax1.legend(loc="best", fontsize=6)
    ax2.legend(loc="best", fontsize=6)
    fig.tight_layout()
    _save(fig, "phantom_fig07_trajectory_comparison")


# ── Figure 8: Estimation error growth ───────────────────────────────────────

def fig08_estimation_error(config):
    print("Fig 8: Estimation error")
    injection = run_single_engagement(config, ramp_rate=0.035, seed=3042)
    baseline = run_baseline_single(config, seed=42)

    traj_inj = injection["trajectory"]
    traj_bl = baseline["trajectory"]

    t_inj = np.array([e["t"] for e in traj_inj])
    err_inj = np.array([e["estimation_error"] for e in traj_inj])
    t_bl = np.array([e["t"] for e in traj_bl])
    err_bl = np.array([e["estimation_error"] for e in traj_bl])

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(t_inj, err_inj, color=BLUE, linewidth=1.2, label="PHANTOM injection")
    ax.plot(t_bl, err_bl, color=GREEN, linestyle="--", linewidth=1.0, label="Baseline")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Estimation Error (m)")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    _save(fig, "phantom_fig08_estimation_error")


# ── Figure 9: Power-law scaling fit ─────────────────────────────────────────

def fig09_scaling_law(sweep_df):
    print("Fig 9: Scaling law")
    rates = sweep_df["ramp_rate"].values
    misses = sweep_df["mean_miss"].values
    stds = sweep_df["std_miss"].values

    peak_idx = int(np.argmax(misses)) + 1
    fit_rates = rates[:peak_idx]
    fit_misses = np.maximum.accumulate(misses[:peak_idx])
    miss_floor = float(np.min(fit_misses)) - 1.0
    fit_excess = fit_misses - miss_floor

    def power_law(x, a, b):
        return a * np.power(x, b)

    popt, _ = curve_fit(power_law, fit_rates, fit_excess, bounds=([0, 0.5], [1e9, 5.0]), maxfev=100000)
    a_fit, b_fit = popt
    predicted = power_law(rates, a_fit, b_fit) + miss_floor
    ss_res = np.sum((misses[:peak_idx] - predicted[:peak_idx]) ** 2)
    ss_tot = np.sum((misses[:peak_idx] - np.mean(misses[:peak_idx])) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.errorbar(rates, misses, yerr=stds, color=BLUE, marker="o", capsize=3, markersize=4, label="Sweep data")
    smooth_rates = np.linspace(rates[0], rates[-1], 200)
    ax.plot(smooth_rates, power_law(smooth_rates, a_fit, b_fit) + miss_floor, color=RED, linewidth=1.5, label="Power-law fit")
    ax.text(
        0.97, 0.05,
        f"$b = {b_fit:.2f}$\n$R^2 = {r2:.3f}$",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.9),
    )
    ax.set_xlabel("Ramp Rate (rad/s)")
    ax.set_ylabel("Mean Miss Distance (m)")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    _save(fig, "phantom_fig09_miss_distance_scaling")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    _ieee_style()
    config = load_phantom_config()

    csv_path = sorted(
        (PROJECT_ROOT / "data" / "results").glob("phantom_ramp_sweep_seed42*.csv")
    )[-1]
    sweep_df = pd.read_csv(csv_path)
    print(f"Loaded sweep CSV: {csv_path.name}")

    print("\nRunning baseline Monte Carlo (N=100)...")
    baseline_results = run_baseline_mc(config, n=100, base_seed=42)

    print("\nGenerating figures...\n")
    fig01_baseline_histogram(config, baseline_results)
    fig02_baseline_trajectory(config)
    fig03_miss_vs_rate(sweep_df)
    fig04_detection_vs_rate(sweep_df)
    fig05_critical_rate_combined(sweep_df)
    fig06_innovation_timeline(config)
    fig07_trajectory_comparison(config)
    fig08_estimation_error(config)
    fig09_scaling_law(sweep_df)

    print("\nAll 9 figures generated in docs/")


if __name__ == "__main__":
    main()

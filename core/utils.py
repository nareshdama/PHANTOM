"""PHANTOM — Shared utilities for geometry, statistics, and I/O workflows.

This module collects the helper functions that multiple PHANTOM phases need
but that do not naturally belong inside the EKF, guidance, injection, or
simulator classes. These utilities support experiment scripts, analysis
pipelines, visualization, and dataset generation while keeping a consistent
mathematical and file-output convention across the project.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import yaml  # type: ignore[import-untyped]
from numpy.random import RandomState

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_CONFIG_SECTIONS = (
    "simulation",
    "monte_carlo",
    "initial_conditions",
    "injection",
    "validation",
)


def _as_vector(vector: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert an input array-like to a 2D float vector for PHANTOM geometry."""
    array = np.asarray(vector, dtype=np.float64)
    if array.shape != (2,):
        raise ValueError(f"Expected shape (2,), got {array.shape}")
    return array


def _resolve_path(path_str: str) -> Path:
    """Resolve PHANTOM-relative paths from the repository root."""
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _timestamp() -> str:
    """Return a filesystem-safe UTC timestamp for PHANTOM outputs."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _build_output_path(output_dir: str, stem: str, suffix: str) -> Path:
    """Create a timestamped PHANTOM output path and ensure its directory exists."""
    directory = _resolve_path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"phantom_{stem}_{_timestamp()}{suffix}"


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi] for numerically stable PHANTOM bearing math.

    Bearing measurements are defined on [-pi, pi] but EKF innovations can
    momentarily exceed this range during rapid geometry changes. Wrapping
    prevents artificial large innovations that would falsely trigger the
    chi-squared gate — a subtle numerical issue that caused incorrect
    detection rates in early PHANTOM prototypes.
    """
    wrapped = float((angle + np.pi) % (2.0 * np.pi) - np.pi)
    if np.isclose(wrapped, -np.pi) and angle < 0.0:
        return float(np.pi)
    return wrapped


def range_bearing_to_xy(
    range_m: float,
    bearing_rad: float,
    origin: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert a seeker measurement into a Cartesian PHANTOM target position.

    Experiment scripts often work in range-bearing space while analysis and
    guidance diagnostics are easier in Cartesian coordinates. This helper makes
    the conversion explicit so PHANTOM's EKF and post-processing tools use the
    same geometry convention everywhere.
    """
    origin_vec = _as_vector(origin)
    return origin_vec + np.array(
        [range_m * np.cos(bearing_rad), range_m * np.sin(bearing_rad)],
        dtype=np.float64,
    )


def los_angle(
    missile_pos: npt.NDArray[np.float64],
    target_pos: npt.NDArray[np.float64],
) -> float:
    """Return the instantaneous line-of-sight angle sigma for PN analysis.

    PHANTOM corrupts the guidance loop through line-of-sight geometry, so a
    single canonical LOS-angle helper avoids silent sign-convention drift
    between EKF studies, guidance diagnostics, and plotting scripts.
    """
    rel = _as_vector(target_pos) - _as_vector(missile_pos)
    return wrap_angle(float(np.arctan2(rel[1], rel[0])))


def xy_to_range_bearing(
    source: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """Convert missile and target Cartesian positions to range and bearing.

    This is the inverse of PHANTOM's polar-to-Cartesian helper and lets
    analysis code compare true geometry, EKF estimates, and injected bearing
    errors using the same physical measurement representation as the seeker.
    """
    source_vec = _as_vector(source)
    target_vec = _as_vector(target)
    rel = target_vec - source_vec
    return float(np.linalg.norm(rel)), los_angle(source_vec, target_vec)


def compute_miss_vector(
    missile_pos: npt.NDArray[np.float64],
    target_pos: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """Return miss distance and miss angle for PHANTOM intercept analysis.

    Miss distance alone tells whether deception succeeded, but miss angle tells
    where the missile was steered. That directional information becomes
    especially important in later PHANTOM phases that aim for controlled miss
    placement rather than merely causing any miss.
    """
    miss_distance, bearing = xy_to_range_bearing(missile_pos, target_pos)
    return miss_distance, float(np.degrees(bearing))


def compute_detection_threshold_crossing(
    innovation_history: list[float],
    chi2_threshold: float = 9.21,
) -> dict[str, float | int]:
    """Summarize when EKF innovations crossed the PHANTOM detection threshold.

    PHANTOM is successful only when the injected signal meaningfully distorts
    guidance while staying mostly below the chi-squared gate. This helper turns
    a raw gamma_k time series into a compact detection-health summary suitable
    for experiment logs and adaptive-controller studies.
    """
    if not innovation_history:
        return {
            "n_crossings": 0,
            "first_crossing": -1.0,
            "max_gamma": 0.0,
            "mean_gamma": 0.0,
            "detection_rate": 0.0,
        }

    innovations = np.asarray(innovation_history, dtype=np.float64)
    crossings = innovations > chi2_threshold
    crossing_indices = np.flatnonzero(crossings)
    return {
        "n_crossings": int(np.sum(crossings)),
        "first_crossing": float(crossing_indices[0]) if crossing_indices.size else -1.0,
        "max_gamma": float(np.max(innovations)),
        "mean_gamma": float(np.mean(innovations)),
        "detection_rate": float(np.mean(crossings)),
    }


def compute_engagement_statistics(results: list[dict[str, Any]]) -> dict[str, float | int]:
    """Aggregate Monte Carlo PHANTOM results into IEEE-ready summary statistics.

    PHANTOM experiments are evaluated over batches of engagements, not just
    single runs. This helper standardizes the exact summary metrics reported by
    experiment scripts and report generators so every table uses the same
    interpretation of miss distance, detection rate, success rate, and
    confidence bounds.
    """
    if not results:
        raise ValueError("results must contain at least one engagement result.")

    miss_distances = np.asarray([result["miss_distance"] for result in results], dtype=np.float64)
    detection_rates = np.asarray([result["detection_rate"] for result in results], dtype=np.float64)
    success_flags = np.asarray([bool(result["success"]) for result in results], dtype=np.float64)
    n_runs = int(miss_distances.size)
    mean_miss = float(np.mean(miss_distances))
    std_miss = float(np.std(miss_distances))

    # 95% confidence interval using standard error of the mean.
    # For IEEE TAES tables, results must be reported as mean ± std with
    # explicit N and CI bounds. This matches the format used in
    # Zarchan (2012) Monte Carlo guidance analyses.
    standard_error = std_miss / np.sqrt(n_runs)
    ci_half_width = 1.96 * standard_error

    return {
        "n_runs": n_runs,
        "mean_miss": mean_miss,
        "std_miss": std_miss,
        "median_miss": float(np.median(miss_distances)),
        "p95_miss": float(np.percentile(miss_distances, 95.0)),
        "min_miss": float(np.min(miss_distances)),
        "max_miss": float(np.max(miss_distances)),
        "mean_detection": float(np.mean(detection_rates)),
        "std_detection": float(np.std(detection_rates)),
        "success_rate": float(np.mean(success_flags)),
        "ci_95_lower": mean_miss - ci_half_width,
        "ci_95_upper": mean_miss + ci_half_width,
    }


def check_monte_carlo_convergence(
    miss_distances: list[float],
    window: int = 50,
    tolerance: float = 5.0,
) -> bool:
    """Check whether PHANTOM Monte Carlo miss statistics have stabilized.

    Large PHANTOM sweeps can consume substantial compute. This convergence test
    lets experiment scripts stop early once the recent running mean agrees with
    the overall mean closely enough that additional samples are unlikely to
    change the engineering conclusion.
    """
    if window <= 0 or tolerance < 0.0 or len(miss_distances) < window:
        return False

    values = np.asarray(miss_distances, dtype=np.float64)
    overall_mean = float(np.mean(values))
    recent_mean = float(np.mean(values[-window:]))
    return abs(recent_mean - overall_mean) <= tolerance


def load_phantom_config(config_path: str = "configs/phantom_config.yaml") -> dict[str, Any]:
    """Load and validate PHANTOM's single source-of-truth configuration.

    Centralized config loading matters because PHANTOM experiments must be
    reproducible across baseline studies, Monte Carlo sweeps, and training-data
    generation. Enforcing one loader prevents scripts from drifting into
    incompatible local assumptions about simulation or validation settings.
    """
    resolved_path = _resolve_path(config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"PHANTOM config not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise KeyError("PHANTOM config must deserialize to a dictionary.")

    missing_sections = [section for section in REQUIRED_CONFIG_SECTIONS if section not in config]
    if missing_sections:
        raise KeyError(f"Missing required PHANTOM config sections: {missing_sections}")

    LOGGER.info("Loaded PHANTOM config from %s", resolved_path)
    return config


def get_default_initial_conditions(config: dict[str, Any]) -> dict[str, npt.NDArray[np.float64]]:
    """Return the canonical PHANTOM engagement geometry as float arrays.

    Many PHANTOM scripts need the exact baseline engagement used for regression
    tests and paper figures. Returning array-form initial conditions from one
    place avoids subtle list-vs-array mismatches across simulator call sites.
    """
    initial_conditions = config["initial_conditions"]
    return {
        key: np.asarray(initial_conditions[key], dtype=np.float64)
        for key in ("missile_pos", "missile_vel", "target_pos", "target_vel")
    }


def randomize_initial_conditions(
    config: dict[str, Any],
    rng: RandomState,
) -> dict[str, npt.NDArray[np.float64]]:
    """Sample reproducible Monte Carlo initial conditions from config ranges.

    PHANTOM's statistical claims depend on many randomized engagements, but
    those runs must remain fully reproducible across workers and hardware. By
    requiring a caller-supplied random state and config-defined ranges, this
    helper keeps Monte Carlo studies both flexible and deterministic.
    """
    defaults = get_default_initial_conditions(config)
    randomization = config["monte_carlo"]["randomization"]

    separation = rng.uniform(
        randomization["initial_separation_min"],
        randomization["initial_separation_max"],
    )
    bearing_deg = rng.uniform(
        randomization["initial_bearing_min_deg"],
        randomization["initial_bearing_max_deg"],
    )
    missile_speed = rng.normal(
        randomization["missile_speed_mean"],
        randomization["missile_speed_std"],
    )
    target_speed = rng.normal(
        randomization["target_speed_mean"],
        randomization["target_speed_std"],
    )

    bearing_rad = float(np.deg2rad(bearing_deg))
    missile_pos = defaults["missile_pos"].copy()
    target_pos = range_bearing_to_xy(separation, bearing_rad, missile_pos)
    missile_vel = np.array([missile_speed, 0.0], dtype=np.float64)

    # Baseline Monte Carlo runs should vary geometry without destroying the
    # null-hypothesis collision course. Choose the target heading so the
    # initial LOS rate is zero, which is the physically correct no-injection
    # baseline that Experiment 1 is meant to validate.
    los_unit = np.array([np.cos(bearing_rad), np.sin(bearing_rad)], dtype=np.float64)
    missile_speed_cross = missile_speed * abs(np.sin(bearing_rad))
    if target_speed <= missile_speed_cross:
        target_speed = missile_speed_cross + 1e-6
    closing_speed = missile_speed * np.cos(bearing_rad) + np.sqrt(
        target_speed**2 - missile_speed_cross**2
    )
    target_vel = missile_vel - closing_speed * los_unit
    return {
        "missile_pos": missile_pos,
        "missile_vel": missile_vel,
        "target_pos": target_pos,
        "target_vel": target_vel,
    }


def save_results_csv(
    results: list[dict[str, Any]],
    filename: str,
    output_dir: str = "data/results",
) -> str:
    """Save PHANTOM engagement summaries as a scalar-only CSV table.

    CSV exports are convenient for quick inspection, report tables, and
    downstream spreadsheet workflows. PHANTOM keeps them scalar-only so
    trajectory-heavy data stays out of the flat file and moves to HDF5 instead.
    """
    scalar_rows = [
        {
            key: value
            for key, value in result.items()
            if isinstance(value, (bool, int, float, str, np.bool_, np.integer, np.floating))
        }
        for result in results
    ]
    output_path = _build_output_path(output_dir, filename, ".csv")
    pd.DataFrame(scalar_rows).to_csv(output_path, index=False)
    LOGGER.info("Saved PHANTOM CSV results to %s", output_path)
    return str(output_path)


def save_results_hdf5(
    results: list[dict[str, Any]],
    filename: str,
    output_dir: str = "data/results",
) -> str:
    """Save full PHANTOM engagement records, including trajectories, to HDF5.

    HDF5 is PHANTOM's archival format for large experiment batches because it
    preserves full telemetry without flattening away time history. That makes
    it suitable for later analysis, visualization, and Phase 3 training-data
    generation from the same underlying runs.
    """
    output_path = _build_output_path(output_dir, filename, ".h5")
    with h5py.File(output_path, "w") as handle:
        for run_id, result in enumerate(results):
            group = handle.create_group(f"run_{run_id:05d}")
            group.attrs["seed"] = int(result.get("seed", -1))
            group.attrs["miss_distance"] = float(result.get("miss_distance", 0.0))
            group.attrs["detection_rate"] = float(result.get("detection_rate", 0.0))
            group.attrs["success"] = bool(result.get("success", False))
            group.attrs["profile_type"] = str(result.get("profile_type", "unknown"))

            trajectory = result.get("trajectory", [])
            if trajectory:
                # HDF5 chunked storage — each run is a separate group to allow
                # partial reads during analysis without loading the full dataset.
                # Critical for the 100K training scenarios in Phase 3.
                for key in trajectory[0]:
                    data = np.asarray([entry[key] for entry in trajectory])
                    group.create_dataset(key, data=data, chunks=True)

    LOGGER.info("Saved PHANTOM HDF5 results to %s", output_path)
    return str(output_path)


def setup_ieee_figure(
    width_inches: float = 3.5,
    height_inches: float = 2.8,
) -> tuple[Any, Any]:
    """Create a publication-ready figure with PHANTOM's IEEE plotting defaults.

    PHANTOM figures are destined for IEEE-style papers and reports, so the
    project benefits from one canonical plotting setup rather than ad hoc
    styling spread across analysis scripts. This helper enforces consistent
    sizing, typography, and paper-context aesthetics from the start.
    """
    sns.set_theme(
        context="paper",
        style="whitegrid",
        rc={"font.family": "serif", "font.size": 10},
    )
    fig, ax = plt.subplots(figsize=(width_inches, height_inches))
    return fig, ax


def save_ieee_figure(fig: Any, filename: str, output_dir: str = "figures") -> str:
    """Save a PHANTOM figure at IEEE publication quality with timestamped naming.

    Publication figures must be reproducible, high-resolution, and easy to
    trace back to the script that generated them. A standardized save helper
    ensures all PHANTOM plots land in a predictable location with consistent
    300 DPI export settings.
    """
    output_path = _build_output_path(output_dir, filename, ".png")
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        format="png",
        pil_kwargs={"dpi": (300, 300)},
    )
    LOGGER.info("Saved PHANTOM figure to %s", output_path)
    return str(output_path)

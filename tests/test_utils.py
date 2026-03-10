"""PHANTOM — Unit tests for the shared utilities module."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from core.utils import (
    check_monte_carlo_convergence,
    compute_detection_threshold_crossing,
    compute_engagement_statistics,
    compute_miss_vector,
    get_default_initial_conditions,
    load_phantom_config,
    los_angle,
    randomize_initial_conditions,
    range_bearing_to_xy,
    save_ieee_figure,
    save_results_csv,
    save_results_hdf5,
    setup_ieee_figure,
    wrap_angle,
    xy_to_range_bearing,
)


def _fake_results() -> list[dict[str, object]]:
    """Create compact PHANTOM-style results for utility I/O tests."""
    trajectory = [
        {
            "t": 0.0,
            "missile_pos": np.array([0.0, 0.0]),
            "missile_vel": np.array([600.0, 0.0]),
            "target_pos": np.array([5000.0, 0.0]),
            "ekf_estimate": np.array([5000.0, 0.0]),
            "ekf_vel_estimate": np.array([-250.0, 0.0]),
            "gamma_k": 1.0,
            "measurement_accepted": True,
            "injection_rate": 0.0,
            "accumulated_angle": 0.0,
            "true_range": 5000.0,
            "estimated_range": 5000.0,
            "estimation_error": 0.0,
        }
    ]
    return [
        {
            "seed": idx,
            "trajectory": trajectory,
            "miss_distance": float(miss),
            "detection_rate": float(idx) / 10.0,
            "success": miss > 150.0,
            "profile_type": "NoInjection",
        }
        for idx, miss in enumerate((100.0, 200.0, 300.0), start=1)
    ]


def test_wrap_angle_within_bounds() -> None:
    """Wrapped angles must remain inside the EKF's principal interval."""
    wrapped_positive = wrap_angle(3.0 * np.pi)
    wrapped_negative = wrap_angle(-3.0 * np.pi)
    wrapped_zero = wrap_angle(0.0)

    assert wrapped_positive == pytest.approx(-np.pi)
    assert wrapped_negative == pytest.approx(np.pi)
    assert wrapped_zero == pytest.approx(0.0)
    for value in (wrapped_positive, wrapped_negative, wrapped_zero):
        assert -np.pi <= value <= np.pi


def test_xy_to_range_bearing_known_geometry() -> None:
    """Axis-aligned geometry should produce exact PHANTOM seeker measurements."""
    range_x, bearing_x = xy_to_range_bearing(np.array([0.0, 0.0]), np.array([1000.0, 0.0]))
    range_y, bearing_y = xy_to_range_bearing(np.array([0.0, 0.0]), np.array([0.0, 1000.0]))

    assert range_x == pytest.approx(1000.0, abs=1e-6)
    assert bearing_x == pytest.approx(0.0, abs=1e-6)
    assert range_y == pytest.approx(1000.0, abs=1e-6)
    assert bearing_y == pytest.approx(np.pi / 2.0, abs=1e-6)
    assert los_angle(np.array([0.0, 0.0]), np.array([0.0, 1000.0])) == pytest.approx(
        np.pi / 2.0, abs=1e-6
    )


def test_range_bearing_roundtrip() -> None:
    """Cartesian-to-polar conversions must be reversible for analysis code."""
    origin = np.array([100.0, -50.0])
    point = np.array([1200.0, 300.0])
    range_m, bearing = xy_to_range_bearing(origin, point)
    reconstructed = range_bearing_to_xy(range_m, bearing, origin)
    np.testing.assert_allclose(reconstructed, point, atol=1e-6)


def test_compute_miss_vector_correct_distance() -> None:
    """Miss vector geometry should preserve both distance and direction."""
    miss_x, angle_x = compute_miss_vector(np.array([0.0, 0.0]), np.array([150.0, 0.0]))
    miss_y, angle_y = compute_miss_vector(np.array([0.0, 0.0]), np.array([0.0, 200.0]))

    assert miss_x == pytest.approx(150.0)
    assert angle_x == pytest.approx(0.0)
    assert miss_y == pytest.approx(200.0)
    assert angle_y == pytest.approx(90.0)


def test_compute_engagement_statistics_correct_mean() -> None:
    """Monte Carlo aggregation must report the correct batch mean."""
    stats = compute_engagement_statistics(_fake_results())
    threshold_stats = compute_detection_threshold_crossing([1.0, 5.0, 10.0, 12.0])

    assert stats["mean_miss"] == pytest.approx(200.0)
    assert stats["n_runs"] == 3
    assert threshold_stats["n_crossings"] == 2
    assert threshold_stats["first_crossing"] == 2.0


def test_compute_engagement_statistics_ci_bounds() -> None:
    """The 95% confidence interval must contain the Monte Carlo mean."""
    results = [
        {
            "miss_distance": float(150.0 + idx),
            "detection_rate": 0.01,
            "success": True,
        }
        for idx in range(100)
    ]
    stats = compute_engagement_statistics(results)

    assert stats["ci_95_lower"] < stats["mean_miss"] < stats["ci_95_upper"]


def test_monte_carlo_convergence_detects_stable_mean() -> None:
    """Stable batches should converge while drifting batches should not."""
    stable = [200.0] * 200
    unstable = list(np.linspace(100.0, 400.0, 200))

    assert check_monte_carlo_convergence(stable) is True
    assert check_monte_carlo_convergence(unstable) is False


def test_load_phantom_config_returns_required_sections() -> None:
    """The PHANTOM config loader must enforce the project schema."""
    config = load_phantom_config("configs/phantom_config.yaml")

    assert "simulation" in config
    assert "monte_carlo" in config
    assert "injection" in config
    assert "validation" in config


def test_load_phantom_config_raises_on_missing_file() -> None:
    """Missing config files must fail loudly instead of falling back silently."""
    with pytest.raises(FileNotFoundError):
        load_phantom_config("nonexistent.yaml")


def test_randomize_initial_conditions_within_bounds() -> None:
    """Randomized Monte Carlo starts must stay within PHANTOM's configured envelope."""
    config = load_phantom_config("configs/phantom_config.yaml")
    defaults = get_default_initial_conditions(config)
    rng = np.random.RandomState(42)
    initial_conditions = randomize_initial_conditions(config, rng)

    separation = np.linalg.norm(
        initial_conditions["target_pos"] - initial_conditions["missile_pos"]
    )
    missile_speed = np.linalg.norm(initial_conditions["missile_vel"])

    assert np.array_equal(defaults["missile_pos"], np.array([0.0, 0.0]))
    assert 4000.0 < separation < 6000.0
    assert 500.0 < missile_speed < 700.0


def test_save_and_load_results_csv(tmp_path: Path) -> None:
    """CSV exports should preserve PHANTOM scalar result fields."""
    results = _fake_results()
    csv_path = save_results_csv(results, "utils_test", output_dir=str(tmp_path))
    hdf5_path = save_results_hdf5(results, "utils_test", output_dir=str(tmp_path))

    dataframe = pd.read_csv(csv_path)
    assert len(dataframe) == 3
    assert "miss_distance" in dataframe.columns

    with h5py.File(hdf5_path, "r") as handle:
        assert "run_00000" in handle
        assert "t" in handle["run_00000"]


def test_setup_ieee_figure_returns_fig_ax(tmp_path: Path) -> None:
    """IEEE figure helpers should return and save publication-ready figures."""
    fig, ax = setup_ieee_figure()
    output_path = save_ieee_figure(fig, "utils_plot", output_dir=str(tmp_path))

    assert fig is not None
    assert ax is not None
    assert fig.get_size_inches()[0] == pytest.approx(3.5)
    assert Path(output_path).exists()
    fig.clf()

"""PHANTOM — Unit tests for the Extended Kalman Filter module.

These tests verify the correctness of the EKF implementation that forms
the basis of the PHANTOM deception strategy. Each test isolates a single
behavior of the filter: initialization, prediction, update gating,
bearing wrapping, statistics tracking, and reset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from core.ekf import ExtendedKalmanFilter, _wrap_angle


# ---------------------------------------------------------------------------
# Fixtures — load config and construct a default EKF instance
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> dict:  # type: ignore[type-arg]
    """Load the PHANTOM configuration from phantom_config.yaml."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "phantom_config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg: dict = yaml.safe_load(fh)  # type: ignore[assignment]
    return cfg


@pytest.fixture()
def default_ekf(config: dict) -> ExtendedKalmanFilter:  # type: ignore[type-arg]
    """Construct an EKF with noise matrices derived from phantom_config.yaml."""
    sim = config["simulation"]
    Q = np.eye(4) * sim["process_noise"]
    R = np.diag([sim["range_noise"] ** 2, sim["bearing_noise"] ** 2])
    return ExtendedKalmanFilter(Q, R, chi2_threshold=sim["chi2_threshold"])


# ---------------------------------------------------------------------------
# TEST 1 — Initialization shapes and threshold
# ---------------------------------------------------------------------------


def test_ekf_initializes_with_correct_shapes(
    default_ekf: ExtendedKalmanFilter,
    config: dict,  # type: ignore[type-arg]
) -> None:
    """Q, R, x_hat, P shapes and chi2_threshold must match the config."""
    ekf = default_ekf
    assert ekf.Q.shape == (4, 4)
    assert ekf.R.shape == (2, 2)
    assert ekf.x_hat.shape == (4,)
    assert ekf.P.shape == (4, 4)
    assert ekf.CHI2_THRESHOLD_99 == config["simulation"]["chi2_threshold"]


# ---------------------------------------------------------------------------
# TEST 2 — Prediction advances position by velocity * dt
# ---------------------------------------------------------------------------


def test_prediction_advances_position_with_velocity(
    default_ekf: ExtendedKalmanFilter,
) -> None:
    """A target at x=1000 moving at 10 m/s should advance 0.2 m in 0.02 s."""
    ekf = default_ekf
    ekf.x_hat = np.array([1000.0, 0.0, 10.0, 0.0])
    ekf.predict(dt=0.02)
    assert abs(ekf.x_hat[0] - 1000.2) < 1e-9


# ---------------------------------------------------------------------------
# TEST 3 — Prediction increases covariance trace
# ---------------------------------------------------------------------------


def test_prediction_increases_covariance(
    default_ekf: ExtendedKalmanFilter,
) -> None:
    """Process noise must inflate the covariance trace during prediction."""
    ekf = default_ekf
    trace_before = float(np.trace(ekf.P))
    ekf.predict(dt=0.02)
    trace_after = float(np.trace(ekf.P))
    assert trace_after > trace_before


# ---------------------------------------------------------------------------
# TEST 4 — Update accepts a valid (low-noise) measurement
# ---------------------------------------------------------------------------


def test_update_accepts_valid_measurement(
    default_ekf: ExtendedKalmanFilter,
) -> None:
    """A measurement very close to the predicted value should be accepted."""
    ekf = default_ekf
    ekf.x_hat = np.array([1000.0, 0.0, 0.0, 0.0])
    ekf.predict(dt=0.02)

    z = np.array([1000.1, 0.0001])
    missile_pos = np.array([0.0, 0.0])
    gamma_k, accepted = ekf.update(z, missile_pos)

    assert accepted is True
    assert gamma_k < 9.21


# ---------------------------------------------------------------------------
# TEST 5 — Chi-squared gate rejects a large innovation
# ---------------------------------------------------------------------------


def test_chi_squared_gate_rejects_large_innovation(
    default_ekf: ExtendedKalmanFilter,
) -> None:
    """A wildly inconsistent measurement should be rejected by the gate."""
    ekf = default_ekf
    ekf.x_hat = np.array([1000.0, 0.0, 0.0, 0.0])
    ekf.predict(dt=0.02)

    z = np.array([1500.0, 0.5])
    missile_pos = np.array([0.0, 0.0])
    gamma_k, accepted = ekf.update(z, missile_pos)

    assert accepted is False
    assert gamma_k > 9.21


# ---------------------------------------------------------------------------
# TEST 6 — Bearing wrapping handles angle discontinuity
# ---------------------------------------------------------------------------


def test_bearing_wrapping_handles_angle_discontinuity() -> None:
    """No wrapped angle should exceed pi in absolute value."""
    test_angles = [
        3.5,
        -3.5,
        np.pi + 0.1,
        -np.pi - 0.1,
        2.0 * np.pi,
        -2.0 * np.pi,
        0.0,
    ]
    for angle in test_angles:
        wrapped = _wrap_angle(float(angle))
        assert abs(wrapped) <= np.pi, f"_wrap_angle({angle}) = {wrapped}, exceeds pi"


# ---------------------------------------------------------------------------
# TEST 7 — Rejection count increments on rejection
# ---------------------------------------------------------------------------


def test_rejection_count_increments_on_rejection(
    default_ekf: ExtendedKalmanFilter,
) -> None:
    """Forcing a large innovation should increment the rejection counter."""
    ekf = default_ekf
    ekf.x_hat = np.array([1000.0, 0.0, 0.0, 0.0])
    ekf.predict(dt=0.02)

    z = np.array([1500.0, 0.5])
    missile_pos = np.array([0.0, 0.0])
    ekf.update(z, missile_pos)

    assert ekf.get_statistics()["rejection_count"] == 1


# ---------------------------------------------------------------------------
# TEST 8 — Reset clears all state
# ---------------------------------------------------------------------------


def test_reset_clears_all_state(
    default_ekf: ExtendedKalmanFilter,
) -> None:
    """After reset, the filter must be indistinguishable from a fresh instance."""
    ekf = default_ekf

    ekf.x_hat = np.array([1000.0, 0.0, 10.0, 0.0])
    ekf.predict(dt=0.02)
    z = np.array([1000.1, 0.0001])
    missile_pos = np.array([0.0, 0.0])
    ekf.update(z, missile_pos)

    ekf.reset()

    assert np.allclose(ekf.x_hat, np.zeros(4))
    assert np.allclose(ekf.P, np.eye(4) * 1000.0)
    assert ekf.get_statistics()["rejection_count"] == 0
    assert ekf.get_statistics()["update_count"] == 0
    assert len(ekf.innovation_history) == 0


# ---------------------------------------------------------------------------
# TEST 9 — Statistics returns correct keys
# ---------------------------------------------------------------------------


def test_statistics_returns_correct_keys(
    default_ekf: ExtendedKalmanFilter,
) -> None:
    """get_statistics() must return all required diagnostic keys."""
    stats = default_ekf.get_statistics()
    expected_keys = {
        "rejection_rate",
        "mean_innovation",
        "max_innovation",
        "rejection_count",
        "update_count",
    }
    assert set(stats.keys()) == expected_keys

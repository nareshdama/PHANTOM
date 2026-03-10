"""PHANTOM — Unit tests for the Proportional Navigation guidance module.

These tests verify the PN guidance law that translates EKF state estimates
into missile acceleration commands. Correct behavior of this module is
critical because it is the pathway through which PHANTOM's EKF corruption
becomes a physical trajectory deviation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from core.guidance import ProportionalNavigation

# ---------------------------------------------------------------------------
# Fixtures — load config and construct a default PN instance
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> dict:  # type: ignore[type-arg]
    """Load the PHANTOM configuration from phantom_config.yaml."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "phantom_config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg: dict = yaml.safe_load(fh)  # type: ignore[assignment]
    return cfg


@pytest.fixture()
def default_pn(config: dict) -> ProportionalNavigation:  # type: ignore[type-arg]
    """Construct a PN guidance object from phantom_config.yaml parameters."""
    sim = config["simulation"]
    return ProportionalNavigation(
        nav_ratio=sim["nav_ratio"],
        max_accel_g=sim["max_accel_g"],
    )


# ---------------------------------------------------------------------------
# TEST 1 — Zero LOS rate on collision course produces zero acceleration
# ---------------------------------------------------------------------------


def test_zero_los_rate_produces_zero_acceleration(
    default_pn: ProportionalNavigation,
) -> None:
    """Head-on collision course has zero LOS rate, so PN produces zero accel."""
    accel = default_pn.compute_acceleration(
        missile_pos=np.array([0.0, 0.0]),
        missile_vel=np.array([600.0, 0.0]),
        target_estimate=np.array([5000.0, 0.0]),
        target_vel_estimate=np.array([-250.0, 0.0]),
    )
    assert np.linalg.norm(accel) < 1e-6


# ---------------------------------------------------------------------------
# TEST 2 — Positive LOS rate produces upward correction
# ---------------------------------------------------------------------------


def test_positive_los_rate_produces_upward_correction(
    default_pn: ProportionalNavigation,
) -> None:
    """Target above the missile velocity axis requires upward correction."""
    accel = default_pn.compute_acceleration(
        missile_pos=np.array([0.0, 0.0]),
        missile_vel=np.array([600.0, 0.0]),
        target_estimate=np.array([5000.0, 100.0]),
        target_vel_estimate=np.array([-250.0, 0.0]),
    )
    assert accel[1] > 0


# ---------------------------------------------------------------------------
# TEST 3 — Acceleration clipped to max_accel
# ---------------------------------------------------------------------------


def test_acceleration_clipped_to_max_accel() -> None:
    """Extreme LOS rate scenario must be clipped to the structural limit."""
    pn = ProportionalNavigation(nav_ratio=5.0, max_accel_g=1.0)
    accel = pn.compute_acceleration(
        missile_pos=np.array([0.0, 0.0]),
        missile_vel=np.array([600.0, 0.0]),
        target_estimate=np.array([100.0, 500.0]),
        target_vel_estimate=np.array([-250.0, -200.0]),
    )
    assert np.linalg.norm(accel) <= 1.0 * 9.81 + 1e-6


# ---------------------------------------------------------------------------
# TEST 4 — Singularity handled at zero range
# ---------------------------------------------------------------------------


def test_singularity_handled_at_zero_range(
    default_pn: ProportionalNavigation,
) -> None:
    """When missile and target coincide, return zero accel without error."""
    accel = default_pn.compute_acceleration(
        missile_pos=np.array([100.0, 200.0]),
        missile_vel=np.array([600.0, 0.0]),
        target_estimate=np.array([100.0, 200.0]),
        target_vel_estimate=np.array([-250.0, 0.0]),
    )
    assert np.array_equal(accel, np.array([0.0, 0.0]))


# ---------------------------------------------------------------------------
# TEST 5 — Zero missile velocity handled safely
# ---------------------------------------------------------------------------


def test_zero_missile_velocity_handled_safely(
    default_pn: ProportionalNavigation,
) -> None:
    """Zero missile velocity must return zero accel — no division by zero."""
    accel = default_pn.compute_acceleration(
        missile_pos=np.array([0.0, 0.0]),
        missile_vel=np.array([0.0, 0.0]),
        target_estimate=np.array([5000.0, 0.0]),
        target_vel_estimate=np.array([-250.0, 0.0]),
    )
    assert np.array_equal(accel, np.array([0.0, 0.0]))


# ---------------------------------------------------------------------------
# TEST 6 — Closing velocity positive on approach
# ---------------------------------------------------------------------------


def test_closing_velocity_positive_on_approach(
    default_pn: ProportionalNavigation,
) -> None:
    """Head-on approach yields positive closing velocity."""
    v_c = default_pn.get_closing_velocity(
        missile_pos=np.array([0.0, 0.0]),
        missile_vel=np.array([600.0, 0.0]),
        target_pos=np.array([5000.0, 0.0]),
        target_vel=np.array([-250.0, 0.0]),
    )
    assert v_c > 0


# ---------------------------------------------------------------------------
# TEST 7 — LOS rate sign convention
# ---------------------------------------------------------------------------


def test_los_rate_sign_convention(
    default_pn: ProportionalNavigation,
) -> None:
    """Target above axis -> positive LOS rate; below -> negative."""
    sigma_above = default_pn.get_los_rate(
        missile_pos=np.array([0.0, 0.0]),
        missile_vel=np.array([600.0, 0.0]),
        target_pos=np.array([5000.0, 100.0]),
        target_vel=np.array([-250.0, 0.0]),
    )
    sigma_below = default_pn.get_los_rate(
        missile_pos=np.array([0.0, 0.0]),
        missile_vel=np.array([600.0, 0.0]),
        target_pos=np.array([5000.0, -100.0]),
        target_vel=np.array([-250.0, 0.0]),
    )
    assert sigma_above > 0, "Target above axis should give positive LOS rate"
    assert sigma_below < 0, "Target below axis should give negative LOS rate"


# ---------------------------------------------------------------------------
# TEST 8 — Nav ratio scales acceleration linearly
# ---------------------------------------------------------------------------


def test_nav_ratio_scales_acceleration_linearly() -> None:
    """Doubling the nav ratio must exactly double the acceleration (no clipping)."""
    pn_n3 = ProportionalNavigation(nav_ratio=3.0, max_accel_g=100.0)
    pn_n6 = ProportionalNavigation(nav_ratio=6.0, max_accel_g=100.0)

    missile_pos = np.array([0.0, 0.0])
    missile_vel = np.array([600.0, 0.0])
    target_pos = np.array([5000.0, 100.0])
    target_vel = np.array([-250.0, 0.0])

    accel_n3 = pn_n3.compute_acceleration(missile_pos, missile_vel, target_pos, target_vel)
    accel_n6 = pn_n6.compute_acceleration(missile_pos, missile_vel, target_pos, target_vel)

    np.testing.assert_allclose(accel_n6, 2.0 * accel_n3, atol=1e-9)

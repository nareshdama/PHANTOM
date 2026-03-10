"""PHANTOM — Unit tests for the False Signal Injection Profile Library.

These tests verify that each injection profile produces the correct false
LOS angular rate as a function of time. Correct behavior here is essential
because the injection profile is the mechanism through which PHANTOM steers
the missile's EKF track away from the true target.
"""

from __future__ import annotations

import pytest

from core.injection import (
    AdaptiveInjection,
    InjectionProfile,
    LinearRampInjection,
    NoInjection,
    PiecewiseInjection,
    StepInjection,
)

# ---------------------------------------------------------------------------
# TEST 1 — NoInjection always returns zero
# ---------------------------------------------------------------------------


def test_no_injection_always_returns_zero() -> None:
    """NoInjection must return exactly 0.0 at every time point."""
    profile = NoInjection()
    for t in [0.0, 1.0, 5.0, 10.0, 30.0]:
        assert profile.get_rate(t) == 0.0


# ---------------------------------------------------------------------------
# TEST 2 — StepInjection returns zero before t_start
# ---------------------------------------------------------------------------


def test_step_injection_zero_before_start() -> None:
    """Step profile must return zero before the activation time."""
    profile = StepInjection(t_start=2.0, amplitude=0.05)
    assert profile.get_rate(0.0) == 0.0
    assert profile.get_rate(1.99) == 0.0


# ---------------------------------------------------------------------------
# TEST 3 — StepInjection returns amplitude at and after t_start
# ---------------------------------------------------------------------------


def test_step_injection_amplitude_after_start() -> None:
    """Step profile must return the full amplitude once t >= t_start."""
    profile = StepInjection(t_start=2.0, amplitude=0.05)
    assert profile.get_rate(2.0) == 0.05
    assert profile.get_rate(5.0) == 0.05
    assert profile.get_rate(10.0) == 0.05


# ---------------------------------------------------------------------------
# TEST 4 — LinearRampInjection returns zero before t_start
# ---------------------------------------------------------------------------


def test_linear_ramp_zero_before_start() -> None:
    """Ramp profile must return zero before the ramp onset time."""
    profile = LinearRampInjection(t_start=1.0, t_end=4.0, ramp_rate=0.032)
    assert profile.get_rate(0.0) == 0.0
    assert profile.get_rate(0.99) == 0.0


# ---------------------------------------------------------------------------
# TEST 5 — LinearRampInjection grows linearly between t_start and t_end
# ---------------------------------------------------------------------------


def test_linear_ramp_grows_linearly() -> None:
    """Ramp rate at time t must equal ramp_rate * (t - t_start)."""
    profile = LinearRampInjection(t_start=1.0, t_end=4.0, ramp_rate=0.032)
    assert abs(profile.get_rate(2.0) - 0.032 * (2.0 - 1.0)) < 1e-9
    assert abs(profile.get_rate(3.0) - 0.032 * (3.0 - 1.0)) < 1e-9


# ---------------------------------------------------------------------------
# TEST 6 — LinearRampInjection holds at max_rate after t_end
# ---------------------------------------------------------------------------


def test_linear_ramp_holds_at_max_after_end() -> None:
    """After t_end the ramp must hold at max_rate = ramp_rate * duration."""
    profile = LinearRampInjection(t_start=1.0, t_end=4.0, ramp_rate=0.032)
    expected_max = 0.032 * (4.0 - 1.0)
    assert abs(profile.get_rate(5.0) - expected_max) < 1e-9
    assert abs(profile.get_rate(10.0) - expected_max) < 1e-9
    assert abs(profile.max_rate - expected_max) < 1e-9


# ---------------------------------------------------------------------------
# TEST 7 — LinearRampInjection accumulated_angle_at_end matches theory
# ---------------------------------------------------------------------------


def test_linear_ramp_accumulated_angle_at_end() -> None:
    """Analytical accumulated angle must equal 0.5 * ramp_rate * duration^2."""
    profile = LinearRampInjection(t_start=1.0, t_end=4.0, ramp_rate=0.032)
    expected = 0.5 * 0.032 * (4.0 - 1.0) ** 2
    assert abs(profile.accumulated_angle_at_end - expected) < 1e-9

    # Numerical integration via get_accumulated_angle should agree closely
    # with the analytical result (trapezoidal rule with fine dt).
    numerical = profile.get_accumulated_angle(t=4.0, dt=0.001)
    assert abs(numerical - expected) < 1e-4

    # Edge case: t=0 should return zero accumulated angle.
    assert profile.get_accumulated_angle(t=0.0, dt=0.02) == 0.0


# ---------------------------------------------------------------------------
# TEST 8 — PiecewiseInjection interpolates between breakpoints
# ---------------------------------------------------------------------------


def test_piecewise_interpolates_between_breakpoints() -> None:
    """Piecewise profile must linearly interpolate between control points."""
    profile = PiecewiseInjection([(1.0, 0.0), (2.0, 0.05), (4.0, 0.05)])
    assert abs(profile.get_rate(1.5) - 0.025) < 1e-9
    assert abs(profile.get_rate(3.0) - 0.05) < 1e-9

    # After the last breakpoint, the rate holds at the final value.
    assert abs(profile.get_rate(5.0) - 0.05) < 1e-9


# ---------------------------------------------------------------------------
# TEST 9 — PiecewiseInjection returns zero before first breakpoint
# ---------------------------------------------------------------------------


def test_piecewise_returns_zero_before_first_breakpoint() -> None:
    """Before the first breakpoint, no injection should be active."""
    profile = PiecewiseInjection([(1.0, 0.0), (2.0, 0.05), (4.0, 0.05)])
    assert profile.get_rate(0.0) == 0.0
    assert profile.get_rate(0.99) == 0.0


# ---------------------------------------------------------------------------
# TEST 10 — All profiles report their class name via profile_type
# ---------------------------------------------------------------------------


def test_all_profiles_have_profile_type_string() -> None:
    """Each profile's .profile_type must return its class name as a string."""
    assert NoInjection().profile_type == "NoInjection"
    assert StepInjection(1.0, 0.05).profile_type == "StepInjection"
    assert LinearRampInjection(1.0, 4.0, 0.032).profile_type == "LinearRampInjection"
    assert PiecewiseInjection([(0.0, 0.0), (1.0, 0.05)]).profile_type == "PiecewiseInjection"
    assert AdaptiveInjection(target_gamma=8.0, kp=0.005).profile_type == "AdaptiveInjection"


# ---------------------------------------------------------------------------
# TEST 11 — AdaptiveInjection stub returns a float without error
# ---------------------------------------------------------------------------


def test_adaptive_injection_returns_float() -> None:
    """The Phase 3 stub must return a float (0.0) without raising."""
    profile = AdaptiveInjection(target_gamma=8.0, kp=0.005)
    result = profile.get_rate(1.0)
    assert isinstance(result, float)
    assert result == 0.0

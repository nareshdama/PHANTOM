"""PHANTOM — Unit tests for the PhantomSimulator engagement engine.

These tests verify that the simulator correctly integrates the EKF, PN
guidance, and injection profiles into a functioning engagement loop, and
that the telemetry and result fields are complete and correct.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from core.injection import (
    LinearRampInjection,
    NoInjection,
    StepInjection,
)
from core.simulator import PhantomSimulator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> dict:  # type: ignore[type-arg]
    """Load the PHANTOM configuration from phantom_config.yaml."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "phantom_config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg: dict = yaml.safe_load(fh)  # type: ignore[assignment]
    return cfg


@pytest.fixture()
def initial_conditions(config: dict) -> dict:  # type: ignore[type-arg]
    """Extract initial conditions from the loaded config."""
    return config["initial_conditions"]


@pytest.fixture()
def simulator(config: dict) -> PhantomSimulator:  # type: ignore[type-arg]
    """Construct a PhantomSimulator with seed=42."""
    return PhantomSimulator(config, seed=42)


# ---------------------------------------------------------------------------
# TEST 1 — Baseline engagement achieves direct hit
# ---------------------------------------------------------------------------


def test_baseline_engagement_achieves_direct_hit(
    simulator: PhantomSimulator,
    initial_conditions: dict,  # type: ignore[type-arg]
) -> None:
    """With no injection, the missile must achieve a direct hit (miss < 5m)."""
    result = simulator.run_engagement(NoInjection(), initial_conditions)
    assert result["miss_distance"] < 5.0
    assert result["detection_rate"] == 0.0


# ---------------------------------------------------------------------------
# TEST 2 — Ramp injection increases miss distance
# ---------------------------------------------------------------------------


def test_ramp_injection_increases_miss_distance(
    config: dict,  # type: ignore[type-arg]
    initial_conditions: dict,  # type: ignore[type-arg]
) -> None:
    """LinearRampInjection must produce a larger miss than the unperturbed baseline."""
    sim_baseline = PhantomSimulator(config, seed=42)
    baseline = sim_baseline.run_engagement(NoInjection(), initial_conditions)

    sim_ramp = PhantomSimulator(config, seed=42)
    ramp = sim_ramp.run_engagement(
        LinearRampInjection(t_start=1.0, t_end=4.0, ramp_rate=0.032),
        initial_conditions,
    )

    assert ramp["miss_distance"] > baseline["miss_distance"]


# ---------------------------------------------------------------------------
# TEST 3 — Step injection triggers EKF detection
# ---------------------------------------------------------------------------


def test_step_injection_triggers_detection(
    simulator: PhantomSimulator,
    initial_conditions: dict,  # type: ignore[type-arg]
) -> None:
    """StepInjection with amplitude 0.1 must trigger some EKF gate rejections."""
    result = simulator.run_engagement(
        StepInjection(t_start=2.0, amplitude=0.1),
        initial_conditions,
    )
    assert result["detection_rate"] > 0.05


# ---------------------------------------------------------------------------
# TEST 4 — Telemetry has correct keys
# ---------------------------------------------------------------------------


def test_telemetry_has_correct_keys(
    simulator: PhantomSimulator,
    initial_conditions: dict,  # type: ignore[type-arg]
) -> None:
    """Every telemetry entry must contain all required fields."""
    result = simulator.run_engagement(NoInjection(), initial_conditions)
    expected_keys = {
        "t",
        "missile_pos",
        "missile_vel",
        "target_pos",
        "ekf_estimate",
        "ekf_vel_estimate",
        "gamma_k",
        "measurement_accepted",
        "injection_rate",
        "accumulated_angle",
        "true_range",
        "estimated_range",
        "estimation_error",
    }
    entry = result["trajectory"][0]
    assert set(entry.keys()) == expected_keys


# ---------------------------------------------------------------------------
# TEST 5 — Result has correct keys
# ---------------------------------------------------------------------------


def test_result_has_correct_keys(
    simulator: PhantomSimulator,
    initial_conditions: dict,  # type: ignore[type-arg]
) -> None:
    """The returned result dict must contain all required fields."""
    result = simulator.run_engagement(NoInjection(), initial_conditions)
    expected_keys = {
        "trajectory",
        "miss_distance",
        "miss_angle",
        "intercept_time",
        "detection_rate",
        "max_innovation",
        "mean_innovation",
        "max_estimation_error",
        "success",
        "profile_type",
    }
    assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# TEST 6 — Simulation time advances correctly
# ---------------------------------------------------------------------------


def test_simulation_time_advances_correctly(
    simulator: PhantomSimulator,
    initial_conditions: dict,  # type: ignore[type-arg]
) -> None:
    """Consecutive telemetry entries must differ by exactly dt = 0.02s."""
    result = simulator.run_engagement(NoInjection(), initial_conditions)
    traj = result["trajectory"]
    for i in range(1, min(len(traj), 100)):
        dt = traj[i]["t"] - traj[i - 1]["t"]
        assert abs(dt - 0.02) < 1e-9


# ---------------------------------------------------------------------------
# TEST 7 — Missile moves in direction of velocity
# ---------------------------------------------------------------------------


def test_missile_moves_in_direction_of_velocity(
    simulator: PhantomSimulator,
    initial_conditions: dict,  # type: ignore[type-arg]
) -> None:
    """With initial vel [600, 0], the missile must advance in +x after one step."""
    result = simulator.run_engagement(NoInjection(), initial_conditions)
    first_pos = result["trajectory"][0]["missile_pos"]
    assert first_pos[0] > initial_conditions["missile_pos"][0]


# ---------------------------------------------------------------------------
# TEST 8 — Seed produces reproducible results
# ---------------------------------------------------------------------------


def test_seed_produces_reproducible_results(
    config: dict,  # type: ignore[type-arg]
    initial_conditions: dict,  # type: ignore[type-arg]
) -> None:
    """Two simulators with the same seed must produce identical miss distances."""
    sim1 = PhantomSimulator(config, seed=42)
    result1 = sim1.run_engagement(NoInjection(), initial_conditions)

    sim2 = PhantomSimulator(config, seed=42)
    result2 = sim2.run_engagement(NoInjection(), initial_conditions)

    assert result1["miss_distance"] == result2["miss_distance"]


# ---------------------------------------------------------------------------
# TEST 9 — Reset allows clean rerun
# ---------------------------------------------------------------------------


def test_reset_allows_clean_rerun(
    config: dict,  # type: ignore[type-arg]
    initial_conditions: dict,  # type: ignore[type-arg]
) -> None:
    """After reset(), the simulator must reproduce its first engagement exactly."""
    sim = PhantomSimulator(config, seed=42)
    result1 = sim.run_engagement(NoInjection(), initial_conditions)

    sim.reset()
    result2 = sim.run_engagement(NoInjection(), initial_conditions)

    assert result1["miss_distance"] == result2["miss_distance"]

    sim_fresh = PhantomSimulator(config, seed=42)
    result3 = sim_fresh.run_engagement(NoInjection(), initial_conditions)
    assert result2["miss_distance"] == result3["miss_distance"]


# ---------------------------------------------------------------------------
# TEST 10 — Max innovation exceeds threshold under step injection
# ---------------------------------------------------------------------------


def test_max_innovation_exceeds_threshold_under_step_injection(
    simulator: PhantomSimulator,
    initial_conditions: dict,  # type: ignore[type-arg]
) -> None:
    """A large-amplitude step injection must produce gamma_k > 9.21."""
    result = simulator.run_engagement(
        StepInjection(t_start=1.0, amplitude=0.5),
        initial_conditions,
    )
    assert result["max_innovation"] > 9.21

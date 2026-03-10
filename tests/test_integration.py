"""PHANTOM — Integration tests for the full engagement pipeline.

These tests verify that the complete PHANTOM pipeline — from config loading
through simulation execution — produces physically correct results for both
baseline (no injection) and PHANTOM-active (ramp injection) scenarios.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.injection import (
    LinearRampInjection,
    NoInjection,
    PiecewiseInjection,
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


# ---------------------------------------------------------------------------
# TEST 1 — Full PHANTOM pipeline: baseline
# ---------------------------------------------------------------------------


def test_full_phantom_pipeline_baseline(
    config: dict,  # type: ignore[type-arg]
    initial_conditions: dict,  # type: ignore[type-arg]
) -> None:
    """Baseline engagement must achieve a direct hit with zero detection."""
    sim = PhantomSimulator(config, seed=42)
    result = sim.run_engagement(NoInjection(), initial_conditions)

    assert result["miss_distance"] < 5.0
    assert result["detection_rate"] == 0.0
    print(f"PHANTOM baseline: {result['miss_distance']:.2f}m miss")


# ---------------------------------------------------------------------------
# TEST 2 — Full PHANTOM pipeline: ramp injection
# ---------------------------------------------------------------------------


def test_full_phantom_pipeline_ramp_injection(
    config: dict,  # type: ignore[type-arg]
    initial_conditions: dict,  # type: ignore[type-arg]
) -> None:
    """Ramp injection must cause >50m miss with <10% detection rate."""
    sim = PhantomSimulator(config, seed=42)
    result = sim.run_engagement(
        LinearRampInjection(t_start=1.0, t_end=4.0, ramp_rate=0.032),
        initial_conditions,
    )

    assert result["miss_distance"] > 50.0
    assert result["detection_rate"] < 0.10
    print(
        f"PHANTOM ramp injection: {result['miss_distance']:.2f}m miss, "
        f"{result['detection_rate'] * 100:.1f}% detected"
    )


# ---------------------------------------------------------------------------
# TEST 3 — All injection profiles run without error
# ---------------------------------------------------------------------------


def test_all_injection_profiles_run_without_error(
    config: dict,  # type: ignore[type-arg]
    initial_conditions: dict,  # type: ignore[type-arg]
) -> None:
    """Every injection profile must complete an engagement without exception."""
    profiles = [
        NoInjection(),
        StepInjection(t_start=2.0, amplitude=0.05),
        LinearRampInjection(t_start=1.0, t_end=4.0, ramp_rate=0.032),
        PiecewiseInjection([(1.0, 0.0), (2.0, 0.03), (4.0, 0.03)]),
    ]

    for profile in profiles:
        sim = PhantomSimulator(config, seed=42)
        result = sim.run_engagement(profile, initial_conditions)
        assert "miss_distance" in result, f"{profile.profile_type} did not return 'miss_distance'"

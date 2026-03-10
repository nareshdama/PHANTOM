"""PHANTOM — Tests for experiment scripts."""

from __future__ import annotations

from pathlib import Path

from experiments.baseline_validation import run_baseline_validation
from experiments.ramp_rate_sweep import evaluate_single_run, run_ramp_rate_sweep
from core.utils import load_phantom_config


def test_baseline_experiment_passes_gate(tmp_path: Path) -> None:
    """Baseline validation should pass its physical-correctness gate quickly."""
    summary = run_baseline_validation(
        seed=42,
        runs=20,
        output="baseline_test",
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
        show_progress=False,
    )

    assert summary["gate_passed"] is True
    assert float(summary["stats"]["mean_miss"]) < 5.0
    assert float(summary["stats"]["mean_detection"]) == 0.0


def test_baseline_experiment_produces_output_files(tmp_path: Path) -> None:
    """Experiment 1 should save both data artifacts and both figures."""
    summary = run_baseline_validation(
        seed=42,
        runs=20,
        output="baseline_outputs",
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
        show_progress=False,
    )

    assert Path(summary["csv_path"]).exists()
    assert Path(summary["hdf5_path"]).exists()
    assert len(summary["figure_paths"]) == 2
    assert all(Path(path_str).exists() for path_str in summary["figure_paths"])


def test_baseline_experiment_reproducible(tmp_path: Path) -> None:
    """Running Experiment 1 twice with the same seed should reproduce the stats."""
    first = run_baseline_validation(
        seed=42,
        runs=20,
        output="baseline_repro_a",
        results_dir=str(tmp_path / "results_a"),
        figures_dir=str(tmp_path / "figures_a"),
        show_progress=False,
    )
    second = run_baseline_validation(
        seed=42,
        runs=20,
        output="baseline_repro_b",
        results_dir=str(tmp_path / "results_b"),
        figures_dir=str(tmp_path / "figures_b"),
        show_progress=False,
    )

    assert float(first["stats"]["mean_miss"]) == float(second["stats"]["mean_miss"])
    assert float(first["stats"]["std_miss"]) == float(second["stats"]["std_miss"])


def test_ramp_sweep_identifies_critical_rate(tmp_path: Path) -> None:
    """The fast sweep should identify a plausible critical operating point."""
    summary = run_ramp_rate_sweep(
        seed=42,
        runs=20,
        workers=2,
        output="ramp_sweep_test",
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
        show_progress=False,
    )

    assert summary["critical_rate"] is not None
    assert 0.010 < float(summary["critical_rate"]) < 0.060


def test_ramp_sweep_monotonic_detection_trend(tmp_path: Path) -> None:
    """Higher ramp rates should be more detectable than lower ramp rates."""
    summary = run_ramp_rate_sweep(
        seed=42,
        runs=10,
        workers=2,
        output="ramp_sweep_detection",
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
        show_progress=False,
    )

    detection_low = float(summary["rate_summaries"][0]["mean_detection"])
    detection_high = float(summary["rate_summaries"][-1]["mean_detection"])
    assert detection_high > detection_low


def test_ramp_sweep_produces_output_files(tmp_path: Path) -> None:
    """The sweep should save its CSV and all three publication figures."""
    summary = run_ramp_rate_sweep(
        seed=42,
        runs=10,
        workers=2,
        output="ramp_sweep_outputs",
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
        show_progress=False,
    )

    assert Path(summary["csv_path"]).exists()
    assert len(summary["figure_paths"]) == 3
    assert all(Path(path_str).exists() for path_str in summary["figure_paths"])
    assert (tmp_path / "figures" / "phantom_fig05_critical_rate_combined.png").exists()


def test_evaluate_single_run_returns_correct_keys() -> None:
    """The multiprocessing worker should return the PHANTOM result fields we need."""
    config = load_phantom_config()
    result = evaluate_single_run((0.032, 42, config))

    assert "miss_distance" in result
    assert "detection_rate" in result
    assert "success" in result
    assert "ramp_rate" in result


def test_ramp_sweep_reproducible_with_same_seed(tmp_path: Path) -> None:
    """The sweep should be exactly reproducible for a fixed seed and run count."""
    first = run_ramp_rate_sweep(
        seed=42,
        runs=10,
        workers=2,
        output="ramp_sweep_repro_a",
        results_dir=str(tmp_path / "results_a"),
        figures_dir=str(tmp_path / "figures_a"),
        show_progress=False,
    )
    second = run_ramp_rate_sweep(
        seed=42,
        runs=10,
        workers=2,
        output="ramp_sweep_repro_b",
        results_dir=str(tmp_path / "results_b"),
        figures_dir=str(tmp_path / "figures_b"),
        show_progress=False,
    )

    assert float(first["critical_rate"]) == float(second["critical_rate"])
    first_miss = [float(summary["mean_miss"]) for summary in first["rate_summaries"]]
    second_miss = [float(summary["mean_miss"]) for summary in second["rate_summaries"]]
    assert first_miss == second_miss

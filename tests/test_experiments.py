"""PHANTOM — Tests for experiment scripts."""

from __future__ import annotations

from pathlib import Path

from experiments.baseline_validation import run_baseline_validation


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

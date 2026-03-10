# PHANTOM API Reference

## Core Modules

### `core.ekf` — Extended Kalman Filter

**`class ExtendedKalmanFilter`**

- `__init__(dt, sigma_range, sigma_bearing, process_noise_std, chi2_threshold)` — Initialize EKF with measurement noise and gating parameters.
- `predict()` — Propagate state estimate forward by `dt` seconds using constant-velocity model.
- `update(z_range, z_bearing, missile_pos)` — Fuse range-bearing measurement; returns `(accepted, gamma_k)` where `accepted` indicates whether the measurement passed the chi-squared gate.
- `get_estimated_target_state() -> np.ndarray` — Return current `[x, y, vx, vy]` estimate.
- `get_innovation_statistic() -> float` — Return most recent normalized innovation `gamma_k`.
- `get_covariance_trace() -> float` — Return trace of the state covariance matrix.
- `reset(target_pos, target_vel)` — Re-initialize filter state from known position and velocity.

### `core.guidance` — Proportional Navigation

**`class ProportionalNavigation`**

- `__init__(nav_ratio, missile_speed, dt)` — Initialize PN law with navigation ratio N, missile speed, and timestep.
- `compute_acceleration(missile_pos, missile_vel, estimated_target_pos, estimated_target_vel) -> np.ndarray` — Compute lateral acceleration command using the PN law: `a_cmd = N * Vc * sigma_dot`.
- `get_closing_velocity() -> float` — Return most recent closing velocity estimate.
- `get_los_rate() -> float` — Return most recent LOS angular rate.

### `core.injection` — False Signal Injection Profiles

**`class NoInjection`** — Baseline (zero injection).

**`class LinearRampInjection`**

- `__init__(ramp_rate, start_time)` — Injection begins at `start_time` with constant LOS rate bias `ramp_rate` rad/s.
- `get_delta_sigma(t) -> float` — Return accumulated false bearing angle at time `t`.
- `get_current_rate(t) -> float` — Return instantaneous injection rate at time `t`.

**`class StepInjection`** — Instantaneous step in bearing angle.

**`class SinusoidalInjection`** — Oscillating injection for testing detection sensitivity.

**`class PiecewiseInjection`** — Arbitrary piecewise-linear injection profile.

**`class AdaptiveInjection`** — Placeholder for Phase 2 adaptive controller.

### `core.simulator` — Engagement Simulator

**`class PhantomSimulator`**

- `__init__(config, seed)` — Initialize simulator with YAML config and RNG seed.
- `run_engagement(injection_profile) -> dict` — Execute one full engagement. Returns dict with keys: `miss_distance`, `detected`, `trajectory`, `final_time`, `profile_name`, `seed`.
- `reset()` — Reset simulator state for a new engagement with the same seed.

### `core.utils` — Utility Functions

- `load_phantom_config(path) -> dict` — Load and validate `phantom_config.yaml`.
- `compute_engagement_statistics(results) -> dict` — Compute aggregate statistics (mean, std, p95, CI) from a list of engagement results.
- `setup_ieee_figure(width, height) -> (fig, ax)` — Create a matplotlib figure with IEEE publication defaults.
- `save_ieee_figure(fig, filename) -> str` — Save figure at 300 DPI with timestamp.
- `save_results_csv(results, filename) -> str` — Save engagement results to CSV.
- `save_results_hdf5(results, filename) -> str` — Save full trajectory data to HDF5.
- `wrap_angle(angle) -> float` — Wrap angle to `[-pi, pi]`.
- `xy_to_range_bearing(dx, dy) -> (range, bearing)` — Cartesian to polar conversion.
- `compute_miss_vector(missile_pos, target_pos) -> (distance, angle)` — Compute miss distance and angle.
- `monte_carlo_convergence(values, threshold) -> bool` — Check if running mean has stabilized.
- `randomize_initial_conditions(config, rng) -> dict` — Sample randomized engagement geometry.

## Experiment Scripts

### `experiments.baseline_validation`

- `main(seed, runs)` — Run N baseline engagements and validate against gate criteria (mean miss < 5 m, p95 < 10 m, detection = 0%).

### `experiments.ramp_rate_sweep`

- `main(seed, runs, fast)` — Sweep ramp rates from 0.005 to 0.100 rad/s, identify the critical rate maximizing miss distance below 5% detection.
- `evaluate_single_run(args) -> dict` — Run one engagement at a given ramp rate and seed.

## Analysis Modules

### `analysis.statistics`

- `compare_baseline_vs_injection(baseline, injection) -> dict` — Welch's t-test, Cohen's d, p-value.
- `fit_miss_distance_scaling(rates, misses) -> dict` — Power-law fit to miss distance vs. ramp rate.
- `compute_critical_rate_confidence_interval(sweep, n_bootstrap) -> dict` — Bootstrap 95% CI on critical rate.
- `compute_ekf_innovation_statistics(trajectory) -> dict` — Innovation gamma_k statistics.
- `compute_scaling_laws(sweep_frame) -> dict` — Empirical vs. theoretical scaling exponents.

### `analysis.visualization`

- `plot_baseline_miss_histogram(results, save) -> fig` — Figure 1.
- `plot_ramp_rate_optimization(sweep_stats, critical_rate, save) -> fig` — Figure 5.
- `plot_innovation_timeline(trajectory, chi2_threshold, save) -> fig` — Figure 6.
- `plot_trajectory_comparison(baseline_traj, injection_traj, save) -> fig` — Figure 7.
- `plot_estimation_error_timeline(enriched_traj, save) -> fig` — Figure 8.
- `plot_miss_distance_scaling(sweep_stats, fit_result, save) -> fig` — Figure 9.

### `analysis.generate_report`

- `generate_phase1_report(baseline_stats, sweep_stats, ...) -> str` — Generate full Markdown validation report.
- `main()` — Entry point: load data, compute stats, generate all figures and report.

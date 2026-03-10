"""PHANTOM — PhantomSimulator Engagement Engine.

This module assembles the three PHANTOM core modules — Extended Kalman Filter,
Proportional Navigation guidance, and false signal injection profiles — into a
complete 2D missile-target engagement simulation. The PhantomSimulator is the
primary research instrument for all PHANTOM experiments, from baseline
validation to Monte Carlo sweeps of injection parameters.

References
----------
Zarchan, P. (2012).
    Tactical and Strategic Missile Guidance (6th ed.). AIAA.
Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001).
    Estimation with Applications to Tracking and Navigation. Wiley.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from core.ekf import ExtendedKalmanFilter
from core.guidance import ProportionalNavigation
from core.injection import InjectionProfile

logger = logging.getLogger(__name__)

TelemetryEntry = dict[str, Any]


class PhantomSimulator:
    """The PhantomSimulator integrates the three PHANTOM core modules into a
    complete 2D engagement loop. It is the primary research instrument —
    every hypothesis in the PHANTOM project is tested by running this
    simulator with different injection profiles and initial conditions,
    then analyzing the resulting miss distances and EKF statistics.

    Parameters
    ----------
    config : dict
        Full PHANTOM configuration dictionary loaded from phantom_config.yaml.
        Must contain 'simulation' and 'validation' sections.
    seed : int
        Random seed for reproducible noise generation. Each simulator instance
        maintains its own numpy RandomState, never touching the global state.

    Example
    -------
    >>> import yaml
    >>> from pathlib import Path
    >>> cfg = yaml.safe_load(Path("configs/phantom_config.yaml").read_text())
    >>> sim = PhantomSimulator(cfg, seed=42)
    >>> from core.injection import NoInjection
    >>> result = sim.run_engagement(NoInjection(), cfg["initial_conditions"])
    >>> result["miss_distance"] < 5.0
    True
    """

    def __init__(self, config: dict, seed: int = 42) -> None:  # type: ignore[type-arg]
        self._config = config
        self._seed = seed
        self._rng = np.random.RandomState(seed)

        sim = config["simulation"]
        validation = config["validation"]
        self._dt: float = float(sim["timestep"])
        self._max_time: float = float(sim["max_time"])
        self._sigma_r: float = float(sim["range_noise"])
        self._sigma_bearing: float = float(sim["bearing_noise"])
        self._intercept_range: float = float(validation["baseline_miss_threshold"])
        self._success_miss_distance: float = float(validation["target_miss_distance"])
        self._success_detection_rate: float = float(validation["max_detection_rate"])

        Q = np.eye(4, dtype=np.float64) * sim["process_noise"]
        R = np.diag([self._sigma_r**2, self._sigma_bearing**2]).astype(np.float64)
        self._ekf = ExtendedKalmanFilter(Q, R, chi2_threshold=sim["chi2_threshold"])
        self._chi2_threshold: float = float(sim["chi2_threshold"])

        self._pn = ProportionalNavigation(
            nav_ratio=sim["nav_ratio"],
            max_accel_g=sim["max_accel_g"],
        )

        logger.info(
            "PhantomSimulator initialized: dt=%.3fs, max_time=%.1fs, seed=%d",
            self._dt,
            self._max_time,
            seed,
        )

    # ------------------------------------------------------------------
    # Primary engagement loop
    # ------------------------------------------------------------------

    def run_engagement(
        self,
        injection_profile: InjectionProfile,
        initial_conditions: dict,  # type: ignore[type-arg]
    ) -> dict:  # type: ignore[type-arg]
        """Run a single missile-target engagement with the given injection profile.

        A single engagement begins with the missile on a near-collision course
        and runs until intercept (range < 5m) or timeout. At each 20ms timestep,
        the PHANTOM injection corrupts the bearing measurement before the EKF
        sees it. The EKF's chi-squared gate either accepts or rejects the
        corrupted measurement — if accepted, the guidance error accumulates;
        if rejected, the missile briefly self-corrects. The balance between
        these two effects determines whether PHANTOM succeeds.

        Parameters
        ----------
        injection_profile : InjectionProfile
            The false signal injection profile to apply during the engagement.
        initial_conditions : dict
            Dictionary with keys 'missile_pos', 'missile_vel', 'target_pos',
            'target_vel', each a list or array of two floats.

        Returns
        -------
        dict
            Engagement result containing trajectory, miss_distance,
            detection_rate, and all other PHANTOM-specified metrics.
        """
        self._ekf.reset()

        missile_pos = np.array(initial_conditions["missile_pos"], dtype=np.float64)
        missile_vel = np.array(initial_conditions["missile_vel"], dtype=np.float64)
        target_pos = np.array(initial_conditions["target_pos"], dtype=np.float64)
        target_vel = np.array(initial_conditions["target_vel"], dtype=np.float64)

        # The EKF begins with a perfect initial track — this isolates the
        # effect of PHANTOM injection from initial acquisition errors.
        self._ekf.x_hat = np.array(
            [target_pos[0], target_pos[1], target_vel[0], target_vel[1]],
            dtype=np.float64,
        )

        accumulated_angle: float = 0.0
        trajectory: list[TelemetryEntry] = []
        active_updates = 0
        active_rejections = 0
        max_steps = int(round(self._max_time / self._dt))

        for step in range(max_steps + 1):
            t = step * self._dt

            # --- True Measurement Generation ---
            # Compute the geometrically true range and bearing from missile to target.
            # This is what a perfect sensor would report. Gaussian noise is added
            # next to model realistic radar measurement uncertainty.
            rel = target_pos - missile_pos
            true_range = float(np.linalg.norm(rel))
            true_bearing = float(np.arctan2(rel[1], rel[0]))

            range_meas = true_range + self._rng.normal(0.0, self._sigma_r)
            bearing_meas = true_bearing + self._rng.normal(0.0, self._sigma_bearing)

            # --- PHANTOM False Signal Injection ---
            # This is where deception occurs. The false LOS rate delta_sigma_dot(t)
            # is integrated into an accumulated false bearing angle delta_sigma(t).
            # The EKF receives bearing_true + delta_sigma + noise — it has no way
            # to distinguish the injected component from legitimate target motion.
            injection_rate = injection_profile.get_rate(t)
            accumulated_angle += injection_rate * self._dt
            bearing_meas += accumulated_angle

            # --- EKF Predict-Update Cycle ---
            # The filter first projects its state estimate forward by dt seconds,
            # then corrects using the (corrupted) measurement. If the innovation
            # gamma_k exceeds the chi-squared gate, the measurement is discarded
            # and the missile relies solely on its prediction — temporarily immune
            # to the PHANTOM injection but also blind to true target motion.
            self._ekf.predict(self._dt)
            z = np.array([range_meas, bearing_meas], dtype=np.float64)
            gamma_k, accepted = self._ekf.update(z, missile_pos)

            if self._is_detection_window_active(injection_profile, t, injection_rate):
                active_updates += 1
                if not accepted:
                    active_rejections += 1

            # --- PN Guidance Command ---
            # The acceleration command is computed from the EKF's estimated target
            # state, not the true state. Under PHANTOM injection, this estimate
            # drifts progressively from reality, and the missile steers toward
            # the phantom intercept point with full confidence.
            ekf_state = self._ekf.get_state()
            accel = self._pn.compute_acceleration(
                missile_pos,
                missile_vel,
                ekf_state[:2],
                ekf_state[2:4],
            )

            # --- Dynamics Update (Euler integration) ---
            missile_vel = missile_vel + accel * self._dt
            missile_pos = missile_pos + missile_vel * self._dt
            target_pos = target_pos + target_vel * self._dt

            # --- Telemetry Logging ---
            entry = self._build_telemetry_entry(
                t,
                missile_pos,
                missile_vel,
                target_pos,
                ekf_state,
                gamma_k,
                accepted,
                injection_rate,
                accumulated_angle,
            )
            trajectory.append(entry)

            # --- Termination Check ---
            # Intercept is declared when the true range drops below 5m — a direct
            # hit by any operational definition. Under successful PHANTOM injection,
            # this condition is never met; the engagement times out with the missile
            # hundreds of meters from the true target.
            if entry["true_range"] < self._intercept_range or t >= self._max_time:
                break

        logger.info(
            "Engagement complete: profile=%s, t=%.2fs, miss=%.2fm",
            injection_profile.profile_type,
            trajectory[-1]["t"],
            trajectory[-1]["true_range"],
        )

        return self._compute_result(
            trajectory,
            injection_profile,
            active_updates=active_updates,
            active_rejections=active_rejections,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_telemetry_entry(
        self,
        t: float,
        missile_pos: npt.NDArray[np.float64],
        missile_vel: npt.NDArray[np.float64],
        target_pos: npt.NDArray[np.float64],
        ekf_state: npt.NDArray[np.float64],
        gamma_k: float,
        accepted: bool,
        injection_rate: float,
        accumulated_angle: float,
    ) -> TelemetryEntry:
        """Construct a single telemetry entry with all required fields.

        Parameters
        ----------
        t : float
            Current simulation time in seconds.
        missile_pos : np.ndarray
            Post-dynamics missile position, shape (2,).
        missile_vel : np.ndarray
            Post-dynamics missile velocity, shape (2,).
        target_pos : np.ndarray
            Post-dynamics target position, shape (2,).
        ekf_state : np.ndarray
            EKF state estimate [x, y, vx, vy], shape (4,).
        gamma_k : float
            Chi-squared innovation statistic from this step.
        accepted : bool
            Whether the EKF accepted the measurement.
        injection_rate : float
            Current false LOS rate in rad/s.
        accumulated_angle : float
            Total accumulated false bearing angle in rad.

        Returns
        -------
        dict
            Telemetry entry with all required keys.
        """
        ekf_pos = ekf_state[:2]
        true_range = float(np.linalg.norm(target_pos - missile_pos))
        estimated_range = float(np.linalg.norm(ekf_pos - missile_pos))
        estimation_error = float(np.linalg.norm(target_pos - ekf_pos))

        return {
            "t": t,
            "missile_pos": missile_pos.copy(),
            "missile_vel": missile_vel.copy(),
            "target_pos": target_pos.copy(),
            "ekf_estimate": ekf_pos.copy(),
            "ekf_vel_estimate": ekf_state[2:4].copy(),
            "gamma_k": gamma_k,
            "measurement_accepted": accepted,
            "injection_rate": injection_rate,
            "accumulated_angle": accumulated_angle,
            "true_range": true_range,
            "estimated_range": estimated_range,
            "estimation_error": estimation_error,
        }

    def _compute_result(
        self,
        trajectory: list[TelemetryEntry],
        injection_profile: InjectionProfile,
        active_updates: int,
        active_rejections: int,
    ) -> dict[str, Any]:
        """Build the engagement result dictionary from the telemetry log.

        Parameters
        ----------
        trajectory : list[dict]
            Full telemetry log from the engagement.
        injection_profile : InjectionProfile
            The injection profile used during the engagement.

        Returns
        -------
        dict
            Engagement result with all PHANTOM-specified fields.
        """
        final = trajectory[-1]
        miss_distance = final["true_range"]

        miss_vec = final["target_pos"] - final["missile_pos"]
        miss_angle = float(np.degrees(np.arctan2(miss_vec[1], miss_vec[0])))

        gamma_values = [entry["gamma_k"] for entry in trajectory]
        detection_rate = active_rejections / active_updates if active_updates > 0 else 0.0

        return {
            "trajectory": trajectory,
            "miss_distance": miss_distance,
            "miss_angle": miss_angle,
            "intercept_time": final["t"],
            "detection_rate": detection_rate,
            "max_innovation": (float(np.max(gamma_values)) if gamma_values else 0.0),
            "mean_innovation": (float(np.mean(gamma_values)) if gamma_values else 0.0),
            "max_estimation_error": max(entry["estimation_error"] for entry in trajectory),
            "success": (
                miss_distance > self._success_miss_distance
                and detection_rate < self._success_detection_rate
            ),
            "profile_type": injection_profile.profile_type,
        }

    def _is_detection_window_active(
        self,
        injection_profile: InjectionProfile,
        t: float,
        injection_rate: float,
    ) -> bool:
        """Return whether this timestep counts toward spoofing detection stats.

        PHANTOM's detection metric is evaluated over the period when a profile is
        actively applying its designed false signal. For ramp profiles this is the
        commanded ramp window; for step profiles it is the post-step interval;
        for profiles without explicit timing metadata, any non-zero rate counts.
        """
        t_start = getattr(injection_profile, "t_start", None)
        t_end = getattr(injection_profile, "t_end", None)

        if t_start is not None and t_end is not None:
            return bool(float(t_start) <= t <= float(t_end))
        if t_start is not None:
            return bool(t >= float(t_start))
        return bool(abs(injection_rate) > 0.0)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the simulator for a clean re-run with a new injection profile.

        Restores the random number generator to its initial seed and resets the
        EKF, ensuring that subsequent engagements with the same profile and
        initial conditions produce identical results.
        """
        self._rng = np.random.RandomState(self._seed)
        self._ekf.reset()
        logger.info("PhantomSimulator reset: seed=%d", self._seed)

    def get_config(self) -> dict:  # type: ignore[type-arg]
        """Return the simulator's configuration dictionary.

        Returns
        -------
        dict
            The full PHANTOM configuration loaded at construction time.
        """
        return copy.deepcopy(self._config)

"""PHANTOM — False Signal Injection Profile Library.

This module is the deception engine of the PHANTOM system. It generates the
false LOS angular rate delta_sigma_dot(t) that gets added to the true bearing
measurement before it reaches the missile's EKF. By shaping this injection
profile, PHANTOM controls whether the deception is detected (step injection)
or slips through the chi-squared gate unnoticed (linear ramp injection).

Five profiles are implemented:

    NoInjection         — Baseline (validates direct hit with miss < 5 m)
    StepInjection       — Abrupt step change (detectable, serves as negative control)
    LinearRampInjection — Slowly increasing rate (PHANTOM's primary strategy)
    PiecewiseInjection  — Multi-segment profile (directional steering control)
    AdaptiveInjection   — Stub for Phase 3 LLM-driven adaptive rate selection

The PhantomSimulator calls profile.get_rate(t) at every timestep to obtain
the false LOS angular rate injection for that moment in the engagement.

References
----------
Zarchan, P. (2012).
    Tactical and Strategic Missile Guidance (6th ed.). AIAA.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class InjectionProfile(ABC):
    """Abstract base class for all PHANTOM false signal injection profiles.

    Every injection profile must implement get_rate(t) which returns the
    false LOS angular rate delta_sigma_dot to be added to the true bearing
    measurement at time t. The PhantomSimulator calls this method at each
    timestep of the engagement simulation.

    Subclasses define the temporal shape of the injection — the key insight
    of PHANTOM is that the shape matters far more than the magnitude.
    """

    @abstractmethod
    def get_rate(self, t: float) -> float:
        """Return the false LOS angular rate at time t.

        Parameters
        ----------
        t : float
            Simulation time in seconds since engagement start.

        Returns
        -------
        float
            False LOS angular rate delta_sigma_dot(t) in rad/s.
        """

    def get_accumulated_angle(self, t: float, dt: float) -> float:
        """Compute the accumulated false bearing angle from t=0 to t.

        Uses the trapezoidal rule to numerically integrate get_rate() over
        the interval [0, t]. The accumulated angle delta_sigma(t) determines
        the total angular displacement the missile's track has been steered
        away from the true target.

        Parameters
        ----------
        t : float
            Upper integration limit in seconds.
        dt : float
            Integration timestep in seconds. Smaller dt yields higher accuracy.

        Returns
        -------
        float
            Accumulated false bearing angle in radians.
        """
        if t <= 0.0:
            return 0.0

        n_steps = max(1, int(t / dt))
        times = np.linspace(0.0, t, n_steps + 1)
        rates = np.array([self.get_rate(float(ti)) for ti in times])

        # Trapezoidal integration — sufficient accuracy for engagement-scale
        # timesteps (dt ~ 0.02 s), and matches the simulator's own integration.
        return float(np.trapz(rates, times))

    @property
    def profile_type(self) -> str:
        """Return the class name as a human-readable profile identifier.

        Returns
        -------
        str
            Name of the concrete injection profile class.
        """
        return type(self).__name__


# ---------------------------------------------------------------------------
# NoInjection — baseline / unperturbed engagement
# ---------------------------------------------------------------------------


class NoInjection(InjectionProfile):
    """Null injection profile — the unperturbed engagement baseline.

    NoInjection returns zero false LOS rate at all times, representing a
    standard PN engagement with no deception active. This profile is used
    in PHANTOM Experiment 1 (baseline validation) to confirm that the missile
    achieves a direct hit (miss distance < 5 m) when no injection is present.

    Any deviation from a direct hit under NoInjection indicates a bug in the
    engagement simulation rather than a deception effect.
    """

    def get_rate(self, t: float) -> float:
        """Return zero — no injection active.

        Parameters
        ----------
        t : float
            Simulation time in seconds (unused).

        Returns
        -------
        float
            Always 0.0 rad/s.
        """
        return 0.0


# ---------------------------------------------------------------------------
# StepInjection — detectable negative control
# ---------------------------------------------------------------------------


class StepInjection(InjectionProfile):
    """Step injection profile — PHANTOM's negative control.

    The step profile serves as PHANTOM's negative control. An abrupt
    change in the false LOS rate creates a large instantaneous innovation
    gamma_k that immediately exceeds the chi-squared threshold of 9.21,
    causing the EKF to reject the corrupted measurement. This confirms
    that naive jamming is detectable and motivates the smooth ramp approach.

    Parameters
    ----------
    t_start : float
        Time in seconds when the step injection activates.
    amplitude : float
        Magnitude of the false LOS angular rate step in rad/s.

    Example
    -------
    >>> profile = StepInjection(t_start=2.0, amplitude=0.05)
    >>> profile.get_rate(1.0)   # Before activation
    0.0
    >>> profile.get_rate(3.0)   # After activation
    0.05
    """

    def __init__(self, t_start: float, amplitude: float) -> None:
        self.t_start: float = float(t_start)
        self.amplitude: float = float(amplitude)

    def get_rate(self, t: float) -> float:
        """Return 0.0 before t_start, then the full step amplitude.

        The abrupt transition from zero to amplitude is what makes this
        profile detectable — the EKF's innovation statistic spikes immediately
        and the chi-squared gate rejects the corrupted measurement.

        Parameters
        ----------
        t : float
            Simulation time in seconds.

        Returns
        -------
        float
            False LOS angular rate in rad/s.
        """
        if t < self.t_start:
            return 0.0
        return self.amplitude


# ---------------------------------------------------------------------------
# LinearRampInjection — PHANTOM's primary deception strategy
# ---------------------------------------------------------------------------


class LinearRampInjection(InjectionProfile):
    """Linear ramp injection — PHANTOM's primary deception strategy.

    The linear ramp is PHANTOM's primary injection strategy. By slowly
    increasing the false LOS rate, the accumulated false bearing angle
    grows quadratically — large enough to steer the missile 200m+ off
    target — while the rate of change remains small enough that each
    individual EKF innovation stays below the chi-squared gate.
    The critical ramp rate (approximately 0.032 rad/s for baseline
    parameters) is the central finding of PHANTOM Phase 2.

    The accumulated false bearing angle grows as:

        delta_sigma(t) = 0.5 * ramp_rate * (t - t_start)^2

    This quadratic growth is the key mathematical insight — a linear rate
    produces a quadratic displacement, which can accumulate to hundreds of
    meters of miss distance while each timestep's innovation increment
    remains within the chi-squared gate.

    Parameters
    ----------
    t_start : float
        Time in seconds when the ramp begins. Loaded from
        phantom_config.yaml ``injection.ramp_tstart``.
    t_end : float
        Time in seconds when the ramp reaches its maximum rate and holds.
        Loaded from phantom_config.yaml ``injection.ramp_tend``.
    ramp_rate : float
        Rate of increase of the false LOS angular rate in rad/s per second.
        The sweep values in phantom_config.yaml ``injection.ramp_rates``
        are used to identify the critical rate in Phase 2 experiments.

    Example
    -------
    >>> profile = LinearRampInjection(t_start=1.0, t_end=4.0, ramp_rate=0.032)
    >>> profile.get_rate(2.5)
    0.048
    >>> profile.max_rate
    0.096
    >>> profile.accumulated_angle_at_end
    0.144
    """

    def __init__(self, t_start: float, t_end: float, ramp_rate: float) -> None:
        self.t_start: float = float(t_start)
        self.t_end: float = float(t_end)
        self.ramp_rate: float = float(ramp_rate)

    def get_rate(self, t: float) -> float:
        """Return the linearly increasing false LOS rate.

        The rate grows linearly from zero at t_start to max_rate at t_end,
        then holds at max_rate. This gradual onset is what keeps each
        individual EKF innovation below the chi-squared threshold.

        Parameters
        ----------
        t : float
            Simulation time in seconds.

        Returns
        -------
        float
            False LOS angular rate in rad/s.
        """
        if t < self.t_start:
            return 0.0

        if t > self.t_end:
            return self.ramp_rate * (self.t_end - self.t_start)

        # Accumulated false bearing — quadratic growth for linear ramp:
        # delta_sigma(t) = 0.5 * ramp_rate * (t - t_start)^2
        # This is what drives the PN guidance command off-course (Zarchan, 2012)
        return self.ramp_rate * (t - self.t_start)

    @property
    def max_rate(self) -> float:
        """Peak false LOS angular rate reached at the end of the ramp.

        Returns
        -------
        float
            Maximum injection rate in rad/s, equal to
            ramp_rate * (t_end - t_start).
        """
        return self.ramp_rate * (self.t_end - self.t_start)

    @property
    def accumulated_angle_at_end(self) -> float:
        """Analytical accumulated false bearing angle at t_end.

        For a linear ramp from t_start to t_end, the integral of
        ramp_rate * (t - t_start) dt yields:

            delta_sigma = 0.5 * ramp_rate * (t_end - t_start)^2

        This is the total angular displacement built up during the
        active ramp phase, before the hold period begins.

        Returns
        -------
        float
            Accumulated false bearing angle in radians.
        """
        # Accumulated false bearing — quadratic growth for linear ramp:
        # delta_sigma(t) = 0.5 * ramp_rate * (t - t_start)^2
        # This is what drives the PN guidance command off-course (Zarchan, 2012)
        duration = self.t_end - self.t_start
        return 0.5 * self.ramp_rate * duration**2


# ---------------------------------------------------------------------------
# PiecewiseInjection — directional steering control
# ---------------------------------------------------------------------------


class PiecewiseInjection(InjectionProfile):
    """Piecewise-linear injection profile for directional steering control.

    The piecewise profile enables directional steering — by controlling
    the sign and timing of injection, PHANTOM can steer the miss vector
    to a specific safe zone rather than a random direction (Phase 2, Exp 5).

    The profile linearly interpolates the false LOS rate between a sequence
    of (time, rate) breakpoints. Before the first breakpoint, the rate is
    zero. After the last breakpoint, the rate holds at the final value.

    Parameters
    ----------
    breakpoints : list[tuple[float, float]]
        Sequence of (time_s, rate_rad_per_s) control points. Must contain
        at least two points and be sorted by time in ascending order.

    Example
    -------
    >>> profile = PiecewiseInjection([(1.0, 0.0), (2.0, 0.05), (4.0, 0.05)])
    >>> profile.get_rate(1.5)
    0.025
    >>> profile.get_rate(3.0)
    0.05
    """

    def __init__(self, breakpoints: list[tuple[float, float]]) -> None:
        if len(breakpoints) < 2:
            raise ValueError("PiecewiseInjection requires at least 2 breakpoints.")
        self.breakpoints: list[tuple[float, float]] = [(float(t), float(r)) for t, r in breakpoints]

    def get_rate(self, t: float) -> float:
        """Return the linearly interpolated false LOS rate at time t.

        Parameters
        ----------
        t : float
            Simulation time in seconds.

        Returns
        -------
        float
            Interpolated false LOS angular rate in rad/s.
        """
        # The piecewise profile enables directional steering — by controlling
        # the sign and timing of injection, PHANTOM can steer the miss vector
        # to a specific safe zone rather than a random direction (Phase 2, Exp 5)

        if t < self.breakpoints[0][0]:
            return 0.0

        if t >= self.breakpoints[-1][0]:
            return self.breakpoints[-1][1]

        # Walk the breakpoint list to find the enclosing segment, then
        # linearly interpolate within that segment.
        for i in range(len(self.breakpoints) - 1):
            t0, r0 = self.breakpoints[i]
            t1, r1 = self.breakpoints[i + 1]
            if t0 <= t < t1:
                frac = (t - t0) / (t1 - t0)
                return r0 + frac * (r1 - r0)

        return self.breakpoints[-1][1]


# ---------------------------------------------------------------------------
# AdaptiveInjection — stub for Phase 3 LLM-driven control
# ---------------------------------------------------------------------------


class AdaptiveInjection(InjectionProfile):
    """Adaptive injection profile — stub for Phase 3 LLM-driven control.

    Full implementation in Phase 3 — PhantomLLMController will override
    this with real-time adaptive rate selection.

    The adaptive profile will observe the EKF's innovation statistic
    gamma_k in real time and adjust the injection rate to maintain
    gamma_k just below the chi-squared threshold. The proportional
    controller gain kp governs how aggressively the rate adapts.

    Parameters
    ----------
    target_gamma : float
        Desired chi-squared statistic to maintain (should be just below
        the 9.21 threshold, e.g. 8.0).
    kp : float
        Proportional controller gain for rate adaptation.

    Example
    -------
    >>> profile = AdaptiveInjection(target_gamma=8.0, kp=0.005)
    >>> profile.get_rate(1.0)
    0.0
    """

    def __init__(self, target_gamma: float, kp: float) -> None:
        self.target_gamma: float = float(target_gamma)
        self.kp: float = float(kp)

    def get_rate(self, t: float) -> float:
        """Return the adaptive false LOS rate (stub — returns 0.0).

        Full implementation in Phase 3 — PhantomLLMController will override
        this with real-time adaptive rate selection based on the observed
        innovation statistic.

        Parameters
        ----------
        t : float
            Simulation time in seconds.

        Returns
        -------
        float
            False LOS angular rate in rad/s. Currently returns 0.0.
        """
        return 0.0

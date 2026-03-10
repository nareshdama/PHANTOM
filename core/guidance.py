"""PHANTOM — Proportional Navigation Guidance Law.

This module implements the 2D Proportional Navigation (PN) guidance law that
steers the missile toward its estimated target. The PN command is the critical
link between EKF estimation and physical trajectory: when PHANTOM injects a
false bearing into the EKF, the corrupted state estimate feeds into this PN
law, producing an acceleration command that drives the missile toward a
phantom intercept point rather than the true aircraft.

The relationship is:

    a_cmd = N * V_c * sigma_dot

Where N is the navigation ratio, V_c is the closing velocity, and sigma_dot
is the line-of-sight (LOS) angular rate. PHANTOM's entire deception acts
through sigma_dot — by corrupting the EKF's bearing estimate, the computed
sigma_dot diverges from truth, and the PN law faithfully steers the missile
away from the real target.

References
----------
Zarchan, P. (2012).
    Tactical and Strategic Missile Guidance (6th ed.). AIAA.
Shneydor, N. A. (1998).
    Missile Guidance and Pursuit. Horwood Publishing.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Acceleration due to gravity — used to convert max_accel from g to m/s².
GRAVITY_MS2: float = 9.81

# When the missile-target range falls below this threshold, the guidance
# geometry becomes singular (division by nearly-zero range). At this point
# the missile is effectively at intercept and no further steering is needed.
SINGULARITY_RANGE_M: float = 1.0

# Minimum missile speed below which the perpendicular acceleration direction
# cannot be determined. Returns zero acceleration to avoid division by zero.
MIN_SPEED_MS: float = 1e-10


class ProportionalNavigation:  # pylint: disable=invalid-name
    """2D Proportional Navigation guidance law for missile steering.

    Proportional Navigation commands a lateral acceleration proportional to
    the line-of-sight (LOS) angular rate and closing velocity. Under ideal
    conditions (constant-velocity target, perfect measurements), PN with
    N >= 3 guarantees a zero-effort-miss intercept. PHANTOM exploits this
    by corrupting the LOS rate estimate through false bearing injection:
    the missile "perfectly" guides itself to the wrong point.

    Parameters
    ----------
    nav_ratio : float
        Navigation ratio N (dimensionless, typically 3–5). Higher values
        produce more aggressive steering but amplify noise. Loaded from
        phantom_config.yaml.
    max_accel_g : float
        Maximum acceleration capability of the missile in g-units. The PN
        command is clipped to this limit, reflecting the physical constraint
        of airframe structural load factor. Loaded from phantom_config.yaml.

    Attributes
    ----------
    NAV_RATIO : float
        Stored navigation ratio.
    MAX_ACCEL_G : float
        Stored maximum acceleration in g-units.

    Example
    -------
    >>> pn = ProportionalNavigation(nav_ratio=4.0, max_accel_g=30.0)
    >>> accel = pn.compute_acceleration(
    ...     missile_pos=np.array([0.0, 0.0]),
    ...     missile_vel=np.array([600.0, 0.0]),
    ...     target_estimate=np.array([5000.0, 100.0]),
    ...     target_vel_estimate=np.array([-250.0, 0.0]),
    ... )
    """

    def __init__(self, nav_ratio: float, max_accel_g: float) -> None:
        self.NAV_RATIO: float = float(nav_ratio)
        self.MAX_ACCEL_G: float = float(max_accel_g)

    # ------------------------------------------------------------------
    # Primary guidance command
    # ------------------------------------------------------------------

    def compute_acceleration(
        self,
        missile_pos: npt.NDArray[np.float64],
        missile_vel: npt.NDArray[np.float64],
        target_estimate: npt.NDArray[np.float64],
        target_vel_estimate: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the PN lateral acceleration command in Cartesian coordinates.

        The PN guidance law steers the missile by zeroing the LOS angular rate.
        When PHANTOM injects a false bearing into the EKF, the estimated target
        position drifts from the true position, causing a false sigma_dot that
        drives the missile toward the phantom intercept point rather than the
        true aircraft.

        The command is decomposed into [ax, ay] perpendicular to the current
        missile velocity vector, consistent with the physical constraint that
        aerodynamic lift acts normal to the flight path.

        Parameters
        ----------
        missile_pos : np.ndarray
            Missile position [x, y] in meters, shape (2,).
        missile_vel : np.ndarray
            Missile velocity [vx, vy] in m/s, shape (2,).
        target_estimate : np.ndarray
            Estimated target position [x, y] in meters, shape (2,). This comes
            from the EKF state estimate — when PHANTOM is active, this estimate
            is corrupted and diverges from the true target position.
        target_vel_estimate : np.ndarray
            Estimated target velocity [vx, vy] in m/s, shape (2,).

        Returns
        -------
        np.ndarray
            Acceleration command [ax, ay] in m/s², shape (2,). Perpendicular
            to the missile velocity vector. Magnitude clipped to max_accel_g.
        """
        missile_pos = np.asarray(missile_pos, dtype=np.float64)
        missile_vel = np.asarray(missile_vel, dtype=np.float64)
        target_estimate = np.asarray(target_estimate, dtype=np.float64)
        target_vel_estimate = np.asarray(target_vel_estimate, dtype=np.float64)

        # Relative position from missile to estimated target
        r_rel = target_estimate - missile_pos
        rng = float(np.linalg.norm(r_rel))

        # Singularity guard — at intercept the guidance geometry degenerates.
        # Below SINGULARITY_RANGE_M the missile is effectively at the target
        # and no further steering correction is meaningful.
        if rng < SINGULARITY_RANGE_M:
            return np.zeros(2, dtype=np.float64)

        # Zero-velocity guard — without a velocity vector, the perpendicular
        # acceleration direction is undefined.
        missile_speed = float(np.linalg.norm(missile_vel))
        if missile_speed < MIN_SPEED_MS:
            return np.zeros(2, dtype=np.float64)

        # --- Closing velocity and LOS angular rate ---
        v_c = self.get_closing_velocity(
            missile_pos, missile_vel, target_estimate, target_vel_estimate
        )
        sigma_dot = self.get_los_rate(
            missile_pos, missile_vel, target_estimate, target_vel_estimate
        )

        # PN lateral acceleration — Zarchan (2012), "Tactical and Strategic
        # Missile Guidance", 6th ed., Eq. 3.14
        a_lateral = self.NAV_RATIO * v_c * sigma_dot

        # Clip to the missile's structural acceleration limit.
        # The airframe can only sustain a finite load factor; exceeding it
        # would mean the commanded maneuver is physically unrealizable.
        max_accel_ms2 = self.MAX_ACCEL_G * GRAVITY_MS2
        a_lateral = float(np.clip(a_lateral, -max_accel_ms2, max_accel_ms2))

        # Decompose into Cartesian [ax, ay] perpendicular to missile velocity.
        # The perpendicular direction (consistent with right-hand coordinate
        # system) is [-vy, vx] / |v|, which points "upward" (positive
        # cross-product direction) relative to the velocity vector.
        perp_x = -missile_vel[1] / missile_speed
        perp_y = missile_vel[0] / missile_speed

        return np.array([a_lateral * perp_x, a_lateral * perp_y], dtype=np.float64)

    # ------------------------------------------------------------------
    # Public helpers — used by analysis/statistics.py to verify
    # PHANTOM's effect on the guidance loop
    # ------------------------------------------------------------------

    def get_los_rate(
        self,
        missile_pos: npt.NDArray[np.float64],
        missile_vel: npt.NDArray[np.float64],
        target_pos: npt.NDArray[np.float64],
        target_vel: npt.NDArray[np.float64],
    ) -> float:
        """Compute the line-of-sight angular rate sigma_dot.

        The LOS angular rate sigma_dot is the time derivative of the angle
        between the missile-target line and a fixed inertial reference. Under
        pure PN with no noise, a zero LOS rate guarantees a collision course.
        PHANTOM's entire deception hinges on corrupting this single quantity.

        Derived from the 2D vector cross product of relative position and
        relative velocity, normalized by range squared:

            sigma_dot = (r_x * v_y - r_y * v_x) / |r|^2

        Parameters
        ----------
        missile_pos : np.ndarray
            Missile position [x, y] in meters, shape (2,).
        missile_vel : np.ndarray
            Missile velocity [vx, vy] in m/s, shape (2,).
        target_pos : np.ndarray
            Target position [x, y] in meters, shape (2,).
        target_vel : np.ndarray
            Target velocity [vx, vy] in m/s, shape (2,).

        Returns
        -------
        float
            LOS angular rate in rad/s. Positive when the LOS angle is
            increasing (target moving counterclockwise relative to missile).
        """
        r_rel = np.asarray(target_pos, dtype=np.float64) - np.asarray(missile_pos, dtype=np.float64)
        v_rel = np.asarray(target_vel, dtype=np.float64) - np.asarray(missile_vel, dtype=np.float64)

        rng_sq = float(np.dot(r_rel, r_rel))

        # Guard against the singularity when missile and target nearly coincide.
        if rng_sq < SINGULARITY_RANGE_M**2:
            return 0.0

        # LOS angular rate — derived from the cross product of relative position
        # and relative velocity, normalized by range squared.
        # sigma_dot = (r_x * v_y - r_y * v_x) / |r|^2
        # This is the 2D scalar cross product r × v divided by |r|^2.
        cross = float(r_rel[0] * v_rel[1] - r_rel[1] * v_rel[0])

        return cross / rng_sq

    def get_closing_velocity(
        self,
        missile_pos: npt.NDArray[np.float64],
        missile_vel: npt.NDArray[np.float64],
        target_pos: npt.NDArray[np.float64],
        target_vel: npt.NDArray[np.float64],
    ) -> float:
        """Compute the closing velocity V_c between missile and target.

        Closing velocity is the rate at which the missile-target range
        decreases. It is computed as the negative projection of the relative
        velocity onto the unit LOS vector. A positive V_c means the missile
        is closing on the target — this is the expected condition during an
        active engagement.

        Parameters
        ----------
        missile_pos : np.ndarray
            Missile position [x, y] in meters, shape (2,).
        missile_vel : np.ndarray
            Missile velocity [vx, vy] in m/s, shape (2,).
        target_pos : np.ndarray
            Target position [x, y] in meters, shape (2,).
        target_vel : np.ndarray
            Target velocity [vx, vy] in m/s, shape (2,).

        Returns
        -------
        float
            Closing velocity in m/s. Positive when the missile is approaching
            the target (range decreasing).
        """
        r_rel = np.asarray(target_pos, dtype=np.float64) - np.asarray(missile_pos, dtype=np.float64)
        v_rel = np.asarray(target_vel, dtype=np.float64) - np.asarray(missile_vel, dtype=np.float64)

        rng = float(np.linalg.norm(r_rel))

        # At intercept the closing velocity concept is undefined.
        if rng < SINGULARITY_RANGE_M:
            return 0.0

        # Closing velocity — negative projection of relative velocity onto
        # the unit LOS vector; positive V_c means missile is closing on target.
        # V_c = -dot(v_rel, r_hat) = -dot(v_rel, r_rel) / |r_rel|
        return -float(np.dot(r_rel, v_rel)) / rng

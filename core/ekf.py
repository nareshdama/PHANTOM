"""PHANTOM — Extended Kalman Filter with Chi-Squared Innovation Gating.

This module implements the 2D constant-velocity Extended Kalman Filter (EKF)
that a Proportional Navigation-guided missile uses to track its target. The
chi-squared innovation gate is the vulnerability PHANTOM exploits: by injecting
false signals whose innovation statistic gamma_k stays just below the gating
threshold, the missile confidently tracks a phantom trajectory rather than the
true target.

References
----------
Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001).
    Estimation with Applications to Tracking and Navigation. Wiley.
Zarchan, P. (2012).
    Tactical and Strategic Missile Guidance (6th ed.). AIAA.
"""

from __future__ import annotations

# Standard matrix names (Q, R, F, H, K, S, P) are universal in Kalman filter
# literature (Bar-Shalom et al., 2001) and would lose clarity if renamed.
# pylint: disable=invalid-name

from typing import Tuple

import numpy as np
import numpy.typing as npt


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to the interval [-pi, pi].

    Bearing innovations can cross the +/-pi discontinuity when the
    line-of-sight angle passes through the negative x-axis. Without
    wrapping, the filter would see a ~2*pi innovation and reject the
    measurement, even though the actual angular change is small.

    Parameters
    ----------
    angle : float
        Angle in radians, potentially outside [-pi, pi].

    Returns
    -------
    float
        Angle wrapped to [-pi, pi].
    """
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class ExtendedKalmanFilter:  # pylint: disable=too-many-instance-attributes
    """2D constant-velocity Extended Kalman Filter with chi-squared gating.

    This filter estimates the kinematic state of a target using noisy range
    and bearing measurements from the missile's onboard seeker. The innovation
    gate allows the filter to reject measurements that are statistically
    inconsistent with the current track — a defense mechanism that, paradoxically,
    becomes PHANTOM's primary attack surface when the injected signal is kept
    just below the acceptance threshold.

    Parameters
    ----------
    Q : np.ndarray
        Process noise covariance matrix, shape (4, 4). Represents the
        uncertainty introduced by unmodeled target maneuvers between updates.
    R : np.ndarray
        Measurement noise covariance matrix, shape (2, 2). Diagonal entries
        correspond to range variance (m^2) and bearing variance (rad^2).
    chi2_threshold : float
        Chi-squared gating threshold. For 2 degrees of freedom at the 99th
        percentile, this is 9.21. Loaded from phantom_config.yaml — never
        hardcoded.

    Attributes
    ----------
    x_hat : np.ndarray
        Estimated state vector [x, y, vx, vy]^T in meters and m/s.
    P : np.ndarray
        State error covariance matrix, shape (4, 4).

    Example
    -------
    >>> Q = np.eye(4) * 0.1
    >>> R = np.diag([10.0**2, 0.00873**2])
    >>> ekf = ExtendedKalmanFilter(Q, R, chi2_threshold=9.21)
    >>> ekf.x_hat = np.array([5000.0, 0.0, -250.0, 0.0])
    >>> ekf.predict(dt=0.02)
    >>> gamma_k, accepted = ekf.update(
    ...     z=np.array([4995.0, 0.001]),
    ...     missile_pos=np.array([0.0, 0.0])
    ... )
    """

    def __init__(
        self,
        Q: npt.NDArray[np.float64],
        R: npt.NDArray[np.float64],
        chi2_threshold: float,
    ) -> None:
        self._n_states: int = 4
        self._n_meas: int = 2

        self.Q: npt.NDArray[np.float64] = np.array(Q, dtype=np.float64)
        self.R: npt.NDArray[np.float64] = np.array(R, dtype=np.float64)

        # Chi-squared gate — 99th percentile threshold for 2 DOF (range + bearing).
        # This value is loaded from phantom_config.yaml at the call site.
        self.CHI2_THRESHOLD_99: float = float(chi2_threshold)

        # State and covariance — initialized with large uncertainty to reflect
        # the filter's ignorance at the start of the engagement.
        self.x_hat: npt.NDArray[np.float64] = np.zeros(self._n_states, dtype=np.float64)
        self.P: npt.NDArray[np.float64] = np.eye(self._n_states, dtype=np.float64) * 1000.0

        # Preserve initial conditions so reset() can restore them exactly.
        self._x_hat_init: npt.NDArray[np.float64] = self.x_hat.copy()
        self._P_init: npt.NDArray[np.float64] = self.P.copy()

        # Engagement statistics — tracked across the full engagement for
        # post-hoc analysis of PHANTOM deception effectiveness.
        self.innovation_history: list[float] = []
        self.rejection_count: int = 0
        self.update_count: int = 0

    # ------------------------------------------------------------------
    # Prediction step
    # ------------------------------------------------------------------

    def predict(self, dt: float) -> None:
        """Propagate the state estimate and covariance forward by one timestep.

        The constant-velocity kinematic model assumes the target maintains its
        current velocity between measurement updates. In reality, targets
        maneuver — the process noise Q accounts for this unmodeled acceleration.
        During the prediction phase, uncertainty always grows because no new
        information is incorporated.

        Parameters
        ----------
        dt : float
            Time step in seconds between the current and next measurement
            epoch. For PHANTOM's 50 Hz engagement, this is typically 0.02 s.
        """
        # State transition — constant velocity model (Bar-Shalom et al., 2001, Eq. 6.2-15)
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Predicted state — position extrapolated using current velocity estimate.
        self.x_hat = F @ self.x_hat

        # Predicted covariance — inflated by process noise to reflect growing
        # uncertainty in the absence of new measurements.
        # P_k|k-1 = F * P_k-1|k-1 * F^T + Q  (Bar-Shalom et al., 2001, Eq. 5.2-12)
        self.P = F @ self.P @ F.T + self.Q

    # ------------------------------------------------------------------
    # Measurement update step
    # ------------------------------------------------------------------

    def update(
        self, z: npt.NDArray[np.float64], missile_pos: npt.NDArray[np.float64]
    ) -> Tuple[float, bool]:
        """Perform the EKF measurement update with chi-squared innovation gating.

        When the measured bearing deviates significantly from the predicted
        value, the innovation statistic gamma_k will exceed the chi-squared
        threshold, causing the filter to reject the measurement. This is the
        central mechanism exploited by the PHANTOM false signal injection
        strategy — by keeping gamma_k just below the gate, injected signals are
        accepted as legitimate, steering the missile's track toward a phantom
        trajectory.

        Parameters
        ----------
        z : np.ndarray
            Measurement vector [range (m), bearing (rad)], shape (2,).
            Range is the slant distance from missile to target; bearing is the
            line-of-sight angle measured from the missile's reference axis.
        missile_pos : np.ndarray
            Current missile position [x, y] in meters, shape (2,). Required to
            compute the predicted measurement h(x_hat) since range and bearing
            are relative quantities.

        Returns
        -------
        Tuple[float, bool]
            (gamma_k, accepted) where gamma_k is the chi-squared innovation
            statistic and accepted indicates whether the measurement passed
            the gate. When accepted is False, the filter performs prediction
            only and the state is not corrected.

        Raises
        ------
        ValueError
            If z does not have shape (2,) or missile_pos does not have shape (2,).
        """
        z = np.asarray(z, dtype=np.float64)
        missile_pos = np.asarray(missile_pos, dtype=np.float64)

        if z.shape != (2,):
            raise ValueError(f"Measurement z must have shape (2,), got {z.shape}")
        if missile_pos.shape != (2,):
            raise ValueError(f"missile_pos must have shape (2,), got {missile_pos.shape}")

        self.update_count += 1

        # --- Predicted measurement h(x_hat) and Jacobian H ---
        z_pred, H = self._compute_measurement_and_jacobian(missile_pos)

        # --- Innovation (measurement residual) ---
        # The innovation y_tilde represents the discrepancy between what the
        # EKF predicted the target's position to be and what the sensor actually
        # measured. Under PHANTOM injection, this discrepancy is deliberately
        # shaped to stay below the chi-squared gate.
        y_tilde = z - z_pred

        # Wrap bearing innovation to [-pi, pi] — prevents discontinuity at +/-pi
        # that would otherwise cause spurious gate rejections when the LOS angle
        # crosses the branch cut of atan2.
        y_tilde[1] = _wrap_angle(float(y_tilde[1]))

        # --- Innovation covariance ---
        # S combines the predicted state uncertainty (projected into measurement
        # space) with the sensor noise. A larger S makes the gate more tolerant,
        # which is why PHANTOM injects gradually — to avoid inflating S and
        # losing deception effectiveness.
        # S_k = H * P_k|k-1 * H^T + R  (Bar-Shalom et al., 2001, Eq. 5.2-15)
        S = H @ self.P @ H.T + self.R
        S_inv = np.linalg.inv(S)

        # --- Chi-squared innovation gate ---
        # gamma_k quantifies how many standard deviations the innovation lies
        # from zero, accounting for correlations between range and bearing.
        # Under the null hypothesis (correct track), gamma_k ~ chi2(2).
        # Chi-squared gate — 99th percentile threshold for 2 DOF (range + bearing)
        gamma_k = float(y_tilde.T @ S_inv @ y_tilde)

        self.innovation_history.append(gamma_k)

        accepted = gamma_k <= self.CHI2_THRESHOLD_99

        if accepted:
            self._apply_kalman_correction(H, S_inv, y_tilde)
        else:
            # Measurement rejected — the filter coasts on prediction alone.
            # Uncertainty continues to grow because no correction is applied.
            # This is the filter's defense against spoofing, and the exact
            # mechanism PHANTOM must avoid triggering.
            self.rejection_count += 1

        return gamma_k, accepted

    def _apply_kalman_correction(
        self,
        H: npt.NDArray[np.float64],
        S_inv: npt.NDArray[np.float64],
        y_tilde: npt.NDArray[np.float64],
    ) -> None:
        """Apply the Kalman gain to correct state and covariance.

        With the measurement accepted by the chi-squared gate, the standard
        linear correction shifts the state estimate toward the measurement.
        The gain K balances prediction uncertainty against measurement noise —
        when the filter is confident in its prediction (small P), K is small
        and the correction is modest.

        Parameters
        ----------
        H : np.ndarray
            Measurement Jacobian, shape (2, 4).
        S_inv : np.ndarray
            Inverse of innovation covariance, shape (2, 2).
        y_tilde : np.ndarray
            Innovation vector, shape (2,).
        """
        # Kalman gain — K_k = P_k|k-1 * H^T * S_k^-1
        # (Bar-Shalom et al., 2001, Eq. 5.2-16)
        K = self.P @ H.T @ S_inv

        # State correction — shift the estimate toward the measurement.
        self.x_hat = self.x_hat + K @ y_tilde

        # Covariance update — standard form.
        # P_k|k = (I - K_k * H) * P_k|k-1  (Bar-Shalom et al., 2001, Eq. 5.2-17)
        I_KH = np.eye(self._n_states) - K @ H
        self.P = I_KH @ self.P

    # ------------------------------------------------------------------
    # Nonlinear measurement model
    # ------------------------------------------------------------------

    def _compute_measurement_and_jacobian(
        self, missile_pos: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute the predicted measurement h(x_hat) and its Jacobian H.

        The nonlinear measurement model converts the Cartesian state into
        polar range and bearing as observed from the missile's position. The
        Jacobian H linearizes this mapping around the current estimate for
        use in the EKF update equations.

        Parameters
        ----------
        missile_pos : np.ndarray
            Missile position [x, y] in meters, shape (2,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (z_pred, H) where z_pred is the predicted measurement
            [range (m), bearing (rad)] and H is the (2, 4) measurement
            Jacobian matrix.
        """
        # Relative position from missile to estimated target
        dx = self.x_hat[0] - missile_pos[0]
        dy = self.x_hat[1] - missile_pos[1]

        # Predicted range — Euclidean distance in the engagement plane.
        # This forms the first component of the measurement model h(x).
        r_pred = float(np.sqrt(dx**2 + dy**2))

        # Guard against division by zero when missile and target nearly coincide
        r_safe = max(r_pred, 1e-10)

        # Predicted bearing — line-of-sight angle from missile to target,
        # measured counterclockwise from the positive x-axis.
        bearing_pred = float(np.arctan2(dy, dx))

        z_pred = np.array([r_pred, bearing_pred], dtype=np.float64)

        # Measurement Jacobian — partial derivatives of h(x) w.r.t. the state.
        # H = dh/dx evaluated at x_hat  (Bar-Shalom et al., 2001, Eq. 6.3-4)
        H = np.zeros((self._n_meas, self._n_states), dtype=np.float64)

        # dr/d(x,y) — range changes linearly with position along the LOS direction.
        H[0, 0] = dx / r_safe
        H[0, 1] = dy / r_safe

        # dtheta/d(x,y) — bearing sensitivity is inversely proportional to range;
        # cross-range displacement at close range produces larger angle changes.
        H[1, 0] = -dy / (r_safe**2)
        H[1, 1] = dx / (r_safe**2)

        # Velocity components do not affect the instantaneous measurement.
        # H[:, 2] and H[:, 3] remain zero.

        return z_pred, H

    # ------------------------------------------------------------------
    # State accessors and statistics
    # ------------------------------------------------------------------

    def get_state(self) -> npt.NDArray[np.float64]:
        """Return the current state estimate [x, y, vx, vy].

        The state vector represents the filter's best estimate of the target's
        position and velocity in the engagement coordinate frame. Under
        successful PHANTOM injection, this estimate diverges from the true
        target state, causing the missile to pursue a phantom trajectory.

        Returns
        -------
        np.ndarray
            State vector [x (m), y (m), vx (m/s), vy (m/s)], shape (4,).
        """
        return self.x_hat.copy()

    def get_statistics(self) -> dict[str, float | int]:
        """Return engagement-level tracking statistics for analysis.

        These statistics quantify the health of the EKF track throughout the
        engagement. The rejection rate is a key PHANTOM performance metric:
        it must stay below ~5% for the deception to remain undetected by
        built-in-test or operator monitoring.

        Returns
        -------
        dict[str, float | int]
            Dictionary with keys:

            - ``rejection_rate``: Fraction of updates rejected by the gate.
            - ``mean_innovation``: Average chi-squared statistic across updates.
            - ``max_innovation``: Peak chi-squared statistic observed.
            - ``rejection_count``: Total measurements rejected.
            - ``update_count``: Total measurement updates attempted.
        """
        n = self.update_count
        return {
            "rejection_rate": self.rejection_count / n if n > 0 else 0.0,
            "mean_innovation": (
                float(np.mean(self.innovation_history)) if self.innovation_history else 0.0
            ),
            "max_innovation": (
                float(np.max(self.innovation_history)) if self.innovation_history else 0.0
            ),
            "rejection_count": self.rejection_count,
            "update_count": self.update_count,
        }

    def reset(self) -> None:
        """Reset the filter to its initial state for a new engagement.

        Restores the state estimate, covariance, and all tracking statistics
        to their values at construction time. Used between Monte Carlo runs
        to ensure each trial begins from identical initial conditions.
        """
        self.x_hat = self._x_hat_init.copy()
        self.P = self._P_init.copy()
        self.innovation_history = []
        self.rejection_count = 0
        self.update_count = 0

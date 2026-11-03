from __future__ import annotations

import numpy as np
from scipy.linalg import LinAlgError
from scipy.linalg.lapack import dpotrf, dpotrs


class KalmanFilter:
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    """

    def __init__(self):
        ndim, dt = 4, 1.0

        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(
        self, measurement: np.ndarray, velocity: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create track from an unassociated measurement.

        Args:
            measurement (ndarray): Bounding box coordinates (x, y, a, h) with
                center position (x, y), aspect ratio a, and height h.
            velocity (ndarray, optional): Initial center velocity (vx, vy) in
                pixels per frame. Defaults to 0.

        Returns:
            Tuple[ndarray, ndarray]: Returns the mean vector (8 dimensional) and
                covariance matrix (8x8 dimensional) of the new track.
                Unobserved velocities are initialized to 0 mean, except the
                center velocity when `velocity` is given.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        if velocity is not None:
            mean_vel[:2] = velocity
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter prediction step.

        Args:
            mean (ndarray): The 8 dimensional mean vector of the object
                state at the previous time step.
            covariance (ndarray): The 8x8 dimensional covariance matrix of
                the object state at the previous time step.

        Returns:
            Tuple[ndarray, ndarray]: Returns the mean vector and
                covariance matrix of the predicted state.
                Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )

        return mean, covariance

    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space.

        Args:
            mean (ndarray): The state's mean vector (8 dimensional array).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            Tuple[ndarray, ndarray]: Returns the projected mean and
                covariance matrix of the given state estimate.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        # _update_mat only selects (x, y, a, h): slicing gives the same values as
        # the matrix products, without them
        return mean[:4].copy(), covariance[:4, :4] + innovation_cov

    def multi_predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter prediction step (Vectorized version).

        Args:
            mean (ndarray): The Nx8 dimensional mean matrix
                of the object states at the previous time step.
            covariance (ndarray): The Nx8x8 dimensional covariance matrices
                of the object states at the previous time step.

        Returns:
            Tuple[ndarray, ndarray]: Returns the mean vector and
                covariance matrix of the predicted state.
                Unobserved velocities are initialized to 0 mean.
        """
        std = np.empty((len(mean), 8))
        std[:, [0, 1, 3]] = (self._std_weight_position * mean[:, 3])[:, None]
        std[:, 2] = 1e-2
        std[:, [4, 5, 7]] = (self._std_weight_velocity * mean[:, 3])[:, None]
        std[:, 6] = 1e-5

        # _motion_mat is the identity plus dt = 1 at (i, i + 4): multiplying by it
        # only adds each velocity to its position. These sums are exactly the
        # products mean @ F.T and F @ cov @ F.T (two non-zero terms per entry),
        # without the (N, 8, 8) matrix products.
        mean = mean.copy()
        mean[:, :4] += mean[:, 4:]
        covariance = covariance.copy()
        covariance[:, :4, :] += covariance[:, 4:, :]
        covariance[:, :, :4] += covariance[:, :, 4:]
        covariance[:, np.arange(8), np.arange(8)] += np.square(std)

        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (ndarray): The 4-dimensional measurement vector (x, y, a, h),
                where (x, y) is the center position, a the aspect ratio,
                and h the height of the bounding box.

        Returns:
            Tuple[ndarray, ndarray]: Returns the measurement-corrected
                state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # The LAPACK routines behind scipy.linalg.cho_factor/cho_solve, called
        # directly: same results without the per-call checks, which cost more
        # than the 4x4 solve. covariance[:, :4] == covariance @ _update_mat.T
        chol_factor, info = dpotrf(projected_cov, lower=1, clean=0)
        if info != 0:
            raise LinAlgError(f"cholesky of the projected covariance failed: {info}")
        kalman_gain, info = dpotrs(chol_factor, covariance[:, :4].T, lower=1)
        kalman_gain = kalman_gain.T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        # Same order np.linalg.multi_dot picks for these shapes: K @ (S @ K.T)
        new_covariance = covariance - np.dot(
            kalman_gain, np.dot(projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance

from __future__ import annotations

from enum import Enum

import numpy as np
import numpy.typing as npt

from supervision.tracker.byte_tracker.kalman_filter import KalmanFilter
from supervision.tracker.byte_tracker.utils import IdCounter


class TrackState(Enum):
    """Enumeration for track state."""

    TENTATIVE = 1
    CONFIRMED = 2
    TENTATIVE_DELETED = 3
    CONFIRMED_DELETED = 4


class SORTTrack:
    """
    A single target track for SORT algorithm.

    Uses Kalman filtering for prediction and update with state space (x, y, a, h).
    """

    def __init__(
        self,
        tlwh: npt.NDArray[np.float32],
        confidence: float,
        class_id: int,
        n_init: int,
        max_age: int,
        id_counter: IdCounter,
    ):
        """
        Initialize a track.

        Args:
            tlwh: Bounding box in format (top left x, top left y, width, height).
            confidence: Detection confidence score.
            class_id: Detection class ID.
            n_init: Number of consecutive detections before track is confirmed.
            max_age: Maximum number of consecutive misses before track is deleted.
            id_counter: Counter for generating unique track IDs.
        """
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = confidence
        self.class_id = class_id
        self._n_init = n_init
        self._max_age = max_age
        self.id_counter = id_counter

        self.mean = None
        self.covariance = None
        self.kalman_filter = None

        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.TENTATIVE
        self.track_id = id_counter.NO_ID
        self.frame_id = 0

    def activate(self, kalman_filter: KalmanFilter, frame_id: int) -> None:
        """Initialize track with Kalman filter."""
        self.kalman_filter = kalman_filter
        self.track_id = self.id_counter.new_id()
        self.frame_id = frame_id

        # Initialize Kalman filter with xyah format
        xyah = self.tlwh_to_xyah(self._tlwh)
        self.mean, self.covariance = self.kalman_filter.initiate(xyah)

    def predict(self, kalman_filter: KalmanFilter) -> None:
        """Propagate the state distribution using Kalman filter prediction."""
        if self.mean is None:
            return

        self.mean, self.covariance = kalman_filter.predict(
            self.mean, self.covariance
        )
        self.age += 1
        self.time_since_update += 1

    @staticmethod
    def multi_predict(
        tracks: list[SORTTrack], shared_kalman: KalmanFilter
    ) -> None:
        """Perform batch prediction for multiple tracks."""
        if len(tracks) == 0:
            return

        multi_mean = []
        multi_covariance = []

        for track in tracks:
            if track.mean is not None:
                multi_mean.append(track.mean.copy())
                multi_covariance.append(track.covariance)

        if len(multi_mean) == 0:
            return

        multi_mean, multi_covariance = shared_kalman.multi_predict(
            np.asarray(multi_mean), np.asarray(multi_covariance)
        )

        idx = 0
        for track in tracks:
            if track.mean is not None:
                track.mean = multi_mean[idx]
                track.covariance = multi_covariance[idx]
                idx += 1

    def update(
        self,
        kalman_filter: KalmanFilter,
        xyxy: npt.NDArray[np.float32],
        confidence: float,
        class_id: int,
        frame_id: int,
    ) -> None:
        """Perform Kalman filter measurement update."""
        # Convert xyxy to tlwh
        tlwh = np.array(
            [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
        )

        # Convert tlwh to xyah for Kalman filter
        xyah = self.tlwh_to_xyah(tlwh)

        self.mean, self.covariance = kalman_filter.update(
            self.mean, self.covariance, xyah
        )

        self._tlwh = tlwh
        self.confidence = confidence
        self.class_id = class_id
        self.frame_id = frame_id

        self.hits += 1
        self.time_since_update = 0

        if self.state == TrackState.TENTATIVE and self.hits >= self._n_init:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        """Mark this track as missed (no association at current time step)."""
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.TENTATIVE_DELETED
        elif self.time_since_update > self._max_age:
            self.state = TrackState.CONFIRMED_DELETED

    def is_tentative(self) -> bool:
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.TENTATIVE

    def is_confirmed(self) -> bool:
        """Returns True if this track is confirmed."""
        return self.state == TrackState.CONFIRMED

    def is_deleted(self) -> bool:
        """Returns True if this track is dead and should be deleted."""
        return self.state in (
            TrackState.TENTATIVE_DELETED,
            TrackState.CONFIRMED_DELETED,
        )

    @property
    def tlwh(self) -> npt.NDArray[np.float32]:
        """Get current position in bounding box format (top left x, top left y, width, height)."""
        if self.mean is None:
            return self._tlwh.copy()

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> npt.NDArray[np.float32]:
        """Get current position in bounding box format (min x, min y, max x, max y)."""
        ret = self.tlwh.copy()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Convert bounding box to format (center x, center y, aspect ratio, height)."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self) -> npt.NDArray[np.float32]:
        """Convert current position to xyah format."""
        return self.tlwh_to_xyah(self.tlwh)

    def __repr__(self) -> str:
        return f"SORTTrack_{self.track_id}_(hits:{self.hits}, age:{self.age})"

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from supervision.tracker.byte_tracker.utils import IdCounter


class CentroidTrack:
    """
    A simple track that stores centroid history for centroid-based tracking.
    """

    def __init__(
        self,
        centroid: npt.NDArray[np.float32],
        xyxy: npt.NDArray[np.float32],
        confidence: float,
        class_id: int,
        id_counter: IdCounter,
    ):
        """
        Initialize a centroid track.

        Args:
            centroid: Center point (x, y) of the detection.
            xyxy: Bounding box in format (min x, min y, max x, max y).
            confidence: Detection confidence score.
            class_id: Detection class ID.
            id_counter: Counter for generating unique track IDs.
        """
        self.centroid = np.asarray(centroid, dtype=np.float32)
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.confidence = confidence
        self.class_id = class_id
        self.track_id = id_counter.new_id()
        self.disappeared = 0

    def update(
        self,
        centroid: npt.NDArray[np.float32],
        xyxy: npt.NDArray[np.float32],
        confidence: float,
        class_id: int,
    ) -> None:
        """
        Update track with new detection.

        Args:
            centroid: New center point.
            xyxy: New bounding box.
            confidence: New confidence score.
            class_id: New class ID.
        """
        self.centroid = np.asarray(centroid, dtype=np.float32)
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.confidence = confidence
        self.class_id = class_id
        self.disappeared = 0

    def mark_disappeared(self) -> None:
        """Mark this track as having disappeared in the current frame."""
        self.disappeared += 1

    def __repr__(self) -> str:
        return (
            f"CentroidTrack_{self.track_id}_(disappeared:{self.disappeared})"
        )

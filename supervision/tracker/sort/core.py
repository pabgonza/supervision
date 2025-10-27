from __future__ import annotations

import numpy as np

from supervision.detection.core import Detections
from supervision.detection.utils.iou_and_nms import box_iou_batch
from supervision.tracker.byte_tracker import matching
from supervision.tracker.byte_tracker.kalman_filter import KalmanFilter
from supervision.tracker.byte_tracker.utils import IdCounter
from supervision.tracker.sort.track import SORTTrack, TrackState


class SORT:
    """
    Simple Online and Realtime Tracking (SORT) algorithm.

    SORT uses Kalman filtering for state prediction and Hungarian algorithm
    for data association based on intersection-over-union (IoU) distance.

    Args:
        max_age: Maximum number of frames to keep alive a track without detections.
        min_hits: Minimum number of associated detections before track is confirmed.
        iou_threshold: Minimum IoU threshold for matching detections to tracks.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.frame_id = 0
        self.kalman_filter = KalmanFilter()
        self.shared_kalman = KalmanFilter()

        self.tracks: list[SORTTrack] = []
        self.id_counter = IdCounter(start_id=1)

    def update_with_detections(self, detections: Detections) -> Detections:
        """
        Updates the tracker with the provided detections.

        Args:
            detections: The detections to process.

        Returns:
            Updated detections with tracker_id assigned.
        """
        self.frame_id += 1

        # Predict new locations of existing tracks
        SORTTrack.multi_predict(self.tracks, self.shared_kalman)

        if len(detections) == 0:
            # Mark all tracks as missed
            for track in self.tracks:
                track.mark_missed()
            self._remove_dead_tracks()
            return Detections.empty()

        # Match detections to tracks
        matches, unmatched_tracks, unmatched_detections = self._associate(detections)

        # Update matched tracks
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kalman_filter,
                detections.xyxy[detection_idx],
                detections.confidence[detection_idx],
                detections.class_id[detection_idx]
                if detections.class_id is not None
                else 0,
                self.frame_id,
            )

        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            self._initiate_track(detections, detection_idx)

        # Remove dead tracks
        self._remove_dead_tracks()

        # Return detections with tracker IDs
        return self._assign_tracker_ids(detections)

    def _associate(
        self, detections: Detections
    ) -> tuple[np.ndarray, tuple, tuple]:
        """Associate detections to existing tracks using IoU distance."""
        if len(self.tracks) == 0:
            return (
                np.empty((0, 2), dtype=int),
                tuple(),
                tuple(range(len(detections))),
            )

        # Get track boxes
        track_boxes = np.array([track.tlbr for track in self.tracks])

        # Calculate IoU cost matrix
        iou_matrix = matching.iou_distance(track_boxes, detections.xyxy)

        # Use Hungarian algorithm for matching
        matches, unmatched_tracks, unmatched_detections = matching.linear_assignment(
            iou_matrix, self.iou_threshold
        )

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(
        self, detections: Detections, detection_idx: int
    ) -> None:
        """Create a new track from a detection."""
        xyxy = detections.xyxy[detection_idx]
        confidence = detections.confidence[detection_idx]
        class_id = (
            detections.class_id[detection_idx]
            if detections.class_id is not None
            else 0
        )

        # Convert xyxy to tlwh
        tlwh = np.array(
            [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
        )

        track = SORTTrack(
            tlwh=tlwh,
            confidence=confidence,
            class_id=class_id,
            n_init=self.min_hits,
            max_age=self.max_age,
            id_counter=self.id_counter,
        )
        track.activate(self.kalman_filter, self.frame_id)
        self.tracks.append(track)

    def _remove_dead_tracks(self) -> None:
        """Remove tracks that are marked as deleted."""
        self.tracks = [track for track in self.tracks if not track.is_deleted()]

    def _assign_tracker_ids(self, detections: Detections) -> Detections:
        """Assign tracker IDs to detections based on confirmed tracks."""
        if len(detections) == 0:
            return detections

        # Get confirmed tracks only
        confirmed_tracks = [
            track for track in self.tracks if track.is_confirmed()
        ]

        if len(confirmed_tracks) == 0:
            return Detections.empty()

        # Match detections to confirmed tracks
        track_boxes = np.array([track.tlbr for track in confirmed_tracks])
        ious = box_iou_batch(detections.xyxy, track_boxes)
        iou_costs = 1 - ious

        matches, _, _ = matching.linear_assignment(iou_costs, 0.5)

        # Initialize tracker_id with -1
        detections.tracker_id = np.full(len(detections), -1, dtype=int)

        # Assign tracker IDs to matched detections
        for detection_idx, track_idx in matches:
            detections.tracker_id[detection_idx] = confirmed_tracks[
                track_idx
            ].track_id

        # Filter out detections without valid track (consistent with ByteTrack)
        return detections[detections.tracker_id != -1]

    def reset(self) -> None:
        """Reset the tracker state."""
        self.frame_id = 0
        self.tracks = []
        self.id_counter.reset()

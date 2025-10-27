from __future__ import annotations

import numpy as np
from scipy.spatial import distance as dist

from supervision.detection.core import Detections
from supervision.tracker.byte_tracker.utils import IdCounter
from supervision.tracker.centroid.track import CentroidTrack


class CentroidTracker:
    """
    Centroid-based object tracker.

    Tracks objects by computing Euclidean distances between centroids
    of detections across frames. Simple and fast but less robust than
    Kalman-based trackers.

    Args:
        max_disappeared: Maximum number of frames a track can disappear
            before being deregistered.
        max_distance: Maximum Euclidean distance for associating detections
            to existing tracks.
    """

    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 50.0,
    ):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        self.tracks: dict[int, CentroidTrack] = {}
        self.id_counter = IdCounter(start_id=1)

    def update_with_detections(self, detections: Detections) -> Detections:
        """
        Updates the tracker with the provided detections.

        Args:
            detections: The detections to process.

        Returns:
            Updated detections with tracker_id assigned.
        """
        # If no detections, mark all tracks as disappeared
        if len(detections) == 0:
            for track in self.tracks.values():
                track.mark_disappeared()
            self._deregister_disappeared_tracks()
            return Detections.empty()

        # Calculate centroids for current detections
        centroids = self._compute_centroids(detections.xyxy)

        # If no existing tracks, register all detections as new tracks
        if len(self.tracks) == 0:
            for i in range(len(detections)):
                self._register_track(
                    centroids[i],
                    detections.xyxy[i],
                    detections.confidence[i],
                    detections.class_id[i]
                    if detections.class_id is not None
                    else 0,
                )
        else:
            # Match detections to existing tracks
            matches, unmatched_detections = self._associate(centroids)

            # Update matched tracks
            for track_id, detection_idx in matches.items():
                self.tracks[track_id].update(
                    centroids[detection_idx],
                    detections.xyxy[detection_idx],
                    detections.confidence[detection_idx],
                    detections.class_id[detection_idx]
                    if detections.class_id is not None
                    else 0,
                )

            # Mark unmatched tracks as disappeared
            for track_id in self.tracks:
                if track_id not in matches:
                    self.tracks[track_id].mark_disappeared()

            # Register new tracks for unmatched detections
            for detection_idx in unmatched_detections:
                self._register_track(
                    centroids[detection_idx],
                    detections.xyxy[detection_idx],
                    detections.confidence[detection_idx],
                    detections.class_id[detection_idx]
                    if detections.class_id is not None
                    else 0,
                )

        # Deregister tracks that have disappeared for too long
        self._deregister_disappeared_tracks()

        # Assign tracker IDs to detections
        return self._assign_tracker_ids(detections, centroids)

    def _compute_centroids(
        self, xyxy: np.ndarray
    ) -> np.ndarray:
        """Compute centroids from bounding boxes."""
        centroids = np.zeros((len(xyxy), 2), dtype=np.float32)
        centroids[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2.0  # center x
        centroids[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2.0  # center y
        return centroids

    def _associate(
        self, detection_centroids: np.ndarray
    ) -> tuple[dict[int, int], list[int]]:
        """
        Associate detections to tracks using Euclidean distance.

        Returns:
            Tuple of (matches dict, list of unmatched detection indices)
        """
        track_ids = list(self.tracks.keys())
        track_centroids = np.array(
            [self.tracks[tid].centroid for tid in track_ids]
        )

        # Compute Euclidean distance matrix
        D = dist.cdist(track_centroids, detection_centroids)

        # Find the minimum distance for each track
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()
        matches = {}

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            # Check if distance is within threshold
            if D[row, col] > self.max_distance:
                continue

            track_id = track_ids[row]
            matches[track_id] = col
            used_rows.add(row)
            used_cols.add(col)

        # Find unmatched detections
        unmatched_detections = [
            i for i in range(len(detection_centroids)) if i not in used_cols
        ]

        return matches, unmatched_detections

    def _register_track(
        self,
        centroid: np.ndarray,
        xyxy: np.ndarray,
        confidence: float,
        class_id: int,
    ) -> None:
        """Register a new track."""
        track = CentroidTrack(
            centroid=centroid,
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            id_counter=self.id_counter,
        )
        self.tracks[track.track_id] = track

    def _deregister_disappeared_tracks(self) -> None:
        """Remove tracks that have disappeared for too long."""
        to_delete = []
        for track_id, track in self.tracks.items():
            if track.disappeared > self.max_disappeared:
                to_delete.append(track_id)

        for track_id in to_delete:
            del self.tracks[track_id]

    def _assign_tracker_ids(
        self, detections: Detections, centroids: np.ndarray
    ) -> Detections:
        """Assign tracker IDs to detections."""
        if len(detections) == 0:
            return detections

        if len(self.tracks) == 0:
            return Detections.empty()

        # Initialize tracker_id with -1
        detections.tracker_id = np.full(len(detections), -1, dtype=int)

        # Get track centroids
        track_ids = list(self.tracks.keys())
        track_centroids = np.array(
            [self.tracks[tid].centroid for tid in track_ids]
        )

        # Match detections to tracks by centroid distance
        D = dist.cdist(centroids, track_centroids)

        for detection_idx in range(len(centroids)):
            if len(track_ids) == 0:
                break

            # Find nearest track
            nearest_track_idx = D[detection_idx].argmin()
            min_distance = D[detection_idx, nearest_track_idx]

            # Assign tracker ID if within threshold
            if min_distance <= self.max_distance:
                detections.tracker_id[detection_idx] = track_ids[
                    nearest_track_idx
                ]

        # Filter out detections without valid track (consistent with ByteTrack)
        return detections[detections.tracker_id != -1]

    def reset(self) -> None:
        """Reset the tracker state."""
        self.tracks = {}
        self.id_counter.reset()

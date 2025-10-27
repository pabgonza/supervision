from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from supervision.detection.core import Detections
from supervision.detection.line_zone import LineZone
from supervision.geometry.core import Point, Position
from supervision.tracker.byte_tracker.core import ByteTrack


class ByteTrackerStep:
    """
    Pipeline step for object tracking using ByteTrack algorithm.

    Tracks detections across frames and adds tracker_id to each detection.
    This step requires detections to already be present in the data dictionary.

    Examples:
        ```python
        import supervision as sv

        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | sv.YOLODetectionStep("yolov8n.pt", conf=0.5)
            | sv.ByteTrackerStep()
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("Tracking")
        )
        pipeline.run()
        ```

        ```python
        # Custom tracker configuration
        import supervision as sv

        yolo_step = sv.YOLODetectionStep("yolov8n.pt")

        pipeline = (
            sv.Pipeline(sv.VideoFileSource("video.mp4"))
            | yolo_step
            | sv.ByteTrackerStep(
                track_activation_threshold=0.3,
                lost_track_buffer=60,
                minimum_matching_threshold=0.85
            )
            | sv.TrackerAnnotatorStep(class_names=yolo_step.model.names)
            | sv.DisplaySink("Object Tracking")
        )
        ```
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        minimum_consecutive_frames: int = 1,
        detections_key: str = "detections",
        metrics_key: str = "tracker_metrics",
    ):
        """
        Initialize ByteTrack tracking step.

        Args:
            track_activation_threshold: Detection confidence threshold for track
                activation. Increasing this improves accuracy and stability but might
                miss true detections. Decreasing increases completeness but risks
                introducing noise.
            lost_track_buffer: Number of frames to buffer when a track is lost.
                Increasing this enhances occlusion handling and reduces track
                fragmentation from brief detection gaps.
            minimum_matching_threshold: Threshold for matching tracks with detections.
                Increasing improves accuracy but risks fragmentation. Decreasing
                improves completeness but risks false positives and drift.
            frame_rate: Frame rate of the video being processed.
            minimum_consecutive_frames: Number of consecutive frames an object must
                be tracked before considered a 'valid' track. Increasing prevents
                accidental tracks from false detections but risks missing shorter
                tracks.
            detections_key: Key in data dict containing Detections object
                (default: 'detections')
            metrics_key: Key to store timing metrics in data dict
                (default: 'tracker_metrics')
        """

        self.detections_key = detections_key
        self.metrics_key = metrics_key
        self.tracker = ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Track detections across frames.

        Args:
            data: Pipeline data containing detections

        Returns:
            Data with updated detections containing tracker_id field
        """
        detections = data.get(self.detections_key)

        if detections is None or len(detections) == 0:
            data[self.metrics_key] = {"processing_time_ms": 0.0}
            return data

        # Measure processing time
        start_time = time.perf_counter()

        # Update tracker with detections
        tracked_detections = self.tracker.update_with_detections(detections)

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - start_time

        # Update data with tracked detections
        data[self.detections_key] = tracked_detections

        # Add tracking metrics to data dict
        data[self.metrics_key] = {"processing_time_ms": elapsed_time * 1000}

        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Process if detections exist."""
        return self.detections_key in data

    def reset(self) -> None:
        """
        Reset the tracker state.

        Useful when processing multiple videos sequentially or when you need
        to restart tracking from scratch.
        """
        self.tracker.reset()


class LineZoneStep:
    """
    Pipeline step for counting objects crossing a line.

    Uses LineZone to track object crossings and maintain in/out counts.
    Requires detections with tracker_id (use ByteTrackerStep before this step).

    Examples:
        ```python
        import supervision as sv

        # Define line crossing zone
        start = sv.Point(x=0, y=500)
        end = sv.Point(x=1920, y=500)

        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | sv.YOLODetectionStep("yolov8n.pt")
            | sv.ByteTrackerStep()
            | sv.LineZoneStep(start=start, end=end)
            | sv.LineZoneAnnotatorStep()
            | sv.DisplaySink("Line Counter")
        )
        pipeline.run()
        ```

        ```python
        # Access crossing counts
        import supervision as sv

        start = sv.Point(x=0, y=500)
        end = sv.Point(x=1920, y=500)

        pipeline = (
            sv.Pipeline(sv.VideoFileSource("video.mp4"))
            | sv.YOLODetectionStep("yolov8n.pt")
            | sv.ByteTrackerStep()
            | sv.LineZoneStep(start=start, end=end)
        )

        for data in pipeline:
            line_zone = data["line_zone"]
            print(f"In: {line_zone.in_count}, Out: {line_zone.out_count}")
            print(f"IDs crossed in: {line_zone.last_crossed_in_ids}")
            print(f"IDs crossed out: {line_zone.last_crossed_out_ids}")
        ```
    """

    def __init__(
        self,
        start: Point,
        end: Point,
        triggering_anchors: Iterable[Position] = (Position.CENTER,),
        minimum_crossing_threshold: int = 1,
        detections_key: str = "detections",
        line_zone_key: str = "line_zone",
    ):
        """
        Initialize line zone counting step.

        Args:
            start: Starting point of the line
            end: Ending point of the line
            triggering_anchors: List of detection anchor positions to consider
                for crossing detection (default: CENTER)
            minimum_crossing_threshold: Number of frames detection must be
                on other side to count as crossing (default: 1)
            detections_key: Key in data dict containing Detections object
                (default: 'detections')
            line_zone_key: Key to store LineZone object in data dict
                (default: 'line_zone')
        """
        self.detections_key = detections_key
        self.line_zone_key = line_zone_key
        self.line_zone = LineZone(
            start=start,
            end=end,
            triggering_anchors=triggering_anchors,
            minimum_crossing_threshold=minimum_crossing_threshold,
        )

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Count objects crossing the line.

        Args:
            data: Pipeline data containing detections with tracker_id

        Returns:
            Data with line_zone object added. Access counts and crossing IDs via:
            - data["line_zone"].in_count
            - data["line_zone"].out_count
            - data["line_zone"].in_count_per_class
            - data["line_zone"].out_count_per_class
            - data["line_zone"].last_crossed_in_ids
            - data["line_zone"].last_crossed_out_ids
        """
        detections = data.get(self.detections_key)

        if detections is not None and len(detections) > 0:
            self.line_zone.trigger(detections)
        else:
            self.line_zone.trigger(data.get(self.detections_key, Detections.empty()))

        data[self.line_zone_key] = self.line_zone

        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Process if detections exist."""
        return self.detections_key in data

    def reset(self) -> None:
        """
        Reset the line zone counts.

        Useful when starting a new video or resetting the counting state.
        """
        # Reset internal counters
        self.line_zone._in_count_per_class.clear()
        self.line_zone._out_count_per_class.clear()
        self.line_zone.crossing_state_history.clear()


class SORTTrackerStep:
    """
    Pipeline step for object tracking using SORT (Simple Online and Realtime Tracking).

    SORT uses Kalman filtering for state prediction and Hungarian algorithm
    for data association. It's simpler and faster than ByteTrack but may be
    less robust in crowded scenes.

    Examples:
        ```python
        import supervision as sv

        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | sv.YOLODetectionStep("yolov8n.pt", conf=0.5)
            | sv.SORTTrackerStep()
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("SORT Tracking")
        )
        pipeline.run()
        ```

        ```python
        # Custom tracker configuration
        import supervision as sv

        yolo_step = sv.YOLODetectionStep("yolov8n.pt")

        pipeline = (
            sv.Pipeline(sv.VideoFileSource("video.mp4"))
            | yolo_step
            | sv.SORTTrackerStep(
                lost_track_buffer=30,
                minimum_consecutive_frames=3,
                iou_threshold=0.3
            )
            | sv.TrackerAnnotatorStep(class_names=yolo_step.model.names)
            | sv.DisplaySink("SORT Tracking")
        )
        ```
    """

    def __init__(
        self,
        lost_track_buffer: int = 30,
        minimum_consecutive_frames: int = 3,
        iou_threshold: float = 0.3,
        detections_key: str = "detections",
        metrics_key: str = "tracker_metrics",
    ):
        """
        Initialize SORT tracking step.

        Args:
            lost_track_buffer: Number of frames to buffer when a track is lost.
            minimum_consecutive_frames: Minimum number of consecutive frames that an object
                must be tracked before it is considered a valid track.
            iou_threshold: Minimum IoU for matching detections to tracks.
            detections_key: Key in data dict containing Detections object
                (default: 'detections')
            metrics_key: Key to store timing metrics in data dict
                (default: 'tracker_metrics')
        """
        from supervision.tracker.sort import SORT

        self.detections_key = detections_key
        self.metrics_key = metrics_key
        self.tracker = SORT(
            lost_track_buffer=lost_track_buffer,
            minimum_consecutive_frames=minimum_consecutive_frames,
            iou_threshold=iou_threshold,
        )

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Track detections across frames.

        Args:
            data: Pipeline data containing detections

        Returns:
            Data with updated detections containing tracker_id field
        """
        detections = data.get(self.detections_key)

        if detections is None or len(detections) == 0:
            data[self.metrics_key] = {"processing_time_ms": 0.0}
            return data

        # Measure processing time
        start_time = time.perf_counter()

        # Update tracker with detections
        tracked_detections = self.tracker.update_with_detections(detections)

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - start_time

        # Update data with tracked detections
        data[self.detections_key] = tracked_detections

        # Add tracking metrics to data dict
        data[self.metrics_key] = {"processing_time_ms": elapsed_time * 1000}

        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Process if detections exist."""
        return self.detections_key in data

    def reset(self) -> None:
        """
        Reset the tracker state.

        Useful when processing multiple videos sequentially or when you need
        to restart tracking from scratch.
        """
        self.tracker.reset()


class CentroidTrackerStep:
    """
    Pipeline step for object tracking using centroid-based tracking.

    Tracks objects by computing Euclidean distances between centroids of
    detections across frames. Simple and fast but less robust than Kalman-based
    trackers in handling occlusions and complex motion patterns.

    Examples:
        ```python
        import supervision as sv

        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | sv.YOLODetectionStep("yolov8n.pt", conf=0.5)
            | sv.CentroidTrackerStep()
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("Centroid Tracking")
        )
        pipeline.run()
        ```

        ```python
        # Custom tracker configuration
        import supervision as sv

        yolo_step = sv.YOLODetectionStep("yolov8n.pt")

        pipeline = (
            sv.Pipeline(sv.VideoFileSource("video.mp4"))
            | yolo_step
            | sv.CentroidTrackerStep(
                lost_track_buffer=50,
                max_distance=100.0
            )
            | sv.TrackerAnnotatorStep(class_names=yolo_step.model.names)
            | sv.DisplaySink("Centroid Tracking")
        )
        ```
    """

    def __init__(
        self,
        lost_track_buffer: int = 30,
        max_distance: float = 50.0,
        detections_key: str = "detections",
        metrics_key: str = "tracker_metrics",
    ):
        """
        Initialize centroid tracking step.

        Args:
            lost_track_buffer: Number of frames to buffer when a track is lost.
            max_distance: Maximum Euclidean distance for associating detections
                to existing tracks.
            detections_key: Key in data dict containing Detections object
                (default: 'detections')
            metrics_key: Key to store timing metrics in data dict
                (default: 'tracker_metrics')
        """
        from supervision.tracker.centroid import CentroidTracker

        self.detections_key = detections_key
        self.metrics_key = metrics_key
        self.tracker = CentroidTracker(
            lost_track_buffer=lost_track_buffer,
            max_distance=max_distance,
        )

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Track detections across frames.

        Args:
            data: Pipeline data containing detections

        Returns:
            Data with updated detections containing tracker_id field
        """
        detections = data.get(self.detections_key)

        if detections is None or len(detections) == 0:
            data[self.metrics_key] = {"processing_time_ms": 0.0}
            return data

        # Measure processing time
        start_time = time.perf_counter()

        # Update tracker with detections
        tracked_detections = self.tracker.update_with_detections(detections)

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - start_time

        # Update data with tracked detections
        data[self.detections_key] = tracked_detections

        # Add tracking metrics to data dict
        data[self.metrics_key] = {"processing_time_ms": elapsed_time * 1000}

        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Process if detections exist."""
        return self.detections_key in data

    def reset(self) -> None:
        """
        Reset the tracker state.

        Useful when processing multiple videos sequentially or when you need
        to restart tracking from scratch.
        """
        self.tracker.reset()

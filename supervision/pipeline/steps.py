from __future__ import annotations

from typing import Any, Callable

import numpy as np

from supervision.annotators.core import BaseAnnotator
from supervision.detection.core import Detections
from supervision.utils.image import resize_image
from supervision.utils.video import FPSMonitor


class FPSCalculatorStep:
    """
    Pipeline step to calculate and add FPS to frame data.

    Tracks frame processing rate and adds 'fps' field to data.

    Examples:
        ```python
        import supervision as sv

        pipeline = (
            sv.Pipeline(source)
            | sv.FPSCalculatorStep()
            | sv.DisplaySink("Video")
        )
        ```
    """

    def __init__(self, window_size: int = 30):
        """
        Initialize FPS calculator.

        Args:
            window_size: Number of frames to average for FPS calculation
        """
        self.fps_monitor = FPSMonitor(sample_size=window_size)

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate FPS and add to data.

        Args:
            data: Pipeline data dictionary

        Returns:
            Data with 'fps' field added
        """
        self.fps_monitor.tick()
        data["fps"] = self.fps_monitor.fps
        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Always process."""
        return True


class AnnotationStep:
    """
    Pipeline step to annotate frames using supervision annotators.

    Applies one or more annotators to frames with detections.

    Examples:
        ```python
        import supervision as sv

        annotator = sv.BoxAnnotator()
        pipeline = (
            sv.Pipeline(source)
            | sv.DetectionStep(model)
            | sv.AnnotationStep(annotator)
            | sv.DisplaySink("Detections")
        )
        ```
    """

    def __init__(
        self,
        annotators: BaseAnnotator | list[BaseAnnotator],
        detections_key: str = "detections",
        output_key: str = "frame",
    ):
        """
        Initialize annotation step.

        Args:
            annotators: Single annotator or list of annotators to apply
            detections_key: Key in data dict containing Detections object
            output_key: Key to store annotated frame (default: overwrites 'frame')
        """
        self.annotators = annotators if isinstance(annotators, list) else [annotators]
        self.detections_key = detections_key
        self.output_key = output_key

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Annotate frame with detections.

        Args:
            data: Pipeline data containing 'frame' and detections

        Returns:
            Data with annotated frame
        """
        frame = data.get("frame")
        detections = data.get(self.detections_key)

        if frame is None:
            return data

        # Make a copy if we might modify the original
        if self.output_key == "frame":
            annotated_frame = frame.copy()
        else:
            annotated_frame = frame

        # Apply all annotators
        if detections is not None and isinstance(detections, Detections):
            for annotator in self.annotators:
                annotated_frame = annotator.annotate(
                    scene=annotated_frame, detections=detections
                )

        data[self.output_key] = annotated_frame
        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Process if frame exists."""
        return "frame" in data


class FilterStep:
    """
    Generic pipeline step to filter data based on a predicate function.

    Examples:
        ```python
        import supervision as sv

        # Only process frames with detections
        def has_detections(data):
            detections = data.get('detections')
            return detections is not None and len(detections) > 0

        pipeline = (
            sv.Pipeline(source)
            | sv.DetectionStep(model)
            | sv.FilterStep(has_detections)
            | sv.AnnotationStep(annotator)
            | sv.DisplaySink("Filtered")
        )
        ```
    """

    def __init__(self, predicate: Callable[[dict[str, Any]], bool]):
        """
        Initialize filter step.

        Args:
            predicate: Function that returns True if data should be processed
        """
        self.predicate = predicate

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Pass through data unchanged."""
        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Apply predicate function."""
        return self.predicate(data)


class TransformStep:
    """
    Generic pipeline step to transform data using a function.

    Examples:
        ```python
        import supervision as sv
        import cv2

        # Convert to grayscale
        def to_grayscale(data):
            frame = data['frame']
            data['frame'] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return data

        pipeline = (
            sv.Pipeline(source)
            | sv.TransformStep(to_grayscale)
            | sv.DisplaySink("Grayscale")
        )
        ```
    """

    def __init__(self, transform_func: Callable[[dict[str, Any]], dict[str, Any]]):
        """
        Initialize transform step.

        Args:
            transform_func: Function to transform data dictionary
        """
        self.transform_func = transform_func

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply transformation function."""
        return self.transform_func(data)

    def filter(self, data: dict[str, Any]) -> bool:
        """Always process."""
        return True


class ResizeStep:
    """
    Pipeline step to resize frames.

    Examples:
        ```python
        import supervision as sv

        pipeline = (
            sv.Pipeline(source)
            | sv.ResizeStep(width=1280, height=720)
            | sv.DisplaySink("Resized")
        )
        ```
    """

    def __init__(
        self,
        width: int | None = None,
        height: int | None = None,
        keep_aspect_ratio: bool = True,
    ):
        """
        Initialize resize step.

        Args:
            width: Target width (None to auto-calculate)
            height: Target height (None to auto-calculate)
            keep_aspect_ratio: Whether to maintain aspect ratio
        """
        self.width = width
        self.height = height
        self.keep_aspect_ratio = keep_aspect_ratio

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Resize frame.

        Args:
            data: Pipeline data containing 'frame'

        Returns:
            Data with resized frame
        """
        frame = data.get("frame")
        if frame is None:
            return data

        resized = resize_image(
            image=frame,
            resolution_wh=(self.width, self.height),
            keep_aspect_ratio=self.keep_aspect_ratio,
        )

        data["frame"] = resized
        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Process if frame exists."""
        return "frame" in data


class DetectionFilterStep:
    """
    Pipeline step to filter detections based on criteria.

    Examples:
        ```python
        import supervision as sv

        # Only keep person detections with confidence > 0.7
        pipeline = (
            sv.Pipeline(source)
            | sv.DetectionStep(model)
            | sv.DetectionFilterStep(
                class_ids=[0],  # person class
                min_confidence=0.7
            )
            | sv.AnnotationStep(annotator)
        )
        ```
    """

    def __init__(
        self,
        class_ids: list[int] | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
        detections_key: str = "detections",
    ):
        """
        Initialize detection filter step.

        Args:
            class_ids: List of class IDs to keep (None = keep all)
            min_confidence: Minimum confidence threshold
            max_confidence: Maximum confidence threshold
            detections_key: Key in data dict containing Detections
        """
        self.class_ids = class_ids
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.detections_key = detections_key

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Filter detections.

        Args:
            data: Pipeline data containing detections

        Returns:
            Data with filtered detections
        """
        detections: Detections = data.get(self.detections_key)

        if detections is None or len(detections) == 0:
            return data

        # Filter by class ID
        if self.class_ids is not None and detections.class_id is not None:
            mask = np.isin(detections.class_id, self.class_ids)
            detections = detections[mask]

        # Filter by confidence
        if self.min_confidence is not None and detections.confidence is not None:
            mask = detections.confidence >= self.min_confidence
            detections = detections[mask]

        if self.max_confidence is not None and detections.confidence is not None:
            mask = detections.confidence <= self.max_confidence
            detections = detections[mask]

        data[self.detections_key] = detections
        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Always process."""
        return True


class CallbackStep:
    """
    Pipeline step that executes a callback function.

    Useful for custom processing, logging, or side effects.

    Examples:
        ```python
        import supervision as sv

        def log_detections(data):
            detections = data.get('detections')
            if detections:
                print(f"Frame {data['frame_number']}: {len(detections)} detections")

        pipeline = (
            sv.Pipeline(source)
            | sv.DetectionStep(model)
            | sv.CallbackStep(log_detections)
            | sv.DisplaySink("Video")
        )
        ```
    """

    def __init__(self, callback: Callable[[dict[str, Any]], None]):
        """
        Initialize callback step.

        Args:
            callback: Function to call with data (doesn't modify data)
        """
        self.callback = callback

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute callback and pass through data."""
        self.callback(data)
        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Always process."""
        return True


class YOLODetectionStep:
    """
    Pipeline step for YOLO object detection using Ultralytics.

    Runs YOLO inference on frames and converts results to supervision Detections.

    Examples:
        ```python
        import supervision as sv

        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | sv.YOLODetectionStep("yolov8n.pt", conf=0.5)
            | sv.AnnotationStep(sv.BoxAnnotator())
            | sv.DisplaySink("Detections")
        )
        pipeline.run()
        ```
    """

    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cuda",
        verbose: bool = False,
    ):
        """
        Initialize YOLO detection step.

        Args:
            model_path: Path to YOLO model file (.pt)
            conf: Confidence threshold for detections
            iou: IOU threshold for NMS
            device: Device for inference ('cuda' or 'cpu')
            verbose: Whether to print verbose output
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLODetectionStep. "
                "Install it with: pip install ultralytics"
            )

        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device = device
        self.verbose = verbose

        # Load model
        self.model = YOLO(model_path)
        self.model.to(device)

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Run YOLO inference on frame.

        Args:
            data: Pipeline data containing 'frame'

        Returns:
            Data with 'detections' field added
        """
        frame = data.get("frame")
        if frame is None:
            return data

        # Run inference
        results = self.model.predict(
            source=frame, conf=self.conf, iou=self.iou, verbose=self.verbose
        )

        # Convert to supervision Detections
        data["detections"] = Detections.from_ultralytics(results[0])

        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Process if frame exists."""
        return "frame" in data


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
            | sv.AnnotationStep(sv.BoxAnnotator())
            | sv.DisplaySink("Tracking")
        )
        pipeline.run()
        ```

        ```python
        # Custom tracker configuration
        import supervision as sv

        pipeline = (
            sv.Pipeline(sv.VideoFileSource("video.mp4"))
            | sv.YOLODetectionStep("yolov8n.pt")
            | sv.ByteTrackerStep(
                track_activation_threshold=0.3,
                lost_track_buffer=60,
                minimum_matching_threshold=0.85
            )
            | sv.AnnotationStep([
                sv.BoxAnnotator(),
                sv.LabelAnnotator()
            ])
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
        """
        from supervision.tracker.byte_tracker.core import ByteTrack

        self.detections_key = detections_key
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
            return data

        # Update tracker with detections
        tracked_detections = self.tracker.update_with_detections(detections)

        # Update data with tracked detections
        data[self.detections_key] = tracked_detections

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

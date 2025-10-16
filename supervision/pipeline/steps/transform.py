from __future__ import annotations

from typing import Any, Callable

import numpy as np

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
            | sv.BoxAnnotatorStep()
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
        input_key: str = "frame",
        output_key: str = "frame",
    ):
        """
        Initialize resize step.

        Args:
            width: Target width (None to auto-calculate)
            height: Target height (None to auto-calculate)
            keep_aspect_ratio: Whether to maintain aspect ratio
            input_key: Key in data dict containing input frame (default: 'frame')
            output_key: Key to store resized frame (default: 'frame')
        """
        self.width = width
        self.height = height
        self.keep_aspect_ratio = keep_aspect_ratio
        self.input_key = input_key
        self.output_key = output_key

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Resize frame.

        Args:
            data: Pipeline data containing input frame

        Returns:
            Data with resized frame
        """
        frame = data.get(self.input_key)
        if frame is None:
            return data

        resized = resize_image(
            image=frame,
            resolution_wh=(self.width, self.height),
            keep_aspect_ratio=self.keep_aspect_ratio,
        )

        data[self.output_key] = resized
        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Process if frame exists."""
        return self.input_key in data


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
            | sv.BoxAnnotatorStep()
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


class LabelFormatterStep:
    """
    Pipeline step to automatically generate labels for detections.

    Creates formatted labels from detection information including tracker ID,
    class name, and confidence score. The generated labels are stored in the
    data dictionary under the 'labels' key for use by LabelAnnotatorStep.

    Examples:
        ```python
        import supervision as sv

        yolo_step = sv.YOLODetectionStep("yolov8n.pt")

        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | yolo_step
            | sv.ByteTrackerStep()
            | sv.LabelFormatterStep(class_names=yolo_step.model.names)
            | sv.LabelAnnotatorStep()
            | sv.DisplaySink("Labeled Detections")
        )
        pipeline.run()
        ```

        ```python
        # Custom label format
        import supervision as sv

        yolo_step = sv.YOLODetectionStep("yolov8n.pt")

        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | yolo_step
            | sv.LabelFormatterStep(
                class_names=yolo_step.model.names,
                show_tracker_id=True,
                show_confidence=True,
                confidence_decimals=3
            )
            | sv.LabelAnnotatorStep()
            | sv.DisplaySink("Formatted Labels")
        )
        ```
    """

    def __init__(
        self,
        class_names: dict[int, str] | None = None,
        show_tracker_id: bool = True,
        show_class: bool = True,
        show_confidence: bool = True,
        confidence_decimals: int = 2,
        detections_key: str = "detections",
        labels_key: str = "labels",
    ):
        """
        Initialize label formatter step.

        Args:
            class_names: Dictionary mapping class IDs to class names.
                If None, class IDs will be used instead of names.
            show_tracker_id: Whether to include tracker ID in labels
            show_class: Whether to include class name/ID in labels
            show_confidence: Whether to include confidence score in labels
            confidence_decimals: Number of decimal places for confidence
            detections_key: Key in data dict containing Detections object
            labels_key: Key to store generated labels in data dict
        """
        self.class_names = class_names
        self.show_tracker_id = show_tracker_id
        self.show_class = show_class
        self.show_confidence = show_confidence
        self.confidence_decimals = confidence_decimals
        self.detections_key = detections_key
        self.labels_key = labels_key

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Generate labels from detections.

        Args:
            data: Pipeline data containing detections

        Returns:
            Data with generated labels added
        """
        detections = data.get(self.detections_key)

        if detections is None or len(detections) == 0:
            data[self.labels_key] = []
            return data

        labels = []
        for i in range(len(detections)):
            label_parts = []

            # Add tracker ID
            if self.show_tracker_id and detections.tracker_id is not None:
                tracker_id = detections.tracker_id[i]
                label_parts.append(f"#{tracker_id}")

            # Add class name or ID
            if self.show_class:
                if detections.class_id is not None:
                    class_id = detections.class_id[i]
                    if self.class_names and class_id in self.class_names:
                        label_parts.append(self.class_names[class_id])
                    else:
                        label_parts.append(f"class_{class_id}")

            # Add confidence
            if self.show_confidence and detections.confidence is not None:
                confidence = detections.confidence[i]
                conf_str = f"{confidence:.{self.confidence_decimals}f}"
                label_parts.append(conf_str)

            # Combine parts
            label = " ".join(label_parts) if label_parts else ""
            labels.append(label)

        data[self.labels_key] = labels
        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Always process."""
        return True

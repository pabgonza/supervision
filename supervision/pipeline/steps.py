from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, Callable

import numpy as np

from supervision.annotators.core import (
    BaseAnnotator,
    BoxAnnotator,
    LabelAnnotator,
    TraceAnnotator,
)
from supervision.annotators.utils import ColorLookup
from supervision.detection.core import Detections
from supervision.detection.line_zone import LineZone, LineZoneAnnotator
from supervision.draw.color import Color, ColorPalette
from supervision.geometry.core import Point, Position
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
            | sv.BoxAnnotatorStep()
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


class BoxAnnotatorStep:
    """
    Pipeline step for drawing bounding boxes using BoxAnnotator.

    Examples:
        ```python
        import supervision as sv

        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | sv.YOLODetectionStep("yolov8n.pt")
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("Boxes")
        )
        pipeline.run()
        ```
    """

    def __init__(
        self,
        thickness: int = 2,
        color: Any = None,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        detections_key: str = "detections",
        output_key: str = "frame",
        copy_frame: bool = True,
    ):
        """
        Initialize BoxAnnotator step.

        Args:
            thickness: Thickness of bounding box lines
            color: Color or ColorPalette for boxes
            color_lookup: Strategy for mapping colors to annotations
                (default: ColorLookup.CLASS). Options: INDEX, CLASS, TRACK
            detections_key: Key in data dict containing Detections
            output_key: Key to store annotated frame
            copy_frame: Whether to copy frame before annotating (default: True)
        """
        if color is None:
            color = ColorPalette.DEFAULT

        self.box_annotator = BoxAnnotator(
            color=color, thickness=thickness, color_lookup=color_lookup
        )
        self.detections_key = detections_key
        self.output_key = output_key
        self.copy_frame = copy_frame

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Draw bounding boxes on frame."""
        frame = data.get("frame")
        detections = data.get(self.detections_key)

        if frame is None:
            return data

        # Copy frame if requested and modifying original
        if self.copy_frame and self.output_key == "frame":
            annotated_frame = frame.copy()
        else:
            annotated_frame = frame

        if detections is not None and isinstance(detections, Detections):
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

        data[self.output_key] = annotated_frame
        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Process if frame exists."""
        return "frame" in data


class LabelAnnotatorStep:
    """
    Pipeline step for drawing labels using LabelAnnotator.

    Reads labels from data dictionary and draws them on detections.
    Use with LabelFormatterStep to generate labels automatically.

    Examples:
        ```python
        import supervision as sv

        yolo_step = sv.YOLODetectionStep("yolov8n.pt")

        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | yolo_step
            | sv.LabelFormatterStep(class_names=yolo_step.model.names)
            | sv.LabelAnnotatorStep()
            | sv.DisplaySink("Labels")
        )
        pipeline.run()
        ```
    """

    def __init__(
        self,
        text_color: Any = None,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        color: Any = None,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        detections_key: str = "detections",
        labels_key: str = "labels",
        output_key: str = "frame",
        copy_frame: bool = True,
    ):
        """
        Initialize LabelAnnotator step.

        Args:
            text_color: Color for text
            text_scale: Scale of text
            text_thickness: Thickness of text
            text_padding: Padding around text
            color: Color or ColorPalette for label background
            color_lookup: Strategy for mapping colors to annotations
                (default: ColorLookup.CLASS). Options: INDEX, CLASS, TRACK
            detections_key: Key in data dict containing Detections
            labels_key: Key in data dict containing labels list
            output_key: Key to store annotated frame
            copy_frame: Whether to copy frame before annotating (default: True)
        """
        if text_color is None:
            text_color = Color.BLACK

        if color is None:
            color = ColorPalette.DEFAULT

        self.label_annotator = LabelAnnotator(
            text_color=text_color,
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_padding=text_padding,
            color=color,
            color_lookup=color_lookup,
        )
        self.detections_key = detections_key
        self.labels_key = labels_key
        self.output_key = output_key
        self.copy_frame = copy_frame

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Draw labels on frame."""
        frame = data.get("frame")
        detections = data.get(self.detections_key)
        labels = data.get(self.labels_key)

        if frame is None:
            return data

        # Copy frame if requested and modifying original
        if self.copy_frame and self.output_key == "frame":
            annotated_frame = frame.copy()
        else:
            annotated_frame = frame

        if (
            detections is not None
            and isinstance(detections, Detections)
            and labels is not None
        ):
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

        data[self.output_key] = annotated_frame
        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Process if frame exists."""
        return "frame" in data


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


class TraceAnnotatorStep:
    """
    Pipeline step for drawing tracking traces/trails using TraceAnnotator.

    Visualizes the path of tracked objects over time. This step requires
    detections with tracker_id to be present in the data dictionary.

    Examples:
        ```python
        import supervision as sv

        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | sv.YOLODetectionStep("yolov8n.pt")
            | sv.ByteTrackerStep()
            | sv.TraceAnnotatorStep(trace_length=50)
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("Tracking with Traces")
        )
        pipeline.run()
        ```

        ```python
        # Custom trace configuration
        import supervision as sv

        pipeline = (
            sv.Pipeline(sv.VideoFileSource("video.mp4"))
            | sv.YOLODetectionStep("yolov8n.pt")
            | sv.ByteTrackerStep()
            | sv.TraceAnnotatorStep(
                trace_length=100,
                thickness=3,
                color=sv.ColorPalette.DEFAULT
            )
            | sv.DisplaySink("Object Tracking with Trails")
        )
        ```
    """

    def __init__(
        self,
        trace_length: int = 30,
        thickness: int = 2,
        color: Any = None,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        detections_key: str = "detections",
        output_key: str = "frame",
    ):
        """
        Initialize TraceAnnotator step.

        Args:
            trace_length: Number of frames to keep in the trace history
            thickness: Thickness of the trace lines
            color: Color or ColorPalette for traces (default: ColorPalette.DEFAULT)
            color_lookup: Strategy for mapping colors to annotations
                (default: ColorLookup.CLASS). Options: INDEX, CLASS, TRACK
            detections_key: Key in data dict containing Detections object
                (default: 'detections')
            output_key: Key to store annotated frame (default: overwrites 'frame')
        """
        if color is None:
            color = ColorPalette.DEFAULT

        self.trace_annotator = TraceAnnotator(
            color=color,
            trace_length=trace_length,
            thickness=thickness,
            color_lookup=color_lookup,
        )
        self.detections_key = detections_key
        self.output_key = output_key

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Draw traces on the frame.

        Args:
            data: Pipeline data containing 'frame' and detections with tracker_id

        Returns:
            Data with annotated frame showing tracking traces
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

        # Draw traces if detections with tracker_id exist
        if (
            detections is not None
            and isinstance(detections, Detections)
            and len(detections) > 0
            and hasattr(detections, "tracker_id")
            and detections.tracker_id is not None
        ):
            annotated_frame = self.trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

        data[self.output_key] = annotated_frame
        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Process if frame exists."""
        return "frame" in data


class TrackerAnnotatorStep:
    """
    Pipeline step that combines box, label, and trace annotations for object tracking.

    This is a convenience step that combines BoxAnnotatorStep, LabelFormatterStep,
    LabelAnnotatorStep, and TraceAnnotatorStep into a single unified step for
    complete tracking visualization.

    Examples:
        ```python
        import supervision as sv

        yolo_step = sv.YOLODetectionStep("yolov8n.pt")

        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | yolo_step
            | sv.ByteTrackerStep()
            | sv.TrackerAnnotatorStep(class_names=yolo_step.model.names)
            | sv.DisplaySink("Tracking")
        )
        pipeline.run()
        ```

        ```python
        # Custom configuration
        import supervision as sv

        yolo_step = sv.YOLODetectionStep("yolov8n.pt")

        pipeline = (
            sv.Pipeline(sv.VideoFileSource("video.mp4"))
            | yolo_step
            | sv.ByteTrackerStep()
            | sv.TrackerAnnotatorStep(
                class_names=yolo_step.model.names,
                box_thickness=3,
                trace_length=100,
                show_confidence=True
            )
            | sv.DisplaySink("Complete Tracking Visualization")
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
        box_thickness: int = 2,
        box_color: Any = None,
        label_text_color: Any = None,
        label_text_scale: float = 0.5,
        label_text_thickness: int = 1,
        label_text_padding: int = 10,
        label_color: Any = None,
        trace_length: int = 30,
        trace_thickness: int = 2,
        trace_color: Any = None,
        color_lookup: ColorLookup = ColorLookup.TRACK,
        detections_key: str = "detections",
        output_key: str = "frame",
        copy_frame: bool = True,
    ):
        """
        Initialize tracker annotator step.

        Args:
            class_names: Dictionary mapping class IDs to class names.
                If None, class IDs will be used instead of names.
            show_tracker_id: Whether to show tracker ID in labels
            show_class: Whether to show class name in labels
            show_confidence: Whether to show confidence score in labels
            confidence_decimals: Number of decimal places for confidence
            box_thickness: Thickness of bounding box lines
            box_color: Color or ColorPalette for boxes (default: ColorPalette.DEFAULT)
            label_text_color: Color for label text (default: Color.BLACK)
            label_text_scale: Scale of label text
            label_text_thickness: Thickness of label text
            label_text_padding: Padding around label text
            label_color: Color or ColorPalette for label background
                (default: ColorPalette.DEFAULT)
            trace_length: Number of frames to keep in trace history
            trace_thickness: Thickness of trace lines
            trace_color: Color or ColorPalette for traces (default: ColorPalette.DEFAULT)
            color_lookup: Strategy for mapping colors to all annotations
                (default: ColorLookup.TRACK). Options: INDEX, CLASS, TRACK.
                Applied to boxes, labels, and traces.
            detections_key: Key in data dict containing Detections object
            output_key: Key to store annotated frame (default: overwrites 'frame')
            copy_frame: Whether to copy frame before annotating (default: True)
        """
        # Set default colors
        if box_color is None:
            box_color = ColorPalette.DEFAULT
        if label_text_color is None:
            label_text_color = Color.BLACK
        if label_color is None:
            label_color = ColorPalette.DEFAULT
        if trace_color is None:
            trace_color = ColorPalette.DEFAULT

        # Initialize annotators
        self.box_annotator = BoxAnnotator(
            color=box_color, thickness=box_thickness, color_lookup=color_lookup
        )
        self.label_annotator = LabelAnnotator(
            text_color=label_text_color,
            text_scale=label_text_scale,
            text_thickness=label_text_thickness,
            text_padding=label_text_padding,
            color=label_color,
            color_lookup=color_lookup,
        )
        self.trace_annotator = TraceAnnotator(
            color=trace_color,
            trace_length=trace_length,
            thickness=trace_thickness,
            color_lookup=color_lookup,
        )

        # Label formatter parameters
        self.class_names = class_names
        self.show_tracker_id = show_tracker_id
        self.show_class = show_class
        self.show_confidence = show_confidence
        self.confidence_decimals = confidence_decimals

        # Keys
        self.detections_key = detections_key
        self.output_key = output_key
        self.copy_frame = copy_frame

    def _format_labels(self, detections: Detections) -> list[str]:
        """Generate formatted labels from detections."""
        if detections is None or len(detections) == 0:
            return []

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

        return labels

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Annotate frame with traces, boxes, and labels.

        Args:
            data: Pipeline data containing 'frame' and detections

        Returns:
            Data with fully annotated frame showing traces, boxes, and labels
        """
        frame = data.get("frame")
        detections = data.get(self.detections_key)

        if frame is None:
            return data

        # Copy frame if requested and modifying original
        if self.copy_frame and self.output_key == "frame":
            annotated_frame = frame.copy()
        else:
            annotated_frame = frame

        if (
            detections is not None
            and isinstance(detections, Detections)
            and len(detections) > 0
        ):
            # 1. Draw traces first (background layer)
            if (
                hasattr(detections, "tracker_id")
                and detections.tracker_id is not None
            ):
                annotated_frame = self.trace_annotator.annotate(
                    scene=annotated_frame, detections=detections
                )

            # 2. Draw bounding boxes
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

            # 3. Generate and draw labels (foreground layer)
            labels = self._format_labels(detections)
            if labels:
                annotated_frame = self.label_annotator.annotate(
                    scene=annotated_frame, detections=detections, labels=labels
                )

        data[self.output_key] = annotated_frame
        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Process if frame exists."""
        return "frame" in data


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
            Data with line_zone object added. Access counts via:
            - data["line_zone"].in_count
            - data["line_zone"].out_count
            - data["line_zone"].in_count_per_class
            - data["line_zone"].out_count_per_class
        """
        detections = data.get(self.detections_key)

        if detections is not None and len(detections) > 0:
            # Trigger line zone with detections
            self.line_zone.trigger(detections)

        # Store line_zone object in data dict
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


class LineZoneAnnotatorStep:
    """
    Pipeline step for visualizing line zone and crossing counts.

    Draws the counting line and displays in/out counts on the frame.
    Requires line_zone object in data dict (from LineZoneStep).

    Examples:
        ```python
        import supervision as sv

        start = sv.Point(x=0, y=500)
        end = sv.Point(x=1920, y=500)

        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | sv.YOLODetectionStep("yolov8n.pt")
            | sv.ByteTrackerStep()
            | sv.LineZoneStep(start=start, end=end)
            | sv.LineZoneAnnotatorStep(
                thickness=4,
                text_scale=1.0,
                custom_in_text="Entered",
                custom_out_text="Exited"
            )
            | sv.DisplaySink("Line Counter")
        )
        ```

        ```python
        # Customize colors and text
        import supervision as sv

        pipeline = (
            sv.Pipeline(sv.VideoFileSource("video.mp4"))
            | sv.YOLODetectionStep("yolov8n.pt")
            | sv.ByteTrackerStep()
            | sv.LineZoneStep(
                start=sv.Point(x=0, y=500),
                end=sv.Point(x=1920, y=500)
            )
            | sv.LineZoneAnnotatorStep(
                color=sv.Color.RED,
                text_color=sv.Color.WHITE,
                thickness=3,
                copy_frame=False
            )
            | sv.DisplaySink("Counting")
        )
        ```
    """

    def __init__(
        self,
        thickness: int = 2,
        color: Color = Color.WHITE,
        text_thickness: int = 2,
        text_color: Color = Color.BLACK,
        text_scale: float = 0.5,
        text_offset: float = 1.5,
        text_padding: int = 10,
        custom_in_text: str | None = None,
        custom_out_text: str | None = None,
        display_in_count: bool = True,
        display_out_count: bool = True,
        display_text_box: bool = True,
        text_orient_to_line: bool = False,
        text_centered: bool = True,
        line_zone_key: str = "line_zone",
        output_key: str = "frame",
        copy_frame: bool = True,
    ):
        """
        Initialize line zone annotator step.

        Args:
            thickness: Line thickness for drawing the zone
            color: Color of the line
            text_thickness: Thickness of count text
            text_color: Color of count text
            text_scale: Scale of count text
            text_offset: Distance of text from line
            text_padding: Padding around text box
            custom_in_text: Custom label for in count (default: "in")
            custom_out_text: Custom label for out count (default: "out")
            display_in_count: Whether to display in count
            display_out_count: Whether to display out count
            display_text_box: Whether to display text background box
            text_orient_to_line: Whether to orient text along line angle
            text_centered: Whether to center text on line
            line_zone_key: Key in data dict containing LineZone object
                (default: 'line_zone')
            output_key: Key to store annotated frame (default: overwrites 'frame')
            copy_frame: Whether to copy frame before annotating (default: True)
        """
        self.line_zone_key = line_zone_key
        self.output_key = output_key
        self.copy_frame = copy_frame

        self.annotator = LineZoneAnnotator(
            thickness=thickness,
            color=color,
            text_thickness=text_thickness,
            text_color=text_color,
            text_scale=text_scale,
            text_offset=text_offset,
            text_padding=text_padding,
            custom_in_text=custom_in_text,
            custom_out_text=custom_out_text,
            display_in_count=display_in_count,
            display_out_count=display_out_count,
            display_text_box=display_text_box,
            text_orient_to_line=text_orient_to_line,
            text_centered=text_centered,
        )

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Annotate frame with line zone and counts.

        Args:
            data: Pipeline data containing 'frame' and 'line_zone'

        Returns:
            Data with annotated frame showing line and crossing counts
        """
        frame = data.get("frame")
        line_zone = data.get(self.line_zone_key)

        if frame is None or line_zone is None:
            return data

        # Copy frame if requested and modifying original
        if self.copy_frame and self.output_key == "frame":
            annotated_frame = frame.copy()
        else:
            annotated_frame = frame

        # Annotate frame with line zone
        annotated_frame = self.annotator.annotate(
            frame=annotated_frame, line_counter=line_zone
        )

        data[self.output_key] = annotated_frame
        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Process if frame and line_zone exist."""
        return "frame" in data and self.line_zone_key in data


class DetectionStrategy(Enum):
    """
    Strategy for handling slow detectors in async detection steps.

    Attributes:
        SKIP_WHEN_BUSY: Skip frames when detector is busy, use last cached result
        USE_LAST_RESULT: Always return cached result, queue frame for async processing
        QUEUE_LATEST: Queue frames for processing, discard old frames when queue full
        SYNCHRONOUS: Traditional synchronous processing (blocks pipeline)
    """

    SKIP_WHEN_BUSY = "skip"
    USE_LAST_RESULT = "cache"
    QUEUE_LATEST = "queue"
    SYNCHRONOUS = "sync"


class AsyncDetectionStep(ABC):
    """
    Base class for asynchronous object detection steps.

    This class provides a framework for running object detection in a separate
    thread to avoid blocking the pipeline when the detector is slower than the
    frame rate. Different strategies can be used to handle the timing mismatch.

    The class uses a worker thread that continuously processes frames from a queue,
    storing results that can be retrieved by the main pipeline thread.

    Important: In asynchronous strategies (SKIP_WHEN_BUSY, USE_LAST_RESULT,
    QUEUE_LATEST), the pipeline returns the processed frame along with its
    detections to ensure perfect alignment. This means downstream steps see
    slightly delayed frames but with synchronized detections, eliminating visual
    misalignment at the cost of additional latency. The SYNCHRONOUS strategy
    processes frames immediately without this delay.

    Examples:
        ```python
        import supervision as sv

        # Create async detector with caching strategy
        detector = sv.AsyncYOLODetectionStep(
            model_path="yolov8n.pt",
            strategy=sv.DetectionStrategy.USE_LAST_RESULT,
            device="cuda"
        )

        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | detector
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("Async Detection")
        )
        pipeline.run()

        # Check metrics
        metrics = detector.get_metrics()
        print(f"Frames processed: {metrics['frames_processed']}")
        print(f"Frames cached: {metrics['frames_cached']}")
        ```
    """

    def __init__(
        self,
        strategy: DetectionStrategy = DetectionStrategy.USE_LAST_RESULT,
        max_queue_size: int = 2,
        inference_timeout: float = 0.5,
    ):
        """
        Initialize async detection step.

        Args:
            strategy: Strategy for handling timing mismatch between detector and source
            max_queue_size: Maximum frames to queue for processing (lower = less latency)
            inference_timeout: Maximum time to wait for inference results (seconds)
        """
        self.strategy = strategy
        self.max_queue_size = max_queue_size
        self.inference_timeout = inference_timeout

        # Threading components
        self._inference_queue: Queue = Queue(maxsize=max_queue_size)
        self._result_lock = Lock()
        self._stop_event = Event()
        self._worker_thread: Thread | None = None

        # State
        self._last_detections: Detections | None = None
        self._last_frame: np.ndarray | None = None
        self._last_frame_timestamp: float = 0.0
        self._inference_in_progress = False
        self._is_running = False

        # Metrics
        self._metrics = {
            "frames_processed": 0,
            "frames_skipped": 0,
            "frames_cached": 0,
            "frames_queued": 0,
            "avg_inference_time_ms": 0.0,
            "queue_full_count": 0,
            "total_inference_time_ms": 0.0,
        }

    @abstractmethod
    def _run_inference(self, frame: np.ndarray) -> Detections:
        """
        Run inference on a frame. Must be implemented by subclasses.

        Args:
            frame: Input frame for detection

        Returns:
            Detections object with detection results
        """
        pass

    def _inference_worker(self) -> None:
        """Worker thread that continuously processes frames from the queue."""
        while not self._stop_event.is_set():
            try:
                # Get frame from queue with timeout
                frame_data = self._inference_queue.get(timeout=0.1)

                if frame_data is None:  # Poison pill
                    break

                frame, timestamp = frame_data

                # Run inference
                start_time = time.time()
                self._inference_in_progress = True

                detections = self._run_inference(frame)

                inference_time_ms = (time.time() - start_time) * 1000

                # Update results and metrics
                with self._result_lock:
                    self._last_detections = detections
                    self._last_frame = frame
                    self._last_frame_timestamp = timestamp
                    self._inference_in_progress = False

                    # Update metrics
                    self._metrics["frames_processed"] += 1
                    self._metrics["total_inference_time_ms"] += inference_time_ms
                    self._metrics["avg_inference_time_ms"] = (
                        self._metrics["total_inference_time_ms"]
                        / self._metrics["frames_processed"]
                    )

                self._inference_queue.task_done()

            except Empty:
                continue
            except Exception as e:
                # Log error but don't crash the worker
                print(f"Error in inference worker: {e}")
                self._inference_in_progress = False
                continue

    def start(self) -> AsyncDetectionStep:
        """
        Start the inference worker thread.

        Returns:
            Self for method chaining
        """
        if not self._is_running:
            self._stop_event.clear()
            self._worker_thread = Thread(
                target=self._inference_worker, name="AsyncDetectionWorker", daemon=True
            )
            self._worker_thread.start()
            self._is_running = True
        return self

    def stop(self) -> None:
        """Stop the inference worker thread and cleanup resources."""
        if self._is_running:
            self._stop_event.set()
            # Send poison pill
            try:
                self._inference_queue.put(None, timeout=1.0)
            except:
                pass

            if self._worker_thread:
                self._worker_thread.join(timeout=2.0)

            self._is_running = False

    def _enqueue_frame(self, frame: np.ndarray) -> bool:
        """
        Try to enqueue a frame for processing.

        Args:
            frame: Frame to enqueue

        Returns:
            True if frame was enqueued, False if queue was full
        """
        try:
            self._inference_queue.put_nowait((frame, time.time()))
            with self._result_lock:
                self._metrics["frames_queued"] += 1
            return True
        except:
            with self._result_lock:
                self._metrics["queue_full_count"] += 1
            return False

    def _clear_queue(self) -> None:
        """Clear old frames from queue."""
        cleared = 0
        while not self._inference_queue.empty():
            try:
                self._inference_queue.get_nowait()
                self._inference_queue.task_done()
                cleared += 1
            except:
                break

    def _use_cached_result(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Return cached detection result with matching frame.

        Args:
            data: Pipeline data

        Returns:
            Data with cached detections and frame added
        """
        with self._result_lock:
            if self._last_detections is not None and self._last_frame is not None:
                data["detections"] = self._last_detections
                data["frame"] = self._last_frame
                self._metrics["frames_cached"] += 1
            else:
                # No cached result yet, return empty detections
                data["detections"] = Detections.empty()

        return data

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Process frame according to selected strategy.

        Args:
            data: Pipeline data containing 'frame'

        Returns:
            Data with 'detections' field added
        """
        frame = data.get("frame")
        if frame is None:
            return data

        # Start worker thread if not running
        if not self._is_running:
            self.start()

        if self.strategy == DetectionStrategy.SYNCHRONOUS:
            # Synchronous mode: block and wait for result
            start_time = time.time()
            detections = self._run_inference(frame)
            inference_time_ms = (time.time() - start_time) * 1000

            data["detections"] = detections
            with self._result_lock:
                self._metrics["frames_processed"] += 1
                self._metrics["total_inference_time_ms"] += inference_time_ms
                self._metrics["avg_inference_time_ms"] = (
                    self._metrics["total_inference_time_ms"]
                    / self._metrics["frames_processed"]
                )

        elif self.strategy == DetectionStrategy.SKIP_WHEN_BUSY:
            # Skip if busy, otherwise enqueue
            if self._inference_in_progress or self._inference_queue.full():
                data = self._use_cached_result(data)
                with self._result_lock:
                    self._metrics["frames_skipped"] += 1
            else:
                self._enqueue_frame(frame)
                data = self._use_cached_result(data)

        elif self.strategy == DetectionStrategy.USE_LAST_RESULT:
            # Always enqueue and return cached result
            self._enqueue_frame(frame)
            data = self._use_cached_result(data)

        elif self.strategy == DetectionStrategy.QUEUE_LATEST:
            # Clear queue and enqueue latest frame
            if self._inference_queue.full():
                self._clear_queue()
            self._enqueue_frame(frame)
            data = self._use_cached_result(data)

        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Process if frame exists."""
        return "frame" in data

    def get_metrics(self) -> dict[str, Any]:
        """
        Get performance metrics.

        Returns:
            Dictionary with metrics:
                - frames_processed: Total frames processed by detector
                - frames_skipped: Frames skipped due to busy detector
                - frames_cached: Times cached result was used
                - frames_queued: Total frames enqueued
                - avg_inference_time_ms: Average inference time in milliseconds
                - queue_full_count: Times queue was full
        """
        with self._result_lock:
            return self._metrics.copy()

    def get_queue_size(self) -> tuple[int, int]:
        """
        Get current queue size and maximum queue size.

        Returns:
            Tuple of (current_size, max_size)
        """
        return (self._inference_queue.qsize(), self.max_queue_size)

    def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        with self._result_lock:
            for key in self._metrics:
                if isinstance(self._metrics[key], (int, float)):
                    self._metrics[key] = 0

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


class AsyncYOLODetectionStep(AsyncDetectionStep):
    """
    Asynchronous YOLO object detection step.

    This step runs YOLO inference in a separate thread, allowing the pipeline
    to continue processing frames even when detection is slower than the frame rate.

    Examples:
        ```python
        import supervision as sv

        # Basic async detection with default caching strategy
        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | sv.AsyncYOLODetectionStep("yolov8n.pt", device="cuda")
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("Async YOLO")
        )
        pipeline.run()
        ```

        ```python
        # Skip frames when detector is busy (lowest latency)
        detector = sv.AsyncYOLODetectionStep(
            model_path="yolov8n.pt",
            strategy=sv.DetectionStrategy.SKIP_WHEN_BUSY,
            conf=0.5,
            device="cuda"
        )

        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | sv.FPSCalculatorStep()
            | detector
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("Low Latency Detection")
        )

        # Monitor metrics
        for data in pipeline:
            metrics = detector.get_metrics()
            print(f"FPS: {data.get('fps', 0):.1f}")
            print(f"Avg inference: {metrics['avg_inference_time_ms']:.1f}ms")
            print(f"Frames skipped: {metrics['frames_skipped']}")
        ```

        ```python
        # Queue latest frames (best accuracy)
        detector = sv.AsyncYOLODetectionStep(
            model_path="yolov8n.pt",
            strategy=sv.DetectionStrategy.QUEUE_LATEST,
            max_queue_size=5,
            device="cuda"
        )
        ```
    """

    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cuda",
        verbose: bool = False,
        strategy: DetectionStrategy = DetectionStrategy.USE_LAST_RESULT,
        max_queue_size: int = 2,
        inference_timeout: float = 0.5,
        warmup: bool = True,
    ):
        """
        Initialize async YOLO detection step.

        Args:
            model_path: Path to YOLO model file (.pt)
            conf: Confidence threshold for detections (0.0-1.0)
            iou: IOU threshold for NMS (0.0-1.0)
            device: Device for inference ('cuda' or 'cpu')
            verbose: Whether to print verbose YOLO output
            strategy: Detection strategy for handling slow inference
            max_queue_size: Maximum frames to queue (lower = less latency)
            inference_timeout: Maximum time to wait for results (seconds)
            warmup: Whether to run warmup inference on model load
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for AsyncYOLODetectionStep. "
                "Install it with: pip install ultralytics"
            )

        # Initialize base class
        super().__init__(
            strategy=strategy,
            max_queue_size=max_queue_size,
            inference_timeout=inference_timeout,
        )

        # YOLO parameters
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device = device
        self.verbose = verbose

        # Load model
        self.model = YOLO(model_path)
        self.model.to(device)

        # Warmup model
        if warmup:
            self._warmup_model()

        # Start worker thread
        self.start()

    def _warmup_model(self) -> None:
        """
        Warmup the model with a dummy inference.

        This ensures CUDA is initialized and model weights are loaded,
        avoiding slow first inference.
        """
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            _ = self.model.predict(
                source=dummy_frame,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )
        except Exception as e:
            print(f"Warning: Model warmup failed: {e}")

    def _run_inference(self, frame: np.ndarray) -> Detections:
        """
        Run YOLO inference on frame.

        Args:
            frame: Input frame for detection

        Returns:
            Detections object with detection results
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            verbose=self.verbose,
        )

        return Detections.from_ultralytics(results[0])

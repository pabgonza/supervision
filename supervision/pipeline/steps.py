from __future__ import annotations

from typing import Any, Callable

import numpy as np

from supervision.annotators.core import (
    BaseAnnotator,
    BoxAnnotator,
    LabelAnnotator,
    TraceAnnotator,
)
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette
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
        detections_key: str = "detections",
        output_key: str = "frame",
        copy_frame: bool = True,
    ):
        """
        Initialize BoxAnnotator step.

        Args:
            thickness: Thickness of bounding box lines
            color: Color or ColorPalette for boxes
            detections_key: Key in data dict containing Detections
            output_key: Key to store annotated frame
            copy_frame: Whether to copy frame before annotating (default: True)
        """
        if color is None:
            color = ColorPalette.DEFAULT

        self.box_annotator = BoxAnnotator(color=color, thickness=thickness)
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
        detections_key: str = "detections",
        output_key: str = "frame",
    ):
        """
        Initialize TraceAnnotator step.

        Args:
            trace_length: Number of frames to keep in the trace history
            thickness: Thickness of the trace lines
            color: Color or ColorPalette for traces (default: ColorPalette.DEFAULT)
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
        self.box_annotator = BoxAnnotator(color=box_color, thickness=box_thickness)
        self.label_annotator = LabelAnnotator(
            text_color=label_text_color,
            text_scale=label_text_scale,
            text_thickness=label_text_thickness,
            text_padding=label_text_padding,
            color=label_color,
        )
        self.trace_annotator = TraceAnnotator(
            color=trace_color,
            trace_length=trace_length,
            thickness=trace_thickness,
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

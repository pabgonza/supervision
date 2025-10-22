from __future__ import annotations

from typing import Any

from supervision.annotators.core import (
    BoxAnnotator,
    LabelAnnotator,
    TraceAnnotator,
)
from supervision.annotators.utils import ColorLookup
from supervision.detection.core import Detections
from supervision.detection.line_zone import LineZoneAnnotator
from supervision.draw.color import Color, ColorPalette


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
        input_key: str = "frame",
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
            input_key: Key in data dict containing input frame (default: 'frame')
            output_key: Key to store annotated frame
            copy_frame: Whether to copy frame before annotating (default: True)
        """
        if color is None:
            color = ColorPalette.DEFAULT

        self.box_annotator = BoxAnnotator(
            color=color, thickness=thickness, color_lookup=color_lookup
        )
        self.detections_key = detections_key
        self.input_key = input_key
        self.output_key = output_key
        self.copy_frame = copy_frame

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Draw bounding boxes on frame."""
        frame = data.get(self.input_key)
        detections = data.get(self.detections_key)

        if frame is None:
            return data

        # Copy frame if requested and modifying original
        if self.copy_frame and self.output_key == self.input_key:
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
        return self.input_key in data


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
        input_key: str = "frame",
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
            input_key: Key in data dict containing input frame (default: 'frame')
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
        self.input_key = input_key
        self.output_key = output_key
        self.copy_frame = copy_frame

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Draw labels on frame."""
        frame = data.get(self.input_key)
        detections = data.get(self.detections_key)
        labels = data.get(self.labels_key)

        if frame is None:
            return data

        # Copy frame if requested and modifying original
        if self.copy_frame and self.output_key == self.input_key:
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
        return self.input_key in data


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
        input_key: str = "frame",
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
            input_key: Key in data dict containing input frame (default: 'frame')
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
        self.input_key = input_key
        self.output_key = output_key

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Draw traces on the frame.

        Args:
            data: Pipeline data containing frame and detections with tracker_id

        Returns:
            Data with annotated frame showing tracking traces
        """
        frame = data.get(self.input_key)
        detections = data.get(self.detections_key)

        if frame is None:
            return data

        # Make a copy if we might modify the original
        if self.output_key == self.input_key:
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
        return self.input_key in data


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
        input_key: str = "frame",
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
            input_key: Key in data dict containing input frame (default: 'frame')
            output_key: Key to store annotated frame (default: overwrites 'frame')
            copy_frame: Whether to copy frame before annotating (default: True)
        """
        self.line_zone_key = line_zone_key
        self.input_key = input_key
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
            data: Pipeline data containing frame and line_zone

        Returns:
            Data with annotated frame showing line and crossing counts
        """
        frame = data.get(self.input_key)
        line_zone = data.get(self.line_zone_key)

        if frame is None or line_zone is None:
            return data

        # Copy frame if requested and modifying original
        if self.copy_frame and self.output_key == self.input_key:
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
        return self.input_key in data and self.line_zone_key in data


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
        input_key: str = "frame",
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
            input_key: Key in data dict containing input frame (default: 'frame')
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
        self.input_key = input_key
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
            data: Pipeline data containing frame and detections

        Returns:
            Data with fully annotated frame showing traces, boxes, and labels
        """
        frame = data.get(self.input_key)
        detections = data.get(self.detections_key)

        if frame is None:
            return data

        # Copy frame if requested and modifying original
        if self.copy_frame and self.output_key == self.input_key:
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
        return self.input_key in data

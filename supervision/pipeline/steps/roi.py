from __future__ import annotations

import copy
from typing import Any

import cv2


class ROIExtractionStep:
    """
    Pipeline step to extract a Region of Interest (ROI) from frames.

    Extracts a rectangular ROI at specified coordinates and stores offset
    information for later coordinate translation. Supports automatic centering
    when coordinates are not specified.

    Examples:
        ```python
        import supervision as sv

        # Extract centered 640x640 ROI
        pipeline = (
            sv.Pipeline(source)
            | sv.ROIExtractionStep(width=640, height=640)
            | sv.YOLODetectionStep(model_path="model.pt", input_key="roi_frame")
            | sv.DisplaySink("ROI Detection")
        )

        # Extract ROI at specific position
        pipeline = (
            sv.Pipeline(source)
            | sv.ROIExtractionStep(x=100, y=200, width=800, height=600)
            | sv.YOLODetectionStep(model_path="model.pt", input_key="roi_frame")
            | sv.DisplaySink("ROI Detection")
        )
        ```

    Note:
        The original frame is preserved in data["frame"]. The extracted ROI
        is stored in the output_key (default: "roi_frame").

    TODO: Add support for multiple ROIs processing in parallel
    """

    def __init__(
        self,
        x: int | None = None,
        y: int | None = None,
        width: int = 640,
        height: int = 640,
        input_key: str = "frame",
        output_key: str = "roi_frame",
    ):
        """
        Initialize ROI extraction step.

        Args:
            x: X coordinate of ROI top-left corner (None = center horizontally)
            y: Y coordinate of ROI top-left corner (None = center vertically)
            width: Width of ROI in pixels (default: 640)
            height: Height of ROI in pixels (default: 640)
            input_key: Key in data dict containing input frame (default: "frame")
            output_key: Key to store extracted ROI (default: "roi_frame")
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"ROI dimensions must be positive, got {width}x{height}")

        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.input_key = input_key
        self.output_key = output_key
        self._roi_x = 0
        self._roi_y = 0

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Extract ROI from frame and store offset information.

        Args:
            data: Pipeline data dictionary

        Returns:
            Data with ROI frame and metadata added
        """
        frame = data.get(self.input_key)
        if frame is None:
            return data

        h, w = frame.shape[:2]

        if self.x is None:
            self._roi_x = max(0, (w - self.width) // 2)
        else:
            self._roi_x = max(0, min(self.x, w - self.width))

        if self.y is None:
            self._roi_y = max(0, (h - self.height) // 2)
        else:
            self._roi_y = max(0, min(self.y, h - self.height))

        roi_x_end = min(self._roi_x + self.width, w)
        roi_y_end = min(self._roi_y + self.height, h)

        roi = frame[self._roi_y : roi_y_end, self._roi_x : roi_x_end]

        data["roi_offset"] = (self._roi_x, self._roi_y)
        data["roi_size"] = (roi.shape[1], roi.shape[0])
        data[self.output_key] = roi

        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Process if input frame exists."""
        return self.input_key in data


class CoordinateTranslationStep:
    """
    Pipeline step to translate detection coordinates from ROI to full frame.

    Translates bounding box coordinates from ROI space back to original frame
    coordinates using offset information stored by ROIExtractionStep.

    Examples:
        ```python
        import supervision as sv

        pipeline = (
            sv.Pipeline(source)
            | sv.ROIExtractionStep(width=640, height=640)
            | sv.YOLODetectionStep(
                model_path="model.pt",
                input_key="roi_frame",
                output_key="roi_detections"
            )
            | sv.CoordinateTranslationStep(
                input_key="roi_detections",
                output_key="detections"
            )
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("Translated Detections")
        )
        ```

    Note:
        This step requires "roi_offset" metadata in the data dictionary,
        which is automatically added by ROIExtractionStep.

    TODO: Add support for multiple ROIs with ID-based translation
    """

    def __init__(
        self,
        input_key: str = "roi_detections",
        output_key: str = "detections",
    ):
        """
        Initialize coordinate translation step.

        Args:
            input_key: Key in data dict containing ROI detections
                (default: "roi_detections")
            output_key: Key to store translated detections (default: "detections")
        """
        self.input_key = input_key
        self.output_key = output_key

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Translate detection coordinates from ROI to full frame.

        Args:
            data: Pipeline data dictionary

        Returns:
            Data with translated detections added
        """
        detections = data.get(self.input_key)
        roi_offset = data.get("roi_offset")

        if detections is None or roi_offset is None or len(detections) == 0:
            if detections is not None:
                data[self.output_key] = detections
            return data

        roi_x, roi_y = roi_offset

        translated_detections = copy.deepcopy(detections)

        if translated_detections.xyxy is not None:
            translated_detections.xyxy[:, [0, 2]] += roi_x
            translated_detections.xyxy[:, [1, 3]] += roi_y

        data[self.output_key] = translated_detections
        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Always process."""
        return True


class ROIVisualizationStep:
    """
    Pipeline step to draw ROI rectangle on frame.

    Visualizes the Region of Interest by drawing a rectangle border using
    offset information from ROIExtractionStep.

    Examples:
        ```python
        import supervision as sv

        pipeline = (
            sv.Pipeline(source)
            | sv.ROIExtractionStep(width=640, height=640)
            | sv.YOLODetectionStep(model_path="model.pt", input_key="roi_frame")
            | sv.CoordinateTranslationStep()
            | sv.ROIVisualizationStep(color=(0, 255, 255), thickness=3)
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("ROI Detection")
        )
        ```

    Note:
        This step modifies the frame in-place. It requires "roi_offset" and
        "roi_size" metadata which are automatically added by ROIExtractionStep.

    TODO: Add support for visualizing multiple ROIs with different colors
    """

    def __init__(
        self,
        color: tuple[int, int, int] = (255, 255, 0),
        thickness: int = 2,
        frame_key: str = "frame",
    ):
        """
        Initialize ROI visualization step.

        Args:
            color: Color of ROI rectangle in BGR format (default: cyan)
            thickness: Thickness of rectangle border in pixels (default: 2)
            frame_key: Key in data dict containing frame to annotate
                (default: "frame")
        """
        self.color = color
        self.thickness = thickness
        self.frame_key = frame_key

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Draw ROI rectangle on frame.

        Args:
            data: Pipeline data dictionary

        Returns:
            Data with ROI rectangle drawn on frame
        """
        frame = data.get(self.frame_key)
        roi_offset = data.get("roi_offset")
        roi_size = data.get("roi_size")

        if frame is None or roi_offset is None or roi_size is None:
            return data

        roi_x, roi_y = roi_offset
        roi_w, roi_h = roi_size

        cv2.rectangle(
            frame,
            (roi_x, roi_y),
            (roi_x + roi_w, roi_y + roi_h),
            self.color,
            self.thickness,
        )

        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Always process."""
        return True

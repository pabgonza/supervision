from __future__ import annotations

import copy
from typing import Any

import cv2


class ROIExtractionStep:
    """
    Pipeline step to extract a Region of Interest (ROI) from frames.

    Extracts a rectangular ROI at specified coordinates.

    Examples:
        ```python
        import supervision as sv

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
        x: int,
        y: int,
        width: int,
        height: int,
        input_key: str = "frame",
        output_key: str = "roi_frame",
    ):
        """
        Initialize ROI extraction step.

        Args:
            x: X coordinate of ROI top-left corner
            y: Y coordinate of ROI top-left corner
            width: Width of ROI in pixels
            height: Height of ROI in pixels
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
        self._roi_x = x
        self._roi_y = y

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Extract ROI from frame.

        Args:
            data: Pipeline data dictionary

        Returns:
            Data with ROI frame added
        """
        frame = data.get(self.input_key)
        if frame is None:
            return data

        h, w = frame.shape[:2]

        self._roi_x = max(0, min(self.x, w - self.width))
        self._roi_y = max(0, min(self.y, h - self.height))

        roi_x_end = min(self._roi_x + self.width, w)
        roi_y_end = min(self._roi_y + self.height, h)

        roi = frame[self._roi_y : roi_y_end, self._roi_x : roi_x_end]

        data[self.output_key] = roi

        return data

    @property
    def roi_x(self) -> int:
        """Get current ROI X offset."""
        return self._roi_x

    @property
    def roi_y(self) -> int:
        """Get current ROI Y offset."""
        return self._roi_y

    @property
    def roi_offset(self) -> tuple[int, int]:
        """Get current ROI offset as (x, y) tuple."""
        return (self._roi_x, self._roi_y)

    def filter(self, data: dict[str, Any]) -> bool:
        """Process if input frame exists."""
        return self.input_key in data


class CoordinateTranslationStep:
    """
    Pipeline step to translate detection coordinates from ROI to full frame.

    Translates bounding box coordinates from ROI space back to original frame
    coordinates using specified offset values.

    Examples:
        ```python
        import supervision as sv

        roi_step = sv.ROIExtractionStep(x=100, y=200, width=640, height=640)

        pipeline = (
            sv.Pipeline(source)
            | roi_step
            | sv.YOLODetectionStep(
                model_path="model.pt",
                input_key="roi_frame",
                output_key="roi_detections"
            )
            | sv.CoordinateTranslationStep(
                offset_x=100,
                offset_y=200,
                input_key="roi_detections",
                output_key="detections"
            )
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("Translated Detections")
        )
        ```

    Note:
        This step is independent of ROIExtractionStep and data dictionary.
        Offset coordinates must be provided at initialization.

    TODO: Add support for multiple ROIs with ID-based translation
    """

    def __init__(
        self,
        offset_x: int = 0,
        offset_y: int = 0,
        input_key: str = "roi_detections",
        output_key: str = "detections",
    ):
        """
        Initialize coordinate translation step.

        Args:
            offset_x: X coordinate offset to add to detections (default: 0)
            offset_y: Y coordinate offset to add to detections (default: 0)
            input_key: Key in data dict containing ROI detections
                (default: "roi_detections")
            output_key: Key to store translated detections (default: "detections")
        """
        self.offset_x = offset_x
        self.offset_y = offset_y
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

        if detections is None or len(detections) == 0:
            if detections is not None:
                data[self.output_key] = detections
            return data

        translated_detections = copy.deepcopy(detections)

        if translated_detections.xyxy is not None:
            translated_detections.xyxy[:, [0, 2]] += self.offset_x
            translated_detections.xyxy[:, [1, 3]] += self.offset_y

        data[self.output_key] = translated_detections
        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Always process."""
        return True


class ROIVisualizationStep:
    """
    Pipeline step to draw ROI rectangle on frame.

    Visualizes the Region of Interest by drawing a rectangle border using
    specified coordinates and dimensions.

    Examples:
        ```python
        import supervision as sv

        pipeline = (
            sv.Pipeline(source)
            | sv.ROIExtractionStep(x=100, y=200, width=640, height=640)
            | sv.YOLODetectionStep(model_path="model.pt", input_key="roi_frame")
            | sv.CoordinateTranslationStep(offset_x=100, offset_y=200)
            | sv.ROIVisualizationStep(
                x=100, y=200, width=640, height=640,
                color=(0, 255, 255), thickness=3
            )
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("ROI Detection")
        )
        ```

    Note:
        This step modifies the frame in-place by default. Use copy_frame=False
        to avoid copying the frame for better performance when chaining multiple
        annotation steps. It is independent of ROIExtractionStep and data dictionary.

    TODO: Add support for visualizing multiple ROIs with different colors
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        color: tuple[int, int, int] = (255, 255, 0),
        thickness: int = 2,
        frame_key: str = "frame",
        copy_frame: bool = True,
    ):
        """
        Initialize ROI visualization step.

        Args:
            x: X coordinate of ROI top-left corner
            y: Y coordinate of ROI top-left corner
            width: Width of ROI in pixels
            height: Height of ROI in pixels
            color: Color of ROI rectangle in BGR format (default: cyan)
            thickness: Thickness of rectangle border in pixels (default: 2)
            frame_key: Key in data dict containing frame to annotate
                (default: "frame")
            copy_frame: Whether to copy the frame before drawing (default: True)
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.thickness = thickness
        self.frame_key = frame_key
        self.copy_frame = copy_frame

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Draw ROI rectangle on frame.

        Args:
            data: Pipeline data dictionary

        Returns:
            Data with ROI rectangle drawn on frame
        """
        frame = data.get(self.frame_key)

        if frame is None:
            return data

        if self.copy_frame:
            frame = frame.copy()

        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            self.color,
            self.thickness,
        )

        data[self.frame_key] = frame

        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Always process."""
        return True

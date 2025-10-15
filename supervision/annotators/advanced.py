"""
Advanced annotator ported from cvutils library with enhanced visualization capabilities.
"""

from __future__ import annotations

import colorsys
from enum import Enum

import cv2
import numpy as np

from supervision.annotators.base import BaseAnnotator
from supervision.detection.core import Detections
from supervision.draw.base import ImageType
from supervision.draw.color import ColorPalette

CV2_FONT = cv2.FONT_HERSHEY_SIMPLEX


class HersheyFonts(Enum):
    """OpenCV Hershey font constants."""

    FONT_HERSHEY_SIMPLEX = 0
    FONT_HERSHEY_PLAIN = 1
    FONT_HERSHEY_DUPLEX = 2
    FONT_HERSHEY_COMPLEX = 3
    FONT_HERSHEY_TRIPLEX = 4
    FONT_HERSHEY_COMPLEX_SMALL = 5
    FONT_HERSHEY_SCRIPT_SIMPLEX = 6
    FONT_HERSHEY_SCRIPT_COMPLEX = 7
    FONT_ITALIC = 16


class AdvancedAnnotator(BaseAnnotator):
    """
    Advanced annotator with adaptive scaling and enhanced visualization features.

    This annotator automatically adjusts text size and line thickness based on
    image size, provides advanced mask visualization with transparency, and includes
    styling options for different visualization needs.

    Attributes:
        font (int): OpenCV font constant for text rendering.
        mask_alpha (float): Transparency value for mask overlays (0.0-1.0).
        text_separation (int): Vertical separation between text lines in pixels.
        font_scale (float | None): Manual font scale override. If None, auto-calculated.
        thickness (int | None): Manual line thickness override. If None,
            auto-calculated.
        color_palette (ColorPalette): Color palette for detections.
        show_labels (bool): Whether to display class labels and confidence scores.
        show_masks (bool): Whether to render detection masks.
        mask_threshold (float): Threshold for converting masks to boolean.
    """

    _SMALL_IMAGE_AREA = 512000
    _LARGE_IMAGE_AREA = 2073600

    def __init__(
        self,
        font: int = HersheyFonts.FONT_HERSHEY_SIMPLEX.value,
        mask_alpha: float = 0.4,
        text_separation: int = 15,
        font_scale: float | None = None,
        thickness: int | None = None,
        color_palette: ColorPalette = ColorPalette.DEFAULT,
        show_labels: bool = True,
        show_masks: bool = True,
        mask_threshold: float = 0.3,
    ):
        """
        Initialize the AdvancedAnnotator.

        Args:
            font: OpenCV font constant for text rendering.
            mask_alpha: Transparency value for mask overlays (0.0-1.0).
            text_separation: Vertical separation between text lines in pixels.
            font_scale: Manual font scale override. If None, auto-calculated
                based on image size.
            thickness: Manual line thickness override. If None, auto-calculated
                based on image size.
            color_palette: Color palette for detections.
            show_labels: Whether to display class labels and confidence scores.
            show_masks: Whether to render detection masks.
            mask_threshold: Threshold for converting masks to boolean.
        """
        self.font = font
        self.mask_alpha = mask_alpha
        self.text_separation = text_separation
        self._manual_font_scale = font_scale
        self._manual_thickness = thickness
        self.color_palette = color_palette
        self.show_labels = show_labels
        self.show_masks = show_masks
        self.mask_threshold = mask_threshold

    def annotate(self, scene: ImageType, detections: Detections) -> ImageType:
        """
        Annotate the scene with advanced visualization features.

        Args:
            scene: The input image to annotate.
            detections: Detections to visualize.

        Returns:
            The annotated image.
        """
        annotated_image = scene.copy()
        image_height, image_width = annotated_image.shape[:2]

        # Calculate adaptive parameters based on image size
        font_scale = self._calculate_font_scale(image_width, image_height)
        thickness = self._calculate_thickness(image_width, image_height)

        # Generate colors for each detection
        colors = self._generate_colors(detections)

        for i in range(len(detections)):
            # Get detection data
            xyxy = detections.xyxy[i]
            class_id = detections.class_id[i] if detections.class_id is not None else 0
            confidence = (
                detections.confidence[i] if detections.confidence is not None else 1.0
            )
            mask = detections.mask[i] if detections.mask is not None else None

            color_bgr = colors[i]

            # Draw bounding box
            self._draw_bounding_box(
                annotated_image, xyxy, color_bgr, thickness
            )

            # Draw mask if available
            if self.show_masks and mask is not None:
                self._draw_mask(
                    annotated_image, mask, xyxy, color_bgr, self.mask_alpha
                )

            # Draw label
            if self.show_labels:
                self._draw_label(
                    annotated_image,
                    xyxy,
                    class_id,
                    confidence,
                    color_bgr,
                    font_scale,
                    thickness,
                )

        return annotated_image

    def _calculate_font_scale(self, width: int, height: int) -> float:
        """Calculate adaptive font scale based on image dimensions."""
        if self._manual_font_scale is not None:
            return self._manual_font_scale

        return max(np.sqrt(height * width) / 2050, 0.3)

    def _calculate_thickness(self, width: int, height: int) -> int:
        """Calculate adaptive line thickness based on image dimensions."""
        if self._manual_thickness is not None:
            return self._manual_thickness

        area = height * width
        if area < self._SMALL_IMAGE_AREA:
            return 1
        elif area > self._LARGE_IMAGE_AREA:
            return 3
        else:
            return 2

    def _generate_colors(self, detections: Detections) -> list[tuple[int, int, int]]:
        """Generate BGR colors for each detection."""
        colors = []

        for i in range(len(detections)):
            class_id = detections.class_id[i] if detections.class_id is not None else 0

            # Use color palette or generate unique color
            if hasattr(self.color_palette, "by_idx"):
                color = self.color_palette.by_idx(class_id)
                color_bgr = (int(color.b), int(color.g), int(color.r))
            else:
                color_bgr = self._create_unique_color(int(class_id))

            colors.append(color_bgr)

        return colors

    def _create_unique_color(
        self, tag: int, hue_step: float = 0.41
    ) -> tuple[int, int, int]:
        """Create a unique BGR color for a given tag using HSV color space."""
        h, v = (tag * hue_step) % 1, 1.0 - (int(tag * hue_step) % 4) / 5.0
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, v)
        return int(255 * b), int(255 * g), int(255 * r)  # BGR format

    def _draw_bounding_box(
        self,
        image: np.ndarray,
        xyxy: np.ndarray,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        """Draw bounding box rectangle."""
        x1, y1, x2, y2 = xyxy.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    def _draw_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        xyxy: np.ndarray,
        color: tuple[int, int, int],
        alpha: float,
    ) -> None:
        """Draw segmentation mask with transparency."""
        # Convert mask to boolean
        if mask.max() <= 1.0:
            boolean_mask = mask > self.mask_threshold
        else:
            boolean_mask = mask.astype(bool)

        # Apply mask overlay
        if boolean_mask.any():
            # Resize mask to match image dimensions if needed
            if mask.shape[:2] != image.shape[:2]:
                # Get bounding box coordinates
                x1, y1, x2, y2 = xyxy.astype(int)

                # Resize mask to bounding box size
                box_height, box_width = y2 - y1, x2 - x1
                if boolean_mask.shape != (box_height, box_width):
                    boolean_mask = cv2.resize(
                        boolean_mask.astype(np.uint8),
                        (box_width, box_height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)

                # Apply mask to the bounding box region
                if (x1 >= 0 and y1 >= 0 and
                    x2 <= image.shape[1] and y2 <= image.shape[0]):
                    roi = image[y1:y2, x1:x2]
                    overlay = roi.copy()
                    overlay[boolean_mask] = color

                    # Blend with original region
                    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
                    image[y1:y2, x1:x2] = roi
            else:
                # Full image mask
                overlay = image.copy()
                overlay[boolean_mask] = color
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    def _draw_label(
        self,
        image: np.ndarray,
        xyxy: np.ndarray,
        class_id: int,
        confidence: float,
        color: tuple[int, int, int],
        font_scale: float,
        thickness: int,
    ) -> None:
        """Draw class label and confidence score."""
        # Format label text
        label_text = f"{int(class_id)}: {confidence:.2f}"

        # Calculate text size and position
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, self.font, font_scale, thickness
        )

        x1, y1 = xyxy[:2].astype(int)

        # Position text above bounding box
        text_x = x1
        text_y = (
            y1 - baseline
            if y1 - text_height - baseline > text_height
            else y1 + text_height
        )

        # Draw text background rectangle
        bg_x1 = text_x
        bg_y1 = text_y - text_height
        bg_x2 = text_x + text_width
        bg_y2 = text_y + baseline

        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

        # Draw text in white for visibility
        text_color = (255, 255, 255)
        cv2.putText(
            image,
            label_text,
            (text_x, text_y),
            self.font,
            font_scale,
            text_color,
            thickness,
        )

    def draw_info_text(
        self,
        scene: ImageType,
        text_lines: list[str],
        position: tuple[int, int] = (20, 20),
        color: tuple[int, int, int] = (0, 0, 255),
    ) -> ImageType:
        """
        Draw informational text on the image.

        Args:
            scene: The input image.
            text_lines: List of text strings to draw.
            position: Starting position (x, y) for the text.
            color: BGR color for the text.

        Returns:
            The image with text drawn.
        """
        annotated_image = scene.copy()
        image_height, image_width = annotated_image.shape[:2]

        font_scale = self._calculate_font_scale(image_width, image_height)
        thickness = self._calculate_thickness(image_width, image_height)

        x_pos, y_pos = position

        for i, text in enumerate(text_lines):
            text_y = y_pos + i * (self.text_separation + 20)  # approx text height
            cv2.putText(
                annotated_image,
                text,
                (x_pos, text_y),
                self.font,
                font_scale,
                color,
                thickness,
            )

        return annotated_image

    def draw_styled_polygon(
        self,
        scene: ImageType,
        points: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 0),
        style: str = "solid",
        gap: int = 20,
    ) -> ImageType:
        """
        Draw styled polygon (solid, dotted, or dashed).

        Args:
            scene: The input image.
            points: Array of polygon vertices in format [[x1, y1], [x2, y2], ...].
            color: BGR color for the polygon.
            style: Line style - "solid", "dotted", or "dashed".
            gap: Gap between dots/dashes for styled lines.

        Returns:
            The image with polygon drawn.
        """
        annotated_image = scene.copy()
        image_height, image_width = annotated_image.shape[:2]
        thickness = self._calculate_thickness(image_width, image_height)

        points_int = points.astype(np.int32)

        if style == "solid":
            cv2.polylines(annotated_image, [points_int], True, color, thickness)
        else:
            # Draw styled polygon by connecting consecutive points
            for i in range(len(points_int)):
                start_point = tuple(points_int[i])
                end_point = tuple(points_int[(i + 1) % len(points_int)])
                self._draw_styled_line(
                    annotated_image,
                    start_point,
                    end_point,
                    color,
                    style,
                    gap,
                    thickness,
                )

        return annotated_image

    def _draw_styled_line(
        self,
        image: np.ndarray,
        start_point: tuple[int, int],
        end_point: tuple[int, int],
        color: tuple[int, int, int],
        style: str,
        gap: int,
        thickness: int,
    ) -> None:
        """Draw styled line between two points."""
        if style == "solid":
            cv2.line(image, start_point, end_point, color, thickness)
            return

        # Calculate line parameters
        dist = np.sqrt(
            (start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2
        )

        if dist == 0:
            return

        # Generate points along the line
        num_points = int(dist / gap)
        for i in range(num_points):
            t = i / num_points if num_points > 0 else 0
            x = int(start_point[0] * (1 - t) + end_point[0] * t)
            y = int(start_point[1] * (1 - t) + end_point[1] * t)

            if style == "dotted":
                cv2.circle(image, (x, y), thickness, color, -1)
            elif style == "dashed" and i % 2 == 0:
                # Draw dashed line segments
                if i + 1 < num_points:
                    next_t = (i + 1) / num_points
                    next_x = int(start_point[0] * (1 - next_t) + end_point[0] * next_t)
                    next_y = int(start_point[1] * (1 - next_t) + end_point[1] * next_t)
                    cv2.line(image, (x, y), (next_x, next_y), color, thickness)


from __future__ import annotations

import numpy as np
import pytest

from supervision.annotators.advanced import AdvancedAnnotator, HersheyFonts
from supervision.detection.core import Detections
from supervision.draw.color import ColorPalette
from test.test_utils import mock_detections


class TestAdvancedAnnotator:
    @pytest.fixture
    def advanced_annotator(self) -> AdvancedAnnotator:
        return AdvancedAnnotator()

    @pytest.fixture
    def sample_image(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_detections(self) -> Detections:
        return mock_detections(
            xyxy=[[100, 100, 200, 200], [250, 150, 350, 250]],
            class_id=[0, 1],
            confidence=[0.8, 0.9],
        )

    def test_annotator_initialization_default(self):
        annotator = AdvancedAnnotator()
        assert annotator.font == HersheyFonts.FONT_HERSHEY_SIMPLEX.value
        assert annotator.mask_alpha == 0.4
        assert annotator.text_separation == 15
        assert annotator._manual_font_scale is None
        assert annotator._manual_thickness is None
        assert annotator.color_palette == ColorPalette.DEFAULT
        assert annotator.show_labels is True
        assert annotator.show_masks is True
        assert annotator.mask_threshold == 0.3

    def test_annotator_initialization_custom(self):
        annotator = AdvancedAnnotator(
            font=HersheyFonts.FONT_HERSHEY_DUPLEX.value,
            mask_alpha=0.6,
            text_separation=20,
            font_scale=1.5,
            thickness=3,
            show_labels=False,
            show_masks=False,
            mask_threshold=0.5,
        )
        assert annotator.font == HersheyFonts.FONT_HERSHEY_DUPLEX.value
        assert annotator.mask_alpha == 0.6
        assert annotator.text_separation == 20
        assert annotator._manual_font_scale == 1.5
        assert annotator._manual_thickness == 3
        assert annotator.show_labels is False
        assert annotator.show_masks is False
        assert annotator.mask_threshold == 0.5

    def test_calculate_font_scale_auto(self, advanced_annotator):
        # Test auto-calculation
        font_scale = advanced_annotator._calculate_font_scale(640, 480)
        expected = max(np.sqrt(480 * 640) / 2050, 0.3)
        assert font_scale == expected

        # Test with small image
        small_scale = advanced_annotator._calculate_font_scale(100, 100)
        assert small_scale == 0.3  # minimum scale

    def test_calculate_font_scale_manual(self):
        annotator = AdvancedAnnotator(font_scale=2.0)
        font_scale = annotator._calculate_font_scale(640, 480)
        assert font_scale == 2.0

    def test_calculate_thickness_auto(self, advanced_annotator):
        # Small image (90000 < 512000)
        thickness = advanced_annotator._calculate_thickness(300, 300)
        assert thickness == 1

        # Medium image (1000000 between 512000 and 2073600)
        thickness = advanced_annotator._calculate_thickness(1000, 1000)
        assert thickness == 2

        # Large image (2073601 > 2073600)
        thickness = advanced_annotator._calculate_thickness(2000, 1500)
        assert thickness == 3

    def test_calculate_thickness_manual(self):
        annotator = AdvancedAnnotator(thickness=5)
        thickness = annotator._calculate_thickness(800, 600)
        assert thickness == 5

    def test_create_unique_color(self, advanced_annotator):
        color1 = advanced_annotator._create_unique_color(0)
        color2 = advanced_annotator._create_unique_color(1)
        color3 = advanced_annotator._create_unique_color(0)

        # Colors should be different for different tags
        assert color1 != color2
        # Same tag should produce same color
        assert color1 == color3
        # Colors should be BGR format with valid values
        assert all(0 <= c <= 255 for c in color1)
        assert len(color1) == 3

    def test_generate_colors(self, advanced_annotator, sample_detections):
        colors = advanced_annotator._generate_colors(sample_detections)

        assert len(colors) == len(sample_detections)
        assert all(len(color) == 3 for color in colors)
        assert all(all(0 <= c <= 255 for c in color) for color in colors)

    def test_annotate_basic(self, advanced_annotator, sample_image, sample_detections):
        annotated_image = advanced_annotator.annotate(sample_image, sample_detections)

        # Check that we get an image back
        assert isinstance(annotated_image, np.ndarray)
        assert annotated_image.shape == sample_image.shape
        assert annotated_image.dtype == sample_image.dtype

        # Image should be modified (not all zeros anymore)
        assert not np.array_equal(annotated_image, sample_image)

    def test_annotate_with_masks(self, advanced_annotator, sample_image):
        # Create detections with masks
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[25:75, 25:75] = True

        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[10:90, 10:90] = True

        detections_with_masks = mock_detections(
            xyxy=[[100, 100, 200, 200], [250, 150, 350, 250]],
            class_id=[0, 1],
            confidence=[0.8, 0.9],
            mask=[mask1, mask2],
        )

        annotated_image = advanced_annotator.annotate(
            sample_image, detections_with_masks
        )

        assert isinstance(annotated_image, np.ndarray)
        assert annotated_image.shape == sample_image.shape
        # Image should be modified due to masks
        assert not np.array_equal(annotated_image, sample_image)

    def test_annotate_no_labels(self, sample_image, sample_detections):
        annotator = AdvancedAnnotator(show_labels=False)
        annotated_image = annotator.annotate(sample_image, sample_detections)

        assert isinstance(annotated_image, np.ndarray)
        assert annotated_image.shape == sample_image.shape

    def test_annotate_no_masks(self, sample_image):
        annotator = AdvancedAnnotator(show_masks=False)

        # Create detections with masks
        mask = np.ones((100, 100), dtype=bool)
        detections_with_masks = mock_detections(
            xyxy=[[100, 100, 200, 200]],
            class_id=[0],
            confidence=[0.8],
            mask=[mask],
        )

        annotated_image = annotator.annotate(sample_image, detections_with_masks)

        assert isinstance(annotated_image, np.ndarray)
        assert annotated_image.shape == sample_image.shape

    def test_annotate_empty_detections(self, advanced_annotator, sample_image):
        empty_detections = Detections.empty()
        annotated_image = advanced_annotator.annotate(sample_image, empty_detections)

        # Should return unchanged image
        assert np.array_equal(annotated_image, sample_image)

    def test_draw_info_text(self, advanced_annotator, sample_image):
        text_lines = ["Info line 1", "Info line 2", "Info line 3"]
        annotated_image = advanced_annotator.draw_info_text(
            sample_image, text_lines, position=(50, 50)
        )

        assert isinstance(annotated_image, np.ndarray)
        assert annotated_image.shape == sample_image.shape
        assert not np.array_equal(annotated_image, sample_image)

    def test_draw_styled_polygon_solid(self, advanced_annotator, sample_image):
        points = np.array([[100, 100], [200, 100], [150, 200]], dtype=np.float32)
        annotated_image = advanced_annotator.draw_styled_polygon(
            sample_image, points, style="solid"
        )

        assert isinstance(annotated_image, np.ndarray)
        assert annotated_image.shape == sample_image.shape
        assert not np.array_equal(annotated_image, sample_image)

    def test_draw_styled_polygon_dotted(self, advanced_annotator, sample_image):
        points = np.array([[100, 100], [200, 100], [150, 200]], dtype=np.float32)
        annotated_image = advanced_annotator.draw_styled_polygon(
            sample_image, points, style="dotted", gap=10
        )

        assert isinstance(annotated_image, np.ndarray)
        assert annotated_image.shape == sample_image.shape
        assert not np.array_equal(annotated_image, sample_image)

    def test_draw_styled_polygon_dashed(self, advanced_annotator, sample_image):
        points = np.array([[100, 100], [200, 100], [150, 200]], dtype=np.float32)
        annotated_image = advanced_annotator.draw_styled_polygon(
            sample_image, points, style="dashed", gap=15
        )

        assert isinstance(annotated_image, np.ndarray)
        assert annotated_image.shape == sample_image.shape
        assert not np.array_equal(annotated_image, sample_image)

    def test_annotate_different_image_sizes(
        self, advanced_annotator, sample_detections
    ):
        # Small image
        small_image = np.zeros((240, 320, 3), dtype=np.uint8)
        annotated_small = advanced_annotator.annotate(small_image, sample_detections)
        assert annotated_small.shape == small_image.shape

        # Large image
        large_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        annotated_large = advanced_annotator.annotate(large_image, sample_detections)
        assert annotated_large.shape == large_image.shape

    def test_annotate_without_confidence(self, advanced_annotator, sample_image):
        detections_no_conf = mock_detections(
            xyxy=[[100, 100, 200, 200]],
            class_id=[0],
        )  # No confidence provided

        annotated_image = advanced_annotator.annotate(sample_image, detections_no_conf)

        assert isinstance(annotated_image, np.ndarray)
        assert annotated_image.shape == sample_image.shape

    def test_annotate_without_class_id(self, advanced_annotator, sample_image):
        detections_no_class = mock_detections(
            xyxy=[[100, 100, 200, 200]],
            confidence=[0.8],
        )  # No class_id provided

        annotated_image = advanced_annotator.annotate(sample_image, detections_no_class)

        assert isinstance(annotated_image, np.ndarray)
        assert annotated_image.shape == sample_image.shape

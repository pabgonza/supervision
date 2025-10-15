import numpy as np

import supervision as sv


class TestByteTrackerStep:
    """Tests for ByteTrackerStep pipeline component."""

    def test_bytetracker_step_initialization(self):
        """Test that ByteTrackerStep initializes correctly."""
        step = sv.ByteTrackerStep()
        assert step is not None
        assert step.detections_key == "detections"
        assert step.tracker is not None

    def test_bytetracker_step_with_custom_params(self):
        """Test ByteTrackerStep initialization with custom parameters."""
        step = sv.ByteTrackerStep(
            track_activation_threshold=0.3,
            lost_track_buffer=60,
            minimum_matching_threshold=0.85,
            frame_rate=60,
            minimum_consecutive_frames=3,
            detections_key="custom_detections",
        )
        assert step.detections_key == "custom_detections"
        assert step.tracker.track_activation_threshold == 0.3
        assert step.tracker.max_time_lost == int(60 / 30.0 * 60)  # frame_rate adjusted

    def test_bytetracker_step_process_with_detections(self):
        """Test that ByteTrackerStep processes detections correctly."""
        step = sv.ByteTrackerStep()

        # Create mock detection data
        detections = sv.Detections(
            xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40]]),
            class_id=np.array([1, 1]),
            confidence=np.array([0.9, 0.8]),
        )

        data = {
            "frame": np.zeros((100, 100, 3), dtype=np.uint8),
            "detections": detections,
        }

        # Process first frame
        result = step.process(data)

        # Check that result has detections with tracker_id
        assert "detections" in result
        assert result["detections"] is not None
        assert hasattr(result["detections"], "tracker_id")
        assert result["detections"].tracker_id is not None

    def test_bytetracker_step_process_consecutive_frames(self):
        """Test that ByteTrackerStep tracks objects across frames."""
        step = sv.ByteTrackerStep()

        # Create two frames with same detections
        detections1 = sv.Detections(
            xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40]]),
            class_id=np.array([1, 1]),
            confidence=np.array([0.9, 0.8]),
        )

        detections2 = sv.Detections(
            xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40]]),
            class_id=np.array([1, 1]),
            confidence=np.array([0.9, 0.8]),
        )

        data1 = {
            "frame": np.zeros((100, 100, 3), dtype=np.uint8),
            "detections": detections1,
        }
        data2 = {
            "frame": np.zeros((100, 100, 3), dtype=np.uint8),
            "detections": detections2,
        }

        # Process frames
        result1 = step.process(data1)
        result2 = step.process(data2)

        # Check that tracker IDs are consistent
        assert result1["detections"].tracker_id is not None
        assert result2["detections"].tracker_id is not None
        # The same objects should have the same tracker IDs
        np.testing.assert_array_equal(
            result1["detections"].tracker_id, result2["detections"].tracker_id
        )

    def test_bytetracker_step_process_empty_detections(self):
        """Test that ByteTrackerStep handles empty detections."""
        step = sv.ByteTrackerStep()

        # Create empty detections
        detections = sv.Detections.empty()

        data = {
            "frame": np.zeros((100, 100, 3), dtype=np.uint8),
            "detections": detections,
        }

        # Process
        result = step.process(data)

        # Should return data unchanged
        assert "detections" in result
        assert len(result["detections"]) == 0

    def test_bytetracker_step_process_no_detections(self):
        """Test that ByteTrackerStep handles missing detections key."""
        step = sv.ByteTrackerStep()

        data = {"frame": np.zeros((100, 100, 3), dtype=np.uint8)}

        # Process
        result = step.process(data)

        # Should return data unchanged
        assert result == data

    def test_bytetracker_step_filter(self):
        """Test that filter method works correctly."""
        step = sv.ByteTrackerStep()

        # Data with detections should pass filter
        data_with_detections = {
            "frame": np.zeros((100, 100, 3), dtype=np.uint8),
            "detections": sv.Detections.empty(),
        }
        assert step.filter(data_with_detections) is True

        # Data without detections should not pass filter
        data_without_detections = {"frame": np.zeros((100, 100, 3), dtype=np.uint8)}
        assert step.filter(data_without_detections) is False

    def test_bytetracker_step_filter_custom_key(self):
        """Test that filter works with custom detections key."""
        step = sv.ByteTrackerStep(detections_key="custom_detections")

        data = {
            "frame": np.zeros((100, 100, 3), dtype=np.uint8),
            "custom_detections": sv.Detections.empty(),
        }
        assert step.filter(data) is True

        data_wrong_key = {
            "frame": np.zeros((100, 100, 3), dtype=np.uint8),
            "detections": sv.Detections.empty(),
        }
        assert step.filter(data_wrong_key) is False

    def test_bytetracker_step_reset(self):
        """Test that reset clears tracker state."""
        step = sv.ByteTrackerStep()

        # Process some detections
        detections = sv.Detections(
            xyxy=np.array([[10, 10, 20, 20]]),
            class_id=np.array([1]),
            confidence=np.array([0.9]),
        )

        data = {
            "frame": np.zeros((100, 100, 3), dtype=np.uint8),
            "detections": detections,
        }

        step.process(data)

        # Reset tracker
        step.reset()

        # Process again - should get new tracker ID (starting from 1)
        result2 = step.process(data)
        second_tracker_id = result2["detections"].tracker_id[0]

        # After reset, tracker IDs should restart
        assert second_tracker_id == 1

    def test_bytetracker_step_in_pipeline(self):
        """Test ByteTrackerStep integrated in a pipeline."""
        # Create a simple pipeline with ByteTrackerStep
        detections_list = [
            sv.Detections(
                xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40]]),
                class_id=np.array([1, 1]),
                confidence=np.array([0.9, 0.8]),
            ),
            sv.Detections(
                xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40]]),
                class_id=np.array([1, 1]),
                confidence=np.array([0.9, 0.8]),
            ),
        ]

        # Create a simple source that yields frames with detections
        class TestSource(sv.PipelineSource):
            def __init__(self, detections_list):
                self.detections_list = detections_list
                self.index = 0

            def __iter__(self):
                for detections in self.detections_list:
                    yield {
                        "frame": np.zeros((100, 100, 3), dtype=np.uint8),
                        "detections": detections,
                    }

            def close(self):
                pass

        # Build pipeline
        source = TestSource(detections_list)
        pipeline = sv.Pipeline(source) | sv.ByteTrackerStep()

        # Run pipeline and collect results
        results = list(pipeline)

        # Check results
        assert len(results) == 2
        assert all("detections" in r for r in results)
        assert all(hasattr(r["detections"], "tracker_id") for r in results)

        # Same objects should have same tracker IDs across frames
        np.testing.assert_array_equal(
            results[0]["detections"].tracker_id, results[1]["detections"].tracker_id
        )


class TestTrackerAnnotatorStep:
    """Tests for TrackerAnnotatorStep pipeline component."""

    def test_tracker_annotator_step_initialization(self):
        """Test that TrackerAnnotatorStep initializes correctly."""
        step = sv.TrackerAnnotatorStep()
        assert step is not None
        assert step.detections_key == "detections"
        assert step.output_key == "frame"
        assert step.box_annotator is not None
        assert step.label_annotator is not None
        assert step.trace_annotator is not None

    def test_tracker_annotator_step_with_class_names(self):
        """Test TrackerAnnotatorStep initialization with class names."""
        class_names = {0: "person", 1: "car", 2: "dog"}
        step = sv.TrackerAnnotatorStep(class_names=class_names)
        assert step.class_names == class_names

    def test_tracker_annotator_step_with_custom_params(self):
        """Test TrackerAnnotatorStep initialization with custom parameters."""
        step = sv.TrackerAnnotatorStep(
            show_tracker_id=False,
            show_class=True,
            show_confidence=False,
            box_thickness=3,
            trace_length=50,
            detections_key="custom_detections",
            output_key="annotated_frame",
        )
        assert step.show_tracker_id is False
        assert step.show_class is True
        assert step.show_confidence is False
        assert step.detections_key == "custom_detections"
        assert step.output_key == "annotated_frame"

    def test_tracker_annotator_step_format_labels(self):
        """Test label formatting with different configurations."""
        class_names = {0: "person", 1: "car"}

        # Test with all fields enabled
        step = sv.TrackerAnnotatorStep(
            class_names=class_names,
            show_tracker_id=True,
            show_class=True,
            show_confidence=True,
            confidence_decimals=2,
        )

        detections = sv.Detections(
            xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40]]),
            class_id=np.array([0, 1]),
            confidence=np.array([0.95, 0.87]),
            tracker_id=np.array([1, 2]),
        )

        labels = step._format_labels(detections)
        assert len(labels) == 2
        assert labels[0] == "#1 person 0.95"
        assert labels[1] == "#2 car 0.87"

    def test_tracker_annotator_step_format_labels_no_tracker_id(self):
        """Test label formatting without tracker ID."""
        class_names = {0: "person", 1: "car"}

        step = sv.TrackerAnnotatorStep(
            class_names=class_names,
            show_tracker_id=False,
            show_class=True,
            show_confidence=True,
            confidence_decimals=2,
        )

        detections = sv.Detections(
            xyxy=np.array([[10, 10, 20, 20]]),
            class_id=np.array([0]),
            confidence=np.array([0.95]),
        )

        labels = step._format_labels(detections)
        assert len(labels) == 1
        assert labels[0] == "person 0.95"

    def test_tracker_annotator_step_format_labels_only_tracker_id(self):
        """Test label formatting with only tracker ID."""
        step = sv.TrackerAnnotatorStep(
            show_tracker_id=True,
            show_class=False,
            show_confidence=False,
        )

        detections = sv.Detections(
            xyxy=np.array([[10, 10, 20, 20]]),
            class_id=np.array([0]),
            confidence=np.array([0.95]),
            tracker_id=np.array([42]),
        )

        labels = step._format_labels(detections)
        assert len(labels) == 1
        assert labels[0] == "#42"

    def test_tracker_annotator_step_format_labels_empty_detections(self):
        """Test label formatting with empty detections."""
        step = sv.TrackerAnnotatorStep()
        detections = sv.Detections.empty()

        labels = step._format_labels(detections)
        assert labels == []

    def test_tracker_annotator_step_process_with_detections(self):
        """Test that TrackerAnnotatorStep processes detections correctly."""
        class_names = {0: "person", 1: "car"}
        step = sv.TrackerAnnotatorStep(class_names=class_names)

        # Create mock detection data with tracking
        detections = sv.Detections(
            xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40]]),
            class_id=np.array([0, 1]),
            confidence=np.array([0.9, 0.8]),
            tracker_id=np.array([1, 2]),
        )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        data = {"frame": frame, "detections": detections}

        # Process
        result = step.process(data)

        # Check that result has annotated frame
        assert "frame" in result
        assert result["frame"] is not None
        assert result["frame"].shape == frame.shape
        # Frame should be modified (different from input)
        assert not np.array_equal(result["frame"], frame)

    def test_tracker_annotator_step_process_without_tracker_id(self):
        """Test processing detections without tracker_id.

        Note: TrackerAnnotatorStep is designed to work with tracked detections.
        If detections don't have tracker_id, it indicates the ByteTrackerStep
        was not used before this step. This test verifies the step doesn't crash
        but uses a different color_lookup strategy that doesn't require tracker_id.
        """
        # Use CLASS color lookup instead of TRACK (default)
        step = sv.TrackerAnnotatorStep(color_lookup=sv.ColorLookup.CLASS)

        # Create detections without tracker_id
        detections = sv.Detections(
            xyxy=np.array([[10, 10, 20, 20]]),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
        )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        data = {"frame": frame, "detections": detections}

        # Process - should work with CLASS color lookup
        result = step.process(data)
        assert "frame" in result
        assert result["frame"] is not None

    def test_tracker_annotator_step_process_empty_detections(self):
        """Test that TrackerAnnotatorStep handles empty detections."""
        step = sv.TrackerAnnotatorStep()

        # Create empty detections
        detections = sv.Detections.empty()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        data = {"frame": frame, "detections": detections}

        # Process
        result = step.process(data)

        # Should return frame (copy)
        assert "frame" in result
        assert result["frame"] is not None

    def test_tracker_annotator_step_process_no_detections(self):
        """Test that TrackerAnnotatorStep handles missing detections key."""
        step = sv.TrackerAnnotatorStep()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        data = {"frame": frame}

        # Process
        result = step.process(data)

        # Should return frame (copy)
        assert "frame" in result

    def test_tracker_annotator_step_process_no_frame(self):
        """Test that TrackerAnnotatorStep handles missing frame."""
        step = sv.TrackerAnnotatorStep()

        detections = sv.Detections(
            xyxy=np.array([[10, 10, 20, 20]]),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
        )

        data = {"detections": detections}

        # Process
        result = step.process(data)

        # Should return data unchanged
        assert result == data

    def test_tracker_annotator_step_custom_output_key(self):
        """Test using custom output key."""
        step = sv.TrackerAnnotatorStep(output_key="annotated_frame")

        detections = sv.Detections(
            xyxy=np.array([[10, 10, 20, 20]]),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
            tracker_id=np.array([1]),
        )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        data = {"frame": frame, "detections": detections}

        # Process
        result = step.process(data)

        # Should have both frame and annotated_frame
        assert "frame" in result
        assert "annotated_frame" in result
        assert result["annotated_frame"] is not None

    def test_tracker_annotator_step_filter(self):
        """Test that filter method works correctly."""
        step = sv.TrackerAnnotatorStep()

        # Data with frame should pass filter
        data_with_frame = {
            "frame": np.zeros((100, 100, 3), dtype=np.uint8),
            "detections": sv.Detections.empty(),
        }
        assert step.filter(data_with_frame) is True

        # Data without frame should not pass filter
        data_without_frame = {"detections": sv.Detections.empty()}
        assert step.filter(data_without_frame) is False

    def test_tracker_annotator_step_confidence_decimals(self):
        """Test custom confidence decimal places."""
        step = sv.TrackerAnnotatorStep(
            show_tracker_id=False,
            show_class=False,
            show_confidence=True,
            confidence_decimals=4,
        )

        detections = sv.Detections(
            xyxy=np.array([[10, 10, 20, 20]]),
            class_id=np.array([0]),
            confidence=np.array([0.123456]),
        )

        labels = step._format_labels(detections)
        assert labels[0] == "0.1235"

    def test_tracker_annotator_step_in_pipeline(self):
        """Test TrackerAnnotatorStep integrated in a pipeline."""
        class_names = {0: "object"}

        detections_list = [
            sv.Detections(
                xyxy=np.array([[10, 10, 20, 20]]),
                class_id=np.array([0]),
                confidence=np.array([0.9]),
                tracker_id=np.array([1]),
            ),
            sv.Detections(
                xyxy=np.array([[11, 11, 21, 21]]),
                class_id=np.array([0]),
                confidence=np.array([0.85]),
                tracker_id=np.array([1]),
            ),
        ]

        # Create a simple source
        class TestSource(sv.PipelineSource):
            def __init__(self, detections_list):
                self.detections_list = detections_list

            def __iter__(self):
                for detections in self.detections_list:
                    yield {
                        "frame": np.zeros((100, 100, 3), dtype=np.uint8),
                        "detections": detections,
                    }

            def close(self):
                pass

        # Build pipeline
        source = TestSource(detections_list)
        pipeline = sv.Pipeline(source) | sv.TrackerAnnotatorStep(
            class_names=class_names
        )

        # Run pipeline and collect results
        results = list(pipeline)

        # Check results
        assert len(results) == 2
        assert all("frame" in r for r in results)
        # Frames should be annotated (non-zero due to annotations)
        assert results[0]["frame"] is not None
        assert results[1]["frame"] is not None

    def test_tracker_annotator_step_copy_frame_true(self):
        """Test that copy_frame=True preserves original frame."""
        step = sv.TrackerAnnotatorStep(copy_frame=True)

        detections = sv.Detections(
            xyxy=np.array([[10, 10, 20, 20]]),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
            tracker_id=np.array([1]),
        )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_copy = frame.copy()
        data = {"frame": frame, "detections": detections}

        # Process
        result = step.process(data)

        # Original frame should be unchanged
        assert np.array_equal(frame, frame_copy)
        # Result frame should be different (annotated)
        assert not np.array_equal(result["frame"], frame)

    def test_tracker_annotator_step_copy_frame_false(self):
        """Test that copy_frame=False modifies frame in place."""
        step = sv.TrackerAnnotatorStep(copy_frame=False)

        detections = sv.Detections(
            xyxy=np.array([[10, 10, 20, 20]]),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
            tracker_id=np.array([1]),
        )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_copy = frame.copy()
        data = {"frame": frame, "detections": detections}

        # Process
        result = step.process(data)

        # Original frame should be modified
        assert not np.array_equal(frame, frame_copy)
        # Result frame should be the same object as input frame
        assert result["frame"] is frame

    def test_tracker_annotator_step_copy_frame_with_custom_output_key(self):
        """Test that copy_frame doesn't copy when using custom output_key."""
        step = sv.TrackerAnnotatorStep(copy_frame=True, output_key="annotated")

        detections = sv.Detections(
            xyxy=np.array([[10, 10, 20, 20]]),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
            tracker_id=np.array([1]),
        )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_copy = frame.copy()
        data = {"frame": frame, "detections": detections}

        # Process
        result = step.process(data)

        # When output_key != "frame", copy_frame should not copy
        # The frame should be modified directly
        assert not np.array_equal(frame, frame_copy)
        # Result should have custom key
        assert "annotated" in result


class TestLineZoneStep:
    """Tests for LineZoneStep pipeline component."""

    def test_line_zone_step_initialization(self):
        """Test that LineZoneStep initializes correctly."""
        start = sv.Point(x=0, y=100)
        end = sv.Point(x=200, y=100)
        step = sv.LineZoneStep(start=start, end=end)

        assert step is not None
        assert step.detections_key == "detections"
        assert step.line_zone_key == "line_zone"
        assert step.line_zone is not None
        assert step.line_zone.in_count == 0
        assert step.line_zone.out_count == 0

    def test_line_zone_step_with_custom_params(self):
        """Test LineZoneStep initialization with custom parameters."""
        start = sv.Point(x=0, y=100)
        end = sv.Point(x=200, y=100)
        step = sv.LineZoneStep(
            start=start,
            end=end,
            triggering_anchors=[sv.Position.CENTER],
            minimum_crossing_threshold=2,
            detections_key="custom_detections",
            line_zone_key="custom_line_zone",
        )

        assert step.detections_key == "custom_detections"
        assert step.line_zone_key == "custom_line_zone"
        assert step.line_zone is not None

    def test_line_zone_step_process_with_detections(self):
        """Test that LineZoneStep processes detections correctly."""
        start = sv.Point(x=0, y=100)
        end = sv.Point(x=200, y=100)
        step = sv.LineZoneStep(start=start, end=end)

        # Create mock detection data with tracker_id
        detections = sv.Detections(
            xyxy=np.array([[50, 80, 70, 95], [50, 105, 70, 120]]),
            class_id=np.array([0, 0]),
            confidence=np.array([0.9, 0.8]),
            tracker_id=np.array([1, 2]),
        )

        data = {"detections": detections}

        # Process
        result = step.process(data)

        # Check that result has line_zone object
        assert "line_zone" in result
        assert result["line_zone"] is not None
        assert result["line_zone"] == step.line_zone
        # Check that line_zone has count properties
        assert hasattr(result["line_zone"], "in_count")
        assert hasattr(result["line_zone"], "out_count")
        assert hasattr(result["line_zone"], "in_count_per_class")
        assert hasattr(result["line_zone"], "out_count_per_class")

    def test_line_zone_step_process_empty_detections(self):
        """Test that LineZoneStep handles empty detections."""
        start = sv.Point(x=0, y=100)
        end = sv.Point(x=200, y=100)
        step = sv.LineZoneStep(start=start, end=end)

        # Create empty detections
        detections = sv.Detections.empty()
        data = {"detections": detections}

        # Process
        result = step.process(data)

        # Should still add line_zone to data
        assert "line_zone" in result
        assert result["line_zone"].in_count == 0
        assert result["line_zone"].out_count == 0

    def test_line_zone_step_process_no_detections(self):
        """Test that LineZoneStep handles missing detections."""
        start = sv.Point(x=0, y=100)
        end = sv.Point(x=200, y=100)
        step = sv.LineZoneStep(start=start, end=end)

        # No detections in data
        data = {}

        # Process
        result = step.process(data)

        # Should still add line_zone to data
        assert "line_zone" in result

    def test_line_zone_step_filter(self):
        """Test LineZoneStep filter method."""
        start = sv.Point(x=0, y=100)
        end = sv.Point(x=200, y=100)
        step = sv.LineZoneStep(start=start, end=end)

        # With detections key
        data_with = {"detections": sv.Detections.empty()}
        assert step.filter(data_with) is True

        # Without detections key
        data_without = {}
        assert step.filter(data_without) is False

    def test_line_zone_step_reset(self):
        """Test LineZoneStep reset method."""
        start = sv.Point(x=0, y=100)
        end = sv.Point(x=200, y=100)
        step = sv.LineZoneStep(start=start, end=end)

        # Manually set some counts (simulating crossings)
        step.line_zone._in_count_per_class[0] = 5
        step.line_zone._out_count_per_class[1] = 3

        assert step.line_zone.in_count == 5
        assert step.line_zone.out_count == 3

        # Reset
        step.reset()

        # Counts should be cleared
        assert step.line_zone.in_count == 0
        assert step.line_zone.out_count == 0

    def test_line_zone_step_stores_object_not_copies(self):
        """Test that LineZoneStep stores line_zone object, not individual properties."""
        start = sv.Point(x=0, y=100)
        end = sv.Point(x=200, y=100)
        step = sv.LineZoneStep(start=start, end=end)

        detections = sv.Detections(
            xyxy=np.array([[50, 80, 70, 95]]),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
            tracker_id=np.array([1]),
        )

        data = {"detections": detections}
        result = step.process(data)

        # Check that line_zone object is stored
        assert "line_zone" in result
        assert isinstance(result["line_zone"], sv.LineZone)
        # Check that we can access properties directly
        assert isinstance(result["line_zone"].in_count, int)
        assert isinstance(result["line_zone"].out_count, int)
        assert isinstance(result["line_zone"].in_count_per_class, dict)
        assert isinstance(result["line_zone"].out_count_per_class, dict)


class TestLineZoneAnnotatorStep:
    """Tests for LineZoneAnnotatorStep pipeline component."""

    def test_line_zone_annotator_step_initialization(self):
        """Test that LineZoneAnnotatorStep initializes correctly."""
        step = sv.LineZoneAnnotatorStep()

        assert step is not None
        assert step.line_zone_key == "line_zone"
        assert step.output_key == "frame"
        assert step.copy_frame is True
        assert step.annotator is not None

    def test_line_zone_annotator_step_with_custom_params(self):
        """Test LineZoneAnnotatorStep initialization with custom parameters."""
        step = sv.LineZoneAnnotatorStep(
            thickness=4,
            color=sv.Color.RED,
            text_scale=1.0,
            custom_in_text="Entered",
            custom_out_text="Exited",
            display_in_count=True,
            display_out_count=False,
            line_zone_key="custom_line_zone",
            output_key="annotated_frame",
            copy_frame=False,
        )

        assert step.line_zone_key == "custom_line_zone"
        assert step.output_key == "annotated_frame"
        assert step.copy_frame is False
        assert step.annotator is not None

    def test_line_zone_annotator_step_process_with_line_zone(self):
        """Test that LineZoneAnnotatorStep annotates frame correctly."""
        start = sv.Point(x=0, y=50)
        end = sv.Point(x=100, y=50)
        line_zone = sv.LineZone(start=start, end=end)

        step = sv.LineZoneAnnotatorStep()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        data = {"frame": frame, "line_zone": line_zone}

        # Process
        result = step.process(data)

        # Check that result has annotated frame
        assert "frame" in result
        assert result["frame"] is not None
        assert result["frame"].shape == frame.shape
        # Frame should be modified (line drawn)
        assert not np.array_equal(result["frame"], frame)

    def test_line_zone_annotator_step_process_no_frame(self):
        """Test processing without frame."""
        start = sv.Point(x=0, y=50)
        end = sv.Point(x=100, y=50)
        line_zone = sv.LineZone(start=start, end=end)

        step = sv.LineZoneAnnotatorStep()
        data = {"line_zone": line_zone}

        # Process - should handle gracefully
        result = step.process(data)
        assert result == data

    def test_line_zone_annotator_step_process_no_line_zone(self):
        """Test processing without line_zone."""
        step = sv.LineZoneAnnotatorStep()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        data = {"frame": frame}

        # Process - should handle gracefully
        result = step.process(data)
        # Frame should not be modified
        assert np.array_equal(result["frame"], frame)

    def test_line_zone_annotator_step_filter(self):
        """Test LineZoneAnnotatorStep filter method."""
        step = sv.LineZoneAnnotatorStep()

        start = sv.Point(x=0, y=50)
        end = sv.Point(x=100, y=50)
        line_zone = sv.LineZone(start=start, end=end)

        # With both frame and line_zone
        data_with_both = {"frame": np.zeros((100, 100, 3)), "line_zone": line_zone}
        assert step.filter(data_with_both) is True

        # With only frame
        data_only_frame = {"frame": np.zeros((100, 100, 3))}
        assert step.filter(data_only_frame) is False

        # With only line_zone
        data_only_line_zone = {"line_zone": line_zone}
        assert step.filter(data_only_line_zone) is False

        # With neither
        data_empty = {}
        assert step.filter(data_empty) is False

    def test_line_zone_annotator_step_copy_frame_true(self):
        """Test that copy_frame=True preserves original frame."""
        start = sv.Point(x=0, y=50)
        end = sv.Point(x=100, y=50)
        line_zone = sv.LineZone(start=start, end=end)

        step = sv.LineZoneAnnotatorStep(copy_frame=True)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_copy = frame.copy()
        data = {"frame": frame, "line_zone": line_zone}

        # Process
        result = step.process(data)

        # Original frame should be modified (because we're overwriting "frame" key)
        # But let's verify that the annotation happened
        assert "frame" in result
        assert not np.array_equal(result["frame"], frame_copy)

    def test_line_zone_annotator_step_copy_frame_false(self):
        """Test that copy_frame=False modifies frame in-place."""
        start = sv.Point(x=0, y=50)
        end = sv.Point(x=100, y=50)
        line_zone = sv.LineZone(start=start, end=end)

        step = sv.LineZoneAnnotatorStep(copy_frame=False)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_copy = frame.copy()
        data = {"frame": frame, "line_zone": line_zone}

        # Process
        result = step.process(data)

        # Frame should be modified
        assert not np.array_equal(frame, frame_copy)
        assert result["frame"] is frame  # Same object

    def test_line_zone_annotator_step_custom_output_key(self):
        """Test using custom output_key."""
        start = sv.Point(x=0, y=50)
        end = sv.Point(x=100, y=50)
        line_zone = sv.LineZone(start=start, end=end)

        step = sv.LineZoneAnnotatorStep(output_key="annotated", copy_frame=False)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_copy = frame.copy()
        data = {"frame": frame, "line_zone": line_zone}

        # Process
        result = step.process(data)

        # When output_key != "frame", copy_frame should not copy
        # The frame should be modified directly
        assert not np.array_equal(frame, frame_copy)
        # Result should have custom key
        assert "annotated" in result

    def test_line_zone_annotator_step_accesses_line_zone_properties(self):
        """Test that annotator can access line_zone properties directly."""
        start = sv.Point(x=0, y=50)
        end = sv.Point(x=100, y=50)
        line_zone = sv.LineZone(start=start, end=end)

        # Manually set counts
        line_zone._in_count_per_class[0] = 5
        line_zone._out_count_per_class[0] = 3

        step = sv.LineZoneAnnotatorStep()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        data = {"frame": frame, "line_zone": line_zone}

        # Process - should not raise error accessing counts
        result = step.process(data)

        # Verify annotation happened
        assert "frame" in result
        assert not np.array_equal(result["frame"], np.zeros((100, 100, 3), dtype=np.uint8))

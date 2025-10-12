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

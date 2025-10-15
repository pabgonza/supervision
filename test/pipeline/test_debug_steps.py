import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import supervision as sv


class TestDebugLoggerStep:
    """Tests for DebugLoggerStep pipeline component."""

    def test_debug_logger_initialization(self):
        """Test that DebugLoggerStep initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.json"
            step = sv.DebugLoggerStep(log_file=log_file)

            assert step is not None
            assert step.log_file == log_file
            assert step.log_interval == 30
            assert step.include_detections_data is False
            assert step.auto_flush is True
            assert log_file.exists()

    def test_debug_logger_custom_params(self):
        """Test DebugLoggerStep initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "custom.json"
            step = sv.DebugLoggerStep(
                log_file=log_file,
                log_interval=60,
                include_detections_data=True,
                custom_metrics=["fps", "sequence_number"],
                auto_flush=False,
            )

            assert step.log_interval == 60
            assert step.include_detections_data is True
            assert step.custom_metrics == ["fps", "sequence_number"]
            assert step.auto_flush is False

    def test_debug_logger_process_basic(self):
        """Test that DebugLoggerStep processes data correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.json"
            step = sv.DebugLoggerStep(log_file=log_file, auto_flush=False)

            # Create test data
            data = {
                "frame": np.zeros((100, 100, 3), dtype=np.uint8),
                "fps": 30.0,
            }

            # Process
            result = step.process(data)

            # Check that data is unchanged (pass-through)
            assert result == data
            assert "frame" in result
            assert result["fps"] == 30.0

            # Check that metrics were buffered
            assert len(step._metrics_buffer) == 1
            assert step._metrics_buffer[0]["frame_number"] == 1
            assert step._metrics_buffer[0]["fps"] == 30.0

    def test_debug_logger_process_with_detections(self):
        """Test that DebugLoggerStep captures detection metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.json"
            step = sv.DebugLoggerStep(log_file=log_file, auto_flush=False)

            # Create detections
            detections = sv.Detections(
                xyxy=np.array([[10, 10, 20, 20], [30, 30, 40, 40], [50, 50, 60, 60]]),
                class_id=np.array([0, 0, 1]),
                confidence=np.array([0.9, 0.8, 0.7]),
                tracker_id=np.array([1, 2, 3]),
            )

            data = {
                "frame": np.zeros((100, 100, 3), dtype=np.uint8),
                "fps": 25.0,
                "detections": detections,
            }

            # Process
            result = step.process(data)

            # Check metrics
            assert len(step._metrics_buffer) == 1
            metrics = step._metrics_buffer[0]

            assert metrics["detection_count"] == 3
            assert metrics["tracked_count"] == 3
            assert metrics["unique_tracker_ids"] == 3
            assert metrics["confidence_mean"] == pytest.approx(0.8, abs=0.01)
            assert metrics["confidence_min"] == 0.7
            assert metrics["confidence_max"] == 0.9
            assert 0 in metrics["class_distribution"]
            assert 1 in metrics["class_distribution"]
            assert metrics["class_distribution"][0] == 2
            assert metrics["class_distribution"][1] == 1

    def test_debug_logger_process_with_detailed_detections(self):
        """Test that detailed detection data is captured when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.json"
            step = sv.DebugLoggerStep(
                log_file=log_file,
                include_detections_data=True,
                auto_flush=False,
            )

            # Create detections
            detections = sv.Detections(
                xyxy=np.array([[10, 10, 20, 20]]),
                class_id=np.array([0]),
                confidence=np.array([0.9]),
            )

            data = {
                "frame": np.zeros((100, 100, 3), dtype=np.uint8),
                "detections": detections,
            }

            # Process
            step.process(data)

            # Check that detailed data is included
            metrics = step._metrics_buffer[0]
            assert "detections" in metrics
            assert "xyxy" in metrics["detections"]
            assert "confidence" in metrics["detections"]
            assert "class_id" in metrics["detections"]
            assert metrics["detections"]["xyxy"] == [[10, 10, 20, 20]]
            assert metrics["detections"]["confidence"] == [0.9]
            assert metrics["detections"]["class_id"] == [0]

    def test_debug_logger_custom_metrics(self):
        """Test that custom metrics are captured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.json"
            step = sv.DebugLoggerStep(
                log_file=log_file,
                custom_metrics=["sequence_number", "roi_size"],
                auto_flush=False,
            )

            data = {
                "frame": np.zeros((100, 100, 3), dtype=np.uint8),
                "sequence_number": 42,
                "roi_size": 1024,
            }

            step.process(data)

            metrics = step._metrics_buffer[0]
            assert metrics["sequence_number"] == 42
            assert metrics["roi_size"] == 1024

    def test_debug_logger_flush(self):
        """Test that flush writes metrics to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.json"
            step = sv.DebugLoggerStep(log_file=log_file, auto_flush=False)

            # Process some frames
            for i in range(5):
                data = {
                    "frame": np.zeros((100, 100, 3), dtype=np.uint8),
                    "fps": 30.0 + i,
                }
                step.process(data)

            # Check buffer has data
            assert len(step._metrics_buffer) == 5

            # Flush
            step.flush()

            # Check buffer is cleared
            assert len(step._metrics_buffer) == 0

            # Check file has data
            with open(log_file, "r") as f:
                data = json.load(f)
            assert len(data) == 5
            assert data[0]["frame_number"] == 1
            assert data[0]["fps"] == 30.0
            assert data[4]["frame_number"] == 5
            assert data[4]["fps"] == 34.0

    def test_debug_logger_auto_flush(self):
        """Test that auto-flush works at specified interval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.json"
            step = sv.DebugLoggerStep(
                log_file=log_file,
                log_interval=3,  # Flush every 3 frames
                auto_flush=True,
            )

            # Process 5 frames
            for i in range(5):
                data = {
                    "frame": np.zeros((100, 100, 3), dtype=np.uint8),
                    "fps": 30.0 + i,
                }
                step.process(data)

            # After 3 frames, should have flushed once
            # Buffer should have 2 frames (4th and 5th)
            assert len(step._metrics_buffer) == 2

            # Check file has first 3 frames
            with open(log_file, "r") as f:
                data = json.load(f)
            assert len(data) == 3
            assert data[0]["frame_number"] == 1
            assert data[2]["frame_number"] == 3

    def test_debug_logger_statistics(self):
        """Test that statistics are calculated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.json"
            step = sv.DebugLoggerStep(log_file=log_file, auto_flush=False)

            # Process frames with varying FPS and detection counts
            fps_values = [30.0, 28.0, 32.0, 29.0, 31.0]
            det_counts = [5, 3, 7, 4, 6]

            for fps, det_count in zip(fps_values, det_counts):
                detections = sv.Detections(
                    xyxy=np.random.rand(det_count, 4) * 100,
                    confidence=np.random.rand(det_count),
                )
                data = {
                    "frame": np.zeros((100, 100, 3), dtype=np.uint8),
                    "fps": fps,
                    "detections": detections,
                }
                step.process(data)

            # Get statistics
            stats = step.get_statistics()

            # Check FPS stats
            assert "fps" in stats
            assert stats["fps"]["mean"] == pytest.approx(np.mean(fps_values), abs=0.01)
            assert stats["fps"]["min"] == min(fps_values)
            assert stats["fps"]["max"] == max(fps_values)
            assert stats["fps"]["count"] == len(fps_values)

            # Check detection count stats
            assert "detection_count" in stats
            assert stats["detection_count"]["mean"] == pytest.approx(
                np.mean(det_counts), abs=0.01
            )
            assert stats["detection_count"]["min"] == min(det_counts)
            assert stats["detection_count"]["max"] == max(det_counts)

    def test_debug_logger_reset(self):
        """Test that reset clears all state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.json"
            step = sv.DebugLoggerStep(log_file=log_file, auto_flush=False)

            # Process some frames
            for i in range(3):
                data = {
                    "frame": np.zeros((100, 100, 3), dtype=np.uint8),
                    "fps": 30.0,
                }
                step.process(data)

            # Check state
            assert step._frame_count == 3
            assert len(step._metrics_buffer) == 3

            # Reset
            step.reset()

            # Check state is cleared
            assert step._frame_count == 0
            assert len(step._metrics_buffer) == 0
            assert len(step._accumulated_stats["fps"]) == 0

    def test_debug_logger_filter(self):
        """Test that filter always returns True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.json"
            step = sv.DebugLoggerStep(log_file=log_file)

            assert step.filter({}) is True
            assert step.filter({"frame": None}) is True

    def test_debug_logger_multiple_flushes(self):
        """Test that multiple flushes append data correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.json"
            step = sv.DebugLoggerStep(log_file=log_file, auto_flush=False)

            # First batch
            for i in range(3):
                step.process({"frame": np.zeros((100, 100, 3)), "fps": 30.0 + i})
            step.flush()

            # Second batch
            for i in range(2):
                step.process({"frame": np.zeros((100, 100, 3)), "fps": 40.0 + i})
            step.flush()

            # Check file has all data
            with open(log_file, "r") as f:
                data = json.load(f)

            assert len(data) == 5
            assert data[0]["fps"] == 30.0
            assert data[2]["fps"] == 32.0
            assert data[3]["fps"] == 40.0
            assert data[4]["fps"] == 41.0
            # Frame numbers should be sequential
            for i in range(5):
                assert data[i]["frame_number"] == i + 1

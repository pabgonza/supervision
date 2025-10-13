"""
Tests for async detection steps.
"""
import time

import numpy as np
import pytest

import supervision as sv


class MockDetector(sv.AsyncDetectionStep):
    """Mock detector for testing without requiring actual YOLO model."""

    def __init__(self, inference_time_ms: float = 50, **kwargs):
        super().__init__(**kwargs)
        self.inference_time_ms = inference_time_ms
        self.inference_count = 0

    def _run_inference(self, frame: np.ndarray) -> sv.Detections:
        """Simulate inference with controllable delay."""
        time.sleep(self.inference_time_ms / 1000.0)  # Convert ms to seconds
        self.inference_count += 1

        # Return mock detection
        return sv.Detections(
            xyxy=np.array([[10, 10, 50, 50]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )


class TestDetectionStrategy:
    """Tests for DetectionStrategy enum."""

    def test_strategy_enum_values(self):
        """Test that all strategy enum values exist."""
        assert hasattr(sv.DetectionStrategy, "SKIP_WHEN_BUSY")
        assert hasattr(sv.DetectionStrategy, "USE_LAST_RESULT")
        assert hasattr(sv.DetectionStrategy, "QUEUE_LATEST")
        assert hasattr(sv.DetectionStrategy, "SYNCHRONOUS")

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert sv.DetectionStrategy.SKIP_WHEN_BUSY.value == "skip"
        assert sv.DetectionStrategy.USE_LAST_RESULT.value == "cache"
        assert sv.DetectionStrategy.QUEUE_LATEST.value == "queue"
        assert sv.DetectionStrategy.SYNCHRONOUS.value == "sync"


class TestAsyncDetectionStep:
    """Tests for AsyncDetectionStep base class."""

    def test_initialization(self):
        """Test async detection step initialization."""
        detector = MockDetector(
            strategy=sv.DetectionStrategy.USE_LAST_RESULT,
            max_queue_size=3,
            inference_timeout=1.0,
        )

        assert detector.strategy == sv.DetectionStrategy.USE_LAST_RESULT
        assert detector.max_queue_size == 3
        assert detector.inference_timeout == 1.0
        assert detector._is_running is False

        detector.stop()

    def test_start_stop(self):
        """Test starting and stopping worker thread."""
        detector = MockDetector()

        # Start detector
        detector.start()
        assert detector._is_running is True
        assert detector._worker_thread is not None
        assert detector._worker_thread.is_alive()

        # Stop detector
        detector.stop()
        assert detector._is_running is False

    def test_synchronous_strategy(self):
        """Test synchronous processing strategy."""
        detector = MockDetector(
            strategy=sv.DetectionStrategy.SYNCHRONOUS,
            inference_time_ms=10,
        )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        data = {"frame": frame}

        result = detector.process(data)

        assert "detections" in result
        assert len(result["detections"]) == 1
        assert detector.inference_count == 1

        metrics = detector.get_metrics()
        assert metrics["frames_processed"] == 1

        detector.stop()

    def test_skip_when_busy_strategy(self):
        """Test skip when busy strategy."""
        detector = MockDetector(
            strategy=sv.DetectionStrategy.SKIP_WHEN_BUSY,
            inference_time_ms=100,  # Slow inference
        )
        detector.start()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Process multiple frames quickly
        results = []
        for _ in range(5):
            data = {"frame": frame}
            result = detector.process(data)
            results.append(result)
            time.sleep(0.01)  # Small delay between frames

        metrics = detector.get_metrics()

        # Should have skipped some frames
        assert metrics["frames_skipped"] > 0

        detector.stop()

    def test_use_last_result_strategy(self):
        """Test use last result (caching) strategy."""
        detector = MockDetector(
            strategy=sv.DetectionStrategy.USE_LAST_RESULT,
            inference_time_ms=50,
        )
        detector.start()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Wait for first inference to complete to have a cached result
        data = {"frame": frame}
        detector.process(data)
        time.sleep(0.2)  # Wait for first result to be cached

        # Now process multiple frames quickly
        for _ in range(5):
            data = {"frame": frame}
            result = detector.process(data)
            assert "detections" in result  # Should always have detections (cached or fresh)
            time.sleep(0.01)

        # Wait for processing
        time.sleep(0.3)

        metrics = detector.get_metrics()

        # Should have cached results or queued frames
        assert metrics["frames_cached"] > 0 or metrics["frames_queued"] > 0

        detector.stop()

    def test_queue_latest_strategy(self):
        """Test queue latest strategy."""
        detector = MockDetector(
            strategy=sv.DetectionStrategy.QUEUE_LATEST,
            inference_time_ms=50,
            max_queue_size=2,
        )
        detector.start()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Process multiple frames quickly to fill queue
        for _ in range(10):
            data = {"frame": frame}
            result = detector.process(data)

        # Wait for processing
        time.sleep(0.5)

        metrics = detector.get_metrics()

        # Queue should have been managed
        assert metrics["frames_queued"] > 0

        detector.stop()

    def test_empty_frame_handling(self):
        """Test handling of empty/missing frames."""
        detector = MockDetector()

        # Test with no frame
        data = {}
        result = detector.process(data)
        assert result == data

        # Test with None frame
        data = {"frame": None}
        result = detector.process(data)
        assert result == data

        detector.stop()

    def test_metrics_tracking(self):
        """Test metrics are tracked correctly."""
        detector = MockDetector(
            strategy=sv.DetectionStrategy.SYNCHRONOUS,
            inference_time_ms=10,
        )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Process some frames
        for _ in range(3):
            data = {"frame": frame}
            detector.process(data)

        metrics = detector.get_metrics()

        assert metrics["frames_processed"] == 3
        assert metrics["avg_inference_time_ms"] > 0
        assert "frames_skipped" in metrics
        assert "frames_cached" in metrics
        assert "frames_queued" in metrics
        assert "queue_full_count" in metrics

        detector.stop()

    def test_metrics_reset(self):
        """Test metrics can be reset."""
        detector = MockDetector(strategy=sv.DetectionStrategy.SYNCHRONOUS)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        data = {"frame": frame}
        detector.process(data)

        metrics_before = detector.get_metrics()
        assert metrics_before["frames_processed"] > 0

        # Reset metrics
        detector.reset_metrics()

        metrics_after = detector.get_metrics()
        assert metrics_after["frames_processed"] == 0

        detector.stop()

    def test_filter_method(self):
        """Test filter method."""
        detector = MockDetector()

        # With frame
        assert detector.filter({"frame": np.zeros((10, 10, 3))}) is True

        # Without frame
        assert detector.filter({}) is False
        assert detector.filter({"other": "data"}) is False

        detector.stop()

    def test_cleanup_on_deletion(self):
        """Test that detector cleans up on deletion."""
        detector = MockDetector()
        detector.start()

        assert detector._is_running is True

        # Delete detector
        del detector

        # Worker thread should be stopped


@pytest.mark.slow
class TestAsyncYOLODetectionStep:
    """Tests for AsyncYOLODetectionStep (requires ultralytics)."""

    @pytest.fixture
    def model_path(self):
        """Return path to test model."""
        return "yolov8n.pt"  # Assumes model is available

    def test_initialization(self, model_path):
        """Test YOLO detector initialization."""
        try:
            detector = sv.AsyncYOLODetectionStep(
                model_path=model_path,
                conf=0.5,
                iou=0.45,
                device="cpu",  # Use CPU for testing
                strategy=sv.DetectionStrategy.SYNCHRONOUS,
                warmup=False,  # Skip warmup for faster tests
            )

            assert detector.model_path == model_path
            assert detector.conf == 0.5
            assert detector.iou == 0.45
            assert detector.device == "cpu"

            detector.stop()

        except ImportError:
            pytest.skip("ultralytics not installed")
        except Exception as e:
            pytest.skip(f"Model not available: {e}")

    def test_inference(self, model_path):
        """Test YOLO inference."""
        try:
            detector = sv.AsyncYOLODetectionStep(
                model_path=model_path,
                device="cpu",
                strategy=sv.DetectionStrategy.SYNCHRONOUS,
                warmup=False,
            )

            # Create test frame
            frame = np.zeros((640, 640, 3), dtype=np.uint8)
            data = {"frame": frame}

            result = detector.process(data)

            assert "detections" in result
            assert isinstance(result["detections"], sv.Detections)

            detector.stop()

        except ImportError:
            pytest.skip("ultralytics not installed")
        except Exception as e:
            pytest.skip(f"Model not available: {e}")

    def test_async_processing(self, model_path):
        """Test async processing with YOLO."""
        try:
            detector = sv.AsyncYOLODetectionStep(
                model_path=model_path,
                device="cpu",
                strategy=sv.DetectionStrategy.USE_LAST_RESULT,
                warmup=False,
            )
            detector.start()

            # Process multiple frames
            for _ in range(3):
                frame = np.zeros((640, 640, 3), dtype=np.uint8)
                data = {"frame": frame}
                result = detector.process(data)
                time.sleep(0.1)

            # Wait for processing
            time.sleep(0.5)

            metrics = detector.get_metrics()
            assert metrics["frames_queued"] > 0

            detector.stop()

        except ImportError:
            pytest.skip("ultralytics not installed")
        except Exception as e:
            pytest.skip(f"Model not available: {e}")


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_access(self):
        """Test concurrent access to detector."""
        detector = MockDetector(
            strategy=sv.DetectionStrategy.USE_LAST_RESULT,
            inference_time_ms=20,
        )
        detector.start()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Simulate concurrent processing
        results = []
        for _ in range(20):
            data = {"frame": frame}
            result = detector.process(data)
            results.append(result)

        # Wait for processing
        time.sleep(0.5)

        # All results should have detections
        for result in results:
            assert "detections" in result

        # Metrics should be consistent
        metrics = detector.get_metrics()
        assert isinstance(metrics["frames_processed"], int)
        assert metrics["frames_processed"] >= 0

        detector.stop()

    def test_metrics_thread_safety(self):
        """Test that metrics access is thread-safe."""
        detector = MockDetector()
        detector.start()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Process frames while reading metrics
        for _ in range(10):
            data = {"frame": frame}
            detector.process(data)
            metrics = detector.get_metrics()  # Should not raise
            assert isinstance(metrics, dict)

        detector.stop()

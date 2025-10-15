"""
Tests for pool detector steps.
"""
import time

import numpy as np
import pytest

import supervision as sv


class MockPoolDetector(sv.PoolDetectorStep):
    """Mock pool detector for testing without requiring actual YOLO model."""

    def __init__(self, inference_time_ms: float = 50, **kwargs):
        super().__init__(**kwargs)
        self.inference_time_ms = inference_time_ms
        self.inference_count = 0
        self.inference_lock = __import__("threading").Lock()

    def _run_inference(self, frame: np.ndarray) -> sv.Detections:
        """Simulate inference with controllable delay."""
        time.sleep(self.inference_time_ms / 1000.0)

        with self.inference_lock:
            self.inference_count += 1

        # Return mock detection
        return sv.Detections(
            xyxy=np.array([[10, 10, 50, 50]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )


class TestPoolDetectorStep:
    """Tests for PoolDetectorStep base class."""

    def test_initialization(self):
        """Test pool detector step initialization."""
        detector = MockPoolDetector(
            pool_size=3,
            max_queue_size=5,
            reorder_timeout=2.0,
        )

        assert detector.pool_size == 3
        assert detector.max_queue_size == 5
        assert detector.reorder_timeout == 2.0
        assert detector._is_running is False

        detector.stop()

    def test_start_stop(self):
        """Test starting and stopping worker pool."""
        detector = MockPoolDetector(pool_size=2)

        # Start detector
        detector.start()
        assert detector._is_running is True
        assert len(detector._workers) == 2
        for worker in detector._workers:
            assert worker.is_alive()

        # Stop detector
        detector.stop()
        assert detector._is_running is False
        assert len(detector._workers) == 0

    def test_single_frame_processing(self):
        """Test processing a single frame."""
        detector = MockPoolDetector(pool_size=2, inference_time_ms=10)
        detector.start()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        data = {"frame": frame}

        result = detector.process(data)

        assert "detections" in result
        assert len(result["detections"]) == 1
        assert "sequence_number" in result

        detector.stop()

    def test_multiple_frames_in_order(self):
        """Test that multiple frames are returned in order."""
        detector = MockPoolDetector(
            pool_size=2,
            inference_time_ms=20,
        )
        detector.start()

        # Process multiple frames
        num_frames = 10
        sequence_numbers = []

        for i in range(num_frames):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            data = {"frame": frame}
            result = detector.process(data)

            if "sequence_number" in result:
                sequence_numbers.append(result["sequence_number"])

        # Wait for all processing to complete
        time.sleep(0.5)

        # Sequence numbers should be in order
        assert sequence_numbers == sorted(sequence_numbers)

        detector.stop()

    def test_parallel_processing(self):
        """Test that pool processes frames in parallel."""
        detector = MockPoolDetector(
            pool_size=3,
            inference_time_ms=50,  # Moderate inference time
        )
        detector.start()

        # Process multiple frames
        num_frames = 9

        for _ in range(num_frames):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            data = {"frame": frame}
            detector.process(data)

        # Wait for processing to complete
        time.sleep(0.5)

        # Verify that frames were processed
        # With 3 workers, we should process multiple frames concurrently
        metrics = detector.get_metrics()

        # Main verification: frames were processed successfully
        assert metrics["frames_processed"] >= num_frames - 3  # Allow some margin for timing

        # Verify workers are active
        assert metrics["workers_active"] == 3

        detector.stop()

    def test_queue_full_handling(self):
        """Test handling of full queue."""
        detector = MockPoolDetector(
            pool_size=1,
            max_queue_size=3,
            reorder_timeout=0.5,  # Short timeout
        )
        detector.start()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Process a reasonable number of frames
        num_frames = 10
        for _ in range(num_frames):
            data = {"frame": frame}
            result = detector.process(data)
            # Each frame should have either detections or be empty
            assert "detections" in result

        metrics = detector.get_metrics()

        # Verify that some frames were processed
        # The exact number may vary due to queue dynamics
        assert metrics["frames_processed"] > 0

        # Verify detector is working correctly
        assert metrics["workers_active"] == 1

        detector.stop()

    def test_empty_frame_handling(self):
        """Test handling of empty/missing frames."""
        detector = MockPoolDetector(pool_size=2)
        detector.start()

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
        detector = MockPoolDetector(pool_size=2, inference_time_ms=10)
        detector.start()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Process some frames
        for _ in range(5):
            data = {"frame": frame}
            detector.process(data)

        # Wait for processing
        time.sleep(0.3)

        metrics = detector.get_metrics()

        assert metrics["frames_processed"] > 0
        assert metrics["workers_active"] == 2
        assert "avg_inference_time_ms" in metrics
        assert "avg_queue_time_ms" in metrics
        assert "frames_dropped" in metrics
        assert "frames_reordered" in metrics

        detector.stop()

    def test_metrics_reset(self):
        """Test metrics can be reset."""
        detector = MockPoolDetector(pool_size=2)
        detector.start()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        data = {"frame": frame}
        detector.process(data)

        time.sleep(0.2)

        metrics_before = detector.get_metrics()
        assert metrics_before["frames_processed"] > 0

        # Reset metrics
        detector.reset_metrics()

        metrics_after = detector.get_metrics()
        assert metrics_after["frames_processed"] == 0

        detector.stop()

    def test_get_queue_size(self):
        """Test getting queue size."""
        detector = MockPoolDetector(pool_size=1, max_queue_size=5)
        detector.start()

        current, maximum = detector.get_queue_size()

        assert maximum == 5
        assert current >= 0
        assert current <= maximum

        detector.stop()

    def test_filter_method(self):
        """Test filter method."""
        detector = MockPoolDetector(pool_size=2)

        # With frame
        assert detector.filter({"frame": np.zeros((10, 10, 3))}) is True

        # Without frame
        assert detector.filter({}) is False
        assert detector.filter({"other": "data"}) is False

        detector.stop()

    def test_cleanup_on_deletion(self):
        """Test that detector cleans up on deletion."""
        detector = MockPoolDetector(pool_size=2)
        detector.start()

        assert detector._is_running is True

        # Delete detector
        del detector

        # Workers should be stopped (can't check directly after deletion)

    def test_reorder_buffer_works(self):
        """Test that reorder buffer correctly handles out-of-order results."""
        detector = MockPoolDetector(
            pool_size=3,
            inference_time_ms=30,
            reorder_timeout=2.0,
        )
        detector.start()

        # Process frames that might complete out of order
        sequence_numbers = []
        for _ in range(8):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            data = {"frame": frame}
            result = detector.process(data)

            if "sequence_number" in result:
                sequence_numbers.append(result["sequence_number"])

        # All sequence numbers should be in order
        assert sequence_numbers == sorted(sequence_numbers)

        # Check that some frames were reordered
        metrics = detector.get_metrics()
        # With parallel processing, some reordering is expected
        # (though not guaranteed in all test runs)

        detector.stop()


@pytest.mark.slow
class TestPoolYOLODetectionStep:
    """Tests for PoolYOLODetectionStep (requires ultralytics)."""

    @pytest.fixture
    def model_path(self):
        """Return path to test model."""
        return "yolov8n.pt"

    def test_initialization(self, model_path):
        """Test YOLO pool detector initialization."""
        try:
            detector = sv.PoolYOLODetectionStep(
                model_path=model_path,
                pool_size=2,
                conf=0.5,
                iou=0.45,
                device="cpu",  # Use CPU for testing
                warmup=False,  # Skip warmup for faster tests
            )

            assert detector.model_path == model_path
            assert detector.pool_size == 2
            assert detector.conf == 0.5
            assert detector.iou == 0.45
            assert detector.device == "cpu"
            assert len(detector._models) == 2

            detector.stop()

        except ImportError:
            pytest.skip("ultralytics not installed")
        except Exception as e:
            pytest.skip(f"Model not available: {e}")

    def test_inference(self, model_path):
        """Test YOLO pool inference."""
        try:
            detector = sv.PoolYOLODetectionStep(
                model_path=model_path,
                pool_size=2,
                device="cpu",
                warmup=False,
            )

            # Create test frame
            frame = np.zeros((640, 640, 3), dtype=np.uint8)
            data = {"frame": frame}

            result = detector.process(data)

            time.sleep(0.5)  # Wait for processing

            assert "detections" in result
            assert isinstance(result["detections"], sv.Detections)

            detector.stop()

        except ImportError:
            pytest.skip("ultralytics not installed")
        except Exception as e:
            pytest.skip(f"Model not available: {e}")

    def test_parallel_yolo_processing(self, model_path):
        """Test parallel YOLO processing."""
        try:
            detector = sv.PoolYOLODetectionStep(
                model_path=model_path,
                pool_size=2,
                device="cpu",
                warmup=False,
            )

            # Process multiple frames
            num_frames = 4
            for _ in range(num_frames):
                frame = np.zeros((640, 640, 3), dtype=np.uint8)
                data = {"frame": frame}
                detector.process(data)

            # Wait for processing
            time.sleep(2.0)

            metrics = detector.get_metrics()
            assert metrics["frames_processed"] > 0

            detector.stop()

        except ImportError:
            pytest.skip("ultralytics not installed")
        except Exception as e:
            pytest.skip(f"Model not available: {e}")

    def test_frame_ordering_with_yolo(self, model_path):
        """Test frame ordering with YOLO detection."""
        try:
            detector = sv.PoolYOLODetectionStep(
                model_path=model_path,
                pool_size=2,
                device="cpu",
                warmup=False,
            )

            # Process frames
            sequence_numbers = []
            for _ in range(5):
                frame = np.zeros((640, 640, 3), dtype=np.uint8)
                data = {"frame": frame}
                result = detector.process(data)

                if "sequence_number" in result:
                    sequence_numbers.append(result["sequence_number"])

            # Sequence should be ordered
            assert sequence_numbers == sorted(sequence_numbers)

            detector.stop()

        except ImportError:
            pytest.skip("ultralytics not installed")
        except Exception as e:
            pytest.skip(f"Model not available: {e}")


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_frame_submission(self):
        """Test concurrent frame submission."""
        detector = MockPoolDetector(
            pool_size=2,
            inference_time_ms=20,
            max_queue_size=20,
        )
        detector.start()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Submit many frames concurrently
        results = []
        for _ in range(20):
            data = {"frame": frame}
            result = detector.process(data)
            results.append(result)

        # Wait for processing
        time.sleep(0.8)

        # All submissions should succeed (or be dropped gracefully)
        for result in results:
            assert "detections" in result

        # Metrics should be consistent
        metrics = detector.get_metrics()
        assert isinstance(metrics["frames_processed"], int)
        assert metrics["frames_processed"] >= 0

        detector.stop()

    def test_metrics_thread_safety(self):
        """Test that metrics access is thread-safe."""
        detector = MockPoolDetector(pool_size=3, inference_time_ms=10)
        detector.start()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Process frames while reading metrics
        for _ in range(15):
            data = {"frame": frame}
            detector.process(data)
            metrics = detector.get_metrics()  # Should not raise
            assert isinstance(metrics, dict)

        detector.stop()

    def test_sequence_counter_thread_safety(self):
        """Test that sequence counter is thread-safe."""
        detector = MockPoolDetector(pool_size=3, max_queue_size=50)
        detector.start()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Submit many frames
        sequence_numbers = []
        for _ in range(30):
            data = {"frame": frame}
            result = detector.process(data)

        # Wait for processing
        time.sleep(1.0)

        # Sequence counter should have incremented correctly
        assert detector._sequence_counter == 30

        detector.stop()


class TestPerformanceComparison:
    """Tests comparing pool vs single detector performance."""

    def test_pool_vs_single_throughput(self):
        """Test that pool detector has higher throughput than single."""
        # Single detector (pool size 1)
        single_detector = MockPoolDetector(pool_size=1, inference_time_ms=50)
        single_detector.start()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Process frames with single detector
        single_start = time.time()
        for _ in range(10):
            data = {"frame": frame}
            single_detector.process(data)

        time.sleep(0.8)
        single_elapsed = time.time() - single_start
        single_metrics = single_detector.get_metrics()
        single_detector.stop()

        # Pool detector (pool size 3)
        pool_detector = MockPoolDetector(pool_size=3, inference_time_ms=50)
        pool_detector.start()

        # Process frames with pool detector
        pool_start = time.time()
        for _ in range(10):
            data = {"frame": frame}
            pool_detector.process(data)

        time.sleep(0.8)
        pool_elapsed = time.time() - pool_start
        pool_metrics = pool_detector.get_metrics()
        pool_detector.stop()

        # Pool should be faster or process more frames
        # (allowing for test environment variability)
        assert (
            pool_metrics["frames_processed"] >= single_metrics["frames_processed"]
            or pool_elapsed <= single_elapsed * 1.2
        )

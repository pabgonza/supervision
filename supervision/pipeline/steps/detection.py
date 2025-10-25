from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any

import numpy as np

from supervision.detection.core import Detections


class DetectionStep(ABC):
    """
    Base class for synchronous object detection steps.

    This abstract class provides a common interface for implementing object
    detection steps that run synchronously (blocking). Subclasses only need to
    implement the `_run_inference` method to add support for different detection
    backends.

    The base class handles the pipeline integration by implementing `process()`
    and `filter()` methods, allowing subclasses to focus solely on the inference
    logic.

    Examples:
        ```python
        import supervision as sv
        import numpy as np

        # Implement custom detection step
        class CustomDetectionStep(sv.DetectionStep):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def _run_inference(self, frame: np.ndarray) -> sv.Detections:
                # Your custom inference logic
                results = self.model.predict(frame)
                return sv.Detections(...)

        # Use in pipeline
        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | CustomDetectionStep(model)
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("Detections")
        )
        pipeline.run()
        ```

        ```python
        # Built-in YOLO detection step
        import supervision as sv

        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | sv.YOLODetectionStep("yolov8n.pt", conf=0.5)
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("YOLO Detections")
        )
        pipeline.run()
        ```
    """

    def __init__(self, input_key: str = "frame", output_key: str = "detections"):
        """
        Initialize detection step.

        Args:
            input_key: Key in data dict containing input frame (default: 'frame')
            output_key: Key to store detections in data dict (default: 'detections')
        """
        self.input_key = input_key
        self.output_key = output_key

    @abstractmethod
    def _run_inference(self, frame: np.ndarray) -> Detections:
        """
        Run inference on a frame. Must be implemented by subclasses.

        Args:
            frame: Input frame for detection

        Returns:
            Detections object with detection results
        """
        pass

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Process frame with object detection.

        Args:
            data: Pipeline data containing input frame

        Returns:
            Data with detections field added
        """
        frame = data.get(self.input_key)
        if frame is None:
            return data

        # Run inference
        detections = self._run_inference(frame)
        data[self.output_key] = detections

        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Process if frame exists."""
        return self.input_key in data


class YOLODetectionStep(DetectionStep):
    """
    Pipeline step for YOLO object detection using Ultralytics.

    This step implements the DetectionStep interface for YOLO models from Ultralytics.
    It runs YOLO inference on frames and converts results to supervision Detections.

    Examples:
        ```python
        import supervision as sv

        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | sv.YOLODetectionStep("yolov8n.pt", conf=0.5)
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("Detections")
        )
        pipeline.run()
        ```
    """

    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        verbose: bool = False,
        input_key: str = "frame",
        output_key: str = "detections",
        metrics_key: str = "yolo_metrics",
    ):
        """
        Initialize YOLO detection step.

        Args:
            model_path: Path to YOLO model file (.pt, .engine, etc.)
            conf: Confidence threshold for detections
            iou: IOU threshold for NMS
            verbose: Whether to print verbose output
            input_key: Key in data dict containing input frame (default: 'frame')
            output_key: Key to store detections in data dict (default: 'detections')
            metrics_key: Key to store timing metrics in data dict (default: 'yolo_metrics')

        Note:
            Ultralytics automatically detects and uses GPU if CUDA is available.
        """
        super().__init__(input_key=input_key, output_key=output_key)

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLODetectionStep. "
                "Install it with: pip install ultralytics"
            )

        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.verbose = verbose
        self.metrics_key = metrics_key

        # Load model (device auto-detected by ultralytics)
        self.model = YOLO(model_path)

    def _run_inference(self, frame: np.ndarray) -> tuple[Detections, dict]:
        """
        Run YOLO inference on frame.

        Args:
            frame: Input frame for detection

        Returns:
            Tuple of (Detections object, speed metrics dict)
        """
        # Run inference
        results = self.model.predict(
            source=frame, conf=self.conf, iou=self.iou, verbose=self.verbose
        )

        # Convert to supervision Detections
        detections = Detections.from_ultralytics(results[0])

        # Extract speed metrics from Ultralytics
        speed = results[
            0
        ].speed  # dict: {'preprocess': X, 'inference': Y, 'postprocess': Z}

        return detections, speed

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Process frame with object detection and timing metrics.

        Args:
            data: Pipeline data containing input frame

        Returns:
            Data with detections and metrics fields added
        """
        frame = data.get(self.input_key)
        if frame is None:
            return data

        # Run inference
        detections, speed = self._run_inference(frame)
        data[self.output_key] = detections
        data[self.metrics_key] = speed

        return data


class YOLOTrackingStep(DetectionStep):
    """
    Pipeline step for YOLO object detection with integrated tracking using Ultralytics.

    This step combines detection and tracking in a single inference call using YOLO's
    built-in tracking capabilities. It uses model.track() instead of model.predict(),
    which internally runs detection and tracking (BoT-SORT or ByteTrack) in one pass.

    This is more efficient than using separate YOLODetectionStep + ByteTrackerStep,
    as tracking is integrated into the model inference pipeline.

    Examples:
        ```python
        import supervision as sv

        # Using BoT-SORT tracker (default)
        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | sv.YOLOTrackingStep("yolov8n.pt", conf=0.5)
            | sv.TrackerAnnotatorStep()
            | sv.DisplaySink("Tracking")
        )
        pipeline.run()
        ```

        ```python
        # Using ByteTrack tracker
        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | sv.YOLOTrackingStep("yolov8n.pt", tracker="bytetrack.yaml")
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("ByteTrack")
        )
        pipeline.run()
        ```
    """

    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        tracker: str = "botsort.yaml",
        persist: bool = True,
        verbose: bool = False,
        input_key: str = "frame",
        output_key: str = "detections",
        metrics_key: str = "yolo_tracking_metrics",
    ):
        """
        Initialize YOLO detection + tracking step.

        Args:
            model_path: Path to YOLO model file (.pt, .engine, etc.)
            conf: Confidence threshold for detections
            iou: IOU threshold for NMS
            tracker: Tracker configuration file ("botsort.yaml" or "bytetrack.yaml")
            persist: Persist tracks between frames for continuous tracking
            verbose: Whether to print verbose output
            input_key: Key in data dict containing input frame (default: 'frame')
            output_key: Key to store detections in data dict (default: 'detections')
            metrics_key: Key to store timing metrics in data dict
                (default: 'yolo_tracking_metrics')

        Note:
            Ultralytics automatically detects and uses GPU if CUDA is available.
            The tracker parameter accepts either "botsort.yaml" (default, more accurate)
            or "bytetrack.yaml" (faster, lighter).
        """
        super().__init__(input_key=input_key, output_key=output_key)

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLOTrackingStep. "
                "Install it with: pip install ultralytics"
            )

        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.tracker = tracker
        self.persist = persist
        self.verbose = verbose
        self.metrics_key = metrics_key

        # Load model (device auto-detected by ultralytics)
        self.model = YOLO(model_path)

    def _run_inference(self, frame: np.ndarray) -> tuple[Detections, dict]:
        """
        Run YOLO detection and tracking on frame.

        Args:
            frame: Input frame for detection and tracking

        Returns:
            Tuple of (Detections object with tracker_id field, speed metrics dict)
        """
        # Run inference with tracking
        results = self.model.track(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            tracker=self.tracker,
            persist=self.persist,
            verbose=self.verbose,
        )

        # Convert to supervision Detections (tracker_id extracted automatically)
        detections = Detections.from_ultralytics(results[0])

        # Extract speed metrics from Ultralytics
        speed = results[0].speed

        return detections, speed

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Process frame with object detection and tracking.

        Args:
            data: Pipeline data containing input frame

        Returns:
            Data with detections (including tracker_id) and metrics fields added
        """
        frame = data.get(self.input_key)
        if frame is None:
            return data

        # Run inference
        detections, speed = self._run_inference(frame)
        data[self.output_key] = detections
        data[self.metrics_key] = speed

        return data


class DetectionStrategy(Enum):
    """
    Strategy for handling slow detectors in async detection steps.

    Attributes:
        SKIP_WHEN_BUSY: Skip frames when detector is busy, use last cached result
        USE_LAST_RESULT: Always return cached result, queue frame for async processing
        QUEUE_LATEST: Queue frames for processing, discard old frames when queue full
        SYNCHRONOUS: Traditional synchronous processing (blocks pipeline)
    """

    SKIP_WHEN_BUSY = "skip"
    USE_LAST_RESULT = "cache"
    QUEUE_LATEST = "queue"
    SYNCHRONOUS = "sync"


class AsyncDetectionStep(ABC):
    """
    Base class for asynchronous object detection steps.

    This class provides a framework for running object detection in a separate
    thread to avoid blocking the pipeline when the detector is slower than the
    frame rate. Different strategies can be used to handle the timing mismatch.

    The class uses a worker thread that continuously processes frames from a queue,
    storing results that can be retrieved by the main pipeline thread.

    Important: In asynchronous strategies (SKIP_WHEN_BUSY, USE_LAST_RESULT,
    QUEUE_LATEST), the pipeline returns the processed frame along with its
    detections to ensure perfect alignment. This means downstream steps see
    slightly delayed frames but with synchronized detections, eliminating visual
    misalignment at the cost of additional latency. The SYNCHRONOUS strategy
    processes frames immediately without this delay.

    Examples:
        ```python
        import supervision as sv

        # Create async detector with caching strategy
        detector = sv.AsyncYOLODetectionStep(
            model_path="yolov8n.pt",
            strategy=sv.DetectionStrategy.USE_LAST_RESULT
        )

        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | detector
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("Async Detection")
        )
        pipeline.run()

        # Check metrics
        metrics = detector.get_metrics()
        print(f"Frames processed: {metrics['frames_processed']}")
        print(f"Frames cached: {metrics['frames_cached']}")
        ```
    """

    def __init__(
        self,
        strategy: DetectionStrategy = DetectionStrategy.USE_LAST_RESULT,
        max_queue_size: int = 2,
        inference_timeout: float = 0.5,
        input_key: str = "frame",
        output_key: str = "detections",
    ):
        """
        Initialize async detection step.

        Args:
            strategy: Strategy for handling timing mismatch between detector and source
            max_queue_size: Maximum frames to queue for processing (lower = less latency)
            inference_timeout: Maximum time to wait for inference results (seconds)
            input_key: Key in data dict containing input frame (default: 'frame')
            output_key: Key to store detections in data dict (default: 'detections')
        """
        self.strategy = strategy
        self.max_queue_size = max_queue_size
        self.inference_timeout = inference_timeout
        self.input_key = input_key
        self.output_key = output_key

        # Threading components
        self._inference_queue: Queue = Queue(maxsize=max_queue_size)
        self._result_lock = Lock()
        self._stop_event = Event()
        self._worker_thread: Thread | None = None

        # State
        self._last_detections: Detections | None = None
        self._last_frame: np.ndarray | None = None
        self._last_frame_timestamp: float = 0.0
        self._inference_in_progress = False
        self._is_running = False

        # Metrics
        self._metrics = {
            "frames_processed": 0,
            "frames_skipped": 0,
            "frames_cached": 0,
            "frames_queued": 0,
            "avg_inference_time_ms": 0.0,
            "queue_full_count": 0,
            "total_inference_time_ms": 0.0,
        }

    @abstractmethod
    def _run_inference(self, frame: np.ndarray) -> Detections:
        """
        Run inference on a frame. Must be implemented by subclasses.

        Args:
            frame: Input frame for detection

        Returns:
            Detections object with detection results
        """
        pass

    def _inference_worker(self) -> None:
        """Worker thread that continuously processes frames from the queue."""
        while not self._stop_event.is_set():
            try:
                # Get frame from queue with timeout
                frame_data = self._inference_queue.get(timeout=0.1)

                if frame_data is None:  # Poison pill
                    break

                frame, timestamp = frame_data

                # Run inference
                start_time = time.time()
                self._inference_in_progress = True

                detections = self._run_inference(frame)

                inference_time_ms = (time.time() - start_time) * 1000

                # Update results and metrics
                with self._result_lock:
                    self._last_detections = detections
                    self._last_frame = frame
                    self._last_frame_timestamp = timestamp
                    self._inference_in_progress = False

                    # Update metrics
                    self._metrics["frames_processed"] += 1
                    self._metrics["total_inference_time_ms"] += inference_time_ms
                    self._metrics["avg_inference_time_ms"] = (
                        self._metrics["total_inference_time_ms"]
                        / self._metrics["frames_processed"]
                    )

                self._inference_queue.task_done()

            except Empty:
                continue
            except Exception as e:
                # Log error but don't crash the worker
                print(f"Error in inference worker: {e}")
                self._inference_in_progress = False
                continue

    def start(self) -> AsyncDetectionStep:
        """
        Start the inference worker thread.

        Returns:
            Self for method chaining
        """
        if not self._is_running:
            self._stop_event.clear()
            self._worker_thread = Thread(
                target=self._inference_worker, name="AsyncDetectionWorker", daemon=True
            )
            self._worker_thread.start()
            self._is_running = True
        return self

    def stop(self) -> None:
        """Stop the inference worker thread and cleanup resources."""
        if self._is_running:
            self._stop_event.set()
            # Send poison pill
            try:
                self._inference_queue.put(None, timeout=1.0)
            except:
                pass

            if self._worker_thread:
                self._worker_thread.join(timeout=2.0)

            self._is_running = False

    def _enqueue_frame(self, frame: np.ndarray) -> bool:
        """
        Try to enqueue a frame for processing.

        Args:
            frame: Frame to enqueue

        Returns:
            True if frame was enqueued, False if queue was full
        """
        try:
            self._inference_queue.put_nowait((frame, time.time()))
            with self._result_lock:
                self._metrics["frames_queued"] += 1
            return True
        except:
            with self._result_lock:
                self._metrics["queue_full_count"] += 1
            return False

    def _clear_queue(self) -> None:
        """Clear old frames from queue."""
        cleared = 0
        while not self._inference_queue.empty():
            try:
                self._inference_queue.get_nowait()
                self._inference_queue.task_done()
                cleared += 1
            except:
                break

    def _use_cached_result(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Return cached detection result with matching frame.

        Args:
            data: Pipeline data

        Returns:
            Data with cached detections and frame added
        """
        with self._result_lock:
            if self._last_detections is not None and self._last_frame is not None:
                data[self.output_key] = self._last_detections
                data[self.input_key] = self._last_frame
                self._metrics["frames_cached"] += 1
            else:
                # No cached result yet, return empty detections
                data[self.output_key] = Detections.empty()

        return data

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Process frame according to selected strategy.

        Args:
            data: Pipeline data containing input frame

        Returns:
            Data with detections field added
        """
        frame = data.get(self.input_key)
        if frame is None:
            return data

        # Start worker thread if not running
        if not self._is_running:
            self.start()

        if self.strategy == DetectionStrategy.SYNCHRONOUS:
            # Synchronous mode: block and wait for result
            start_time = time.time()
            detections = self._run_inference(frame)
            inference_time_ms = (time.time() - start_time) * 1000

            data[self.output_key] = detections
            with self._result_lock:
                self._metrics["frames_processed"] += 1
                self._metrics["total_inference_time_ms"] += inference_time_ms
                self._metrics["avg_inference_time_ms"] = (
                    self._metrics["total_inference_time_ms"]
                    / self._metrics["frames_processed"]
                )

        elif self.strategy == DetectionStrategy.SKIP_WHEN_BUSY:
            # Skip if busy, otherwise enqueue
            if self._inference_in_progress or self._inference_queue.full():
                data = self._use_cached_result(data)
                with self._result_lock:
                    self._metrics["frames_skipped"] += 1
            else:
                self._enqueue_frame(frame)
                data = self._use_cached_result(data)

        elif self.strategy == DetectionStrategy.USE_LAST_RESULT:
            # Always enqueue and return cached result
            self._enqueue_frame(frame)
            data = self._use_cached_result(data)

        elif self.strategy == DetectionStrategy.QUEUE_LATEST:
            # Clear queue and enqueue latest frame
            if self._inference_queue.full():
                self._clear_queue()
            self._enqueue_frame(frame)
            data = self._use_cached_result(data)

        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Process if frame exists."""
        return self.input_key in data

    def get_metrics(self) -> dict[str, Any]:
        """
        Get performance metrics.

        Returns:
            Dictionary with metrics:
                - frames_processed: Total frames processed by detector
                - frames_skipped: Frames skipped due to busy detector
                - frames_cached: Times cached result was used
                - frames_queued: Total frames enqueued
                - avg_inference_time_ms: Average inference time in milliseconds
                - queue_full_count: Times queue was full
        """
        with self._result_lock:
            return self._metrics.copy()

    def get_queue_size(self) -> tuple[int, int]:
        """
        Get current queue size and maximum queue size.

        Returns:
            Tuple of (current_size, max_size)
        """
        return (self._inference_queue.qsize(), self.max_queue_size)

    def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        with self._result_lock:
            for key in self._metrics:
                if isinstance(self._metrics[key], (int, float)):
                    self._metrics[key] = 0

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


class AsyncYOLODetectionStep(AsyncDetectionStep):
    """
    Asynchronous YOLO object detection step.

    This step runs YOLO inference in a separate thread, allowing the pipeline
    to continue processing frames even when detection is slower than the frame rate.

    Examples:
        ```python
        import supervision as sv

        # Basic async detection with default caching strategy
        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | sv.AsyncYOLODetectionStep("yolov8n.pt")
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("Async YOLO")
        )
        pipeline.run()
        ```

        ```python
        # Skip frames when detector is busy (lowest latency)
        detector = sv.AsyncYOLODetectionStep(
            model_path="yolov8n.pt",
            strategy=sv.DetectionStrategy.SKIP_WHEN_BUSY,
            conf=0.5
        )

        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | sv.FPSCalculatorStep()
            | detector
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("Low Latency Detection")
        )

        # Monitor metrics
        for data in pipeline:
            metrics = detector.get_metrics()
            print(f"FPS: {data.get('fps', 0):.1f}")
            print(f"Avg inference: {metrics['avg_inference_time_ms']:.1f}ms")
            print(f"Frames skipped: {metrics['frames_skipped']}")
        ```

        ```python
        # Queue latest frames (best accuracy)
        detector = sv.AsyncYOLODetectionStep(
            model_path="yolov8n.pt",
            strategy=sv.DetectionStrategy.QUEUE_LATEST,
            max_queue_size=5
        )
        ```
    """

    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        verbose: bool = False,
        strategy: DetectionStrategy = DetectionStrategy.USE_LAST_RESULT,
        max_queue_size: int = 2,
        inference_timeout: float = 0.5,
        warmup: bool = True,
        input_key: str = "frame",
        output_key: str = "detections",
    ):
        """
        Initialize async YOLO detection step.

        Args:
            model_path: Path to YOLO model file (.pt, .engine, etc.)
            conf: Confidence threshold for detections (0.0-1.0)
            iou: IOU threshold for NMS (0.0-1.0)
            verbose: Whether to print verbose YOLO output
            strategy: Detection strategy for handling slow inference
            max_queue_size: Maximum frames to queue (lower = less latency)
            inference_timeout: Maximum time to wait for results (seconds)
            warmup: Whether to run warmup inference on model load
            input_key: Key in data dict containing input frame (default: 'frame')
            output_key: Key to store detections in data dict (default: 'detections')

        Note:
            Ultralytics automatically detects and uses GPU if CUDA is available.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for AsyncYOLODetectionStep. "
                "Install it with: pip install ultralytics"
            )

        # Initialize base class
        super().__init__(
            strategy=strategy,
            max_queue_size=max_queue_size,
            inference_timeout=inference_timeout,
            input_key=input_key,
            output_key=output_key,
        )

        # YOLO parameters
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.verbose = verbose

        # Load model (device auto-detected by ultralytics)
        self.model = YOLO(model_path)

        # Warmup model
        if warmup:
            self._warmup_model()

        # Start worker thread
        self.start()

    def _warmup_model(self) -> None:
        """
        Warmup the model with a dummy inference.

        This ensures CUDA is initialized and model weights are loaded,
        avoiding slow first inference.
        """
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            _ = self.model.predict(
                source=dummy_frame,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )
        except Exception as e:
            print(f"Warning: Model warmup failed: {e}")

    def _run_inference(self, frame: np.ndarray) -> Detections:
        """
        Run YOLO inference on frame.

        Args:
            frame: Input frame for detection

        Returns:
            Detections object with detection results
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            verbose=self.verbose,
        )

        return Detections.from_ultralytics(results[0])


class PoolDetectorStep(ABC):
    """
    Base class for pool-based object detection steps.

    This class provides a framework for running object detection using a pool
    of parallel workers. Multiple detector instances run in separate threads,
    processing frames concurrently to maximize throughput on multi-GPU systems
    or when GPU utilization is low with a single detector.

    The class maintains frame ordering at the output by using sequence numbers
    and a reordering buffer. Frames are tagged with sequence numbers on input,
    processed in parallel by workers, and reordered before output.

    Important: Frames are returned in the same order they were received, even
    though they may be processed out of order internally. This ensures that
    downstream pipeline steps see frames in the correct sequence.

    Examples:
        ```python
        import supervision as sv

        # Create pool detector with 3 workers
        detector = sv.PoolYOLODetectionStep(
            model_path="yolov8n.pt",
            pool_size=3,
            max_queue_size=10
        )

        pipeline = (
            sv.Pipeline(sv.VideoFileSource("video.mp4"))
            | detector
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("Pool Detection")
        )
        pipeline.run()

        # Check metrics
        metrics = detector.get_metrics()
        print(f"Frames processed: {metrics['frames_processed']}")
        print(f"Avg queue time: {metrics['avg_queue_time_ms']:.2f}ms")
        ```
    """

    def __init__(
        self,
        pool_size: int = 2,
        max_queue_size: int = 10,
        reorder_timeout: float = 1.0,
        input_key: str = "frame",
        output_key: str = "detections",
    ):
        """
        Initialize pool detector step.

        Args:
            pool_size: Number of parallel detector workers (default: 2)
            max_queue_size: Maximum frames to queue for processing (default: 10)
            reorder_timeout: Maximum time to wait for expected frame (seconds).
                Frames are always returned in strict order. If the expected frame
                is not available within this timeout, empty detections are returned.
            input_key: Key in data dict containing input frame (default: 'frame')
            output_key: Key to store detections in data dict (default: 'detections')
        """
        self.pool_size = pool_size
        self.max_queue_size = max_queue_size
        self.reorder_timeout = reorder_timeout
        self.input_key = input_key
        self.output_key = output_key

        # Threading components
        self._frame_queue: Queue = Queue(maxsize=max_queue_size)
        self._result_queue: Queue = Queue()
        self._workers: list[Thread] = []
        self._reorder_buffer: dict[int, tuple] = {}
        self._stop_event = Event()

        # Sequence tracking
        self._sequence_counter = 0
        self._sequence_lock = Lock()
        self._next_expected_sequence = 0

        # State
        self._is_running = False

        # Metrics
        self._metrics_lock = Lock()
        self._metrics = {
            "frames_processed": 0,
            "frames_dropped": 0,
            "frames_reordered": 0,
            "avg_queue_time_ms": 0.0,
            "avg_inference_time_ms": 0.0,
            "avg_reorder_delay_ms": 0.0,
            "queue_full_count": 0,
            "workers_active": 0,
            "total_queue_time_ms": 0.0,
            "total_inference_time_ms": 0.0,
            "total_reorder_delay_ms": 0.0,
        }

    @abstractmethod
    def _run_inference(self, frame: np.ndarray) -> Detections:
        """
        Run inference on a frame. Must be implemented by subclasses.

        Args:
            frame: Input frame for detection

        Returns:
            Detections object with detection results
        """
        pass

    def _worker(self, worker_id: int) -> None:
        """
        Worker thread that continuously processes frames from the queue.

        Args:
            worker_id: Unique identifier for this worker
        """
        while not self._stop_event.is_set():
            try:
                # Get frame from queue
                frame_data = self._frame_queue.get(timeout=0.1)

                if frame_data is None:  # Poison pill
                    break

                sequence_num, frame, enqueue_time = frame_data

                # Calculate queue time
                queue_time = time.time() - enqueue_time

                # Run inference
                inference_start = time.time()
                detections = self._run_inference(frame)
                inference_time = time.time() - inference_start

                # Put result in output queue
                result_timestamp = time.time()
                self._result_queue.put(
                    (
                        sequence_num,
                        detections,
                        frame,
                        enqueue_time,
                        queue_time,
                        inference_time,
                        result_timestamp,
                    )
                )

                self._frame_queue.task_done()

            except Empty:
                continue
            except Exception as e:
                # Log error but don't crash the worker
                print(f"Error in worker {worker_id}: {e}")
                continue

    def _get_next_ordered_result(self) -> tuple | None:
        """
        Get the next result in strict order from the reorder buffer.

        Waits for the next expected sequence number. If the expected frame
        is not available within reorder_timeout, returns None.

        Returns:
            Tuple with result data, or None if timeout reached
        """
        timeout_start = time.time()

        while True:
            # Check if we have the next expected sequence in buffer
            if self._next_expected_sequence in self._reorder_buffer:
                result = self._reorder_buffer.pop(self._next_expected_sequence)
                self._next_expected_sequence += 1
                return result

            # Try to get more results from workers
            try:
                result = self._result_queue.get(timeout=0.01)
                seq_num = result[0]

                if seq_num == self._next_expected_sequence:
                    # Perfect! This is the one we're waiting for
                    self._next_expected_sequence += 1
                    return result
                else:
                    # Store for later (out of order result)
                    self._reorder_buffer[seq_num] = result
                    with self._metrics_lock:
                        self._metrics["frames_reordered"] += 1

            except Empty:
                # Check if timeout reached
                elapsed = time.time() - timeout_start
                if elapsed > self.reorder_timeout:
                    # Timeout - give up waiting for expected frame
                    return None
                # Continue waiting
                continue

    def start(self) -> PoolDetectorStep:
        """
        Start the worker pool.

        Returns:
            Self for method chaining
        """
        if not self._is_running:
            self._stop_event.clear()

            # Create and start workers
            for i in range(self.pool_size):
                worker = Thread(
                    target=self._worker,
                    name=f"PoolDetectorWorker-{i}",
                    args=(i,),
                    daemon=True,
                )
                worker.start()
                self._workers.append(worker)

            self._is_running = True

            with self._metrics_lock:
                self._metrics["workers_active"] = self.pool_size

        return self

    def stop(self) -> None:
        """Stop the worker pool and cleanup resources."""
        if self._is_running:
            self._stop_event.set()

            # Send poison pills to workers
            for _ in range(self.pool_size):
                try:
                    self._frame_queue.put(None, timeout=1.0)
                except:
                    pass

            # Wait for workers to finish
            for worker in self._workers:
                worker.join(timeout=2.0)

            self._workers.clear()
            self._is_running = False

            with self._metrics_lock:
                self._metrics["workers_active"] = 0

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Process frame using the detector pool.

        Args:
            data: Pipeline data containing input frame

        Returns:
            Data with detections field added. The returned frame is the
            one that was actually processed (may be slightly delayed to
            maintain order).
        """
        frame = data.get(self.input_key)
        if frame is None:
            return data

        # Start workers if not running
        if not self._is_running:
            self.start()

        # Assign sequence number
        with self._sequence_lock:
            sequence_num = self._sequence_counter
            self._sequence_counter += 1

        # Try to enqueue frame
        enqueue_time = time.time()
        try:
            self._frame_queue.put_nowait((sequence_num, frame, enqueue_time))
        except:
            # Queue full - drop frame
            with self._metrics_lock:
                self._metrics["queue_full_count"] += 1
                self._metrics["frames_dropped"] += 1
            # Return data with empty detections
            data[self.output_key] = Detections.empty()
            return data

        # Get next ordered result
        result = self._get_next_ordered_result()

        if result is not None:
            (
                seq_num,
                detections,
                processed_frame,
                orig_enqueue_time,
                queue_time,
                inference_time,
                result_timestamp,
            ) = result

            # Calculate reorder delay
            reorder_delay = (time.time() - result_timestamp) * 1000
            queue_time_ms = queue_time * 1000
            inference_time_ms = inference_time * 1000

            # Update data
            data[self.output_key] = detections
            data[self.input_key] = processed_frame
            data["sequence_number"] = seq_num

            # Update metrics
            with self._metrics_lock:
                self._metrics["frames_processed"] += 1
                count = self._metrics["frames_processed"]

                self._metrics["total_queue_time_ms"] += queue_time_ms
                self._metrics["avg_queue_time_ms"] = (
                    self._metrics["total_queue_time_ms"] / count
                )

                self._metrics["total_inference_time_ms"] += inference_time_ms
                self._metrics["avg_inference_time_ms"] = (
                    self._metrics["total_inference_time_ms"] / count
                )

                self._metrics["total_reorder_delay_ms"] += reorder_delay
                self._metrics["avg_reorder_delay_ms"] = (
                    self._metrics["total_reorder_delay_ms"] / count
                )
        else:
            # Timeout - return empty detections
            data[self.output_key] = Detections.empty()

        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Process if frame exists."""
        return self.input_key in data

    def get_metrics(self) -> dict[str, Any]:
        """
        Get performance metrics.

        Returns:
            Dictionary with metrics:
                - frames_processed: Total frames successfully processed
                - frames_dropped: Frames dropped due to full queue
                - frames_reordered: Frames that needed reordering
                - avg_queue_time_ms: Average time in queue before processing
                - avg_inference_time_ms: Average inference time
                - avg_reorder_delay_ms: Average time spent in reorder buffer
                - queue_full_count: Times queue was full
                - workers_active: Number of active workers
        """
        with self._metrics_lock:
            return self._metrics.copy()

    def get_queue_size(self) -> tuple[int, int]:
        """
        Get current queue size and maximum queue size.

        Returns:
            Tuple of (current_size, max_size)
        """
        return (self._frame_queue.qsize(), self.max_queue_size)

    def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        with self._metrics_lock:
            for key in self._metrics:
                if key != "workers_active":
                    if isinstance(self._metrics[key], (int, float)):
                        self._metrics[key] = 0

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


class PoolYOLODetectionStep(PoolDetectorStep):
    """
    Pool-based YOLO object detection step with parallel processing.

    This step creates a pool of YOLO detectors running in parallel threads,
    allowing for higher throughput when processing video streams. Each worker
    has its own model instance to avoid lock contention.

    Frames are processed in parallel but returned in the original order,
    ensuring proper synchronization with downstream pipeline steps.

    Examples:
        ```python
        import supervision as sv

        # Basic pool detection with 3 workers
        pipeline = (
            sv.Pipeline(sv.VideoFileSource("video.mp4"))
            | sv.PoolYOLODetectionStep(
                model_path="yolov8n.pt",
                pool_size=3
            )
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("Pool Detection")
        )
        pipeline.run()
        ```

        ```python
        # High-throughput configuration with monitoring
        detector = sv.PoolYOLODetectionStep(
            model_path="yolov8n.pt",
            pool_size=4,
            max_queue_size=20,
            reorder_timeout=2.0,
            conf=0.5
        )

        pipeline = (
            sv.Pipeline(sv.VideoFileSource("video.mp4"))
            | sv.FPSCalculatorStep()
            | detector
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("High Throughput Detection")
        )

        # Monitor performance
        for data in pipeline:
            metrics = detector.get_metrics()
            fps = data.get("fps", 0)
            print(f"FPS: {fps:.1f}, Processed: {metrics['frames_processed']}")
        ```
    """

    def __init__(
        self,
        model_path: str,
        pool_size: int = 2,
        conf: float = 0.25,
        iou: float = 0.45,
        verbose: bool = False,
        max_queue_size: int = 10,
        reorder_timeout: float = 1.0,
        warmup: bool = True,
        input_key: str = "frame",
        output_key: str = "detections",
    ):
        """
        Initialize pool YOLO detection step.

        Args:
            model_path: Path to YOLO model file (.pt, .engine, etc.)
            pool_size: Number of parallel detector workers (default: 2)
            conf: Confidence threshold for detections (0.0-1.0)
            iou: IOU threshold for NMS (0.0-1.0)
            verbose: Whether to print verbose YOLO output
            max_queue_size: Maximum frames to queue (default: 10)
            reorder_timeout: Maximum time to wait for expected frame (seconds).
                Frames are always returned in strict order.
            warmup: Whether to run warmup inference on each model
            input_key: Key in data dict containing input frame (default: 'frame')
            output_key: Key to store detections in data dict (default: 'detections')

        Note:
            Ultralytics automatically detects and uses GPU if CUDA is available.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for PoolYOLODetectionStep. "
                "Install it with: pip install ultralytics"
            )

        # Initialize base class
        super().__init__(
            pool_size=pool_size,
            max_queue_size=max_queue_size,
            reorder_timeout=reorder_timeout,
            input_key=input_key,
            output_key=output_key,
        )

        # YOLO parameters
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.verbose = verbose

        # Create one model per worker (device auto-detected by ultralytics)
        self._models: list = []
        for i in range(pool_size):
            model = YOLO(model_path)
            self._models.append(model)

            # Warmup model
            if warmup:
                self._warmup_model(model, i)

        # Start worker pool
        self.start()

    def _warmup_model(self, model, worker_id: int) -> None:
        """
        Warmup a model with dummy inference.

        Args:
            model: YOLO model to warmup
            worker_id: Worker ID for logging
        """
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            _ = model.predict(
                source=dummy_frame, conf=self.conf, iou=self.iou, verbose=False
            )
        except Exception as e:
            print(f"Warning: Model warmup failed for worker {worker_id}: {e}")

    def _run_inference(self, frame: np.ndarray) -> Detections:
        """
        Run YOLO inference on frame.

        Args:
            frame: Input frame for detection

        Returns:
            Detections object with detection results
        """
        # Get worker ID from thread name
        thread_name = threading.current_thread().name
        try:
            worker_id = int(thread_name.split("-")[-1])
        except (ValueError, IndexError):
            worker_id = 0

        # Get corresponding model
        model = self._models[worker_id % len(self._models)]

        # Run inference
        results = model.predict(
            source=frame, conf=self.conf, iou=self.iou, verbose=self.verbose
        )

        return Detections.from_ultralytics(results[0])

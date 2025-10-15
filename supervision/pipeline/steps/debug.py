from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from supervision.detection.core import Detections


class DebugLoggerStep:
    """
    Pipeline step for logging frame-by-frame metrics to a file for debugging and analysis.

    This step collects metrics from each frame (FPS, detection counts, timing information,
    etc.) and periodically writes them to a JSON file. The logged data can be used for
    performance analysis and visualization.

    The logger captures:
    - Frame number and timestamp
    - FPS (if available in data)
    - Detection counts (total detections, per-class counts)
    - Processing times (detection, tracking, annotation, etc.)
    - Custom metrics from the data dictionary
    - Queue sizes and other async metrics

    Examples:
        ```python
        import supervision as sv

        # Basic usage with default settings (log every 30 frames)
        pipeline = (
            sv.Pipeline(sv.WebcamSource())
            | sv.FPSCalculatorStep()
            | sv.YOLODetectionStep("yolov8n.pt")
            | sv.DebugLoggerStep(log_file="debug.json")
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("Debug Mode")
        )
        pipeline.run()
        ```

        ```python
        # Custom log interval and metrics
        debug_logger = sv.DebugLoggerStep(
            log_file="metrics.json",
            log_interval=60,  # Log every 60 frames
            include_detections_data=True,  # Include detailed detection info
            custom_metrics=["sequence_number", "roi_size"]
        )

        pipeline = (
            sv.Pipeline(sv.VideoFileSource("video.mp4"))
            | sv.FPSCalculatorStep()
            | sv.AsyncYOLODetectionStep("yolov8n.pt")
            | sv.ByteTrackerStep()
            | debug_logger
            | sv.DisplaySink("Debug")
        )
        pipeline.run()

        # Get summary statistics
        stats = debug_logger.get_statistics()
        print(f"Average FPS: {stats['fps']['mean']:.2f}")
        print(f"Average detections: {stats['detection_count']['mean']:.2f}")
        ```
    """

    def __init__(
        self,
        log_file: str | Path,
        log_interval: int = 30,
        include_detections_data: bool = False,
        custom_metrics: list[str] | None = None,
        auto_flush: bool = True,
    ):
        """
        Initialize debug logger step.

        Args:
            log_file: Path to output JSON file for logging metrics
            log_interval: Number of frames between log writes to file (default: 30)
            include_detections_data: Whether to include detailed detection info
                (bounding boxes, confidences, class IDs) in logs (default: False)
            custom_metrics: List of additional keys from data dict to log
            auto_flush: Whether to flush logs to file automatically at interval
                (default: True). If False, call flush() manually.
        """
        self.log_file = Path(log_file)
        self.log_interval = log_interval
        self.include_detections_data = include_detections_data
        self.custom_metrics = custom_metrics or []
        self.auto_flush = auto_flush

        # State
        self._frame_count = 0
        self._start_time = time.time()
        self._metrics_buffer: list[dict[str, Any]] = []
        self._accumulated_stats: dict[str, list[float]] = {
            "fps": [],
            "detection_count": [],
            "processing_time_ms": [],
        }

        # Create log file directory if it doesn't exist
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize log file with empty list
        with open(self.log_file, "w") as f:
            json.dump([], f)

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Process frame and collect metrics.

        Args:
            data: Pipeline data dictionary

        Returns:
            Unchanged data dictionary (pass-through)
        """
        self._frame_count += 1
        frame_time = time.time()

        # Build metrics for this frame
        metrics = {
            "frame_number": self._frame_count,
            "timestamp": frame_time,
            "elapsed_time": frame_time - self._start_time,
        }

        # Extract FPS if available
        fps = data.get("fps")
        if fps is not None:
            metrics["fps"] = float(fps)
            self._accumulated_stats["fps"].append(float(fps))

        # Extract detection information
        detections: Detections | None = data.get("detections")
        if detections is not None:
            detection_count = len(detections)
            metrics["detection_count"] = detection_count
            self._accumulated_stats["detection_count"].append(detection_count)

            # Class distribution
            if detections.class_id is not None:
                unique, counts = np.unique(detections.class_id, return_counts=True)
                metrics["class_distribution"] = {
                    int(class_id): int(count)
                    for class_id, count in zip(unique, counts)
                }

            # Confidence statistics
            if detections.confidence is not None and len(detections.confidence) > 0:
                metrics["confidence_mean"] = float(np.mean(detections.confidence))
                metrics["confidence_min"] = float(np.min(detections.confidence))
                metrics["confidence_max"] = float(np.max(detections.confidence))

            # Tracker IDs if available
            if detections.tracker_id is not None:
                tracked_count = np.sum(detections.tracker_id >= 0)
                metrics["tracked_count"] = int(tracked_count)
                metrics["unique_tracker_ids"] = len(
                    np.unique(detections.tracker_id[detections.tracker_id >= 0])
                )

            # Include detailed detection data if requested
            if self.include_detections_data:
                metrics["detections"] = {
                    "xyxy": detections.xyxy.tolist() if detections.xyxy is not None else None,
                    "confidence": detections.confidence.tolist() if detections.confidence is not None else None,
                    "class_id": detections.class_id.tolist() if detections.class_id is not None else None,
                    "tracker_id": detections.tracker_id.tolist() if detections.tracker_id is not None else None,
                }

        # Extract timing metrics (common keys used in async/pool steps)
        timing_keys = [
            "detection_time_ms",
            "tracking_time_ms",
            "annotation_time_ms",
            "avg_inference_time_ms",
            "avg_queue_time_ms",
            "avg_reorder_delay_ms",
        ]
        for key in timing_keys:
            if key in data:
                metrics[key] = float(data[key])
                if "processing_time_ms" in key or "time_ms" in key:
                    self._accumulated_stats.setdefault(key, []).append(float(data[key]))

        # Extract queue/async metrics
        queue_keys = [
            "frames_processed",
            "frames_skipped",
            "frames_cached",
            "frames_queued",
            "frames_dropped",
            "queue_full_count",
        ]
        for key in queue_keys:
            if key in data:
                metrics[key] = int(data[key])

        # Extract custom metrics
        for key in self.custom_metrics:
            if key in data:
                value = data[key]
                # Convert numpy types to native Python types
                if isinstance(value, (np.integer, np.floating)):
                    metrics[key] = float(value)
                elif isinstance(value, np.ndarray):
                    metrics[key] = value.tolist()
                else:
                    metrics[key] = value

        # Add to buffer
        self._metrics_buffer.append(metrics)

        # Auto-flush to file if interval reached
        if self.auto_flush and len(self._metrics_buffer) >= self.log_interval:
            self.flush()

        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Always process."""
        return True

    def flush(self) -> None:
        """
        Flush buffered metrics to log file.

        This method writes all accumulated metrics to the JSON file.
        It's called automatically based on log_interval if auto_flush is True,
        but can also be called manually.
        """
        if not self._metrics_buffer:
            return

        # Read existing data
        try:
            with open(self.log_file, "r") as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []

        # Append new metrics
        existing_data.extend(self._metrics_buffer)

        # Write back to file
        with open(self.log_file, "w") as f:
            json.dump(existing_data, f, indent=2)

        # Clear buffer
        self._metrics_buffer.clear()

    def get_statistics(self) -> dict[str, dict[str, float]]:
        """
        Get summary statistics for accumulated metrics.

        Returns:
            Dictionary with statistics (mean, min, max, std) for each metric
        """
        stats = {}

        for metric_name, values in self._accumulated_stats.items():
            if values:
                stats[metric_name] = {
                    "mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "std": float(np.std(values)),
                    "count": len(values),
                }

        return stats

    def reset(self) -> None:
        """
        Reset all counters and buffers.

        Useful when starting a new logging session without creating a new instance.
        """
        self._frame_count = 0
        self._start_time = time.time()
        self._metrics_buffer.clear()
        for key in self._accumulated_stats:
            self._accumulated_stats[key].clear()

    def __del__(self):
        """Ensure metrics are flushed on deletion."""
        try:
            if self._metrics_buffer:
                self.flush()
        except Exception:
            pass  # Ignore errors during cleanup

"""
ROI-based Pool Detection + Tracking + Line Crossing Counter Demo

Demonstrates object detection in a configurable Region of Interest (ROI) using parallel
detection pool combined with tracking and line crossing detection. This approach
allows processing only a specific area of interest while maintaining the original
frame context for visualization.

Features:
- Configurable ROI (position and size) for detection
- Parallel detection pool for high throughput
- Coordinate translation between ROI and full frame
- ROI visualization with cyan rectangle
- Line crossing counting with socket and file logging
- Configuration via YAML file (with comments support)

Usage:
    # Using default config.yaml
    python roi_pool_line_zone_demo.py

    # Using custom config file
    python roi_pool_line_zone_demo.py --config my_config.yaml

    # Override config file settings via command line
    python roi_pool_line_zone_demo.py --config config.yaml --pool-size 6 --conf 0.5
"""

import argparse
import logging
import socket
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

import supervision as sv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_point(point_str: str) -> sv.Point:
    """Parse point from string format 'x,y'."""
    try:
        x, y = map(int, point_str.split(","))
        return sv.Point(x=x, y=y)
    except Exception:
        raise ValueError(f"Invalid point format: {point_str}. Expected 'x,y'")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from: {config_path}")
    return config


class ROIExtractionStep:
    """
    Pipeline step to extract a configurable ROI from the frame.

    Extracts a rectangular ROI from specified coordinates and stores
    the offset coordinates for later translation.
    """

    def __init__(
        self,
        x: int | None = None,
        y: int | None = None,
        width: int = 640,
        height: int = 640,
    ):
        """
        Initialize ROI extraction step.

        Args:
            x: X coordinate of ROI top-left corner (None = center horizontally)
            y: Y coordinate of ROI top-left corner (None = center vertically)
            width: Width of ROI (default: 640)
            height: Height of ROI (default: 640)
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.roi_x = 0
        self.roi_y = 0

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Extract ROI from frame."""
        frame = data.get("frame")
        if frame is None:
            return data

        # Store original frame
        data["original_frame"] = frame

        # Calculate ROI coordinates
        h, w = frame.shape[:2]

        # Use configured position or center
        if self.x is None:
            self.roi_x = max(0, (w - self.width) // 2)
        else:
            self.roi_x = max(0, min(self.x, w - self.width))

        if self.y is None:
            self.roi_y = max(0, (h - self.height) // 2)
        else:
            self.roi_y = max(0, min(self.y, h - self.height))

        # Ensure ROI fits within frame
        roi_x_end = min(self.roi_x + self.width, w)
        roi_y_end = min(self.roi_y + self.height, h)

        # Extract ROI
        roi = frame[self.roi_y : roi_y_end, self.roi_x : roi_x_end]

        # Store ROI coordinates and replace frame with ROI
        data["roi_offset"] = (self.roi_x, self.roi_y)
        data["roi_size"] = (roi.shape[1], roi.shape[0])  # (width, height)
        data["frame"] = roi

        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Always process."""
        return True


class CoordinateTranslationStep:
    """
    Pipeline step to translate detection coordinates from ROI to full frame.

    Translates bounding boxes and other geometric data from ROI coordinates
    back to original frame coordinates.
    """

    def __init__(self, detections_key: str = "detections"):
        """
        Initialize coordinate translation step.

        Args:
            detections_key: Key in data dict containing Detections object
        """
        self.detections_key = detections_key

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Translate detection coordinates from ROI to full frame."""
        detections = data.get(self.detections_key)
        roi_offset = data.get("roi_offset")

        if detections is None or roi_offset is None or len(detections) == 0:
            return data

        roi_x, roi_y = roi_offset

        # Translate bounding boxes
        if detections.xyxy is not None:
            detections.xyxy[:, [0, 2]] += roi_x  # x coordinates
            detections.xyxy[:, [1, 3]] += roi_y  # y coordinates

        # Translate masks if present
        if detections.mask is not None:
            # This would require more complex mask translation
            # For now, we just note that masks are in ROI coordinates
            pass

        data[self.detections_key] = detections
        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Always process."""
        return True


class ROIVisualizationStep:
    """
    Pipeline step to draw ROI rectangle on original frame.

    Draws a cyan rectangle showing the ROI area on the original full frame.
    """

    def __init__(self, color: tuple = (255, 255, 0), thickness: int = 2):
        """
        Initialize ROI visualization step.

        Args:
            color: Color of ROI rectangle in BGR (default: cyan)
            thickness: Thickness of rectangle border
        """
        self.color = color
        self.thickness = thickness

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Draw ROI rectangle on frame."""
        frame = data.get("frame")
        roi_offset = data.get("roi_offset")
        original_frame = data.get("original_frame")

        if frame is None or roi_offset is None or original_frame is None:
            return data

        roi_x, roi_y = roi_offset
        roi_h, roi_w = frame.shape[:2]

        # Draw ROI rectangle on original frame
        cv2.rectangle(
            original_frame,
            (roi_x, roi_y),
            (roi_x + roi_w, roi_y + roi_h),
            self.color,
            self.thickness,
        )

        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Always process."""
        return True


class FrameRestoreStep:
    """
    Pipeline step to restore original frame for annotation.

    Replaces the ROI frame with the original full frame so annotations
    are drawn on the full frame with translated coordinates.
    """

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Restore original frame."""
        original_frame = data.get("original_frame")
        if original_frame is not None:
            data["frame"] = original_frame

        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Always process."""
        return True


class LineCrossingLoggerStep:
    """
    Pipeline step to log line crossing events to socket and file.

    Monitors line crossing counts and logs increments with timestamps to both
    a TCP socket and a rotating log file.
    """

    def __init__(self, socket_port: int = 7777, log_file: str = "count.log"):
        """
        Initialize line crossing logger.

        Args:
            socket_port: TCP port for socket server
            log_file: Path to rotating log file
        """
        self.socket_port = socket_port
        self.log_file = log_file

        # Track previous counts
        self.prev_in_count = 0
        self.prev_out_count = 0

        # Setup rotating file logger
        self.file_logger = logging.getLogger("LineCrossingLogger")
        self.file_logger.setLevel(logging.INFO)

        # Create rotating file handler (10MB max, 5 backups)
        handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        self.file_logger.addHandler(handler)

        # Socket server setup
        self.socket_clients = []
        self.socket_lock = threading.Lock()
        self.server_socket = None

        # Start socket server in background thread
        self._start_socket_server()

    def _start_socket_server(self):
        """Start TCP socket server in background thread."""

        def accept_clients():
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(
                    socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
                )
                self.server_socket.bind(("0.0.0.0", self.socket_port))
                self.server_socket.listen(5)
                self.server_socket.settimeout(1.0)  # Non-blocking accept

                logger.info(
                    f"Socket server listening on port {self.socket_port}"
                )

                while True:
                    try:
                        client, addr = self.server_socket.accept()
                        with self.socket_lock:
                            self.socket_clients.append(client)
                        logger.info(f"Client connected: {addr}")
                    except socket.timeout:
                        continue
                    except Exception as e:
                        logger.error(f"Socket accept error: {e}")
                        break

            except Exception as e:
                logger.error(f"Failed to start socket server: {e}")

        server_thread = threading.Thread(target=accept_clients, daemon=True)
        server_thread.start()

    def _send_to_socket(self, message: str):
        """Send message to all connected socket clients."""
        with self.socket_lock:
            disconnected = []
            for client in self.socket_clients:
                try:
                    client.sendall((message + "\n").encode("utf-8"))
                except Exception:
                    disconnected.append(client)

            # Remove disconnected clients
            for client in disconnected:
                self.socket_clients.remove(client)
                try:
                    client.close()
                except Exception:
                    pass

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Monitor line crossings and log events."""
        line_zone = data.get("line_zone")

        if line_zone is None:
            return data

        # Get current counts
        in_count = line_zone.in_count
        out_count = line_zone.out_count

        # Calculate increments
        in_increment = in_count - self.prev_in_count
        out_increment = out_count - self.prev_out_count

        # Log if there's any change
        if in_increment != 0 or out_increment != 0:
            timestamp = datetime.now().isoformat()
            message = f"{timestamp},{in_increment},{out_increment}"

            # Log to file
            self.file_logger.info(message)

            # Send to socket
            self._send_to_socket(message)

            # Print to console
            logger.info(
                f"Line crossing: IN +{in_increment}, OUT +{out_increment}"
            )

        # Update previous counts
        self.prev_in_count = in_count
        self.prev_out_count = out_count

        return data

    def filter(self, data: dict[str, Any]) -> bool:
        """Always process."""
        return True

    def cleanup(self):
        """Cleanup socket server and close connections."""
        with self.socket_lock:
            for client in self.socket_clients:
                try:
                    client.close()
                except Exception:
                    pass
            self.socket_clients.clear()

        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass


def main():
    """Run ROI-based pool detection with line zone counting demo."""
    parser = argparse.ArgumentParser(
        description="ROI-based pool detection with tracking and line crossing counter"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )

    # Allow command-line overrides of config file
    parser.add_argument("--device", type=str, help="Device for inference (cuda/cpu)")
    parser.add_argument("--source", type=str, help="Source type (webcam/file/stream)")
    parser.add_argument("--input", type=str, help="Input source path/URL")
    parser.add_argument("--model", type=str, help="YOLO model path")
    parser.add_argument("--conf", type=float, help="Confidence threshold")
    parser.add_argument("--pool-size", type=int, help="Number of parallel workers")
    parser.add_argument("--roi-size", type=int, help="ROI size (square)")
    parser.add_argument("--output", type=str, help="Output video path")

    args = parser.parse_args()

    # Load config from file
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    # Override config with command-line arguments
    if args.device:
        config["device"] = args.device
    if args.source:
        config["source"] = args.source
    if args.input:
        config["input"] = args.input
    if args.model:
        config["model"] = args.model
    if args.conf is not None:
        config["conf"] = args.conf
    if args.pool_size:
        config["pool_size"] = args.pool_size
    if args.roi_size:
        config["roi_size"] = args.roi_size
    if args.output:
        config["output"] = args.output

    # Print configuration
    print("=" * 60)
    print("ROI-based Pool Detection + Line Zone Counter Demo")
    print("=" * 60)
    print(f"Source: {config.get('source', 'webcam')}")
    print(f"Input: {config.get('input', 'default')}")
    print(f"Model: {config['model']}")

    # Print ROI configuration
    roi_cfg = config.get('roi', {})
    roi_x = roi_cfg.get('x', 'centered')
    roi_y = roi_cfg.get('y', 'centered')
    roi_w = roi_cfg.get('width', 640)
    roi_h = roi_cfg.get('height', 640)
    print(f"ROI: x={roi_x}, y={roi_y}, {roi_w}x{roi_h}")

    print(f"Pool Size: {config['pool_size']} workers")
    print(f"Device: {config['device']}")
    print(f"Confidence: {config['conf']}")
    print(f"Socket Port: {config.get('socket_port', 7777)}")
    print(f"Log File: {config.get('log_file', 'count.log')}")
    print("=" * 60)
    print()

    # Create source based on config
    source_type = config.get("source", "webcam")
    if source_type == "webcam":
        source = sv.WebcamSource(camera_id=0)
    elif source_type == "file":
        source = sv.VideoFileSource(video_path=config["input"])
    elif source_type == "stream":
        source = sv.StreamSource(stream_url=config["input"])
    else:
        raise ValueError(f"Unknown source type: {source_type}")

    # Get video info from source
    import cv2

    try:
        cap = source.cap if hasattr(source, "cap") else None
        if cap is not None and cap.isOpened():
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Validate values
            if fps <= 0 or fps > 120:
                fps = 30
            if width <= 0 or height <= 0:
                width, height = 1920, 1080
        else:
            fps, width, height = 30, 1920, 1080
    except Exception:
        fps, width, height = 30, 1920, 1080

    print(f"Video info: {width}x{height} @ {fps} fps\n")

    # Parse line points from config
    line_start = parse_point(config.get("line_start", "640,0"))
    line_end = parse_point(config.get("line_end", "640,720"))

    print(f"Line: ({line_start.x}, {line_start.y}) -> ({line_end.x}, {line_end.y})")
    print("=" * 60)
    print()

    # Create pipeline steps

    # 1. ROI extraction step
    roi_config = config.get("roi", {})
    roi_step = ROIExtractionStep(
        x=roi_config.get("x"),
        y=roi_config.get("y"),
        width=roi_config.get("width", 640),
        height=roi_config.get("height", 640),
    )

    # 2. Pool detector (operates on ROI)
    detector = sv.PoolYOLODetectionStep(
        model_path=config["model"],
        pool_size=config["pool_size"],
        device=config["device"],
        conf=config["conf"],
        max_queue_size=config.get("queue_size", 10),
        warmup=True,
    )

    # 3. Coordinate translation step (ROI -> full frame)
    coord_translate = CoordinateTranslationStep()

    # 4. Tracker (requires frames in order!)
    tracker = sv.ByteTrackerStep(
        track_activation_threshold=config.get("track_threshold", 0.25),
        lost_track_buffer=config.get("lost_buffer", 30),
        minimum_matching_threshold=config.get("match_threshold", 0.8),
    )

    # 5. Line zone step (with translated coordinates)
    line_zone_step = sv.LineZoneStep(
        start=line_start,
        end=line_end,
    )

    # 6. ROI visualization step
    roi_viz = ROIVisualizationStep(color=(255, 255, 0), thickness=2)  # Cyan in BGR

    # 7. Frame restore step (switch back to original frame for annotation)
    frame_restore = FrameRestoreStep()

    # 8. Line crossing logger
    crossing_logger = LineCrossingLoggerStep(
        socket_port=config.get("socket_port", 7777),
        log_file=config.get("log_file", "count.log"),
    )

    # Build pipeline
    pipeline = sv.Pipeline(source)

    # Add FPS calculator if requested
    if config.get("show_fps", False):
        pipeline = pipeline | sv.FPSCalculatorStep()

    # Add all processing steps
    pipeline = (
        pipeline
        | roi_step  # Extract centered ROI
        | detector  # Detect on ROI (parallel pool)
        | coord_translate  # Translate coordinates to full frame
        | tracker  # Track objects (requires ordered frames!)
        | line_zone_step  # Count line crossings
        | roi_viz  # Draw ROI rectangle on original frame
        | frame_restore  # Restore original frame for annotation
        | sv.TrackerAnnotatorStep(  # Draw tracking visualization
            trace_length=30, trace_thickness=2, copy_frame=False
        )
        | sv.LineZoneAnnotatorStep(  # Draw line and counts
            color=sv.Color.WHITE,
            thickness=config.get("line_thickness", 4),
            text_scale=0.8,
            custom_in_text=config.get("in_text"),
            custom_out_text=config.get("out_text"),
            copy_frame=False,
        )
        | crossing_logger  # Log crossing events
    )

    # Add metrics overlay if requested
    if config.get("show_metrics", False):

        def add_metrics_overlay(data):
            """Add metrics text overlay to frame."""
            frame = data.get("frame")
            if frame is None:
                return data

            metrics = detector.get_metrics()
            queue_current, queue_max = detector.get_queue_size()
            detections = data.get("detections", sv.Detections.empty())
            num_tracked = (
                len(set(detections.tracker_id))
                if detections.tracker_id is not None
                else 0
            )

            # Get line zone counts
            line_zone = data.get("line_zone")
            in_count = line_zone.in_count if line_zone is not None else 0
            out_count = line_zone.out_count if line_zone is not None else 0

            # Get ROI info from data
            roi_size = data.get("roi_size", (640, 640))
            roi_offset = data.get("roi_offset", (0, 0))

            # Get tracking time from current frame
            tracking_time = data.get("tracker_processing_time_ms", 0.0)

            # Create metrics text
            lines = [
                f"Workers: {metrics['workers_active']}",
                f"ROI: {roi_size[0]}x{roi_size[1]} at ({roi_offset[0]},{roi_offset[1]})",
                f"Detections: {len(detections)}",
                f"Tracked: {num_tracked}",
                f"IN Count: {in_count}",
                f"OUT Count: {out_count}",
                f"Processed: {metrics['frames_processed']}",
                f"Dropped: {metrics['frames_dropped']}",
                f"Input Queue: {queue_current}/{queue_max}",
                f"Avg Inference: {metrics['avg_inference_time_ms']:.1f}ms",
                f"Tracking: {tracking_time:.1f}ms",
            ]

            # Draw semi-transparent background
            overlay = frame.copy()
            bg_height = len(lines) * 25 + 15
            cv2.rectangle(overlay, (5, 5), (320, bg_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Draw text on frame
            y_offset = 25
            for line in lines:
                cv2.putText(
                    frame,
                    line,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                y_offset += 25

            data["frame"] = frame
            return data

        pipeline = pipeline | sv.CallbackStep(add_metrics_overlay)

    # Add sink
    output_path = config.get("output")
    if output_path:
        # File sink
        pipeline = pipeline | sv.VideoFileSink(
            output_path=output_path,
            fps=fps,
            width=width,
            height=height,
        )

    # Always add display sink
    pipeline = pipeline | sv.DisplaySink(window_name="ROI Detection + Line Counter")

    # Run pipeline
    print("Starting pipeline... Press 'q' to quit\n")
    try:
        frame_count = 0
        for data in pipeline:
            frame_count += 1

            # Print metrics every 30 frames
            if frame_count % 30 == 0:
                detector_metrics = detector.get_metrics()
                detections = data.get("detections", sv.Detections.empty())
                num_tracked = (
                    len(set(detections.tracker_id))
                    if detections.tracker_id is not None
                    else 0
                )

                # Get line zone counts
                line_zone = data.get("line_zone")
                in_count = line_zone.in_count if line_zone is not None else 0
                out_count = line_zone.out_count if line_zone is not None else 0

                # Get tracking time from current frame
                tracking_time = data.get("tracker_processing_time_ms", 0.0)

                print(
                    f"Frame {frame_count:4d} | "
                    f"Detections: {len(detections):2d} | Tracked: {num_tracked:2d} | "
                    f"IN: {in_count:3d} | OUT: {out_count:3d} | "
                    f"Inference: {detector_metrics['avg_inference_time_ms']:5.1f}ms | "
                    f"Tracking: {tracking_time:5.1f}ms"
                )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        # Cleanup
        crossing_logger.cleanup()

    # Print final metrics
    detector_metrics = detector.get_metrics()
    print("\n" + "=" * 60)
    print("Detector Metrics")
    print("=" * 60)
    print(f"Total Frames:         {detector_metrics['frames_processed']}")
    print(f"Frames Dropped:       {detector_metrics['frames_dropped']}")
    print(f"Avg Inference Time:   {detector_metrics['avg_inference_time_ms']:.2f} ms")
    print(f"Workers Active:       {detector_metrics['workers_active']}")

    # Print tracker metrics
    import utils
    tracker_metrics = tracker.get_metrics()
    utils.print_tracker_metrics(tracker_metrics)

    # Get final line zone counts
    try:
        if line_zone is not None:
            print()
            print(f"Line Zone Counts:")
            print(f"  Objects IN:  {in_count}")
            print(f"  Objects OUT: {out_count}")
            print(f"  Net Count:   {in_count - out_count}")
    except Exception:
        pass

    print("=" * 60)

    if output_path:
        print(f"\nVideo saved to: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()

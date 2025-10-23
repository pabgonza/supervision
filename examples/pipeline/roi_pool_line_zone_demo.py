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
from typing import Any

import supervision as sv
import utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
                f"Line crossing: IN +{in_increment}, OUT +{out_increment}, IDs crossed in: {line_zone.last_crossed_in_ids}, IDs crossed out: {line_zone.last_crossed_out_ids}"
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
        config = utils.load_yaml_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    # Override config with command-line arguments
    video_cfg = config.get("video", {})
    detector_cfg = config.get("detector", {})
    pool_cfg = config.get("pool", {})

    if args.device:
        detector_cfg["device"] = args.device
    if args.input:
        video_cfg["input"] = args.input
    if args.model:
        detector_cfg["model_path"] = args.model
    if args.conf is not None:
        detector_cfg["confidence_threshold"] = args.conf
    if args.pool_size:
        pool_cfg["pool_size"] = args.pool_size
    if args.output:
        config.setdefault("output", {})["file_path"] = args.output

    # Print configuration
    print("=" * 60)
    print("ROI-based Pool Detection + Line Zone Counter Demo")
    print("=" * 60)
    print(f"Input: {video_cfg.get('input')}")
    print(f"Model: {detector_cfg.get('model_path')}")

    # Print ROI configuration
    rois = config.get('rois', [])
    if rois:
        roi = rois[0]
        print(f"ROI: x={roi.get('x', 'centered')}, y={roi.get('y', 'centered')}, {roi.get('w')}x{roi.get('h')}")

    print(f"Pool Size: {pool_cfg.get('pool_size', 10)} workers")
    print(f"Device: {detector_cfg.get('device', 'cuda')}")
    print(f"Confidence: {detector_cfg.get('confidence_threshold', 0.4)}")

    logging_cfg = config.get('logging', {})
    print(f"Socket Port: {logging_cfg.get('socket_port', 7777)}")
    print(f"Log File: {logging_cfg.get('log_file', 'count.log')}")
    print("=" * 60)
    print()

    # Create source using new utility function
    source = utils.create_source(video_cfg.get("input"))

    # Get video info with fallbacks from config
    fps, width, height = utils.get_video_info_with_fallbacks(
        source,
        fallback_fps=video_cfg.get("fallback_fps", 30),
        fallback_resolution=tuple(video_cfg.get("fallback_resolution", [1920, 1080]))
    )

    print(f"Video info: {width}x{height} @ {fps} fps\n")

    # Parse line points from config
    lines = config.get("counting_lines", [])
    if lines:
        line = lines[0]
        line_start = sv.Point(x=line["start"][0], y=line["start"][1])
        line_end = sv.Point(x=line["end"][0], y=line["end"][1])
    else:
        line_start = sv.Point(x=640, y=0)
        line_end = sv.Point(x=640, y=720)

    print(f"Line: ({line_start.x}, {line_start.y}) -> ({line_end.x}, {line_end.y})")
    print("=" * 60)
    print()

    # Create pipeline steps

    # 1. ROI extraction step (extracts ROI to 'roi_frame' key)
    rois_config = config.get("rois", [])
    if rois_config:
        roi = rois_config[0]
        roi_step = sv.ROIExtractionStep(
            x=roi.get("x"),
            y=roi.get("y"),
            width=roi.get("w", 640),
            height=roi.get("h", 640),
            input_key="frame",
            output_key="roi_frame",
        )

    # 2. Pool detector (operates on ROI frame, outputs to 'roi_detections')
    detector = sv.PoolYOLODetectionStep(
        model_path=detector_cfg.get("model_path"),
        pool_size=pool_cfg.get("pool_size", 10),
        conf=detector_cfg.get("confidence_threshold", 0.4),
        max_queue_size=pool_cfg.get("queue_size", 120),
        warmup=True,
        input_key="roi_frame" if rois_config else "frame",
        output_key="roi_detections" if rois_config else "detections",
    )

    # 3. Coordinate translation step (translates ROI detections to full frame)
    if rois_config:
        coord_translate = sv.CoordinateTranslationStep(
            input_key="roi_detections",
            output_key="detections",
        )

    # 4. Tracker (requires frames in order!)
    tracker_cfg = config.get("tracker", {})
    tracker = sv.ByteTrackerStep(
        track_activation_threshold=tracker_cfg.get("track_activation_threshold", 0.25),
        lost_track_buffer=tracker_cfg.get("lost_track_buffer", 30),
        minimum_matching_threshold=tracker_cfg.get("minimum_matching_threshold", 0.8),
        detections_key="detections",
    )

    # 5. Line zone step (with translated coordinates)
    line_zone_step = sv.LineZoneStep(
        start=line_start,
        end=line_end,
        detections_key="detections",
    )

    # 6. ROI visualization step
    if rois_config:
        roi_viz = sv.ROIVisualizationStep(color=(255, 255, 0), thickness=2)

    # 7. Line crossing logger
    crossing_logger = LineCrossingLoggerStep(
        socket_port=logging_cfg.get("socket_port", 7777),
        log_file=logging_cfg.get("log_file", "count.log"),
    )

    # Build pipeline
    pipeline = sv.Pipeline(source)

    # Add FPS calculator
    display_cfg = config.get("display", {})
    if display_cfg.get("show_fps", True):
        pipeline = pipeline | sv.FPSCalculatorStep()

    # Add all processing steps
    if rois_config:
        pipeline = pipeline | roi_step | detector | coord_translate
    else:
        pipeline = pipeline | detector

    pipeline = pipeline | tracker | line_zone_step

    if rois_config:
        pipeline = pipeline | roi_viz

    pipeline = (
        pipeline
        | sv.TrackerAnnotatorStep(
            trace_length=30, trace_thickness=2, copy_frame=False, detections_key="detections"
        )
        | sv.LineZoneAnnotatorStep(
            color=sv.Color.WHITE,
            thickness=config.get("line_visualization", {}).get("thickness", 4),
            text_scale=0.8,
            custom_in_text=config.get("line_visualization", {}).get("in_text"),
            custom_out_text=config.get("line_visualization", {}).get("out_text"),
            copy_frame=False,
        )
        | crossing_logger
    )

    # Add metrics overlay if requested
    if display_cfg.get("show_metrics", False):
        metrics_cfg = display_cfg.get("metrics", {})
        metrics_callback = utils.create_metrics_overlay_callback(
            position=metrics_cfg.get("position", "top-left"),
            font_scale=metrics_cfg.get("font_scale", 0.5),
            color=tuple(metrics_cfg.get("color", [0, 255, 0])),
            bg_opacity=metrics_cfg.get("background_opacity", 0.6),
            show_fps=display_cfg.get("show_fps", True),
            show_detections=metrics_cfg.get("show_detections", True),
            show_tracked=metrics_cfg.get("show_tracked", True),
            show_line_counts=metrics_cfg.get("show_line_counts", True),
            show_roi_info=metrics_cfg.get("show_roi_info", True) and rois_config,
            show_inference_time=metrics_cfg.get("show_inference_time", True),
            show_tracking_time=metrics_cfg.get("show_tracking_time", True),
            show_pool_metrics=metrics_cfg.get("show_pool_metrics", False),
        )
        pipeline = pipeline | sv.CallbackStep(metrics_callback)

    # Add sink
    output_cfg = config.get("output", {})
    if output_cfg.get("enabled", False):
        pipeline = pipeline | sv.VideoFileSink(
            output_path=output_cfg.get("file_path", "output.mp4"),
            fps=fps,
            width=width,
            height=height,
        )

    # Add display sink if enabled
    if display_cfg.get("enabled", True):
        window_name = display_cfg.get("window_name") or "ROI Detection + Line Counter"
        pipeline = pipeline | sv.DisplaySink(window_name=window_name)

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

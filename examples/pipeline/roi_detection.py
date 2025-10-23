"""
ROI-based Detection Demo

Demonstrates object detection in a configurable Region of Interest (ROI).
Reads configuration from YAML file with command-line overrides support.

Features:
- Configurable ROI (position and size)
- Detection only in ROI area
- Coordinate translation from ROI to full frame
- ROI visualization with rectangle
- Configuration via YAML file
- Command-line parameter overrides

Usage:
    # Using default config.yaml
    python roi_detection.py

    # Using custom config file
    python roi_detection.py --config my_config.yaml

    # Override config parameters
    python roi_detection.py --model yolov8n.pt --conf 0.5

    # Override multiple parameters
    python roi_detection.py --config config.yaml --model yolo11s.pt --conf 0.3 --device cpu
"""

import argparse
import logging
from pathlib import Path

import cv2
import yaml

import supervision as sv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from: {config_path}")
    return config


def main():
    parser = argparse.ArgumentParser(
        description="ROI-based detection with YAML configuration"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="YOLO model path (overrides config)",
    )

    parser.add_argument(
        "--conf",
        type=float,
        help="Confidence threshold (overrides config)",
    )

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    if args.model:
        config["model"] = args.model
    if args.conf is not None:
        config["conf"] = args.conf

    print("=" * 60)
    print("ROI-based Detection Demo")
    print("=" * 60)
    print(f"Config file: {args.config}")
    print(f"Source: {config.get('source', 'webcam')}")
    print(f"Input: {config.get('input', 'default')}")
    print(f"Model: {config['model']}")
    print(f"Device: {config.get('device', 'cuda')}")
    print(f"Confidence: {config.get('conf', 0.25)}")

    roi_cfg = config.get('roi', {})
    roi_x = roi_cfg.get('x', 'centered')
    roi_y = roi_cfg.get('y', 'centered')
    roi_w = roi_cfg.get('width', 640)
    roi_h = roi_cfg.get('height', 640)
    print(f"ROI: x={roi_x}, y={roi_y}, size={roi_w}x{roi_h}")
    print("=" * 60)
    print()

    source_type = config.get("source", "webcam")
    if source_type == "webcam":
        source = sv.WebcamSource(camera_id=0)
    elif source_type == "file":
        source = sv.VideoFileSource(video_path=config["input"])
    elif source_type == "stream":
        source = sv.StreamSource(stream_url=config["input"])
    else:
        raise ValueError(f"Unknown source type: {source_type}")

    try:
        cap = source.cap if hasattr(source, "cap") else None
        if cap is not None and cap.isOpened():
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if fps <= 0 or fps > 120:
                fps = 30
            if width <= 0 or height <= 0:
                width, height = 1920, 1080
        else:
            fps, width, height = 30, 1920, 1080
    except Exception:
        fps, width, height = 30, 1920, 1080

    print(f"Video info: {width}x{height} @ {fps} fps\n")

    roi_step = sv.ROIExtractionStep(
        x=roi_cfg.get("x"),
        y=roi_cfg.get("y"),
        width=roi_cfg.get("width", 640),
        height=roi_cfg.get("height", 640),
        input_key="frame",
        output_key="roi_frame",
    )

    detector = sv.YOLODetectionStep(
        model_path=config["model"],
        conf=config.get("conf", 0.25),
        verbose=False,
        input_key="roi_frame",
        output_key="roi_detections",
    )

    coord_translate = sv.CoordinateTranslationStep(
        input_key="roi_detections",
        output_key="detections",
    )

    roi_viz = sv.ROIVisualizationStep(color=(255, 255, 0), thickness=2)

    pipeline = sv.Pipeline(source)

    if config.get("show_fps", False):
        pipeline = pipeline | sv.FPSCalculatorStep()

    pipeline = (
        pipeline
        | roi_step
        | detector
        | coord_translate
        | roi_viz
        | sv.BoxAnnotatorStep(detections_key="detections", copy_frame=False)
    )

    if config.get("show_metrics", False):
        def add_metrics_overlay(data):
            frame = data.get("frame")
            if frame is None:
                return data

            detections = data.get("detections", sv.Detections.empty())
            roi_size = data.get("roi_size", (640, 640))
            roi_offset = data.get("roi_offset", (0, 0))

            lines = [
                f"ROI: {roi_size[0]}x{roi_size[1]} at ({roi_offset[0]},{roi_offset[1]})",
                f"Detections: {len(detections)}",
            ]

            y_offset = 25
            for line in lines:
                cv2.putText(
                    frame,
                    line,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                y_offset += 30

            data["frame"] = frame
            return data

        pipeline = pipeline | sv.CallbackStep(add_metrics_overlay)

    output_path = config.get("output")
    if output_path:
        pipeline = pipeline | sv.VideoFileSink(
            output_path=output_path,
            fps=fps,
            width=width,
            height=height,
        )

    pipeline = pipeline | sv.DisplaySink(window_name="ROI Detection")

    print("Starting detection... Press 'q' to quit\n")
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    if output_path:
        print(f"\nVideo saved to: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()

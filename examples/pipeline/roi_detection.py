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

import supervision as sv
import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        config = utils.load_yaml_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    # Override config with command-line arguments
    video_cfg = config.get("video", {})
    detector_cfg = config.get("detector", {})

    if args.model:
        detector_cfg["model_path"] = args.model
    if args.conf is not None:
        detector_cfg["confidence_threshold"] = args.conf

    print("=" * 60)
    print("ROI-based Detection Demo")
    print("=" * 60)
    print(f"Config file: {args.config}")
    print(f"Input: {video_cfg.get('input')}")
    print(f"Model: {detector_cfg.get('model_path')}")
    print(f"Device: {detector_cfg.get('device', 'cuda')}")
    print(f"Confidence: {detector_cfg.get('confidence_threshold', 0.4)}")

    rois = config.get('rois', [])
    if rois:
        roi = rois[0]
        print(f"ROI: x={roi.get('x', 'centered')}, y={roi.get('y', 'centered')}, size={roi.get('w')}x{roi.get('h')}")
    print("=" * 60)
    print()

    # Create source using new utility
    source = utils.create_source(video_cfg.get("input"))

    # Get video info with fallbacks
    fps, width, height = utils.get_video_info_with_fallbacks(
        source,
        fallback_fps=video_cfg.get("fallback_fps", 30),
        fallback_resolution=tuple(video_cfg.get("fallback_resolution", [1920, 1080]))
    )

    print(f"Video info: {width}x{height} @ {fps} fps\n")

    # ROI extraction step
    roi_step = None
    if rois:
        roi = rois[0]
        roi_step = sv.ROIExtractionStep(
            x=roi.get("x"),
            y=roi.get("y"),
            width=roi.get("w", 640),
            height=roi.get("h", 640),
            input_key="frame",
            output_key="roi_frame",
        )

    # Detector (keep reference for class names)
    yolo_step = sv.YOLODetectionStep(
        model_path=detector_cfg.get("model_path"),
        conf=detector_cfg.get("confidence_threshold", 0.4),
        verbose=False,
        input_key="roi_frame" if rois else "frame",
        output_key="roi_detections" if rois else "detections",
    )

    # Coordinate translation (only if using ROI)
    if rois:
        coord_translate = sv.CoordinateTranslationStep(
            input_key="roi_detections",
            output_key="detections",
        )

    # ROI visualization
    if rois:
        roi_viz = sv.ROIVisualizationStep(color=(255, 255, 0), thickness=2)

    # Build pipeline
    pipeline = sv.Pipeline(source)

    # Add FPS calculator
    display_cfg = config.get("display", {})
    if display_cfg.get("show_fps", True):
        pipeline = pipeline | sv.FPSCalculatorStep()

    # Add processing steps
    if rois:
        pipeline = pipeline | roi_step | yolo_step | coord_translate | roi_viz
    else:
        pipeline = pipeline | yolo_step

    # Add annotations (boxes, labels, and confidence)
    pipeline = pipeline | sv.DetectionAnnotatorStep(
        detections_key="detections",
        class_names=yolo_step.model.names,  # Pass class names from YOLO model
        copy_frame=False
    )

    # Add metrics overlay if requested
    if display_cfg.get("show_metrics", False):
        metrics_cfg = display_cfg.get("metrics", {})
        metrics_callback = utils.create_metrics_overlay_callback(
            position=metrics_cfg.get("position", "top-left"),
            font_scale=metrics_cfg.get("font_scale", 0.6),
            color=tuple(metrics_cfg.get("color", [0, 255, 0])),
            bg_opacity=metrics_cfg.get("background_opacity", 0.6),
            show_fps=display_cfg.get("show_fps", True),
            show_detections=metrics_cfg.get("show_detections", True),
            show_tracked=False,
            show_line_counts=False,
            show_roi_info=metrics_cfg.get("show_roi_info", True) and rois,
            show_inference_time=metrics_cfg.get("show_inference_time", True),
            show_tracking_time=False,
            show_pool_metrics=False,
        )
        pipeline = pipeline | sv.CallbackStep(metrics_callback)

    # Add output sink if configured
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
        window_name = display_cfg.get("window_name") or "ROI Detection"
        pipeline = pipeline | sv.DisplaySink(window_name=window_name)

    print("Starting detection... Press 'q' to quit\n")
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    if output_cfg.get("enabled", False):
        print(f"\nVideo saved to: {output_cfg.get('file_path', 'output.mp4')}")

    print("\nDone!")


if __name__ == "__main__":
    main()

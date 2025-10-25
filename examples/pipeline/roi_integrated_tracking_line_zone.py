"""
ROI-based Detection with Integrated Tracking and Line Counting Demo

Demonstrates object detection in a ROI with integrated YOLO tracking and line crossing.
Uses YOLOTrackingStep for efficient single-pass detection+tracking via Ultralytics.

Features:
- Configurable ROI (position and size)
- Detection and tracking in ROI area (single step)
- Coordinate translation from ROI to full frame
- Line crossing detection and counting
- ROI and line visualization
- Configuration via YAML file
- Command-line parameter overrides

Differences from roi_detection_tracking_line_zone.py:
- Uses YOLOTrackingStep instead of YOLODetectionStep + ByteTrackerStep
- More efficient (single model.track() call vs model.predict() + ByteTrack)
- Supports both BoT-SORT and ByteTrack trackers

Usage:
    # Using default config.yaml
    python roi_integrated_tracking_line_zone.py

    # Using custom config file
    python roi_integrated_tracking_line_zone.py --config my_config.yaml

    # Override config parameters
    python roi_integrated_tracking_line_zone.py --model yolov8n.pt --conf 0.5

    # Use ByteTrack instead of BoT-SORT
    python roi_integrated_tracking_line_zone.py --tracker bytetrack.yaml

    # Save output to video file
    python roi_integrated_tracking_line_zone.py --output result.avi
"""

import argparse
import logging
from copy import deepcopy

import utils

import supervision as sv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(
        description="ROI-based detection with integrated tracking and line counting"
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

    parser.add_argument(
        "--tracker",
        type=str,
        choices=["botsort.yaml", "bytetrack.yaml"],
        help="Tracker type: botsort.yaml (default) or bytetrack.yaml",
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Enable display window with input frame, roi and detections",
    )

    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Show detailed metrics overlay on frame",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output video file path (overrides config and enables output)",
    )

    parser.add_argument(
        "--save-metrics",
        type=str,
        help="Save frame metrics to JSON file (e.g., metrics.json)",
    )

    return parser.parse_args()


def main():
    args = get_args()

    try:
        config = utils.load_yaml_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    # Override config with command-line arguments
    video_cfg = config.get("video", {})
    detector_cfg = config.get("detector", {})
    tracker_cfg = config.get("tracker", {})
    display_cfg = config.get("display", {})
    output_cfg = config.get("output", {})

    if args.model:
        detector_cfg["model_path"] = args.model
    if args.conf is not None:
        detector_cfg["confidence_threshold"] = args.conf
    if args.tracker:
        tracker_cfg["tracker_type"] = args.tracker
    if args.display:
        display_cfg["enabled"] = args.display
    if args.show_metrics:
        display_cfg["show_metrics"] = args.show_metrics
    if args.output:
        output_cfg["enabled"] = True
        output_cfg["file_path"] = args.output

    # Check for ROIs in config
    rois = config.get("rois", [])
    if not rois:
        logger.error("Please provide at least one ROI in the configuration file")
        return
    roi = rois[0]

    # Validate ROI has all required parameters
    roi_x = roi.get("x")
    roi_y = roi.get("y")
    roi_w = roi.get("w")
    roi_h = roi.get("h")

    if roi_x is None or roi_y is None or roi_w is None or roi_h is None:
        logger.error("ROI must specify 'x', 'y', 'w', and 'h' parameters")
        return

    # Check for counting lines in config
    counting_lines = config.get("counting_lines", [])
    if not counting_lines:
        logger.error(
            "Please provide at least one counting line in the configuration file"
        )
        return
    line = counting_lines[0]

    # Validate line has all required parameters
    line_start = line.get("start")
    line_end = line.get("end")

    if not line_start or not line_end or len(line_start) != 2 or len(line_end) != 2:
        logger.error(
            "Counting line must specify 'start' and 'end' as [x, y] coordinates"
        )
        return

    # Get tracker type
    tracker_type = tracker_cfg.get("tracker_type", "botsort.yaml")

    print("=" * 60)
    print("ROI-based Detection with Integrated Tracking and Line Counting")
    print("=" * 60)
    print(f"Config file: {args.config}")
    print(f"Input: {video_cfg.get('input')}")
    print(f"Model: {detector_cfg.get('model_path')}")
    print(f"Device: {detector_cfg.get('device', 'cuda')}")
    print(f"Confidence: {detector_cfg.get('confidence_threshold', 0.4)}")
    print(f"Tracker: {tracker_type}")
    print(f"ROI: x={roi_x}, y={roi_y}, size={roi_w}x{roi_h}")
    print(f"Line: {line_start} -> {line_end}")
    print("=" * 60)
    print()

    # Create source_________________________________________
    source = utils.create_source(video_cfg.get("input"))

    # Get video info with fallbacks
    fps, width, height = utils.get_video_info_with_fallbacks(
        source,
        fallback_fps=video_cfg.get("fallback_fps", 30),
        fallback_resolution=tuple(video_cfg.get("fallback_resolution", [1920, 1080])),
    )
    print(f"Video info: {width}x{height} @ {fps} fps\n")

    # Pipeline steps_________________________________________

    # ROI extraction
    roi_step = sv.ROIExtractionStep(
        x=roi_x,
        y=roi_y,
        width=roi_w,
        height=roi_h,
        input_key="frame",
        output_key="roi_frame",
    )

    # Integrated detection + tracking (single step)
    yolo_tracking_step = sv.YOLOTrackingStep(
        model_path=detector_cfg.get("model_path"),
        conf=detector_cfg.get("confidence_threshold", 0.4),
        tracker=tracker_type,
        persist=True,
        verbose=False,
        input_key="roi_frame",
        output_key="roi_detections",
        metrics_key="yolo_tracking_metrics",
    )

    # Coordinate translation
    coord_translate = sv.CoordinateTranslationStep(
        offset_x=roi_x,
        offset_y=roi_y,
        input_key="roi_detections",
        output_key="detections",
    )

    # Copy all detections before line zone (for visualization)
    def copy_all_detections(data):
        if "detections" in data:
            data["all_detections"] = deepcopy(data["detections"])
        return data

    copy_detections_step = sv.CallbackStep(copy_all_detections)

    # Line counting
    line_zone_step = sv.LineZoneStep(
        start=sv.Point(x=line_start[0], y=line_start[1]),
        end=sv.Point(x=line_end[0], y=line_end[1]),
        detections_key="detections",
        line_zone_key="line_zone",
    )

    # Build pipeline
    pipeline = (
        sv.Pipeline(source)
        | roi_step
        | yolo_tracking_step  # Single step for detection + tracking
        | coord_translate
        | copy_detections_step
        | line_zone_step
    )

    # Add FPS calculator
    pipeline = pipeline | sv.FPSCalculatorStep()

    # Add annotations if display or output is enabled
    if display_cfg.get("enabled", False) or output_cfg.get("enabled", False):
        # ROI visualization
        pipeline = pipeline | sv.ROIVisualizationStep(
            x=roi_x,
            y=roi_y,
            width=roi_w,
            height=roi_h,
            color=(255, 255, 0),
            thickness=2,
            copy_frame=False,
        )

        # Draw all detections in white (boxes and label backgrounds)
        pipeline = pipeline | sv.DetectionAnnotatorStep(
            detections_key="all_detections",
            class_names=yolo_tracking_step.model.names,
            box_color=sv.Color.WHITE,
            box_thickness=1,
            label_text_color=sv.Color.BLACK,
            label_color=sv.Color.WHITE,
            show_class=True,
            show_confidence=True,
            copy_frame=False,
        )

        # Tracking annotations (boxes, labels with tracker IDs, and traces)
        pipeline = pipeline | sv.TrackerAnnotatorStep(
            detections_key="detections",
            class_names=yolo_tracking_step.model.names,
            show_tracker_id=True,
            show_class=True,
            show_confidence=True,
            trace_length=30,
            trace_thickness=2,
            box_thickness=2,
            copy_frame=False,
            color_lookup=sv.ColorLookup.TRACK,
        )

        # Line zone visualization
        line_vis_cfg = config.get("line_visualization", {})
        pipeline = pipeline | sv.LineZoneAnnotatorStep(
            color=utils.parse_color(line_vis_cfg.get("color", "white")),
            thickness=line_vis_cfg.get("thickness", 4),
            custom_in_text=line_vis_cfg.get("in_text"),
            custom_out_text=line_vis_cfg.get("out_text"),
            copy_frame=False,
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
                show_tracked=metrics_cfg.get("show_tracked", True),
                show_line_counts=metrics_cfg.get("show_line_counts", True),
                show_roi_info=metrics_cfg.get("show_roi_info", True) and rois,
                show_inference_time=metrics_cfg.get("show_inference_time", True),
                show_tracking_time=False,  # Integrated in yolo_tracking_metrics
                show_pool_metrics=False,
            )
            pipeline = pipeline | sv.CallbackStep(metrics_callback)

    # Add sink(s) to pipeline
    sinks = []
    if display_cfg.get("enabled", False):
        default_name = "ROI Integrated Tracking + Line Counting"
        window_name = display_cfg.get("window_name") or default_name
        sinks.append(sv.DisplaySink(window_name=window_name))

    if output_cfg.get("enabled", False):
        sinks.append(
            sv.VideoFileSink(
                output_path=output_cfg.get("file_path", "output.mp4"),
                fps=fps,
                width=width,
                height=height,
                codec="XVID",
            )
        )

    if len(sinks) == 1:
        pipeline = pipeline | sinks[0]
    elif len(sinks) > 1:
        pipeline = pipeline | sv.MultiSink(sinks)

    # Run pipeline___________________________________________
    print("Starting integrated tracking... Press 'q' to quit\n")

    stats_n_frames = 30
    metrics_data = []

    try:
        frame_count = 0
        fps_average = 0
        for data in pipeline:
            frame_count += 1
            fps_average += data.get("fps", 0.0)

            # Save metrics if requested
            if args.save_metrics:
                line_zone = data.get("line_zone")
                all_detections = data.get("all_detections")
                detections = data.get("detections")
                if detections and detections.tracker_id is not None:
                    valid_ids = detections.tracker_id[detections.tracker_id >= 0]
                    tracked_count = len(set(valid_ids))
                else:
                    tracked_count = 0
                frame_metrics = {
                    "frame": frame_count,
                    "fps": data.get("fps", 0.0),
                    "detections": len(all_detections) if all_detections else 0,
                    "tracked_objects": tracked_count,
                    "line_in_count": line_zone.in_count if line_zone else 0,
                    "line_out_count": line_zone.out_count if line_zone else 0,
                }

                # Add integrated tracking metrics if available
                yolo_tracking_metrics = data.get("yolo_tracking_metrics", {})
                if yolo_tracking_metrics:
                    frame_metrics["inference_time_ms"] = sum(
                        yolo_tracking_metrics.values()
                    )

                metrics_data.append(frame_metrics)

            # Print stats every stats_n_frames frames
            if frame_count % stats_n_frames == 0:
                line_zone = data.get("line_zone")
                in_count = line_zone.in_count if line_zone else 0
                out_count = line_zone.out_count if line_zone else 0
                avg_fps = fps_average / stats_n_frames
                print(
                    f"Frame {frame_count:5d} | FPS: {avg_fps:5.1f} | "
                    f"In: {in_count:3d} | Out: {out_count:3d}"
                )
                fps_average = 0
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    if output_cfg.get("enabled", False):
        print(f"\nVideo saved to: {output_cfg.get('file_path', 'output.mp4')}")

    # Save metrics to JSON file
    if args.save_metrics and metrics_data:
        utils.save_metrics_to_json(metrics_data, args.save_metrics)
        print(f"Metrics saved to: {args.save_metrics}")

        # Generate plot
        plot_path = utils.plot_metrics_from_json(args.save_metrics)
        print(f"Metrics plot saved to: {plot_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()

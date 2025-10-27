"""
Multi-Tracker Comparison Demo

Demonstrates and compares different tracking algorithms in supervision:
- ByteTrack: Advanced tracker with Kalman filtering and score fusion
- SORT: Simple Online and Realtime Tracking with Kalman filtering
- Centroid: Fast centroid-based tracking using Euclidean distance

Features:
- Configurable tracker selection via command-line
- Side-by-side comparison mode
- ROI-based detection
- Line crossing detection and counting
- Performance metrics for each tracker
- Configuration via YAML file

Usage:
    # Use ByteTrack (default)
    python tracker_comparison_demo.py

    # Use SORT tracker
    python tracker_comparison_demo.py --tracker sort

    # Use Centroid tracker
    python tracker_comparison_demo.py --tracker centroid

    # Compare all trackers side-by-side
    python tracker_comparison_demo.py --compare

    # Custom config and parameters
    python tracker_comparison_demo.py --tracker sort --config config.yaml --conf 0.5
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
        description="Multi-tracker comparison demo"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )

    parser.add_argument(
        "--tracker",
        type=str,
        choices=["bytetrack", "sort", "centroid"],
        default="bytetrack",
        help="Tracker algorithm to use (default: bytetrack)",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all three trackers side-by-side (requires display)",
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
        "--display",
        action="store_true",
        help="Enable display window"
    )

    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Show detailed metrics overlay on frame"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output video file path"
    )

    return parser.parse_args()


def create_tracker_step(tracker_type, tracker_cfg):
    """Create tracker step based on type using new config structure."""
    if tracker_type == "bytetrack":
        params = tracker_cfg.get("bytetrack", {})
        return sv.ByteTrackerStep(
            track_activation_threshold=params.get("track_activation_threshold", 0.25),
            lost_track_buffer=params.get("lost_track_buffer", 30),
            minimum_matching_threshold=params.get("minimum_matching_threshold", 0.8),
            minimum_consecutive_frames=params.get("minimum_consecutive_frames", 1),
            detections_key="detections",
        )
    elif tracker_type == "sort":
        params = tracker_cfg.get("sort", {})
        return sv.SORTTrackerStep(
            lost_track_buffer=params.get("lost_track_buffer", 30),
            minimum_consecutive_frames=params.get("minimum_consecutive_frames", 3),
            iou_threshold=params.get("iou_threshold", 0.3),
            detections_key="detections",
        )
    elif tracker_type == "centroid":
        params = tracker_cfg.get("centroid", {})
        return sv.CentroidTrackerStep(
            lost_track_buffer=params.get("lost_track_buffer", 30),
            max_distance=params.get("max_distance", 50.0),
            detections_key="detections",
        )
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")


def create_pipeline(source, config, tracker_type, yolo_step=None):
    """Create a complete pipeline with the specified tracker."""
    detector_cfg = config.get("detector", {})
    tracker_cfg = config.get("tracker", {})

    # Get ROI config
    rois = config.get("rois", [])
    if not rois:
        raise ValueError("No ROI configured")
    roi = rois[0]
    roi_x, roi_y, roi_w, roi_h = roi["x"], roi["y"], roi["w"], roi["h"]

    # Get line config
    counting_lines = config.get("counting_lines", [])
    if not counting_lines:
        raise ValueError("No counting line configured")
    line = counting_lines[0]
    line_start, line_end = line["start"], line["end"]

    # ROI extraction
    roi_step = sv.ROIExtractionStep(
        x=roi_x, y=roi_y, width=roi_w, height=roi_h,
        input_key="frame", output_key="roi_frame",
    )

    # Detector (create new or reuse)
    if yolo_step is None:
        yolo_step = sv.YOLODetectionStep(
            model_path=detector_cfg.get("model_path"),
            conf=detector_cfg.get("confidence_threshold", 0.4),
            verbose=False,
            input_key="roi_frame",
            output_key="roi_detections",
        )

    # Coordinate translation
    coord_translate = sv.CoordinateTranslationStep(
        offset_x=roi_x, offset_y=roi_y,
        input_key="roi_detections", output_key="detections",
    )

    # Copy detections before tracking
    def copy_all_detections(data):
        if "detections" in data:
            data["all_detections"] = deepcopy(data["detections"])
        return data

    copy_step = sv.CallbackStep(copy_all_detections)

    # Create tracker
    tracker_step = create_tracker_step(tracker_type, tracker_cfg)

    # Line counting
    line_zone_step = sv.LineZoneStep(
        start=sv.Point(x=line_start[0], y=line_start[1]),
        end=sv.Point(x=line_end[0], y=line_end[1]),
        detections_key="detections",
    )

    # Build pipeline
    pipeline = (
        sv.Pipeline(source)
        | roi_step
        | yolo_step
        | coord_translate
        | copy_step
        | tracker_step
        | line_zone_step
        | sv.FPSCalculatorStep()
    )

    return pipeline, yolo_step


def add_annotations(pipeline, config, yolo_step, tracker_name=None):
    """Add visualization annotations to pipeline."""
    rois = config.get("rois", [])
    roi = rois[0]
    roi_x, roi_y, roi_w, roi_h = roi["x"], roi["y"], roi["w"], roi["h"]

    # ROI visualization
    pipeline = pipeline | sv.ROIVisualizationStep(
        x=roi_x, y=roi_y, width=roi_w, height=roi_h,
        color=(255, 255, 0), thickness=2, copy_frame=False
    )

    # All detections (white boxes)
    pipeline = pipeline | sv.DetectionAnnotatorStep(
        detections_key="all_detections",
        class_names=yolo_step.model.names,
        box_color=sv.Color.WHITE,
        box_thickness=1,
        label_color=sv.Color.WHITE,
        show_class=True,
        show_confidence=True,
        copy_frame=False
    )

    # Tracked detections with traces
    pipeline = pipeline | sv.TrackerAnnotatorStep(
        detections_key="detections",
        class_names=yolo_step.model.names,
        show_tracker_id=True,
        show_class=True,
        show_confidence=True,
        trace_length=30,
        trace_thickness=2,
        box_thickness=2,
        copy_frame=False
    )

    # Line zone
    line_vis_cfg = config.get("line_visualization", {})
    pipeline = pipeline | sv.LineZoneAnnotatorStep(
        color=utils.parse_color(line_vis_cfg.get("color", "white")),
        thickness=line_vis_cfg.get("thickness", 4),
        custom_in_text=line_vis_cfg.get("in_text"),
        custom_out_text=line_vis_cfg.get("out_text"),
        copy_frame=False
    )

    # Add tracker name label
    if tracker_name:
        def add_tracker_label(data):
            import cv2
            frame = data.get("frame")
            if frame is not None:
                cv2.putText(
                    frame,
                    f"Tracker: {tracker_name}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                )
            return data
        pipeline = pipeline | sv.CallbackStep(add_tracker_label)

    return pipeline


def main():
    args = get_args()

    try:
        config = utils.load_yaml_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    # Override config
    video_cfg = config.get("video", {})
    detector_cfg = config.get("detector", {})
    display_cfg = config.get("display", {})
    output_cfg = config.get("output", {})

    if args.model:
        detector_cfg["model_path"] = args.model
    if args.conf is not None:
        detector_cfg["confidence_threshold"] = args.conf
    if args.display:
        display_cfg["enabled"] = True
    if args.show_metrics:
        display_cfg["show_metrics"] = True
    if args.output:
        output_cfg["enabled"] = True
        output_cfg["file_path"] = args.output

    print("=" * 60)
    print("Multi-Tracker Comparison Demo")
    print("=" * 60)
    print(f"Config file: {args.config}")
    print(f"Input: {video_cfg.get('input')}")
    print(f"Model: {detector_cfg.get('model_path')}")
    print(f"Confidence: {detector_cfg.get('confidence_threshold', 0.4)}")

    if args.compare:
        print("Mode: Comparison (ByteTrack vs SORT vs Centroid)")
    else:
        print(f"Tracker: {args.tracker.upper()}")

    print("=" * 60)
    print()

    # Tracker descriptions
    tracker_info = {
        "bytetrack": "Advanced tracker with Kalman + score fusion (best for crowded scenes)",
        "sort": "Simple Online Realtime Tracking with Kalman (fast and reliable)",
        "centroid": "Centroid-based tracking with Euclidean distance (fastest, simple scenes)",
    }

    if not args.compare:
        print(f"ℹ️  {tracker_info[args.tracker]}\n")

    # Create source
    source = utils.create_source(video_cfg.get("input"))

    # Get video info
    fps, width, height = utils.get_video_info_with_fallbacks(
        source,
        fallback_fps=video_cfg.get("fallback_fps", 30),
        fallback_resolution=tuple(video_cfg.get("fallback_resolution", [1920, 1080]))
    )
    print(f"Video info: {width}x{height} @ {fps} fps\n")

    # Create pipeline
    pipeline, yolo_step = create_pipeline(source, config, args.tracker)

    # Add annotations
    if display_cfg.get("enabled", False) or output_cfg.get("enabled", False):
        pipeline = add_annotations(
            pipeline,
            config,
            yolo_step,
            tracker_name=args.tracker.upper()
        )

        # Add metrics overlay
        if display_cfg.get("show_metrics", False):
            metrics_cfg = display_cfg.get("metrics", {})
            metrics_callback = utils.create_metrics_overlay_callback(
                position=metrics_cfg.get("position", "top-left"),
                font_scale=metrics_cfg.get("font_scale", 0.6),
                color=tuple(metrics_cfg.get("color", [0, 255, 0])),
                bg_opacity=metrics_cfg.get("background_opacity", 0.6),
                show_fps=True,
                show_detections=True,
                show_tracked=True,
                show_line_counts=True,
                show_roi_info=True,
                show_inference_time=True,
                show_tracking_time=True,
                show_pool_metrics=False,
            )
            pipeline = pipeline | sv.CallbackStep(metrics_callback)

    # Add sinks
    sinks = []
    if display_cfg.get("enabled", False):
        window_name = f"{args.tracker.upper()} Tracker Demo"
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

    # Run pipeline
    print("Starting tracking... Press 'q' to quit\n")
    print("Tracker Performance Comparison:")
    print("-" * 60)

    stats_n_frames = 30
    try:
        frame_count = 0
        fps_sum = 0
        tracking_time_sum = 0

        for data in pipeline:
            frame_count += 1
            fps_sum += data.get("fps", 0.0)

            tracker_metrics = data.get("tracker_metrics", {})
            if "processing_time_ms" in tracker_metrics:
                tracking_time_sum += tracker_metrics["processing_time_ms"]

            if frame_count % stats_n_frames == 0:
                line_zone = data.get("line_zone")
                in_count = line_zone.in_count if line_zone else 0
                out_count = line_zone.out_count if line_zone else 0
                avg_fps = fps_sum / stats_n_frames
                avg_tracking_time = tracking_time_sum / stats_n_frames

                detections = data.get("detections")
                if detections and detections.tracker_id is not None:
                    valid_ids = detections.tracker_id[detections.tracker_id >= 0]
                    tracked_count = len(set(valid_ids))
                else:
                    tracked_count = 0

                print(
                    f"Frame {frame_count:5d} | FPS: {avg_fps:5.1f} | "
                    f"Tracking: {avg_tracking_time:4.1f}ms | "
                    f"Tracked: {tracked_count:3d} | "
                    f"In: {in_count:3d} | Out: {out_count:3d}"
                )
                fps_sum = 0
                tracking_time_sum = 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    if output_cfg.get("enabled", False):
        print(f"\nVideo saved to: {output_cfg.get('file_path')}")

    print("\n" + "=" * 60)
    print("Tracker Comparison Summary:")
    print("-" * 60)
    print(f"{'Tracker':<12} | {'Speed':<12} | {'Accuracy':<12} | {'Use Case'}")
    print("-" * 60)
    print(f"{'ByteTrack':<12} | {'Medium':<12} | {'High':<12} | Crowded scenes, occlusions")
    print(f"{'SORT':<12} | {'Fast':<12} | {'Medium':<12} | General purpose, reliable")
    print(f"{'Centroid':<12} | {'Very Fast':<12} | {'Low-Medium':<12} | Simple scenes, speed critical")
    print("=" * 60)
    print("\nDone!")


if __name__ == "__main__":
    main()

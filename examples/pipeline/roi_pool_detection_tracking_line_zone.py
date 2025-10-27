"""
ROI-based Pool Detection with Tracking and Line Counting Demo

Demonstrates parallel object detection in a ROI using multiple detector workers,
combined with object tracking and line crossing counting. Uses PoolYOLODetectionStep
for high-throughput parallel processing.

Features:
- Configurable ROI (position and size)
- Parallel pool detection with multiple workers
- Detection only in ROI area
- Coordinate translation from ROI to full frame
- Object tracking across frames
- Line crossing detection and counting
- Pool performance metrics (workers active, queue size, frames dropped)
- ROI and line visualization
- Configuration via YAML file
- Command-line parameter overrides
- Metrics export to JSON with automatic plotting

Usage:
    # Using default config.yaml
    python roi_pool_detection_tracking_line_zone.py

    # Using custom config file
    python roi_pool_detection_tracking_line_zone.py --config my_config.yaml

    # Override config parameters
    python roi_pool_detection_tracking_line_zone.py --model yolov8n.pt --conf 0.5

    # Configure pool size and tracker
    python roi_pool_detection_tracking_line_zone.py --pool-size 4 --queue-size 20

    # Change tracker type
    python roi_pool_detection_tracking_line_zone.py --tracker sort

    # Save output to video file
    python roi_pool_detection_tracking_line_zone.py --output result.avi

    # Save metrics to JSON and generate plots
    python roi_pool_detection_tracking_line_zone.py --save-metrics metrics.json
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
        description="ROI-based detection with tracking and line counting"
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

    parser.add_argument(
        "--pool-size",
        type=int,
        help="Number of parallel detector workers (overrides config, default: 4)",
    )

    parser.add_argument(
        "--queue-size",
        type=int,
        help="Maximum frames to queue for processing (overrides config, default: 10)",
    )

    parser.add_argument(
        "--tracker",
        type=str,
        choices=["bytetrack", "sort", "centroid"],
        help="Tracker type (overrides config)",
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
    pool_cfg = config.get("pool", {})
    display_cfg = config.get("display", {})
    output_cfg = config.get("output", {})

    if args.model:
        detector_cfg["model_path"] = args.model
    if args.conf is not None:
        detector_cfg["confidence_threshold"] = args.conf
    if args.pool_size is not None:
        pool_cfg["pool_size"] = args.pool_size
    if args.queue_size is not None:
        pool_cfg["queue_size"] = args.queue_size
    if args.tracker is not None:
        tracker_cfg["type"] = args.tracker
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

    print("=" * 60)
    print("ROI-based Pool Detection with Tracking and Line Counting Demo")
    print("=" * 60)
    print(f"Config file: {args.config}")
    print(f"Input: {video_cfg.get('input')}")
    print(f"Model: {detector_cfg.get('model_path')}")
    print(f"Device: {detector_cfg.get('device', 'cuda')}")
    print(f"Confidence: {detector_cfg.get('confidence_threshold', 0.4)}")
    print(f"Tracker: {tracker_cfg.get('type', 'bytetrack')}")
    print(f"Pool Size: {pool_cfg.get('pool_size', 4)} workers")
    print(f"Queue Size: {pool_cfg.get('queue_size', 10)} frames")
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

    # Pool detector (keep reference for class names)
    pool_detector = sv.PoolYOLODetectionStep(
        model_path=detector_cfg.get("model_path"),
        pool_size=pool_cfg.get("pool_size", 4),
        conf=detector_cfg.get("confidence_threshold", 0.4),
        max_queue_size=pool_cfg.get("queue_size", 10),
        warmup=True,
        input_key="roi_frame",
        output_key="roi_detections",
        metrics_key="pool_metrics",
    )

    # Coordinate translation
    coord_translate = sv.CoordinateTranslationStep(
        offset_x=roi_x,
        offset_y=roi_y,
        input_key="roi_detections",
        output_key="detections",
    )

    # Copy all detections before tracking (tracker filters them)
    def copy_all_detections(data):
        if "detections" in data:
            data["all_detections"] = deepcopy(data["detections"])
        return data

    copy_detections_step = sv.CallbackStep(copy_all_detections)

    # Object tracking - automatically selects tracker based on config
    tracker_step = utils.create_tracker_step_from_config(
        config, detections_key="detections"
    )

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
        | pool_detector
        | coord_translate
        | copy_detections_step
        | tracker_step
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

        # Draw all detections in white (before tracking filtered them)
        pipeline = pipeline | sv.DetectionAnnotatorStep(
            detections_key="all_detections",
            class_names=pool_detector._models[0].names
            if pool_detector._models
            else None,
            box_color=sv.Color.GREY,
            box_thickness=1,
            label_color=sv.Color.GREY,
            show_class=True,
            show_confidence=True,
            copy_frame=False,
        )

        # Tracking annotations (boxes, labels with tracker IDs, and traces)
        pipeline = pipeline | sv.TrackerAnnotatorStep(
            detections_key="detections",
            class_names=pool_detector._models[0].names
            if pool_detector._models
            else None,
            show_tracker_id=True,
            show_class=True,
            show_confidence=True,
            trace_length=30,
            trace_thickness=2,
            box_thickness=2,
            copy_frame=False,
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
                show_tracking_time=metrics_cfg.get("show_tracking_time", True),
                show_pool_metrics=True,
            )
            pipeline = pipeline | sv.CallbackStep(metrics_callback)

    # Add sink(s) to pipeline
    sinks = []
    if display_cfg.get("enabled", False):
        default_name = "ROI Detection + Tracking + Line Counting"
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
    print("Starting detection... Press 'q' to quit\n")

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

                # Add inference time if available (from pool_metrics)
                pool_metrics = data.get("pool_metrics", {})
                if pool_metrics:
                    frame_metrics["inference_time_ms"] = pool_metrics.get(
                        "avg_inference_time_ms", 0
                    )

                # Add tracking time if available
                tracker_metrics = data.get("tracker_metrics", {})
                if tracker_metrics and "processing_time_ms" in tracker_metrics:
                    frame_metrics["tracking_time_ms"] = tracker_metrics[
                        "processing_time_ms"
                    ]

                # Add pool metrics
                if pool_metrics:
                    frame_metrics["workers_active"] = pool_metrics.get(
                        "workers_active", 0
                    )
                    frame_metrics["frames_dropped_total"] = pool_metrics.get(
                        "frames_dropped", 0
                    )
                    frame_metrics["frames_reordered_total"] = pool_metrics.get(
                        "frames_reordered", 0
                    )
                    frame_metrics["avg_queue_time_ms"] = pool_metrics.get(
                        "avg_queue_time_ms", 0
                    )
                    frame_metrics["avg_reorder_delay_ms"] = pool_metrics.get(
                        "avg_reorder_delay_ms", 0
                    )
                    frame_metrics["queue_size_current"] = pool_metrics.get(
                        "queue_current", 0
                    )
                    frame_metrics["queue_size_max"] = pool_metrics.get("queue_max", 0)

                # Add objects that crossed the line
                crossed_objects = []
                if line_zone and detections:
                    # Objects that crossed IN
                    for tracker_id in line_zone.last_crossed_in_ids:
                        idx = None
                        if detections.tracker_id is not None:
                            matches = detections.tracker_id == tracker_id
                            if matches.any():
                                idx = matches.argmax()

                        if idx is not None:
                            xyxy = detections.xyxy[idx]
                            x, y, x2, y2 = xyxy
                            w, h = x2 - x, y2 - y
                            crossed_objects.append(
                                {
                                    "tracker_id": int(tracker_id),
                                    "direction": "in",
                                    "x": float(x),
                                    "y": float(y),
                                    "w": float(w),
                                    "h": float(h),
                                }
                            )

                    # Objects that crossed OUT
                    for tracker_id in line_zone.last_crossed_out_ids:
                        idx = None
                        if detections.tracker_id is not None:
                            matches = detections.tracker_id == tracker_id
                            if matches.any():
                                idx = matches.argmax()

                        if idx is not None:
                            xyxy = detections.xyxy[idx]
                            x, y, x2, y2 = xyxy
                            w, h = x2 - x, y2 - y
                            crossed_objects.append(
                                {
                                    "tracker_id": int(tracker_id),
                                    "direction": "out",
                                    "x": float(x),
                                    "y": float(y),
                                    "w": float(w),
                                    "h": float(h),
                                }
                            )

                frame_metrics["crossed_objects"] = crossed_objects
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

    # Print final pool metrics summary
    if pool_detector:
        final_metrics = pool_detector.get_metrics()
        print("\n" + "=" * 60)
        print("Pool Detector Performance Summary")
        print("=" * 60)
        print(f"Total Frames Processed:   {final_metrics['frames_processed']}")
        print(f"Frames Dropped:           {final_metrics['frames_dropped']}")
        print(f"Frames Reordered:         {final_metrics['frames_reordered']}")
        print(f"Queue Full Count:         {final_metrics['queue_full_count']}")
        print(
            f"Avg Inference Time:       {final_metrics['avg_inference_time_ms']:.2f} ms"
        )
        print(f"Avg Queue Time:           {final_metrics['avg_queue_time_ms']:.2f} ms")
        print(
            f"Avg Reorder Delay:        {final_metrics['avg_reorder_delay_ms']:.2f} ms"
        )
        print(f"Workers Active (final):   {final_metrics['workers_active']}")
        print("=" * 60)

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

"""
Centroid Tracker Demo

Simple demonstration of centroid-based object tracking.
Tracks objects by computing Euclidean distances between detection centroids.
Very fast but less robust than Kalman-based trackers.

Features:
- Fastest tracking option (no Kalman filtering)
- Simple Euclidean distance matching
- Good for non-crowded scenes with predictable motion
- Minimal computational overhead

Usage:
    # Basic usage with default config
    python centroid_tracker_demo.py

    # Custom config
    python centroid_tracker_demo.py --config my_config.yaml

    # Override parameters
    python centroid_tracker_demo.py --max-disappeared 50 --max-distance 100

    # Enable display and save output
    python centroid_tracker_demo.py --display --output centroid_tracking.avi
"""

import argparse
import logging

import utils

import supervision as sv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Centroid tracker demo")

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--max-disappeared",
        type=int,
        help="Maximum frames track can disappear (default: 30)",
    )

    parser.add_argument(
        "--max-distance",
        type=float,
        help="Maximum centroid distance for matching in pixels (default: 50.0)",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="YOLO model path",
    )

    parser.add_argument(
        "--conf",
        type=float,
        help="Detection confidence threshold",
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Enable display window",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output video file path",
    )

    return parser.parse_args()


def main():
    args = get_args()

    try:
        config = utils.load_yaml_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    # Config sections
    video_cfg = config.get("video", {})
    detector_cfg = config.get("detector", {})
    display_cfg = config.get("display", {})
    output_cfg = config.get("output", {})

    # Override with command-line args
    if args.model:
        detector_cfg["model_path"] = args.model
    if args.conf is not None:
        detector_cfg["confidence_threshold"] = args.conf
    if args.display:
        display_cfg["enabled"] = True
    if args.output:
        output_cfg["enabled"] = True
        output_cfg["file_path"] = args.output

    # Centroid tracker parameters
    max_disappeared = args.max_disappeared if args.max_disappeared is not None else 30
    max_distance = args.max_distance if args.max_distance is not None else 50.0

    print("=" * 60)
    print("Centroid Tracker Demo")
    print("=" * 60)
    print(f"Input: {video_cfg.get('input')}")
    print(f"Model: {detector_cfg.get('model_path')}")
    print(f"Confidence: {detector_cfg.get('confidence_threshold', 0.4)}")
    print(f"Centroid Tracker Parameters:")
    print(f"  - max_disappeared: {max_disappeared}")
    print(f"  - max_distance: {max_distance} pixels")
    print("=" * 60)
    print()

    # Create source
    source = utils.create_source(video_cfg.get("input"))

    # Get video info
    fps, width, height = utils.get_video_info_with_fallbacks(
        source,
        fallback_fps=video_cfg.get("fallback_fps", 30),
        fallback_resolution=tuple(video_cfg.get("fallback_resolution", [1920, 1080]))
    )

    # Create detection step
    yolo_step = sv.YOLODetectionStep(
        model_path=detector_cfg.get("model_path"),
        conf=detector_cfg.get("confidence_threshold", 0.4),
        verbose=False,
    )

    # Create Centroid tracker step
    tracker_step = sv.CentroidTrackerStep(
        max_disappeared=max_disappeared,
        max_distance=max_distance,
    )

    # Build pipeline
    pipeline = sv.Pipeline(source) | yolo_step | tracker_step | sv.FPSCalculatorStep()

    # Add annotations if needed
    if display_cfg.get("enabled", False) or output_cfg.get("enabled", False):
        pipeline = pipeline | sv.TrackerAnnotatorStep(
            class_names=yolo_step.model.names,
            show_tracker_id=True,
            show_class=True,
            show_confidence=True,
            trace_length=30,
            trace_thickness=2,
            box_thickness=2,
        )

    # Add sinks
    sinks = []
    if display_cfg.get("enabled", False):
        sinks.append(sv.DisplaySink(window_name="Centroid Tracker"))

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
    print("Starting centroid tracking... Press 'q' to quit\n")
    print("NOTE: Centroid tracker is best for simple scenes with low crowd density")
    print("      and predictable object motion. Use SORT or ByteTrack for complex scenes.\n")

    try:
        frame_count = 0
        for data in pipeline:
            frame_count += 1

            if frame_count % 30 == 0:
                detections = data.get("detections")
                if detections and detections.tracker_id is not None:
                    valid_ids = detections.tracker_id[detections.tracker_id >= 0]
                    unique_ids = len(set(valid_ids))
                else:
                    unique_ids = 0

                tracker_metrics = data.get("tracker_metrics", {})
                tracking_time = tracker_metrics.get("processing_time_ms", 0)

                print(
                    f"Frame {frame_count:5d} | "
                    f"FPS: {data.get('fps', 0):5.1f} | "
                    f"Tracking: {tracking_time:4.1f}ms | "
                    f"Tracked: {unique_ids:3d}"
                )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    if output_cfg.get("enabled", False):
        print(f"\nVideo saved to: {output_cfg.get('file_path')}")

    print("\nDone!")


if __name__ == "__main__":
    main()

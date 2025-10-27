"""
SORT Tracker Demo

Simple demonstration of SORT (Simple Online and Realtime Tracking) tracker.
SORT uses Kalman filtering for state prediction and Hungarian algorithm for
data association based on IoU distance.

Features:
- Fast and reliable tracking for general-purpose applications
- Kalman filter for motion prediction
- Configurable via YAML file
- Works well with moderate crowd density

Usage:
    # Basic usage with default config
    python sort_tracker_demo.py

    # Custom config
    python sort_tracker_demo.py --config my_config.yaml

    # Override parameters
    python sort_tracker_demo.py --max-age 50 --min-hits 5 --iou 0.4

    # Enable display and save output
    python sort_tracker_demo.py --display --output sort_tracking.avi
"""

import argparse
import logging

import utils

import supervision as sv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="SORT tracker demo")

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--lost-track-buffer",
        type=int,
        help="Frames to buffer when track is lost (default: 30)",
    )

    parser.add_argument(
        "--minimum-consecutive-frames",
        type=int,
        help="Consecutive frames to confirm track (default: 3)",
    )

    parser.add_argument(
        "--iou",
        type=float,
        help="Minimum IoU for matching (default: 0.3)",
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

    # SORT tracker parameters
    lost_track_buffer = getattr(args, 'lost_track_buffer', None)
    if lost_track_buffer is None:
        lost_track_buffer = 30
    minimum_consecutive_frames = getattr(args, 'minimum_consecutive_frames', None)
    if minimum_consecutive_frames is None:
        minimum_consecutive_frames = 3
    iou_threshold = args.iou if args.iou is not None else 0.3

    print("=" * 60)
    print("SORT Tracker Demo")
    print("=" * 60)
    print(f"Input: {video_cfg.get('input')}")
    print(f"Model: {detector_cfg.get('model_path')}")
    print(f"Confidence: {detector_cfg.get('confidence_threshold', 0.4)}")
    print(f"SORT Parameters:")
    print(f"  - lost_track_buffer: {lost_track_buffer}")
    print(f"  - minimum_consecutive_frames: {minimum_consecutive_frames}")
    print(f"  - iou_threshold: {iou_threshold}")
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

    # Copy all detections before tracking (tracker filters them)
    def copy_all_detections(data):
        if "detections" in data:
            from copy import deepcopy
            data["all_detections"] = deepcopy(data["detections"])
        return data

    copy_step = sv.CallbackStep(copy_all_detections)

    # Create SORT tracker step
    tracker_step = sv.SORTTrackerStep(
        lost_track_buffer=lost_track_buffer,
        minimum_consecutive_frames=minimum_consecutive_frames,
        iou_threshold=iou_threshold,
    )

    # Build pipeline
    pipeline = (
        sv.Pipeline(source)
        | yolo_step
        | copy_step
        | tracker_step
        | sv.FPSCalculatorStep()
    )

    # Add annotations if needed
    if display_cfg.get("enabled", False) or output_cfg.get("enabled", False):
        # Draw all detections in white (before tracking filtered them)
        pipeline = pipeline | sv.DetectionAnnotatorStep(
            detections_key="all_detections",
            class_names=yolo_step.model.names,
            box_color=sv.Color.GREY,
            box_thickness=1,
            label_color=sv.Color.GREY,
            show_class=True,
            show_confidence=True,
            copy_frame=False,
        )
        
        # Draw tracked detections with colors and traces
        pipeline = pipeline | sv.TrackerAnnotatorStep(
            detections_key="detections",
            class_names=yolo_step.model.names,
            show_tracker_id=True,
            show_class=True,
            show_confidence=True,
            trace_length=30,
            trace_thickness=2,
            box_thickness=2,
            copy_frame=False,
            color_lookup=sv.ColorLookup.TRACK
        )
        

    # Add sinks
    sinks = []
    if display_cfg.get("enabled", False):
        sinks.append(sv.DisplaySink(window_name="SORT Tracker"))

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
    print("Starting SORT tracking... Press 'q' to quit\n")

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

                print(
                    f"Frame {frame_count:5d} | "
                    f"FPS: {data.get('fps', 0):5.1f} | "
                    f"Tracked: {unique_ids:3d}"
                )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    if output_cfg.get("enabled", False):
        print(f"\nVideo saved to: {output_cfg.get('file_path')}")

    print("\nDone!")


if __name__ == "__main__":
    main()

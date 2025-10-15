"""
Debug Logger Line Crossing Demo

This script demonstrates advanced DebugLoggerStep usage with object tracking
and line crossing detection. This example captures comprehensive metrics
including tracking IDs, line crossing counts, and timing information.

The demo shows:
- YOLO detection with ByteTracker
- Line crossing detection and counting
- Comprehensive metric collection (FPS, detections, tracking, line counts)
- Custom metrics (line crossing data)
- Periodic logging to JSON file
- Summary statistics at the end

Usage:
    # Using webcam with default horizontal line at y=400
    python debug_logger_line_crossing_demo.py --model yolov8n.pt

    # Using video file
    python debug_logger_line_crossing_demo.py --source file --input traffic.mp4 \
        --model yolov8n.pt

    # Using stream
    python debug_logger_line_crossing_demo.py --source stream \
        --input rtsp://camera/stream --model yolov8n.pt

    # Custom line position (diagonal from bottom-left to top-right)
    python debug_logger_line_crossing_demo.py --model yolov8n.pt \
        --line-start 0,720 --line-end 1280,0

    # Vertical line in middle of 1280x720 frame
    python debug_logger_line_crossing_demo.py --model yolov8n.pt \
        --line-start 640,0 --line-end 640,720

    # Windows PowerShell (use quotes):
    python debug_logger_line_crossing_demo.py --model yolov8n.pt `
        --line-start "640,0" --line-end "640,720"

    # Full configuration with output video
    python debug_logger_line_crossing_demo.py \
        --source file \
        --input traffic.mp4 \
        --model yolov8n.pt \
        --output tracked.mp4 \
        --line-start 0,300 \
        --line-end 1920,300 \
        --log-interval 60 \
        --log-file ./temp/traffic_metrics.json
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import supervision as sv

# Add examples directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    add_fps_arguments,
    add_sink_arguments,
    add_source_arguments,
    create_line_extract_callback,
    create_sink_from_args,
    create_source_from_args,
    get_video_info_from_source,
    parse_point,
    print_source_info,
)


def main():
    parser = argparse.ArgumentParser(
        description="Debug logger line crossing demo with tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Add standard arguments from utils
    add_source_arguments(parser)
    add_sink_arguments(parser)
    add_fps_arguments(parser)

    # Add demo-specific arguments
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO model (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="./temp/debug_line_crossing_metrics.json",
        help="Path to output log file (default: ./temp/debug_line_crossing_metrics.json)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=30,
        help="Number of frames between log writes (default: 30)",
    )
    parser.add_argument(
        "--include-detections",
        action="store_true",
        help="Include detailed detection data in logs (increases file size)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference: cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--line-start",
        type=str,
        default="0,400",
        help="Line start point as 'x,y' in pixels (default: 0,400)",
    )
    parser.add_argument(
        "--line-end",
        type=str,
        default="1920,400",
        help="Line end point as 'x,y' in pixels (default: 1920,400)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: None)",
    )

    args = parser.parse_args()

    # Parse line points
    line_start = parse_point(args.line_start)
    line_end = parse_point(args.line_end)

    # Create log file directory
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create source
    print("\n" + "=" * 60)
    print("DEBUG LOGGER LINE CROSSING DEMO")
    print("=" * 60)
    source = create_source_from_args(args)
    print_source_info(args)

    # Get video info for sink
    fps, width, height = get_video_info_from_source(source)

    # Print configuration
    print("Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Confidence: {args.conf}")
    print(f"  Device: {args.device}")
    print(f"  Line: ({line_start.x},{line_start.y}) -> ({line_end.x},{line_end.y})")
    print(f"  Log file: {args.log_file}")
    print(f"  Log interval: {args.log_interval} frames")
    print(f"  Include detections: {args.include_detections}")
    if args.max_frames:
        print(f"  Max frames: {args.max_frames}")
    print()

    # Create debug logger with custom metrics
    debug_logger = sv.DebugLoggerStep(
        log_file=args.log_file,
        log_interval=args.log_interval,
        include_detections_data=args.include_detections,
        custom_metrics=["line_in", "line_out"],  # Track line crossing counts
    )

    # Create sink
    sink = create_sink_from_args(
        args,
        window_name="Debug Line Crossing Demo (Press 'q' to quit)",
        fps=fps,
        width=width,
        height=height,
    )

    # Build pipeline
    print("Building pipeline...")

    # Create YOLO detection step (need to keep reference for class names)
    yolo_step = sv.YOLODetectionStep(
        model_path=args.model,
        conf=args.conf,
        device=args.device,
    )

    # Start with source
    pipeline = sv.Pipeline(source) | sv.FPSCalculatorStep()

    # Add detection
    pipeline = pipeline | yolo_step

    # Add tracking
    pipeline = pipeline | sv.ByteTrackerStep()

    # Add line zone step
    line_zone_step = sv.LineZoneStep(start=line_start, end=line_end)
    pipeline = pipeline | line_zone_step

    # Extract line counts for debug logger using utility function
    pipeline = pipeline | sv.CallbackStep(create_line_extract_callback())

    # Add debug logger
    pipeline = pipeline | debug_logger

    # Add annotations
    pipeline = (
        pipeline
        | sv.TrackerAnnotatorStep(
            class_names=yolo_step.model.names  # Pass class names for display
        )
        | sv.LineZoneAnnotatorStep()  # Show line with crossing counts
        | sink
    )

    print("Starting pipeline...")
    print("-" * 60)

    # Run pipeline
    frame_count = 0
    try:
        for data in pipeline:
            frame_count += 1

            # Print live statistics every 30 frames
            if frame_count % 30 == 0:
                fps_current = data.get("fps", 0)
                detections = data.get("detections")
                det_count = len(detections) if detections else 0

                # Get tracking info
                tracked = 0
                if detections and detections.tracker_id is not None:
                    tracked = np.sum(detections.tracker_id >= 0)

                # Get line crossing counts from line_zone
                line_zone = data.get("line_zone")
                line_in = line_zone.in_count if line_zone else 0
                line_out = line_zone.out_count if line_zone else 0

                print(
                    f"Frame {frame_count:4d} | "
                    f"FPS: {fps_current:5.1f} | "
                    f"Detections: {det_count:3d} | "
                    f"Tracked: {tracked:3d} | "
                    f"In: {line_in:3d} | Out: {line_out:3d}"
                )

            # Stop if max frames reached
            if args.max_frames and frame_count >= args.max_frames:
                print(f"\nReached max frames ({args.max_frames}), stopping...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    # Ensure final metrics are written
    debug_logger.flush()

    # Print summary statistics
    print("-" * 60)
    print(f"\nProcessed {frame_count} frames")
    print(f"Metrics saved to: {args.log_file}")

    # Get final line counts
    print("\nLine Crossing Summary:")
    print("-" * 60)
    print(f"  Total IN:  {line_zone_step.line_zone.in_count}")
    print(f"  Total OUT: {line_zone_step.line_zone.out_count}")
    print()

    print("Pipeline Statistics:")
    print("-" * 60)

    stats = debug_logger.get_statistics()
    for metric_name, metric_stats in stats.items():
        print(f"\n{metric_name}:")
        print(f"  Mean: {metric_stats['mean']:.2f}")
        print(f"  Min:  {metric_stats['min']:.2f}")
        print(f"  Max:  {metric_stats['max']:.2f}")
        print(f"  Std:  {metric_stats['std']:.2f}")

    print("\n" + "=" * 60)
    print("Next steps:")
    print(f"  1. Plot metrics:    python examples/pipeline/plot_debug_metrics.py {args.log_file}")
    print(f"  2. View raw data:   cat {args.log_file}")
    print(
        f"  3. Compare graphs:  python examples/pipeline/plot_debug_metrics.py {args.log_file} "
        "--metrics fps detections tracking"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()

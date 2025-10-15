"""
Debug Logger Detection Demo

This script demonstrates how to use the DebugLoggerStep to collect and log
performance metrics during pipeline execution with YOLO detection.

The demo shows:
- Real-time detection with YOLO
- Frame-by-frame metric collection
- Periodic logging to JSON file
- Summary statistics at the end

Usage:
    # Using webcam (default)
    python debug_logger_detection_demo.py

    # Using video file
    python debug_logger_detection_demo.py --source file --input video.mp4

    # Using stream
    python debug_logger_detection_demo.py --source stream --input rtsp://...

    # Custom log interval (every 60 frames instead of 30)
    python debug_logger_detection_demo.py --log-interval 60

    # Include detailed detection data
    python debug_logger_detection_demo.py --include-detections

    # Save output video
    python debug_logger_detection_demo.py --output output.mp4
"""

import argparse
import sys
from pathlib import Path

import supervision as sv

# Add examples directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    add_fps_arguments,
    add_sink_arguments,
    add_source_arguments,
    create_sink_from_args,
    create_source_from_args,
    get_video_info_from_source,
    print_source_info,
)


def main():
    parser = argparse.ArgumentParser(
        description="Debug logger detection demo",
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
        default="./temp/debug_detection_metrics.json",
        help="Path to output log file (default: ./temp/debug_detection_metrics.json)",
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
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: None)",
    )

    args = parser.parse_args()

    # Create log file directory
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create source
    print("\n" + "=" * 60)
    print("DEBUG LOGGER DETECTION DEMO")
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
    print(f"  Log file: {args.log_file}")
    print(f"  Log interval: {args.log_interval} frames")
    print(f"  Include detections: {args.include_detections}")
    if args.max_frames:
        print(f"  Max frames: {args.max_frames}")
    print()

    # Create debug logger
    debug_logger = sv.DebugLoggerStep(
        log_file=args.log_file,
        log_interval=args.log_interval,
        include_detections_data=args.include_detections,
    )

    # Create sink
    sink = create_sink_from_args(
        args,
        window_name="Debug Detection Demo (Press 'q' to quit)",
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

    pipeline = (
        sv.Pipeline(source)
        | sv.FPSCalculatorStep()
        | yolo_step
        | debug_logger
        | sv.LabelFormatterStep(labels=yolo_step.model.names)  # Format class names
        | sv.BoxAnnotatorStep()
        | sv.LabelAnnotatorStep()  # Show formatted labels
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

                print(
                    f"Frame {frame_count:4d} | "
                    f"FPS: {fps_current:5.1f} | "
                    f"Detections: {det_count:3d}"
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
    print("\nSummary Statistics:")
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
    print("=" * 60)


if __name__ == "__main__":
    main()

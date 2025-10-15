"""
YOLO object detection and tracking pipeline demo.

This example demonstrates how to use YOLODetectionStep with ByteTrackerStep
for real-time object detection and tracking with webcam, video file, or stream.

Requirements:
    pip install ultralytics

Usage:
    # Download a YOLO model first (or use your own):
    # https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

    # Run with webcam
    python examples/pipeline/yolo_tracking_demo.py --model yolov8n.pt

    # Run with video file
    python examples/pipeline/yolo_tracking_demo.py --source file --model yolov8n.pt --input video.mp4

    # Run with stream
    python examples/pipeline/yolo_tracking_demo.py --source stream --model yolov8n.pt --input rtsp://camera/stream

    # With output
    python examples/pipeline/yolo_tracking_demo.py --model yolov8n.pt --output tracked.mp4

    # Custom tracker settings
    python examples/pipeline/yolo_tracking_demo.py --model yolov8n.pt --track-threshold 0.3 --lost-buffer 60

Press 'q' or ESC to quit.
"""

import argparse

import supervision as sv

import utils


def main():
    parser = argparse.ArgumentParser(
        description="YOLO detection and tracking pipeline demo"
    )

    # Add standard arguments
    utils.add_source_arguments(parser)
    utils.add_sink_arguments(parser)
    utils.add_fps_arguments(parser)

    # Add YOLO-specific arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO model file (.pt)",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IOU threshold for NMS (default: 0.45)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)",
    )

    # Add tracker-specific arguments
    parser.add_argument(
        "--track-threshold",
        type=float,
        default=0.25,
        help="Track activation threshold (default: 0.25)",
    )

    parser.add_argument(
        "--lost-buffer",
        type=int,
        default=30,
        help="Lost track buffer frames (default: 30)",
    )

    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.8,
        help="Minimum matching threshold (default: 0.8)",
    )

    args = parser.parse_args()

    # Print configuration
    print("YOLO Tracking Demo")
    print("-" * 40)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Detection confidence: {args.conf}")
    print(f"Tracking activation threshold: {args.track_threshold}")
    utils.print_source_info(args)

    # Create source
    source = utils.create_source_from_args(args)

    # Get video info from source for proper output configuration
    fps, width, height = utils.get_video_info_from_source(source)

    # Create YOLO detection step
    yolo_step = sv.YOLODetectionStep(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )

    # Build pipeline
    pipeline = sv.Pipeline(source)

    # Add FPS calculator if requested
    if args.show_fps:
        pipeline = pipeline | sv.FPSCalculatorStep()

    # Create tracker step
    tracker_step = sv.ByteTrackerStep(
        track_activation_threshold=args.track_threshold,
        lost_track_buffer=args.lost_buffer,
        minimum_matching_threshold=args.match_threshold,
    )

    # Add detection, tracking, and annotation
    pipeline = (
        pipeline
        | yolo_step
        | tracker_step
        | sv.TrackerAnnotatorStep(
            class_names=yolo_step.model.names,
            trace_length=50,
            trace_thickness=2,
            copy_frame=False,
        )
    )

    # Add sink(s) with video info from source
    sink = utils.create_sink_from_args(args, "YOLO Tracking", fps, width, height)
    pipeline = pipeline | sink

    # Run pipeline
    print("Starting tracking...")
    try:
        pipeline.run()
    except (KeyboardInterrupt, StopIteration):
        pass
    finally:
        # Print tracker metrics
        tracker_metrics = tracker_step.get_metrics()
        utils.print_tracker_metrics(tracker_metrics)

        if args.output:
            print(f"\nTracking stopped. Video saved to: {args.output}")
        else:
            print("\nTracking stopped.")


if __name__ == "__main__":
    main()

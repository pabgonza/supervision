"""
YOLO integrated detection + tracking pipeline demo.

This example demonstrates how to use YOLOTrackingStep for efficient object
detection and tracking in a single step. This is more efficient than using
separate YOLODetectionStep + ByteTrackerStep as tracking is integrated into
the YOLO model inference.

Requirements:
    pip install ultralytics

Usage:
    # Download a YOLO model first (or use your own):
    # https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

    # Run with webcam using BoT-SORT tracker (default)
    python examples/pipeline/yolo_integrated_tracking_demo.py --model yolov8n.pt

    # Run with ByteTrack tracker
    python examples/pipeline/yolo_integrated_tracking_demo.py --model yolov8n.pt --tracker bytetrack.yaml

    # Run with video file
    python examples/pipeline/yolo_integrated_tracking_demo.py --source file --model yolov8n.pt --input video.mp4

    # With output
    python examples/pipeline/yolo_integrated_tracking_demo.py --model yolov8n.pt --output tracked.mp4

Press 'q' or ESC to quit.
"""

import argparse

import supervision as sv

import utils


def main():
    parser = argparse.ArgumentParser(
        description="YOLO integrated detection + tracking pipeline demo"
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
        "--tracker",
        type=str,
        default="botsort.yaml",
        choices=["botsort.yaml", "bytetrack.yaml"],
        help="Tracker type: botsort.yaml (default, more accurate) or bytetrack.yaml (faster)",
    )

    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Disable track persistence between frames",
    )

    args = parser.parse_args()

    # Print configuration
    print("YOLO Integrated Tracking Demo")
    print("-" * 40)
    print(f"Model: {args.model}")
    print(f"Tracker: {args.tracker}")
    print(f"Detection confidence: {args.conf}")
    print(f"Track persistence: {not args.no_persist}")
    utils.print_source_info(args)

    # Create source
    source = utils.create_source_from_args(args)

    # Get video info from source for proper output configuration
    fps, width, height = utils.get_video_info_from_source(source)

    # Create YOLO tracking step (detection + tracking in one)
    yolo_tracking_step = sv.YOLOTrackingStep(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        tracker=args.tracker,
        persist=not args.no_persist,
        verbose=False,
    )

    # Build pipeline
    pipeline = sv.Pipeline(source)

    # Add FPS calculator if requested
    if args.show_fps:
        pipeline = pipeline | sv.FPSCalculatorStep()

    # Add tracking and annotation
    pipeline = (
        pipeline
        | yolo_tracking_step
        | sv.TrackerAnnotatorStep(
            class_names=yolo_tracking_step.model.names,
            trace_length=50,
            trace_thickness=2,
            copy_frame=False,
            color_lookup=sv.ColorLookup.TRACK,  # Use CLASS instead of TRACK default
        )
    )

    # Add sink(s) with video info from source
    sink = utils.create_sink_from_args(
        args, "YOLO Integrated Tracking", fps, width, height
    )
    pipeline = pipeline | sink

    # Run pipeline
    print("Starting integrated tracking...")
    print(
        f"Note: Using {args.tracker.replace('.yaml', '').upper()} tracker built into YOLO"
    )
    try:
        pipeline.run()
    except (KeyboardInterrupt, StopIteration):
        pass
    finally:
        if args.output:
            print(f"\nTracking stopped. Video saved to: {args.output}")
        else:
            print("\nTracking stopped.")


if __name__ == "__main__":
    main()

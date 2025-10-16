"""
Simple YOLO detection example.

Minimal code to run YOLO detection on webcam, video file, or stream
with box annotations.

Requirements:
    pip install ultralytics

Usage:
    # Webcam with YOLO
    python examples/pipeline/yolo_simple.py --model yolov8n.pt

    # Video file
    python examples/pipeline/yolo_simple.py --model yolov8n.pt --source file --input video.mp4

    # RTSP stream
    python examples/pipeline/yolo_simple.py --model yolov8n.pt --source stream --input rtsp://camera/stream

    # With output and FPS
    python examples/pipeline/yolo_simple.py --model yolov8n.pt --output detections.mp4 --show-fps

    # Save without display
    python examples/pipeline/yolo_simple.py --model yolov8n.pt --output detections.mp4 --no-display

Note: Download YOLO models from https://github.com/ultralytics/ultralytics
Press 'q' or ESC to quit.
"""

import argparse

import supervision as sv

import utils


def main():
    parser = argparse.ArgumentParser(
        description="Simple YOLO detection pipeline example"
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
        help="Path to YOLO model file (e.g., yolov8n.pt)",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)",
    )

    args = parser.parse_args()

    # Print configuration
    print("YOLO Detection Demo")
    print("-" * 40)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Confidence: {args.conf}")
    utils.print_source_info(args)

    # Create source
    source = utils.create_source_from_args(args)

    # Get video info from source for proper output configuration
    fps, width, height = utils.get_video_info_from_source(source)

    # Build detection pipeline
    pipeline = sv.Pipeline(source)

    # Add FPS calculator if requested
    if args.show_fps:
        pipeline = pipeline | sv.FPSCalculatorStep()

    # Add detection and annotation
    pipeline = (
        pipeline
        | sv.YOLODetectionStep(args.model, conf=args.conf)
        | sv.BoxAnnotatorStep()
    )

    # Add sink(s) with video info from source
    sink = utils.create_sink_from_args(args, "YOLO Detection", fps, width, height)
    pipeline = pipeline | sink

    # Run
    print("Starting detection...")
    try:
        pipeline.run()
    except (KeyboardInterrupt, StopIteration):
        if args.output:
            print(f"\nDetection stopped. Video saved to: {args.output}")
        else:
            print("\nDetection stopped.")


if __name__ == "__main__":
    main()

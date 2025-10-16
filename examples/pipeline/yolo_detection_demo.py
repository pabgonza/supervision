"""
YOLO object detection pipeline demo.

This example demonstrates how to use YOLODetectionStep for real-time
object detection with webcam, video file, or RTSP stream.

Requirements:
    pip install ultralytics

Usage:
    # Download a YOLO model first (or use your own):
    # https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

    # Run with webcam (default)
    python examples/pipeline/yolo_detection_demo.py --model yolov8n.pt

    # Run with video file
    python examples/pipeline/yolo_detection_demo.py --model yolov8n.pt --source file --input video.mp4

    # Run with RTSP stream
    python examples/pipeline/yolo_detection_demo.py --model yolov8n.pt --source stream --input rtsp://camera.local/stream

    # With output
    python examples/pipeline/yolo_detection_demo.py --model yolov8n.pt --output detections.mp4 --show-fps

    # Save without display
    python examples/pipeline/yolo_detection_demo.py --model yolov8n.pt --output detections.mp4 --no-display

    # Use CPU instead of GPU
    python examples/pipeline/yolo_detection_demo.py --model yolov8n.pt --device cpu

Press 'q' or ESC to quit.
"""

import argparse

import supervision as sv

import utils


def main():
    parser = argparse.ArgumentParser(description="YOLO detection pipeline demo")

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
        help="Confidence threshold (default: 0.25)",
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

    args = parser.parse_args()

    # Print configuration
    print("YOLO Detection Demo")
    print("-" * 40)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Confidence threshold: {args.conf}")
    utils.print_source_info(args)

    # Create source
    source = utils.create_source_from_args(args)

    # Get video info from source for proper output configuration
    fps, width, height = utils.get_video_info_from_source(source)

    # Build pipeline
    pipeline = sv.Pipeline(source)

    # Add FPS calculator if requested
    if args.show_fps:
        pipeline = pipeline | sv.FPSCalculatorStep()

    # Add detection and annotation steps
    pipeline = (
        pipeline
        | sv.YOLODetectionStep(
            model_path=args.model,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )
        | sv.BoxAnnotatorStep()
    )

    # Add sink(s) with video info from source
    sink = utils.create_sink_from_args(args, "YOLO Detection", fps, width, height)
    pipeline = pipeline | sink

    # Run pipeline
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

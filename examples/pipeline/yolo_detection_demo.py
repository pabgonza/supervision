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
    python examples/pipeline/yolo_detection_demo.py --model yolov8n.pt \
        --source file --input video.mp4

    # Run with RTSP stream
    python examples/pipeline/yolo_detection_demo.py --model yolov8n.pt \
        --source stream --stream-url rtsp://camera.local/stream

    # Run with custom settings
    python examples/pipeline/yolo_detection_demo.py --model yolov8n.pt \
        --conf 0.5 --camera 0 --show-fps

    # Use CPU instead of GPU
    python examples/pipeline/yolo_detection_demo.py --model yolov8n.pt --device cpu
"""

import argparse

import supervision as sv


def main():
    parser = argparse.ArgumentParser(description="YOLO detection pipeline demo")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO model file (.pt)",
    )

    parser.add_argument(
        "--source",
        type=str,
        choices=["webcam", "file", "stream"],
        default="webcam",
        help="Source type (default: webcam)",
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Path to video file (for file source)",
    )

    parser.add_argument(
        "--stream-url",
        type=str,
        help="Stream URL (for stream source, e.g., rtsp://camera.local/stream)",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera ID (default: 0, for webcam source)",
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

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera width (default: 640)",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera height (default: 480)",
    )

    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Show FPS counter",
    )

    args = parser.parse_args()

    print(f"Loading YOLO model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Source: {args.source}")
    if args.source == "webcam":
        print(f"Camera: {args.camera} ({args.width}x{args.height})")
    elif args.source == "file":
        print(f"Input file: {args.input}")
    elif args.source == "stream":
        print(f"Stream URL: {args.stream_url}")
    print("Press 'q' or ESC to quit\n")

    # Create source based on type
    if args.source == "webcam":
        pipeline_source = sv.WebcamSource(
            camera_id=args.camera, width=args.width, height=args.height
        )
    elif args.source == "file":
        if not args.input:
            raise ValueError("--input required for file source")
        pipeline_source = sv.VideoFileSource(args.input)
    elif args.source == "stream":
        if not args.stream_url:
            raise ValueError("--stream-url required for stream source")
        pipeline_source = sv.StreamSource(args.stream_url)
    else:
        raise ValueError(f"Unknown source type: {args.source}")

    # Build pipeline
    pipeline = sv.Pipeline(pipeline_source)

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
            device=args.device,
            verbose=False,
        )
        | sv.BoxAnnotatorStep()
    )

    # Add display sink
    if args.show_fps:
        pipeline = pipeline | sv.DisplaySink("YOLO Detection", show_fps=True)
    else:
        pipeline = pipeline | sv.DisplaySink("YOLO Detection")

    # Run pipeline
    try:
        pipeline.run()
    except (KeyboardInterrupt, StopIteration):
        print("\nDetection stopped")


if __name__ == "__main__":
    main()

"""
YOLO object detection pipeline demo.

This example demonstrates how to use YOLODetectionStep for real-time
object detection with webcam, displaying annotated results.

Requirements:
    pip install ultralytics

Usage:
    # Download a YOLO model first (or use your own):
    # https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

    # Run with default model
    python examples/yolo_detection_demo.py --model yolov8n.pt

    # Run with custom settings
    python examples/yolo_detection_demo.py --model yolov8n.pt --conf 0.5 --camera 0

    # Use CPU instead of GPU
    python examples/yolo_detection_demo.py --model yolov8n.pt --device cpu
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
        "--camera",
        type=int,
        default=0,
        help="Camera ID (default: 0)",
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
    print(f"Camera: {args.camera} ({args.width}x{args.height})")
    print("Press 'q' or ESC to quit\n")

    # Build pipeline
    pipeline = (
        sv.Pipeline(
            sv.WebcamSource(camera_id=args.camera, width=args.width, height=args.height)
        )
        | sv.YOLODetectionStep(
            model_path=args.model,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        | sv.BoxAnnotatorStep()
    )

    # Add FPS calculator if requested
    if args.show_fps:
        # Rebuild pipeline with FPS
        pipeline = (
            sv.Pipeline(
                sv.WebcamSource(
                    camera_id=args.camera, width=args.width, height=args.height
                )
            )
            | sv.FPSCalculatorStep()
            | sv.YOLODetectionStep(
                model_path=args.model,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )
            | sv.BoxAnnotatorStep()
            | sv.DisplaySink("YOLO Detection", show_fps=True)
        )
    else:
        pipeline = pipeline | sv.DisplaySink("YOLO Detection")

    # Run pipeline
    try:
        pipeline.run()
    except (KeyboardInterrupt, StopIteration):
        print("\nDetection stopped")


if __name__ == "__main__":
    main()

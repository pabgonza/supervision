"""
YOLO object detection and tracking pipeline demo.

This example demonstrates how to use YOLODetectionStep with ByteTrackerStep
for real-time object detection and tracking with webcam or video files.

Requirements:
    pip install ultralytics

Usage:
    # Download a YOLO model first (or use your own):
    # https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

    # Run with webcam
    python examples/pipeline/yolo_tracking_demo.py --source webcam --model yolov8n.pt

    # Run with video file
    python examples/pipeline/yolo_tracking_demo.py --source file \
        --model yolov8n.pt --input video.mp4

    # Custom tracker settings
    python examples/pipeline/yolo_tracking_demo.py --source webcam \
        --model yolov8n.pt --track-threshold 0.3 --lost-buffer 60
"""

import argparse

import supervision as sv


def webcam_tracking(
    model_path: str,
    camera_id: int = 0,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cuda",
    track_activation_threshold: float = 0.25,
    lost_track_buffer: int = 30,
    minimum_matching_threshold: float = 0.8,
    width: int | None = None,
    height: int | None = None,
):
    """
    Demo pipeline with webcam source, detection, and tracking.

    Args:
        model_path: Path to YOLO model file
        camera_id: Camera device ID
        conf: Confidence threshold for detections
        iou: IOU threshold for NMS
        device: Device for inference
        track_activation_threshold: Tracker activation threshold
        lost_track_buffer: Frames to buffer lost tracks
        minimum_matching_threshold: Minimum matching threshold for tracker
        width: Camera width (None for default)
        height: Camera height (None for default)
    """
    print(f"Loading YOLO model: {model_path}")
    print(f"Device: {device}")
    if width and height:
        print(f"Camera: {camera_id} ({width}x{height})")
    else:
        print(f"Camera: {camera_id} (default resolution)")
    print(f"Detection confidence: {conf}")
    print(f"Tracking activation threshold: {track_activation_threshold}")
    print("Press 'q' or ESC to quit\n")

    # Build pipeline with optional resolution
    webcam_kwargs = {"camera_id": camera_id}
    if width is not None:
        webcam_kwargs["width"] = width
    if height is not None:
        webcam_kwargs["height"] = height

    # Create YOLO detection step
    yolo_step = sv.YOLODetectionStep(
        model_path=model_path,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )

    pipeline = (
        sv.Pipeline(sv.WebcamSource(**webcam_kwargs))
        | sv.FPSCalculatorStep()
        | yolo_step
        | sv.ByteTrackerStep(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )
        | sv.TraceAnnotatorStep(trace_length=50, thickness=2)
        | sv.BoxAnnotatorStep(copy_frame=False)
        | sv.LabelFormatterStep(class_names=yolo_step.model.names)
        | sv.LabelAnnotatorStep(copy_frame=False)
        | sv.DisplaySink("YOLO Tracking", show_fps=True)
    )

    # Run pipeline
    try:
        pipeline.run()
    except (KeyboardInterrupt, StopIteration):
        print("\nTracking stopped")


def file_tracking(
    model_path: str,
    video_path: str,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cuda",
    track_activation_threshold: float = 0.25,
    lost_track_buffer: int = 30,
    minimum_matching_threshold: float = 0.8,
):
    """
    Demo pipeline with video file source, detection, and tracking.

    Args:
        model_path: Path to YOLO model file
        video_path: Path to video file
        conf: Confidence threshold for detections
        iou: IOU threshold for NMS
        device: Device for inference
        track_activation_threshold: Tracker activation threshold
        lost_track_buffer: Frames to buffer lost tracks
        minimum_matching_threshold: Minimum matching threshold for tracker
    """
    print(f"Loading YOLO model: {model_path}")
    print(f"Video file: {video_path}")
    print(f"Device: {device}")
    print(f"Detection confidence: {conf}")
    print(f"Tracking activation threshold: {track_activation_threshold}")
    print("Press 'q' or ESC to quit\n")

    # Create YOLO detection step
    yolo_step = sv.YOLODetectionStep(
        model_path=model_path,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )

    # Build pipeline
    pipeline = (
        sv.Pipeline(sv.VideoFileSource(video_path))
        | sv.FPSCalculatorStep()
        | yolo_step
        | sv.ByteTrackerStep(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )
        | sv.TraceAnnotatorStep(trace_length=50, thickness=2)
        | sv.BoxAnnotatorStep(copy_frame=False)
        | sv.LabelFormatterStep(class_names=yolo_step.model.names)
        | sv.LabelAnnotatorStep(copy_frame=False)
        | sv.DisplaySink("YOLO Tracking", show_fps=True)
    )

    # Run pipeline
    try:
        pipeline.run()
    except (KeyboardInterrupt, StopIteration):
        print("\nTracking stopped")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO detection and tracking pipeline demo"
    )

    parser.add_argument(
        "--source",
        type=str,
        choices=["webcam", "file"],
        default="webcam",
        help="Source type (default: webcam)",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO model file (.pt)",
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Path to video file (for file source)",
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

    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Camera width (default: webcam default)",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Camera height (default: webcam default)",
    )

    args = parser.parse_args()

    try:
        if args.source == "webcam":
            webcam_tracking(
                model_path=args.model,
                camera_id=args.camera,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                track_activation_threshold=args.track_threshold,
                lost_track_buffer=args.lost_buffer,
                minimum_matching_threshold=args.match_threshold,
                width=args.width,
                height=args.height,
            )
        elif args.source == "file":
            if not args.input:
                print("Error: --input required for file source")
                return
            file_tracking(
                model_path=args.model,
                video_path=args.input,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                track_activation_threshold=args.track_threshold,
                lost_track_buffer=args.lost_buffer,
                minimum_matching_threshold=args.match_threshold,
            )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()

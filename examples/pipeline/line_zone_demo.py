"""
YOLO object detection, tracking, and line crossing counter demo.

This example demonstrates how to use YOLODetectionStep with ByteTrackerStep
and LineZoneStep for real-time object detection, tracking, and counting
objects crossing a line from webcam, video file, or RTSP stream.

Requirements:
    pip install ultralytics

Usage:
    # Download a YOLO model first (or use your own):
    # https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

    # Run with webcam (horizontal line at y=400)
    python examples/pipeline/line_zone_demo.py --source webcam --model yolov8n.pt

    # Run with video file
    python examples/pipeline/line_zone_demo.py --source file \
        --model yolov8n.pt --input video.mp4

    # Run with RTSP stream (vertical line in middle of 720p stream)
    python examples/pipeline/line_zone_demo.py --source stream \
        --model yolov8n.pt --stream-url rtsp://192.168.1.100/stream \
        --line-start 640,0 --line-end 640,720

    # Custom line position (diagonal line)
    python examples/pipeline/line_zone_demo.py --source webcam \
        --model yolov8n.pt --line-start 0,0 --line-end 640,480

    # Custom colors and text
    python examples/pipeline/line_zone_demo.py --source webcam \
        --model yolov8n.pt --line-color red --in-text "Entered" --out-text "Exited"
"""

import argparse

import supervision as sv


def parse_point(point_str: str) -> sv.Point:
    """Parse point from string format 'x,y'."""
    try:
        x, y = map(int, point_str.split(","))
        return sv.Point(x=x, y=y)
    except Exception:
        raise ValueError(f"Invalid point format: {point_str}. Expected 'x,y'")


def parse_color(color_str: str) -> sv.Color:
    """Parse color from string (name or hex)."""
    color_map = {
        "white": sv.Color.WHITE,
        "black": sv.Color.BLACK,
        "red": sv.Color.RED,
        "green": sv.Color.GREEN,
        "blue": sv.Color.BLUE,
        "yellow": sv.Color.YELLOW,
    }

    color_lower = color_str.lower()
    if color_lower in color_map:
        return color_map[color_lower]

    # Try hex color
    try:
        if color_str.startswith("#"):
            color_str = color_str[1:]
        r = int(color_str[0:2], 16)
        g = int(color_str[2:4], 16)
        b = int(color_str[4:6], 16)
        return sv.Color(r=r, g=g, b=b)
    except Exception:
        raise ValueError(
            f"Invalid color: {color_str}. "
            "Use color name (white, black, red, green, blue, yellow) "
            "or hex code (#RRGGBB)"
        )


def line_counter(
    model_path: str,
    source: str = "webcam",
    video_path: str | None = None,
    stream_url: str | None = None,
    camera_id: int = 0,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cuda",
    track_activation_threshold: float = 0.25,
    lost_track_buffer: int = 30,
    minimum_matching_threshold: float = 0.8,
    line_start: sv.Point | None = None,
    line_end: sv.Point | None = None,
    line_color: sv.Color = sv.Color.WHITE,
    line_thickness: int = 4,
    custom_in_text: str | None = None,
    custom_out_text: str | None = None,
    width: int | None = None,
    height: int | None = None,
):
    """
    Demo pipeline with detection, tracking, and line counting.

    Args:
        model_path: Path to YOLO model file
        source: Source type ('webcam', 'file', or 'stream')
        video_path: Path to video file (for file source)
        stream_url: Stream URL (for stream source)
        camera_id: Camera device ID (for webcam source)
        conf: Confidence threshold for detections
        iou: IOU threshold for NMS
        device: Device for inference
        track_activation_threshold: Tracker activation threshold
        lost_track_buffer: Frames to buffer lost tracks
        minimum_matching_threshold: Minimum matching threshold for tracker
        line_start: Starting point of line (default: left center)
        line_end: Ending point of line (default: right center)
        line_color: Color of the line
        line_thickness: Thickness of the line
        custom_in_text: Custom text for in count (default: "in")
        custom_out_text: Custom text for out count (default: "out")
        width: Camera width (None for default, webcam only)
        height: Camera height (None for default, webcam only)
    """
    print(f"Loading YOLO model: {model_path}")
    print(f"Device: {device}")
    print(f"Source: {source}")

    # Create source based on type
    if source == "webcam":
        webcam_kwargs = {"camera_id": camera_id}
        if width is not None:
            webcam_kwargs["width"] = width
        if height is not None:
            webcam_kwargs["height"] = height
        if width and height:
            print(f"Camera: {camera_id} ({width}x{height})")
        else:
            print(f"Camera: {camera_id} (default resolution)")
        pipeline_source = sv.WebcamSource(**webcam_kwargs)
    elif source == "file":
        if not video_path:
            raise ValueError("--input required for file source")
        print(f"Video file: {video_path}")
        pipeline_source = sv.VideoFileSource(video_path)
    elif source == "stream":
        if not stream_url:
            raise ValueError("--stream-url required for stream source")
        print(f"Stream URL: {stream_url}")
        pipeline_source = sv.StreamSource(stream_url)
    else:
        raise ValueError(f"Unknown source type: {source}")

    print(f"Detection confidence: {conf}")
    print(f"Tracking activation threshold: {track_activation_threshold}")

    # Default line position (horizontal center)
    if line_start is None:
        line_start = sv.Point(x=0, y=400)
    if line_end is None:
        line_end = sv.Point(x=1920, y=400)

    print(f"Line: ({line_start.x}, {line_start.y}) -> ({line_end.x}, {line_end.y})")
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
        sv.Pipeline(pipeline_source)
        | sv.FPSCalculatorStep()
        | yolo_step
        | sv.ByteTrackerStep(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )
        | sv.LineZoneStep(
            start=line_start,
            end=line_end,
        )
        | sv.TrackerAnnotatorStep(
            class_names=yolo_step.model.names,
            trace_length=50,
            trace_thickness=2,
            copy_frame=False,
        )
        | sv.LineZoneAnnotatorStep(
            color=line_color,
            thickness=line_thickness,
            text_scale=0.8,
            custom_in_text=custom_in_text,
            custom_out_text=custom_out_text,
            copy_frame=False,
        )
        | sv.DisplaySink("Line Counter", show_fps=True)
    )

    # Run pipeline
    try:
        pipeline.run()
    except (KeyboardInterrupt, StopIteration):
        print("\nLine counting stopped")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO detection, tracking, and line crossing counter demo"
    )

    parser.add_argument(
        "--source",
        type=str,
        choices=["webcam", "file", "stream"],
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
        "--line-start",
        type=str,
        help="Line start point as 'x,y' (default: left center)",
    )

    parser.add_argument(
        "--line-end",
        type=str,
        help="Line end point as 'x,y' (default: right center)",
    )

    parser.add_argument(
        "--line-color",
        type=str,
        default="white",
        help="Line color (name or hex, default: white)",
    )

    parser.add_argument(
        "--line-thickness",
        type=int,
        default=4,
        help="Line thickness (default: 4)",
    )

    parser.add_argument(
        "--in-text",
        type=str,
        default=None,
        help="Custom text for in count (default: 'in')",
    )

    parser.add_argument(
        "--out-text",
        type=str,
        default=None,
        help="Custom text for out count (default: 'out')",
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

    # Parse line points
    line_start = parse_point(args.line_start) if args.line_start else None
    line_end = parse_point(args.line_end) if args.line_end else None

    # Parse line color
    line_color = parse_color(args.line_color)

    try:
        line_counter(
            model_path=args.model,
            source=args.source,
            video_path=args.input,
            stream_url=args.stream_url,
            camera_id=args.camera,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            track_activation_threshold=args.track_threshold,
            lost_track_buffer=args.lost_buffer,
            minimum_matching_threshold=args.match_threshold,
            line_start=line_start,
            line_end=line_end,
            line_color=line_color,
            line_thickness=args.line_thickness,
            custom_in_text=args.in_text,
            custom_out_text=args.out_text,
            width=args.width,
            height=args.height,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()

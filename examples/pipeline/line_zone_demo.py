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
    python examples/pipeline/line_zone_demo.py --model yolov8n.pt

    # Run with video file
    python examples/pipeline/line_zone_demo.py --source file --model yolov8n.pt --input video.mp4

    # Run with RTSP stream (vertical line in middle of 720p stream)
    python examples/pipeline/line_zone_demo.py --source stream --model yolov8n.pt \
        --input rtsp://192.168.1.100/stream --line-start 640,0 --line-end 640,720

    # With output
    python examples/pipeline/line_zone_demo.py --model yolov8n.pt --output counting.mp4

    # Custom line position (diagonal line)
    python examples/pipeline/line_zone_demo.py --model yolov8n.pt --line-start 0,0 --line-end 640,480

    # Custom colors and text
    python examples/pipeline/line_zone_demo.py --model yolov8n.pt --line-color red \
        --in-text "Entered" --out-text "Exited"

Press 'q' or ESC to quit.
"""

import argparse

import supervision as sv

import utils


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


def main():
    parser = argparse.ArgumentParser(
        description="YOLO detection, tracking, and line crossing counter demo"
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

    # Add line zone-specific arguments
    parser.add_argument(
        "--line-start",
        type=str,
        help="Line start point as 'x,y' (default: 0,400)",
    )

    parser.add_argument(
        "--line-end",
        type=str,
        help="Line end point as 'x,y' (default: 1920,400)",
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

    args = parser.parse_args()

    # Print configuration
    print("Line Zone Counter Demo")
    print("-" * 40)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Detection confidence: {args.conf}")
    print(f"Tracking activation threshold: {args.track_threshold}")

    # Parse line points (default: horizontal center)
    line_start = parse_point(args.line_start) if args.line_start else sv.Point(x=0, y=400)
    line_end = parse_point(args.line_end) if args.line_end else sv.Point(x=1920, y=400)

    # Parse line color
    line_color = parse_color(args.line_color)

    print(f"Line: ({line_start.x}, {line_start.y}) -> ({line_end.x}, {line_end.y})")
    utils.print_source_info(args)

    # Create source
    source = utils.create_source_from_args(args)

    # Get video info from source for proper output configuration
    fps, width, height = utils.get_video_info_from_source(source)
    print(f"Video info: {width}x{height} @ {fps} fps")

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

    # Add detection, tracking, line zone, and annotation
    pipeline = (
        pipeline
        | yolo_step
        | tracker_step
        | sv.LineZoneStep(
            start=line_start,
            end=line_end,
        )
        | sv.TrackerAnnotatorStep(
            class_names=yolo_step.model.names,
            trace_length=30,
            trace_thickness=2,
            copy_frame=False,
        )
        | sv.LineZoneAnnotatorStep(
            color=line_color,
            thickness=args.line_thickness,
            text_scale=0.8,
            custom_in_text=args.in_text,
            custom_out_text=args.out_text,
            copy_frame=False,
        )
    )

    # Add sink(s) with video info from source
    sink = utils.create_sink_from_args(
        args,
        window_name="Line Counter",
        fps=fps,
        width=width,
        height=height
    )
    pipeline = pipeline | sink

    # Run pipeline
    print("Starting line counting...")
    try:
        pipeline.run()
    except (KeyboardInterrupt, StopIteration):
        pass
    finally:
        # Print tracker metrics
        tracker_metrics = tracker_step.get_metrics()
        utils.print_tracker_metrics(tracker_metrics)

        if args.output:
            print(f"\nLine counting stopped. Video saved to: {args.output}")
        else:
            print("\nLine counting stopped.")


if __name__ == "__main__":
    main()

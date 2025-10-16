"""
Pool Detection + Tracking + Line Crossing Counter Demo

Demonstrates parallel object detection using PoolYOLODetectionStep combined
with object tracking and line crossing detection. This is ideal for high-throughput
scenarios like monitoring multiple entry/exit points or busy intersections.

The parallel detection pool provides higher FPS while maintaining accurate
tracking and counting through strict frame ordering.

Usage:
    # Webcam with horizontal line
    python pool_line_zone_demo.py --model yolov8n.pt --pool-size 3

    # Video file with custom line position
    python pool_line_zone_demo.py --source file --input video.mp4 --model yolov8n.pt \
        --pool-size 3 --line-start 0,400 --line-end 1920,400

    # RTSP stream with vertical line and metrics
    python pool_line_zone_demo.py --source stream --input rtsp://camera/stream \
        --model yolov8n.pt --pool-size 4 --line-start 640,0 --line-end 640,720 \
        --show-metrics

    # With output video
    python pool_line_zone_demo.py --source file --input video.mp4 --model yolov8n.pt \
        --pool-size 3 --output counted.mp4

    # High throughput configuration
    python pool_line_zone_demo.py --source stream --input rtsp://192.168.1.100/stream \
        --model yolov8n.pt --pool-size 6 --conf 0.4 --show-metrics
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
    """Run pool detection with line zone counting demo."""
    parser = argparse.ArgumentParser(
        description="Pool detection with tracking and line crossing counter"
    )

    # Add standard arguments
    utils.add_source_arguments(parser)
    utils.add_sink_arguments(parser)
    utils.add_fps_arguments(parser)

    # Add pool-specific arguments
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path"
    )

    parser.add_argument(
        "--pool-size",
        type=int,
        default=3,
        help="Number of parallel detector workers (default: 3)"
    )

    parser.add_argument(
        "--queue-size",
        type=int,
        default=10,
        help="Maximum frames to queue for processing (default: 10)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Confidence threshold (default: 0.3)"
    )

    # Add tracker-specific arguments
    parser.add_argument(
        "--track-threshold",
        type=float,
        default=0.25,
        help="Track activation threshold (default: 0.25)"
    )

    parser.add_argument(
        "--lost-buffer",
        type=int,
        default=30,
        help="Lost track buffer frames (default: 30)"
    )

    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.8,
        help="Minimum matching threshold (default: 0.8)"
    )

    # Add line zone-specific arguments
    parser.add_argument(
        "--line-start",
        type=str,
        help="Line start point as 'x,y' (default: 0,400)"
    )

    parser.add_argument(
        "--line-end",
        type=str,
        help="Line end point as 'x,y' (default: 1920,400)"
    )

    parser.add_argument(
        "--line-color",
        type=str,
        default="white",
        help="Line color (name or hex, default: white)"
    )

    parser.add_argument(
        "--line-thickness",
        type=int,
        default=4,
        help="Line thickness (default: 4)"
    )

    parser.add_argument(
        "--in-text",
        type=str,
        default=None,
        help="Custom text for in count (default: 'in')"
    )

    parser.add_argument(
        "--out-text",
        type=str,
        default=None,
        help="Custom text for out count (default: 'out')"
    )

    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Display pool performance metrics on frame"
    )

    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("Pool Detection + Line Zone Counter Demo")
    print("=" * 60)
    utils.print_source_info(args)
    print(f"Model: {args.model}")
    print(f"Pool Size: {args.pool_size} workers")
    print(f"Queue Size: {args.queue_size} frames")
    print(f"Device: {args.device}")
    print(f"Confidence: {args.conf}")

    # Parse line points (default: horizontal center)
    line_start = parse_point(args.line_start) if args.line_start else sv.Point(x=0, y=400)
    line_end = parse_point(args.line_end) if args.line_end else sv.Point(x=1920, y=400)

    # Parse line color
    line_color = parse_color(args.line_color)

    print(f"Line: ({line_start.x}, {line_start.y}) -> ({line_end.x}, {line_end.y})")
    print("=" * 60)
    print()

    # Create source
    source = utils.create_source_from_args(args)

    # Get video info from source
    fps, width, height = utils.get_video_info_from_source(source)
    print(f"Video info: {width}x{height} @ {fps} fps\n")

    # Create pool detector with strict ordering (essential for tracking!)
    detector = sv.PoolYOLODetectionStep(
        model_path=args.model,
        pool_size=args.pool_size,
        conf=args.conf,
        max_queue_size=args.queue_size,
        warmup=True,
    )

    # Create tracker (requires frames in order!)
    tracker = sv.ByteTrackerStep(
        track_activation_threshold=args.track_threshold,
        lost_track_buffer=args.lost_buffer,
        minimum_matching_threshold=args.match_threshold,
    )

    # Create line zone step
    line_zone_step = sv.LineZoneStep(
        start=line_start,
        end=line_end,
    )

    # Build pipeline
    pipeline = sv.Pipeline(source)

    # Add FPS calculator if requested
    if args.show_fps:
        pipeline = pipeline | sv.FPSCalculatorStep()

    # Add detection, tracking, and line zone
    pipeline = (
        pipeline
        | detector  # Pool detection with STRICT ordering
        | tracker  # Tracking requires ordered frames!
        | line_zone_step  # Count crossings
        | sv.TrackerAnnotatorStep(
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

    # Add metrics overlay if requested
    if args.show_metrics:

        def add_metrics_overlay(data):
            """Add metrics text overlay to frame."""
            import cv2

            frame = data.get("frame")
            if frame is None:
                return data

            metrics = detector.get_metrics()
            queue_current, queue_max = detector.get_queue_size()
            detections = data.get("detections", sv.Detections.empty())
            num_tracked = (
                len(set(detections.tracker_id))
                if detections.tracker_id is not None
                else 0
            )

            # Get line zone counts
            line_zone = data.get("line_zone")
            in_count = line_zone.in_count if line_zone is not None else 0
            out_count = line_zone.out_count if line_zone is not None else 0

            # Get reorder buffer size (frames being processed/reordered)
            reorder_buffer_size = len(detector._reorder_buffer) if hasattr(detector, '_reorder_buffer') else 0
            result_queue_size = detector._result_queue.qsize() if hasattr(detector, '_result_queue') else 0

            # Create metrics text
            lines = [
                f"Workers: {metrics['workers_active']}",
                f"Detections: {len(detections)}",
                f"Tracked: {num_tracked}",
                f"IN Count: {in_count}",
                f"OUT Count: {out_count}",
                f"Processed: {metrics['frames_processed']}",
                f"Dropped: {metrics['frames_dropped']}",
                f"Reordered: {metrics['frames_reordered']}",
                f"Input Queue: {queue_current}/{queue_max}",
                f"Result Queue: {result_queue_size}",
                f"Reorder Buffer: {reorder_buffer_size}",
                f"Avg Inference: {metrics['avg_inference_time_ms']:.1f}ms",
                f"Avg Queue Time: {metrics['avg_queue_time_ms']:.1f}ms",
            ]

            # Draw semi-transparent background
            overlay = frame.copy()
            bg_height = len(lines) * 25 + 15
            cv2.rectangle(overlay, (5, 5), (320, bg_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Draw text on frame
            y_offset = 25
            for line in lines:
                cv2.putText(
                    frame,
                    line,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                y_offset += 25

            data["frame"] = frame
            return data

        pipeline = pipeline | sv.CallbackStep(add_metrics_overlay)

    # Add sink(s)
    sink = utils.create_sink_from_args(
        args,
        "Pool Detection + Line Counter",
        fps,
        width,
        height
    )
    pipeline = pipeline | sink

    # Run pipeline
    print("Starting pipeline... Press 'q' to quit\n")
    try:
        frame_count = 0
        for data in pipeline:
            frame_count += 1

            # Print metrics every 30 frames
            if frame_count % 30 == 0:
                detector_metrics = detector.get_metrics()
                detections = data.get("detections", sv.Detections.empty())
                num_tracked = (
                    len(set(detections.tracker_id))
                    if detections.tracker_id is not None
                    else 0
                )

                # Get line zone counts
                line_zone = data.get("line_zone")
                in_count = line_zone.in_count if line_zone is not None else 0
                out_count = line_zone.out_count if line_zone is not None else 0

                print(
                    f"Frame {frame_count:4d} | "
                    f"Detections: {len(detections):2d} | Tracked: {num_tracked:2d} | "
                    f"IN: {in_count:3d} | OUT: {out_count:3d} | "
                    f"Inference: {detector_metrics['avg_inference_time_ms']:5.1f}ms"
                )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    # Print final metrics
    detector_metrics = detector.get_metrics()
    print("\n" + "=" * 60)
    print("Final Metrics")
    print("=" * 60)
    print(f"Total Frames:         {detector_metrics['frames_processed']}")
    print(f"Frames Dropped:       {detector_metrics['frames_dropped']}")
    print(f"Frames Reordered:     {detector_metrics['frames_reordered']}")
    print(f"Queue Full Count:     {detector_metrics['queue_full_count']}")
    print(f"Avg Inference Time:   {detector_metrics['avg_inference_time_ms']:.2f} ms")
    print(f"Avg Queue Time:       {detector_metrics['avg_queue_time_ms']:.2f} ms")
    print(f"Avg Reorder Delay:    {detector_metrics['avg_reorder_delay_ms']:.2f} ms")
    print(f"Workers Active:       {detector_metrics['workers_active']}")

    # Get final line zone counts from last data
    try:
        line_zone = data.get("line_zone")
        if line_zone is not None:
            in_count = line_zone.in_count
            out_count = line_zone.out_count
            print()
            print(f"Line Zone Counts:")
            print(f"  Objects IN:  {in_count}")
            print(f"  Objects OUT: {out_count}")
            print(f"  Net Count:   {in_count - out_count}")
    except:
        pass

    print("=" * 60)

    # Why STRICT_ORDER matters
    if detector_metrics['frames_reordered'] == 0:
        print("\n✅ Perfect frame ordering maintained!")
        print("   This is crucial for tracking and counting - frames must arrive")
        print("   in sequence for accurate line crossing detection.")
    else:
        print(f"\n⚠️  {detector_metrics['frames_reordered']} frames were reordered")
        print("   But thanks to STRICT_ORDER, they were delivered in sequence!")
        print("   Tracking and counting remained accurate.")

    if args.output:
        print(f"\nVideo saved to: {args.output}")

    print("\nDone!")


if __name__ == "__main__":
    main()

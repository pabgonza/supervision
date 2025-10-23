"""
Pool Detection Demo

Demonstrates parallel object detection using PoolYOLODetectionStep.
Uses multiple detector workers to process frames in parallel while
maintaining frame order at the output.

Usage:
    # Webcam
    python pool_detection_demo.py --model yolov8n.pt --pool-size 3

    # Video file
    python pool_detection_demo.py --source file --input video.mp4 --pool-size 3

    # Stream
    python pool_detection_demo.py --source stream --input rtsp://camera/stream --pool-size 2

    # With output
    python pool_detection_demo.py --source file --input video.mp4 --output result.mp4 --pool-size 3

    # With metrics overlay
    python pool_detection_demo.py --source webcam --show-metrics --pool-size 2
"""

import argparse

import supervision as sv
import utils


def main():
    """Run pool detection demo."""
    parser = argparse.ArgumentParser(
        description="Pool-based YOLO detection with parallel processing"
    )

    # Add standard arguments
    utils.add_source_arguments(parser)
    utils.add_sink_arguments(parser)

    # Add pool-specific arguments
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO model file",
    )

    parser.add_argument(
        "--pool-size",
        type=int,
        default=2,
        help="Number of parallel detector workers",
    )

    parser.add_argument(
        "--queue-size",
        type=int,
        default=10,
        help="Maximum frames to queue for processing",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference",
    )

    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Display performance metrics on video",
    )

    args = parser.parse_args()

    # Print configuration
    print("Pool Detection Demo")
    print("=" * 60)
    utils.print_source_info(args)
    print(f"Model: {args.model}")
    print(f"Pool size: {args.pool_size} workers")
    print(f"Queue size: {args.queue_size} frames")
    print(f"Device: {args.device}")
    print(f"Confidence: {args.conf}")
    print("=" * 60)
    print()

    # Create source
    source = utils.create_source_from_args(args)

    # Get video info from source
    fps, width, height = utils.get_video_info_from_source(source)
    print(f"Video info: {width}x{height} @ {fps} fps\n")

    # Create pool detector
    detector = sv.PoolYOLODetectionStep(
        model_path=args.model,
        pool_size=args.pool_size,
        max_queue_size=args.queue_size,
        conf=args.conf,
        warmup=True,
    )

    # Build pipeline
    pipeline = sv.Pipeline(source)
    pipeline = pipeline | detector | sv.BoxAnnotatorStep(thickness=2)

    # Add metrics overlay if requested
    if args.show_metrics:
        metrics_callback = utils.create_metrics_overlay_callback(
            position="top-left",
            font_scale=0.6,
            color=(0, 255, 0),
            bg_opacity=0.6,
            show_fps=False,
            show_detections=True,
            show_tracked=False,
            show_line_counts=False,
            show_roi_info=False,
            show_inference_time=True,
            show_tracking_time=False,
            show_pool_metrics=True,
        )
        # Pass detector reference in data for pool metrics
        def add_detector_to_data(data):
            data["pool_detector"] = detector
            return data

        pipeline = pipeline | sv.CallbackStep(add_detector_to_data) | sv.CallbackStep(metrics_callback)

    # Add sink(s)
    sink = utils.create_sink_from_args(args, "Pool Detection", fps, width, height)
    pipeline = pipeline | sink

    # Run pipeline
    print("Starting pipeline... Press 'q' to quit\n")
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Print final metrics
        metrics = detector.get_metrics()
        print("\n" + "=" * 60)
        print("Final Performance Metrics")
        print("=" * 60)
        print(f"Frames Processed:     {metrics['frames_processed']}")
        print(f"Frames Dropped:       {metrics['frames_dropped']}")
        print(f"Frames Reordered:     {metrics['frames_reordered']}")
        print(f"Queue Full Count:     {metrics['queue_full_count']}")
        print(f"Avg Inference Time:   {metrics['avg_inference_time_ms']:.2f} ms")
        print(f"Avg Queue Time:       {metrics['avg_queue_time_ms']:.2f} ms")
        print(f"Avg Reorder Delay:    {metrics['avg_reorder_delay_ms']:.2f} ms")
        print(f"Workers Active:       {metrics['workers_active']}")
        print("=" * 60)

        # Stop detector
        detector.stop()
        print("\nDone!")


if __name__ == "__main__":
    main()

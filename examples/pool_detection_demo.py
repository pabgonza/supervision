"""
Pool Detection Demo

Demonstrates parallel object detection using PoolYOLODetectionStep.
Uses multiple detector workers to process frames in parallel while
maintaining frame order at the output.

Usage:
    python pool_detection_demo.py --source video.mp4 --pool-size 3
    python pool_detection_demo.py --source 0 --pool-size 2  # Webcam
"""

import argparse

import supervision as sv


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pool-based YOLO detection with parallel processing"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source (file path or camera index)",
    )
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
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video file path (optional)",
    )
    return parser.parse_args()


def main():
    """Run pool detection demo."""
    args = parse_arguments()

    # Parse source
    try:
        source_index = int(args.source)
        source = sv.WebcamSource(camera_id=source_index)
        print(f"Using webcam source: {source_index}")
    except ValueError:
        # Check if it's a stream URL (rtsp://, rtmp://, http://)
        if args.source.startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
            source = sv.StreamSource(stream_url=args.source)
            print(f"Using stream source: {args.source}")
        else:
            source = sv.VideoFileSource(video_path=args.source)
            print(f"Using video file source: {args.source}")

    # Create pool detector
    print(f"\nInitializing pool detector:")
    print(f"  - Model: {args.model}")
    print(f"  - Pool size: {args.pool_size} workers")
    print(f"  - Queue size: {args.queue_size} frames")
    print(f"  - Device: {args.device}")
    print(f"  - Confidence: {args.conf}")
    print()

    detector = sv.PoolYOLODetectionStep(
        model_path=args.model,
        pool_size=args.pool_size,
        max_queue_size=args.queue_size,
        conf=args.conf,
        device=args.device,
        warmup=True,
    )

    # Build pipeline
    pipeline_steps = [
        sv.FPSCalculatorStep(),
        detector,
        sv.BoxAnnotatorStep(thickness=2),
    ]

    # Add metrics overlay if requested
    if args.show_metrics:

        def add_metrics_overlay(data):
            """Add metrics text overlay to frame."""
            import cv2

            frame = data.get("frame")
            if frame is None:
                return data

            fps = data.get("fps", 0)
            metrics = detector.get_metrics()
            queue_current, queue_max = detector.get_queue_size()

            # Create metrics text
            lines = [
                f"FPS: {fps:.1f}",
                f"Workers: {metrics['workers_active']}",
                f"Processed: {metrics['frames_processed']}",
                f"Dropped: {metrics['frames_dropped']}",
                f"Queue: {queue_current}/{queue_max}",
                f"Avg Inference: {metrics['avg_inference_time_ms']:.1f}ms",
                f"Avg Queue Time: {metrics['avg_queue_time_ms']:.1f}ms",
                f"Reordered: {metrics['frames_reordered']}",
            ]

            # Draw text on frame
            y_offset = 30
            for line in lines:
                cv2.putText(
                    frame,
                    line,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                y_offset += 25

            data["frame"] = frame
            return data

        pipeline_steps.append(sv.CallbackStep(add_metrics_overlay))

    # Create pipeline with appropriate sink
    pipeline = sv.Pipeline(source)
    for step in pipeline_steps:
        pipeline = pipeline | step

    if args.output:
        pipeline = pipeline | sv.VideoFileSink(target_path=args.output)
        print(f"Saving output to: {args.output}")
    else:
        pipeline = pipeline | sv.DisplaySink(window_name="Pool Detection")

    # Run pipeline
    print("\nStarting pipeline...")
    print("Press 'q' to quit\n")

    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Print final metrics
        metrics = detector.get_metrics()
        print("\n" + "=" * 50)
        print("Final Performance Metrics:")
        print("=" * 50)
        print(f"Frames Processed:     {metrics['frames_processed']}")
        print(f"Frames Dropped:       {metrics['frames_dropped']}")
        print(f"Frames Reordered:     {metrics['frames_reordered']}")
        print(f"Queue Full Count:     {metrics['queue_full_count']}")
        print(f"Avg Inference Time:   {metrics['avg_inference_time_ms']:.2f} ms")
        print(f"Avg Queue Time:       {metrics['avg_queue_time_ms']:.2f} ms")
        print(f"Avg Reorder Delay:    {metrics['avg_reorder_delay_ms']:.2f} ms")
        print(f"Workers Active:       {metrics['workers_active']}")
        print("=" * 50)

        # Stop detector
        detector.stop()
        print("\nDone!")


if __name__ == "__main__":
    main()

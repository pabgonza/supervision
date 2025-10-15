"""
Pool Detection + Tracking Demo

Demonstrates the main use case for PoolDetectorStep: parallel detection
with strict frame ordering for accurate object tracking.

This example shows why STRICT_ORDER is essential for tracking applications.

Usage:
    # Webcam
    python pool_detection_tracking.py --model yolov8n.pt --pool-size 3

    # Video file
    python pool_detection_tracking.py --source file --input video.mp4 --pool-size 3

    # Stream
    python pool_detection_tracking.py --source stream --input rtsp://camera/stream --pool-size 3

    # With output
    python pool_detection_tracking.py --source file --input video.mp4 --output tracked.mp4 --pool-size 3

    # With metrics overlay (shows detailed pool performance)
    python pool_detection_tracking.py --source stream --input rtsp://camera/stream --show-metrics --pool-size 4

    # Metrics explained:
    # - Input Queue: Frames waiting to be assigned to workers (usually 0 if workers are fast)
    # - Result Queue: Processed frames waiting to be reordered
    # - Reorder Buffer: Out-of-order frames waiting for correct sequence
"""

import argparse

import supervision as sv

import utils


def main():
    """Run pool detection with tracking demo."""
    parser = argparse.ArgumentParser(description="Pool detection with tracking")

    # Add standard arguments
    utils.add_source_arguments(parser)
    utils.add_sink_arguments(parser)

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
        help="Pool size"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Confidence threshold"
    )

    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Display performance metrics on frame"
    )

    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("Pool Detection + Tracking Demo")
    print("=" * 60)
    utils.print_source_info(args)
    print(f"Model: {args.model}")
    print(f"Pool Size: {args.pool_size} workers")
    print(f"Device: {args.device}")
    print(f"Confidence: {args.conf}")
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
        device=args.device,
        conf=args.conf,
        max_queue_size=10,
    )

    # Create tracker (requires frames in order!)
    tracker = sv.ByteTrackerStep()

    # Create annotators
    box_annotator = sv.BoxAnnotatorStep(thickness=2)
    label_annotator = sv.LabelAnnotatorStep(text_scale=0.5)

    # Build pipeline
    pipeline = (
        sv.Pipeline(source)
        | detector  # Pool detection with STRICT ordering
        | tracker  # Tracking requires ordered frames!
        | box_annotator  # Draw boxes
        | label_annotator  # Draw track IDs
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

            # Get reorder buffer size (frames being processed/reordered)
            reorder_buffer_size = len(detector._reorder_buffer) if hasattr(detector, '_reorder_buffer') else 0
            result_queue_size = detector._result_queue.qsize() if hasattr(detector, '_result_queue') else 0

            # Get tracking time from current frame
            tracking_time = data.get("tracker_processing_time_ms", 0.0)

            # Create metrics text
            lines = [
                f"Workers: {metrics['workers_active']}",
                f"Detections: {len(detections)}",
                f"Tracked: {num_tracked}",
                f"Processed: {metrics['frames_processed']}",
                f"Dropped: {metrics['frames_dropped']}",
                f"Reordered: {metrics['frames_reordered']}",
                f"Input Queue: {queue_current}/{queue_max}",
                f"Result Queue: {result_queue_size}",
                f"Reorder Buffer: {reorder_buffer_size}",
                f"Avg Inference: {metrics['avg_inference_time_ms']:.1f}ms",
                f"Avg Queue Time: {metrics['avg_queue_time_ms']:.1f}ms",
                f"Tracking: {tracking_time:.1f}ms",
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
    sink = utils.create_sink_from_args(args, "Pool Detection + Tracking", fps, width, height)
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

                # Get tracking time from current frame
                tracking_time = data.get("tracker_processing_time_ms", 0.0)

                print(
                    f"Frame {frame_count:4d} | "
                    f"Detections: {len(detections):2d} | Tracked: {num_tracked:2d} | "
                    f"Queue: {detector_metrics['frames_reordered']:3d} reordered | "
                    f"Inference: {detector_metrics['avg_inference_time_ms']:5.1f}ms | "
                    f"Tracking: {tracking_time:5.1f}ms"
                )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    # Print final metrics
    detector_metrics = detector.get_metrics()
    print("\n" + "=" * 60)
    print("Detector Metrics")
    print("=" * 60)
    print(f"Total Frames:         {detector_metrics['frames_processed']}")
    print(f"Frames Dropped:       {detector_metrics['frames_dropped']}")
    print(f"Frames Reordered:     {detector_metrics['frames_reordered']}")
    print(f"Queue Full Count:     {detector_metrics['queue_full_count']}")
    print(f"Avg Inference Time:   {detector_metrics['avg_inference_time_ms']:.2f} ms")
    print(f"Avg Queue Time:       {detector_metrics['avg_queue_time_ms']:.2f} ms")
    print(f"Avg Reorder Delay:    {detector_metrics['avg_reorder_delay_ms']:.2f} ms")
    print(f"Workers Active:       {detector_metrics['workers_active']}")
    print("=" * 60)

    # Print tracker metrics
    tracker_metrics = tracker.get_metrics()
    utils.print_tracker_metrics(tracker_metrics)

    # Why STRICT_ORDER matters
    if detector_metrics['frames_reordered'] == 0:
        print("\n✅ Perfect frame ordering maintained!")
        print("   This is crucial for tracking - frames must arrive in sequence")
        print("   for tracker to maintain object identities correctly.")
    else:
        print(f"\n⚠️  {detector_metrics['frames_reordered']} frames were reordered")
        print("   But thanks to STRICT_ORDER, they were delivered in sequence!")

    if args.output:
        print(f"\nVideo saved to: {args.output}")

    print("\nDone!")


if __name__ == "__main__":
    main()

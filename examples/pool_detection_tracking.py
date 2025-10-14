"""
Pool Detection + Tracking Demo

Demonstrates the main use case for PoolDetectorStep: parallel detection
with strict frame ordering for accurate object tracking.

This example shows why STRICT_ORDER is essential for tracking applications.

Usage:
    python pool_detection_tracking.py --source video.mp4 --pool-size 3
"""

import argparse

import supervision as sv


def main():
    """Run pool detection with tracking demo."""
    parser = argparse.ArgumentParser(description="Pool detection with tracking")
    parser.add_argument("--source", type=str, default="output.mp4", help="Video source")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model")
    parser.add_argument("--pool-size", type=int, default=3, help="Pool size")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    args = parser.parse_args()

    print("=" * 60)
    print("Pool Detection + Tracking Demo")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"Model: {args.model}")
    print(f"Pool Size: {args.pool_size} workers")
    print(f"Device: {args.device}")
    print("=" * 60)
    print()

    # Create source
    if args.source.startswith(('rtsp://', 'rtmp://', 'http://')):
        source = sv.StreamSource(stream_url=args.source)
    else:
        source = sv.VideoFileSource(video_path=args.source)

    # Create pool detector with strict ordering (essential for tracking!)
    detector = sv.PoolYOLODetectionStep(
        model_path=args.model,
        pool_size=args.pool_size,
        device=args.device,
        conf=0.3,
        max_queue_size=10,
    )

    # Create tracker (requires frames in order!)
    tracker = sv.ByteTrackerStep()

    # Create annotators
    box_annotator = sv.BoxAnnotatorStep(thickness=2)
    label_annotator = sv.LabelAnnotatorStep(text_scale=0.5)

    # FPS calculator
    fps_calculator = sv.FPSCalculatorStep()

    # Build pipeline
    pipeline = (
        sv.Pipeline(source)
        | fps_calculator  # Calculate FPS
        | detector  # Pool detection with STRICT ordering
        | tracker  # Tracking requires ordered frames!
        | box_annotator  # Draw boxes
        | label_annotator  # Draw track IDs
    )

    # Add sink
    if args.output:
        pipeline = pipeline | sv.VideoFileSink(target_path=args.output)
        print(f"Saving output to: {args.output}\n")
    else:
        pipeline = pipeline | sv.DisplaySink(window_name="Pool Detection + Tracking")
        print("Press 'q' to quit\n")

    # Run pipeline
    try:
        frame_count = 0
        for data in pipeline:
            frame_count += 1

            # Print metrics every 30 frames
            if frame_count % 30 == 0:
                detector_metrics = detector.get_metrics()
                fps = data.get("fps", 0)
                detections = data.get("detections", sv.Detections.empty())
                num_tracked = len(set(detections.tracker_id)) if detections.tracker_id is not None else 0

                print(f"Frame {frame_count:4d} | FPS: {fps:5.1f} | "
                      f"Detections: {len(detections):2d} | Tracked: {num_tracked:2d} | "
                      f"Queue: {detector_metrics['frames_reordered']:3d} reordered | "
                      f"Inference: {detector_metrics['avg_inference_time_ms']:5.1f}ms")

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
    print("=" * 60)

    # Why STRICT_ORDER matters
    if detector_metrics['frames_reordered'] == 0:
        print("\n✅ Perfect frame ordering maintained!")
        print("   This is crucial for tracking - frames must arrive in sequence")
        print("   for tracker to maintain object identities correctly.")
    else:
        print(f"\n⚠️  {detector_metrics['frames_reordered']} frames were reordered")
        print("   But thanks to STRICT_ORDER, they were delivered in sequence!")

    print("\nDone!")


if __name__ == "__main__":
    main()

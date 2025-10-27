"""
Force high queue by using larger model or simulating slow processing.
"""

import time
import threading
import numpy as np
import supervision as sv

print("=" * 60)
print("Slow Worker Test - Force High Queue")
print("=" * 60)

# Configuration - using SMALLEST pool with LARGER model
POOL_SIZE = 1  # Single worker
QUEUE_SIZE = 30
BURST_FRAMES = 200

print(f"Pool Size: {POOL_SIZE} worker (bottleneck)")
print(f"Queue Size: {QUEUE_SIZE} frames")
print(f"Burst Frames: {BURST_FRAMES}")
print("=" * 60)

# Try to use a larger model if available, otherwise use yolov8n
try:
    # Try yolov8m (medium) for slower inference
    model_path = "yolov8m.pt"
    print(f"\nAttempting to use larger model: {model_path}")
except:
    model_path = "yolov8n.pt"
    print(f"\nUsing standard model: {model_path}")

# Create pool detector
pool_detector = sv.PoolYOLODetectionStep(
    model_path=model_path,
    pool_size=POOL_SIZE,
    conf=0.25,
    max_queue_size=QUEUE_SIZE,
    warmup=True,
)

# Monitor queue in background thread
queue_samples = []
monitoring = True

def monitor_queue():
    """Background thread to sample queue size continuously."""
    global monitoring
    sample_count = 0
    while monitoring:
        queue_current, queue_max = pool_detector.get_queue_size()
        metrics = pool_detector.get_metrics()
        timestamp = time.time()
        queue_samples.append({
            'sample': sample_count,
            'time': timestamp,
            'queue_size': queue_current,
            'queue_max': queue_max,
            'frames_processed': metrics.get('frames_processed', 0),
            'frames_dropped': metrics.get('frames_dropped', 0),
        })
        sample_count += 1
        time.sleep(0.002)  # Sample every 2ms

# Start monitoring thread
print("\nStarting queue monitor thread...")
monitor_thread = threading.Thread(target=monitor_queue, daemon=True)
monitor_thread.start()

time.sleep(0.1)  # Let monitoring start

print("Starting FAST burst generation (no delays)...\n")
start_time = time.time()

# Generate synthetic frames generator - NO DELAYS
def burst_source():
    for i in range(BURST_FRAMES):
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        yield {"frame": frame}
        # Absolutely no delay - flood the queue

# Build pipeline
pipeline = sv.Pipeline(burst_source()) | pool_detector

# Process frames and show progress
frame_count = 0
last_print = 0
for data in pipeline:
    frame_count += 1

    # Print progress every 20 frames
    if frame_count - last_print >= 20:
        pool_metrics = data.get("pool_metrics", {})
        print(f"Processed {frame_count}/{BURST_FRAMES} frames | "
              f"Queue: {pool_metrics.get('queue_current', 0)}/{pool_metrics.get('queue_max', 0)} | "
              f"Dropped: {pool_metrics.get('frames_dropped', 0)}")
        last_print = frame_count

burst_duration = time.time() - start_time
monitoring = False  # Stop monitoring
monitor_thread.join(timeout=2.0)

print(f"\nBurst completed: {BURST_FRAMES} frames in {burst_duration:.2f}s ({BURST_FRAMES/burst_duration:.0f} FPS)")

# Analyze queue samples
print("\n" + "=" * 60)
print("QUEUE MONITORING RESULTS")
print("=" * 60)

if queue_samples:
    max_queue_seen = max(s['queue_size'] for s in queue_samples)
    avg_queue = sum(s['queue_size'] for s in queue_samples) / len(queue_samples)

    # Find peak queue moments
    high_queue_samples = [s for s in queue_samples if s['queue_size'] >= max_queue_seen * 0.7]

    print(f"\nTotal samples taken:    {len(queue_samples)}")
    print(f"Max queue size seen:    {max_queue_seen}/{QUEUE_SIZE}")
    print(f"Avg queue size:         {avg_queue:.2f}")

    if max_queue_seen > 5:
        print(f"\n✅ HIGH QUEUE DETECTED! Peak moments (queue >= {int(max_queue_seen*0.7)}):")
        for sample in high_queue_samples[:15]:
            print(f"  Sample {sample['sample']:4d}: queue = {sample['queue_size']:2d}/{sample['queue_max']} | "
                  f"processed = {sample['frames_processed']:3d} | dropped = {sample['frames_dropped']}")
    elif max_queue_seen > 0:
        print(f"\n⚠️  Queue increased to {max_queue_seen} but didn't fill significantly")
        non_zero = [s for s in queue_samples if s['queue_size'] > 0]
        print(f"   Samples with queue > 0: {len(non_zero)} ({100*len(non_zero)/len(queue_samples):.1f}%)")
    else:
        print(f"\n❌ Queue never increased - pool too fast for this test")

    # Show queue size distribution
    queue_distribution = {}
    for s in queue_samples:
        size = s['queue_size']
        queue_distribution[size] = queue_distribution.get(size, 0) + 1

    print(f"\nQueue Size Distribution:")
    for size in sorted(queue_distribution.keys())[:15]:  # Show first 15 buckets
        count = queue_distribution[size]
        percent = 100 * count / len(queue_samples)
        bar = '█' * min(50, int(percent))
        print(f"  Queue={size:2d}: {bar:50s} {count:4d} ({percent:5.1f}%)")

# Final pool metrics
final_metrics = pool_detector.get_metrics()
print(f"\nFinal Pool Metrics:")
print(f"  Frames processed:   {final_metrics['frames_processed']}")
print(f"  Frames dropped:     {final_metrics['frames_dropped']}")
print(f"  Queue full count:   {final_metrics['queue_full_count']}")
print(f"  Avg inference:      {final_metrics['avg_inference_time_ms']:.1f}ms")

if final_metrics['frames_dropped'] > 0:
    print(f"\n✅ SUCCESS: Dropped {final_metrics['frames_dropped']} frames - queue was overloaded!")
elif max_queue_seen > 10:
    print(f"\n✅ SUCCESS: Queue reached {max_queue_seen} - metrics are working!")
else:
    print(f"\n⚠️  Pool too efficient - even 1 worker can handle burst")

print("\n" + "=" * 60)

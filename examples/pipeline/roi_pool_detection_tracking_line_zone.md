# ROI-based Pool Detection with Tracking and Line Counting

Complete example demonstrating parallel object detection using a pool of YOLO models within a Region of Interest (ROI), combined with object tracking, line crossing detection, and comprehensive metrics tracking.

## Features

- **Parallel Detection**: Multiple YOLO detector workers processing frames concurrently
- **ROI Processing**: Detection only within defined region of interest
- **Object Tracking**: ByteTrack, SORT, or Centroid tracker support
- **Line Crossing**: Count objects crossing virtual lines
- **Performance Metrics**: Real-time pool performance monitoring
- **Metrics Export**: JSON export with automatic plot generation

## Architecture

```
Video Source
    ↓
ROI Extraction (crop to region)
    ↓
Pool Detection (parallel YOLO workers)
    ↓
Coordinate Translation (ROI → full frame)
    ↓
Object Tracking (ByteTrack/SORT/Centroid)
    ↓
Line Zone Counting
    ↓
Annotations + Metrics Overlay
    ↓
Display / Video Output
```

## Configuration

### Pool Settings (`config.yaml`)

```yaml
pool:
  pool_size: 4      # Number of parallel detector workers
  queue_size: 10    # Maximum frames to queue for processing
```

### Tracker Settings

```yaml
tracker:
  type: bytetrack  # Options: bytetrack, sort, centroid

  bytetrack:
    track_activation_threshold: 0.25
    lost_track_buffer: 30
    minimum_matching_threshold: 0.8
    minimum_consecutive_frames: 1
```

### ROI Settings

```yaml
rois:
  - id: roi_1
    x: 320       # Top-left X coordinate
    y: 20        # Top-left Y coordinate
    w: 640       # Width
    h: 640       # Height
    enabled: true
```

## Usage Examples

### Basic Usage

```bash
# Use default config.yaml
python roi_pool_detection_tracking_line_zone.py
```

### Custom Configuration

```bash
# Use custom config file
python roi_pool_detection_tracking_line_zone.py --config my_config.yaml

# Override specific parameters
python roi_pool_detection_tracking_line_zone.py \
    --model yolov8n.pt \
    --conf 0.5 \
    --pool-size 4 \
    --queue-size 20
```

### Change Tracker

```bash
# Use SORT tracker instead of ByteTrack
python roi_pool_detection_tracking_line_zone.py --tracker sort

# Use Centroid tracker
python roi_pool_detection_tracking_line_zone.py --tracker centroid
```

### Display and Output

```bash
# Enable display window
python roi_pool_detection_tracking_line_zone.py --display

# Show metrics overlay
python roi_pool_detection_tracking_line_zone.py --display --show-metrics

# Save to video file
python roi_pool_detection_tracking_line_zone.py --output result.avi
```

### Metrics Export

```bash
# Save metrics to JSON and generate plots
python roi_pool_detection_tracking_line_zone.py --save-metrics metrics.json

# This creates:
# - metrics.json (raw data)
# - metrics.png (visualization plots)
```

## Pool Performance Tuning

### Choosing Pool Size

**Rule of thumb**:
- CPU-only: 2-4 workers
- Single GPU: 4-6 workers
- Multi-GPU: 8+ workers (distribute across GPUs)

**Monitor metrics**:
- High `workers_active` = good GPU utilization
- Low `workers_active` = pool size too large or GPU bottleneck

### Queue Size

**Small queue (5-10)**:
- Lower latency
- May drop frames under load
- Use for real-time applications

**Large queue (20-50)**:
- Higher throughput
- More latency
- Use for offline processing

**Monitor metrics**:
- `queue_size_current` near `queue_size_max` = increase queue size
- `frames_dropped` increasing = queue full, increase size or pool_size

### Bottleneck Identification

From metrics overlay or JSON export:

| Symptom | Cause | Solution |
|---------|-------|----------|
| High queue, low workers | Detection slow | Increase pool_size |
| Low queue, high workers | Source slow | Faster video source |
| Increasing dropped frames | Queue overflow | Increase queue_size |
| Workers < pool_size | Very fast detection | Reduce pool_size |

## Metrics Explanation

### Frame Metrics (per-frame)

- `fps`: Frames per second throughput
- `detections`: Number of objects detected in ROI
- `tracked_objects`: Number of active tracked objects
- `line_in_count`: Objects crossed line inward
- `line_out_count`: Objects crossed line outward

### Pool Metrics (per-frame snapshot)

- `workers_active`: Number of active detector workers
- `queue_size_current`: Current frames in queue
- `queue_size_max`: Maximum queue capacity
- `frames_dropped_total`: Cumulative dropped frames
- `frames_reordered_total`: Cumulative reordered frames
- `avg_queue_time_ms`: Average time frames wait in queue
- `avg_inference_time_ms`: Average detection time per worker
- `avg_reorder_delay_ms`: Average time spent reordering

### Tracker Metrics

- `inference_time_ms`: Detection inference time
- `tracking_time_ms`: Tracking processing time

## Performance Expectations

### Throughput

| Setup | Expected FPS |
|-------|-------------|
| Single detector (baseline) | 20-30 |
| Pool (size=2) | 35-50 |
| Pool (size=4) | 60-100 |
| Pool (size=8) | 120-180 |

*Actual performance depends on model size, GPU, and resolution*

### Memory Usage

- Each YOLO model: ~500MB-2GB GPU VRAM
- Each queued frame: ~2-10MB RAM
- **Formula**: `Total VRAM ≈ model_size × pool_size`

**Example**: YOLOv8n (500MB) with pool_size=4:
- VRAM usage: ~2GB
- Safe for 6GB+ GPUs

## Troubleshooting

### High Frame Drops

**Problem**: `frames_dropped` increasing rapidly

**Solutions**:
1. Increase `queue_size` in config
2. Increase `pool_size` to process faster
3. Reduce video resolution or frame rate
4. Use smaller YOLO model (e.g., yolov8n instead of yolov8x)

### Low FPS Despite Multiple Workers

**Problem**: `workers_active` = pool_size but FPS still low

**Solutions**:
1. Check GPU utilization (may be maxed out)
2. Reduce pool_size if GPU is bottleneck
3. Use lighter YOLO model
4. Enable TensorRT if available

### Workers Not All Active

**Problem**: `workers_active` < `pool_size`

**Possible causes**:
- Detection is very fast (not a problem)
- Queue is empty (source too slow)
- Reorder timeout too short

**Solutions**:
- If source is slow: reduce pool_size
- If reorder timeout: increase `reorder_timeout` in code

### Memory Errors

**Problem**: CUDA out of memory

**Solutions**:
1. Reduce `pool_size`
2. Use smaller YOLO model
3. Reduce video resolution
4. Clear GPU cache: `torch.cuda.empty_cache()`

## Comparison with Single Detector

### When to Use Pool Detection

✅ **Use pool detection when:**
- Processing high-resolution video (1080p+)
- Offline processing (batch jobs)
- GPU underutilized with single detector
- Need maximum throughput

❌ **Avoid pool detection when:**
- Real-time low-latency required
- Limited GPU memory
- Video source is bottleneck
- Processing low-resolution streams

### Migration from Single Detector

Replace this:
```python
detector = sv.YOLODetectionStep(
    model_path="yolov8n.pt",
    conf=0.4
)
```

With this:
```python
detector = sv.PoolYOLODetectionStep(
    model_path="yolov8n.pt",
    pool_size=4,
    conf=0.4,
    max_queue_size=10
)
```

Everything else remains the same!

## Command-Line Reference

```
usage: roi_pool_detection_tracking_line_zone.py [-h] [--config CONFIG]
                                                 [--model MODEL] [--conf CONF]
                                                 [--display] [--show-metrics]
                                                 [--output OUTPUT]
                                                 [--save-metrics SAVE_METRICS]
                                                 [--pool-size POOL_SIZE]
                                                 [--queue-size QUEUE_SIZE]
                                                 [--tracker {bytetrack,sort,centroid}]

optional arguments:
  --config CONFIG       Path to YAML configuration file (default: config.yaml)
  --model MODEL         YOLO model path (overrides config)
  --conf CONF           Confidence threshold (overrides config)
  --display             Enable display window
  --show-metrics        Show detailed metrics overlay
  --output OUTPUT       Output video file path
  --save-metrics FILE   Save metrics to JSON file
  --pool-size N         Number of parallel detector workers (default: 4)
  --queue-size N        Maximum frames to queue (default: 10)
  --tracker TYPE        Tracker type: bytetrack, sort, centroid
```

## See Also

- `roi_detection_tracking_line_zone.py` - Single detector version
- `roi_pool_line_zone_demo.py` - Pool detection with socket logging
- `config.yaml` - Full configuration reference

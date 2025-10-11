"""
Simple YOLO detection example.

Minimal code to run YOLO detection on webcam with annotations.

Requirements:
    pip install ultralytics

Usage:
    python examples/yolo_simple.py

Note: Make sure you have a YOLO model file (e.g., yolov8n.pt) in the current directory.
You can download it from: https://github.com/ultralytics/ultralytics
"""

import supervision as sv

# Create annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Build detection pipeline
pipeline = (
    sv.Pipeline(sv.WebcamSource(camera_id=0, width=640, height=480))
    | sv.FPSCalculatorStep()
    | sv.YOLODetectionStep("yolov8n.pt", conf=0.5, device="cuda")
    | sv.AnnotationStep([box_annotator, label_annotator])
    | sv.DisplaySink("YOLO Detection", show_fps=True)
)

# Run
print("Starting YOLO detection... Press 'q' or ESC to quit")
pipeline.run()
print("Detection stopped")

"""
Simple example: Webcam capture with display and video saving.

This demonstrates the minimal code needed to capture from webcam,
display in real-time, and save to a video file.

Press 'q' or ESC to stop.
"""

import supervision as sv

# Create pipeline with webcam source
pipeline = sv.Pipeline(sv.WebcamSource(camera_id=0, width=640, height=480))

# Add FPS calculator
pipeline = pipeline | sv.FPSCalculatorStep()

# Add multi-sink for simultaneous display and save
pipeline = pipeline | sv.MultiSink(
    [
        sv.DisplaySink("Webcam", show_fps=True),
        sv.VideoFileSink("output.mp4", fps=30, width=640, height=480),
    ]
)

# Run
print("Recording... Press 'q' or ESC to stop")
pipeline.run()
print("Video saved to: output.mp4")

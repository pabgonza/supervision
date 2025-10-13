# Camera Calibration

This example demonstrates how to perform camera calibration using a chessboard pattern to correct lens distortion in images and video streams.

## Overview

Camera calibration is the process of estimating the intrinsic and extrinsic parameters of a camera. These parameters are essential for:

- Correcting lens distortion
- Accurate 3D reconstruction
- Object measurement in images
- Augmented reality applications
- Accurate object detection and tracking

### Types of Distortion

Modern cameras, especially those with wide-angle lenses, introduce distortions:

1. **Radial Distortion**: Causes straight lines to appear curved, especially near the edges
   - Barrel distortion (negative radial distortion)
   - Pincushion distortion (positive radial distortion)

2. **Tangential Distortion**: Occurs when the lens is not perfectly aligned with the image sensor

### Calibration Parameters

The calibration process determines:

- **Camera Matrix (Intrinsic Parameters)**:
  - `fx`, `fy`: Focal lengths in pixel units
  - `cx`, `cy`: Optical center (principal point)

- **Distortion Coefficients**:
  - `k1`, `k2`, `k3`: Radial distortion coefficients
  - `p1`, `p2`: Tangential distortion coefficients

## Requirements

```bash
pip install opencv-python numpy
```

## Materials Needed

1. **Chessboard Pattern**: A printed chessboard calibration pattern
   - Download: [OpenCV Chessboard Pattern](https://github.com/opencv/opencv/blob/4.x/doc/pattern.png)
   - Standard size: 9x6 inner corners (10x7 squares)
   - Print on flat, rigid surface (mount on cardboard/foam board)
   - Ensure pattern is perfectly flat

2. **Camera**: The camera you want to calibrate

3. **Good Lighting**: Uniform lighting without glare on the pattern

## Preparing Calibration Images

### Guidelines for Capturing Images

1. **Image Quantity**: Capture 15-30 images
2. **Pattern Coverage**: Fill the frame with the chessboard in different positions
3. **Varied Angles**: Rotate and tilt the pattern (or camera) at different angles
4. **Edge Coverage**: Include images where the pattern is near the edges and corners
5. **Focus**: Keep the pattern in focus and sharp
6. **Avoid Motion Blur**: Use sufficient lighting to keep shutter speed fast

### Recommended Positions

Capture images with the pattern in:
- Center of frame
- All four corners
- All four edges
- Various tilt angles (0°, ±15°, ±30°)
- Various rotation angles
- Different distances from camera

### Example Capture Session

```
calibration_images/
├── img_001.jpg  # Center, straight on
├── img_002.jpg  # Top left corner
├── img_003.jpg  # Top right corner
├── img_004.jpg  # Bottom left corner
├── img_005.jpg  # Bottom right corner
├── img_006.jpg  # Top edge, tilted
├── img_007.jpg  # Left edge, tilted
├── img_008.jpg  # Right edge, tilted
├── img_009.jpg  # Bottom edge, tilted
├── img_010.jpg  # Center, rotated 15°
├── ...
└── img_020.jpg
```

## Usage

### Basic Calibration

```bash
python calibrate_camera.py \
    --images ./calibration_images \
    --width 9 \
    --height 6 \
    --output camera_calibration.json
```

### Arguments

- `--images`: Directory containing calibration images (required)
- `--width`: Number of inner corners along the width (required)
- `--height`: Number of inner corners along the height (required)
- `--square-size`: Size of chessboard squares in your preferred unit (default: 1.0)
- `--output`: Output JSON file for calibration coefficients (default: camera_calibration.json)
- `--show-corners`: Display detected corners during processing
- `--test-image`: Test calibration on a specific image

### Example with Visualization

```bash
python calibrate_camera.py \
    --images ./calibration_images \
    --width 9 \
    --height 6 \
    --show-corners \
    --test-image ./calibration_images/img_001.jpg \
    --output my_camera.json
```

## Step-by-Step Procedure

### Step 1: Download and Print Chessboard

1. Download the chessboard pattern:
   ```bash
   wget https://github.com/opencv/opencv/blob/4.x/doc/pattern.png
   ```

2. Print the pattern on A4 or Letter size paper

3. Mount on rigid surface (cardboard, foam board, or acrylic sheet)

4. Ensure the pattern is completely flat

5. Count the **inner corners** (not squares):
   - Standard OpenCV pattern: 9 wide × 6 high inner corners

### Step 2: Capture Calibration Images

1. Create a directory for images:
   ```bash
   mkdir calibration_images
   ```

2. Capture 15-30 images following the guidelines above

3. Save images with sequential naming (img_001.jpg, img_002.jpg, etc.)

### Step 3: Run Calibration

1. Run the calibration script:
   ```bash
   python calibrate_camera.py \
       --images ./calibration_images \
       --width 9 \
       --height 6 \
       --show-corners
   ```

2. The script will:
   - Detect chessboard corners in each image
   - Display progress for each image
   - Calculate calibration parameters
   - Display calibration results
   - Save coefficients to JSON file

### Step 4: Verify Results

1. Check the re-projection error:
   - Good calibration: error < 0.5 pixels
   - Acceptable: error < 1.0 pixels
   - If error > 1.0, consider recapturing images

2. Test on a sample image:
   ```bash
   python calibrate_camera.py \
       --images ./calibration_images \
       --width 9 \
       --height 6 \
       --test-image ./calibration_images/img_001.jpg
   ```

3. Visually inspect the undistorted image:
   - Straight lines should appear straight
   - Edges should look more natural
   - No excessive warping

## Using Calibration Results

### Load Calibration

```python
from calibrate_camera import CameraCoefficients, undistort_image
import cv2

# Load calibration
coeffs = CameraCoefficients.load("camera_calibration.json")
print(coeffs)

# Undistort an image
image = cv2.imread("test_image.jpg")
undistorted = undistort_image(image, coeffs, crop=True)
cv2.imwrite("undistorted.jpg", undistorted)
```

### Real-time Video Undistortion

```python
import cv2
from calibrate_camera import CameraCoefficients, undistort_image

# Load calibration
coeffs = CameraCoefficients.load("camera_calibration.json")

# Open video stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Undistort frame
    undistorted = undistort_image(frame, coeffs, crop=False)

    # Display
    cv2.imshow('Undistorted', undistorted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Integration with Supervision

```python
import cv2
import supervision as sv
from calibrate_camera import CameraCoefficients, undistort_image

# Load calibration
coeffs = CameraCoefficients.load("camera_calibration.json")

# Process video
def callback(frame: np.ndarray, _: int) -> np.ndarray:
    # Undistort frame
    frame = undistort_image(frame, coeffs, crop=False)

    # Your detection/tracking pipeline here
    # ...

    return frame

sv.process_video(
    source_path="input.mp4",
    target_path="output.mp4",
    callback=callback
)
```

## Troubleshooting

### No Corners Detected

- **Issue**: Script reports "Failed to find corners"
- **Solutions**:
  - Verify corner count matches your pattern (width × height)
  - Ensure better lighting (avoid shadows and glare)
  - Check that pattern is in focus
  - Try different camera settings
  - Ensure pattern is fully visible in frame

### High Re-projection Error

- **Issue**: Error > 1.0 pixels
- **Solutions**:
  - Capture more images (aim for 25-30)
  - Ensure better coverage of frame edges
  - Include more varied angles
  - Check for motion blur in images
  - Ensure pattern is perfectly flat
  - Retake images with better lighting

### Poor Undistortion Results

- **Issue**: Undistorted images look wrong
- **Solutions**:
  - Verify you're using correct camera/resolution
  - Recalibrate with better quality images
  - Check that pattern corner count is correct
  - Ensure pattern squares are actually square

### Pattern Not Flat

- **Issue**: Corners detected but high error
- **Solutions**:
  - Mount pattern on rigid surface
  - Use heavier backing material
  - Ensure pattern doesn't curl at edges

## Tips for Best Results

1. **Pattern Quality**:
   - Use high-quality printer
   - Mount on rigid, flat surface
   - Avoid glossy paper (causes reflections)

2. **Image Capture**:
   - Use tripod for stability
   - Capture at native camera resolution
   - Use good, uniform lighting
   - Avoid shadows on pattern
   - Keep pattern in focus

3. **Coverage**:
   - Cover entire frame in different positions
   - Include corner and edge positions
   - Vary distance from camera
   - Include various angles

4. **Number of Images**:
   - Minimum: 10 images
   - Recommended: 20-30 images
   - More images generally improve results

5. **Verification**:
   - Always check re-projection error
   - Test on real images from your application
   - Verify straight lines are corrected

## References

- [OpenCV Camera Calibration Tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [OpenCV Calibration Pattern](https://github.com/opencv/opencv/blob/4.x/doc/pattern.png)
- [Roboflow Inference Examples](https://github.com/roboflow/inference/tree/main/examples/camera-calibration)

## Advanced Usage

### Custom Square Size

If you need accurate measurements, specify the actual square size:

```bash
python calibrate_camera.py \
    --images ./calibration_images \
    --width 9 \
    --height 6 \
    --square-size 25.0  # 25mm squares
```

### Programmatic Usage

```python
from calibrate_camera import (
    extract_chessboard_corners,
    calibrate_camera,
    undistort_image,
    CameraCoefficients
)

# Extract corners
objpoints, imgpoints, image_size = extract_chessboard_corners(
    image_dir="./calibration_images",
    chessboard_size=(9, 6),
    square_size=1.0,
    show_corners=False
)

# Calibrate
coeffs = calibrate_camera(objpoints, imgpoints, image_size)

# Save
coeffs.save("my_calibration.json")

# Use
image = cv2.imread("test.jpg")
undistorted = undistort_image(image, coeffs, crop=True)
```

## Output Format

The calibration is saved as JSON:

```json
{
  "fx": 1423.45,
  "fy": 1421.89,
  "cx": 960.12,
  "cy": 540.34,
  "k1": -0.341234,
  "k2": 0.123456,
  "p1": 0.001234,
  "p2": -0.000567,
  "k3": -0.012345,
  "width": 1920,
  "height": 1080,
  "error": 0.234567
}
```

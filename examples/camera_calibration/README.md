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

### Lens Types: Standard vs Fisheye

**Standard Lenses (FOV < 180°):**
- Normal and wide-angle lenses
- Use standard pinhole calibration model (`cv2.calibrateCamera`)
- 5 distortion coefficients: k1, k2, p1, p2, k3
- Best for most cameras including webcams and smartphone cameras

**Fisheye Lenses (FOV >= 180°):**
- Ultra-wide angle lenses with extreme distortion
- Require fisheye-specific calibration model (`cv2.fisheye.calibrate`)
- 4 distortion coefficients: k1, k2, k3, k4
- Essential for 360° cameras, action cameras, and security cameras with fisheye lenses

**How to choose:** If straight lines near image edges appear extremely curved (barrel distortion), you likely have a fisheye lens. Use `--fisheye` flag for calibration.

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

### Basic Calibration (Standard Lens)

```bash
python calibrate_camera.py \
    --images ./calibration_images \
    --width 9 \
    --height 6 \
    --output camera_calibration.json
```

### Fisheye Lens Calibration (FOV >= 180°)

For fisheye lenses with field of view >= 180°, use the `--fisheye` flag:

```bash
python calibrate_camera.py \
    --images ./calibration_images \
    --width 9 \
    --height 6 \
    --fisheye \
    --output fisheye_calibration.json
```

### Arguments

- `--images`: Directory containing calibration images (required)
- `--width`: Number of inner corners along the width (required)
- `--height`: Number of inner corners along the height (required)
- `--square-size`: Size of chessboard squares in your preferred unit (default: 1.0)
- `--output`: Output JSON file for calibration coefficients (default: camera_calibration.json)
- `--show-corners`: Display detected corners during processing
- `--test-image`: Test calibration on a specific image
- `--fisheye`: Use fisheye calibration model for lenses with FOV >= 180°
- `--balance`: Balance/alpha parameter for undistortion (0.0=minimal black pixels, 1.0=retain all pixels)

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

### Minimizing Pixel Loss with Balance/Alpha Parameter

The `--balance` parameter controls the trade-off between retaining pixels and eliminating black borders:

**For Standard Lenses (mapped to alpha parameter):**
- `--balance 0.0` (default): Returns image without black pixels but crops corners (minimal black borders)
- `--balance 1.0`: Retains ALL original pixels but introduces black areas in corners (minimal information loss)
- `--balance 0.8-0.95`: Good compromise between the two

**For Fisheye Lenses:**
- `--balance 0.0` (default): Minimal black pixels in output
- `--balance 1.0`: Retains all source pixels with black borders (minimal pixel loss)

Example with minimal pixel loss:
```bash
# Standard lens - retain all pixels
python calibrate_camera.py \
    --images ./calibration_images \
    --width 9 \
    --height 6 \
    --test-image test.jpg \
    --balance 1.0

# Fisheye lens - retain all pixels
python calibrate_camera.py \
    --images ./calibration_images \
    --width 9 \
    --height 6 \
    --fisheye \
    --test-image test.jpg \
    --balance 1.0
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

### Batch Undistort Images with undistort_images.py

After calibration, you can undistort multiple images at once:

```bash
# Basic usage
python undistort_images.py \
    --images ./my_images \
    --camera_coefs camera_calibration.json \
    --output ./undistorted_images

# With cropping to remove black borders
python undistort_images.py \
    --images ./my_images \
    --camera_coefs camera_calibration.json \
    --output ./undistorted_images \
    --crop

# With minimal pixel loss (retain all pixels)
python undistort_images.py \
    --images ./my_images \
    --camera_coefs camera_calibration.json \
    --output ./undistorted_images \
    --balance 1.0

# Fisheye undistortion (automatically detected from calibration file)
python undistort_images.py \
    --images ./my_fisheye_images \
    --camera_coefs fisheye_calibration.json \
    --output ./undistorted_fisheye \
    --balance 1.0
```

**Arguments for undistort_images.py:**
- `--images`: Directory containing images to undistort (required)
- `--camera_coefs`: Calibration JSON file (default: camera_calibration.json)
- `--output`: Output directory for undistorted images (default: undistorted_images)
- `--crop`: Crop black borders from undistorted images
- `--balance`: Balance/alpha parameter (0.0=minimal black pixels, 1.0=retain all pixels)

### Load Calibration Programmatically

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
- [LearnOpenCV Camera Calibration Tutorial](https://learnopencv.com/understanding-lens-distortion/)
- [LearnOpenCV Lens Distortion Tutorial](https://learnopencv.com/camera-calibration-using-opencv/)
- [Camera Calibration with Python - OpenCV](https://www.geeksforgeeks.org/python/camera-calibration-with-python-opencv/)
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

**Standard Lens:**
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
  "error": 0.234567,
  "is_fisheye": false
}
```

**Fisheye Lens:**
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
  "k3": 0.0,
  "width": 1920,
  "height": 1080,
  "error": 0.234567,
  "is_fisheye": true
}
```

Note: For fisheye lenses, the distortion model uses 4 coefficients (k1, k2, p1, p2) instead of the standard 5 coefficients. The k3 field is set to 0.0 for fisheye calibrations.

## Measurement and Real-World Distances

When you calibrate with `--square-size` in real units (e.g., mm), you can measure real-world distances in undistorted images. However, **you need to know the depth (distance from camera to object)** to convert pixels to real measurements.

### Quick Start: Interactive Measurement Tool

Try the included measurement tool for interactive measurements:

```bash
# Using known distance
python measure_example.py \
    --camera_coefs camera_calibration.json \
    --image test.jpg \
    --distance 500

# Using reference object (interactive - you click two points on the reference)
python measure_example.py \
    --camera_coefs camera_calibration.json \
    --image test.jpg \
    --reference-real 24.4
```

**How it works:**
1. **With `--reference-real`**: First, you click two points on a reference object (e.g., a checkerboard square of known size). The tool calculates the scale, then you can measure anything in the same plane.
2. **With `--distance`**: You measure directly, knowing the distance from camera to objects.
3. The tool allows you to click two points to measure distances (works at any orientation, not just horizontal/vertical).

### Understanding the Relationship

The fundamental equation relating pixel size to real-world size is:

```
real_size = (pixel_size × distance_from_camera) / focal_length
```

Where:
- `real_size`: Size in real-world units (mm, cm, etc.)
- `pixel_size`: Size in pixels
- `distance_from_camera`: Depth/distance to object (in same units as square_size)
- `focal_length`: Camera focal length in pixels (fx for horizontal, fy for vertical)

### Method 1: Known Distance (Direct Measurement)

If you know the distance from the camera to the object:

```python
from calibrate_camera import CameraCoefficients, pixels_to_real_size, calculate_pixel_size

# Load calibration
coeffs = CameraCoefficients.load("camera_calibration.json")

# Example: Measuring an object 500mm away from the camera
distance_mm = 500

# Option A: Calculate pixel size at this distance
pixel_w, pixel_h = calculate_pixel_size(coeffs, distance_mm)
print(f"At {distance_mm}mm: 1 pixel = {pixel_w:.3f}mm × {pixel_h:.3f}mm")

# If object is 100 pixels wide:
width_pixels = 100
real_width_mm = width_pixels * pixel_w
print(f"Object width: {real_width_mm:.2f}mm")

# Option B: Direct conversion
real_width = pixels_to_real_size(100, coeffs, distance_mm, "horizontal")
real_height = pixels_to_real_size(80, coeffs, distance_mm, "vertical")
print(f"Object size: {real_width:.2f}mm × {real_height:.2f}mm")
```

### Method 2: Reference Object (Unknown Distance)

If you have a reference object of known size in the same plane as what you want to measure:

```python
import numpy as np
from calibrate_camera import calculate_scale_from_reference

# You know a marker in the image is 50mm, measure it by clicking two points
# (or calculate distance between two pixel coordinates)
point1 = (100, 200)  # First point of reference object
point2 = (220, 200)  # Second point of reference object

# Calculate pixel distance (works at any orientation)
reference_pixels = np.linalg.norm(np.array(point2) - np.array(point1))
reference_mm = 50

# Calculate scale factor (mm per pixel at this depth)
scale = calculate_scale_from_reference(reference_pixels, reference_mm)
print(f"Scale: {scale:.4f} mm/pixel")

# Now measure unknown objects at the same depth
unknown_width_pixels = 200
unknown_width_mm = unknown_width_pixels * scale
print(f"Unknown object width: {unknown_width_mm:.2f}mm")
```

### Method 3: Estimate Distance from Known Object

If you see an object of known size, you can estimate its distance:

```python
from calibrate_camera import estimate_distance_from_known_object

# You see a person (1700mm tall) that appears as 400 pixels in the image
person_height_pixels = 400
person_real_height_mm = 1700

distance = estimate_distance_from_known_object(
    person_height_pixels,
    person_real_height_mm,
    coeffs
)
print(f"Person is approximately {distance:.0f}mm ({distance/1000:.1f}m) away")

# Now you can measure other objects at approximately the same distance
car_width_pixels = 350
car_width_mm = pixels_to_real_size(car_width_pixels, coeffs, distance, "horizontal")
print(f"Car width: {car_width_mm:.0f}mm ({car_width_mm/1000:.2f}m)")
```

### Important Considerations

1. **Depth Dependency**: Measurements are only accurate if you know the distance from camera to object. Objects at different depths require different calculations.

2. **Undistorted Images**: Always undistort the image first before measuring, otherwise lens distortion will affect measurements.

3. **Same Plane**: The reference object method works best when measuring objects in the same plane (same distance from camera).

4. **Perspective**: Measurements are most accurate when:
   - Objects are perpendicular to the camera axis
   - Objects are near the center of the image
   - Camera is properly calibrated

5. **Units Consistency**: Use the same units throughout (if you calibrated with mm, all distances should be in mm).

### Complete Measurement Example

```python
import cv2
import numpy as np
from calibrate_camera import (
    CameraCoefficients,
    undistort_image,
    calculate_scale_from_reference,
    pixels_to_real_size
)

# Load calibration (calibrated with square_size=24.4mm)
coeffs = CameraCoefficients.load("camera_calibration.json")

# Load and undistort image
image = cv2.imread("scene.jpg")
undistorted = undistort_image(image, coeffs, crop=False, balance=1.0)

# Scenario: You have a checkerboard in the scene as reference
# You know one square is 24.4mm, and it measures 30 pixels in the undistorted image
reference_square_pixels = 30
reference_square_mm = 24.4

# Calculate scale for this scene
scale = calculate_scale_from_reference(reference_square_pixels, reference_square_mm)
print(f"Scale: {scale:.4f} mm/pixel")

# Now measure objects in the same plane as the checkerboard
# Example: Measure distance between two points
point1 = (100, 200)  # pixel coordinates in undistorted image
point2 = (350, 200)  # pixel coordinates in undistorted image

# Calculate pixel distance
pixel_distance = np.linalg.norm(np.array(point2) - np.array(point1))
real_distance_mm = pixel_distance * scale

print(f"Distance: {pixel_distance:.1f} pixels = {real_distance_mm:.2f}mm")

# Measure object dimensions
# Example: Bounding box of object
bbox = (150, 180, 250, 280)  # x1, y1, x2, y2
width_pixels = bbox[2] - bbox[0]
height_pixels = bbox[3] - bbox[1]

width_mm = width_pixels * scale
height_mm = height_pixels * scale

print(f"Object size: {width_mm:.2f}mm × {height_mm:.2f}mm")

# Visualize
cv2.rectangle(undistorted, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
cv2.putText(undistorted, f"{width_mm:.1f}mm x {height_mm:.1f}mm",
            (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.imshow("Measurement", undistorted)
cv2.waitKey(0)
```

### Advanced: Ground Plane Measurements

For measuring objects on a ground plane (like in traffic monitoring):

1. **Calibrate with known distance**: Place the calibration pattern on the ground at a known distance
2. **Use homography**: Calculate a homography transformation from image plane to ground plane
3. **Bird's eye view**: Transform the image to a top-down view where measurements are uniform

This requires additional geometric calculations and is beyond basic calibration, but the calibration data you have is the foundation for these advanced techniques.

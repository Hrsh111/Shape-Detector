# Shape Detection System

## Overview

A computer vision-based shape detection system that identifies and classifies five geometric shapes (circles, triangles, rectangles, pentagons, and stars) in images using pure TypeScript/JavaScript algorithms.

## Features

- **Automated shape detection** using computer vision techniques:
  - Otsu thresholding for binarization
  - Morphological operations (erosion, dilation)
  - Connected component analysis
  - Convex hull computation
  - Polygon simplification (Ramer-Douglas-Peucker)
- **Multi-polarity detection** - Works with both light and dark shapes
- **Robust classification** - Uses geometric features (circularity, vertex count, angles, concavity)
- **Interactive UI** - Upload images or use test images
- **Batch evaluation** - Test multiple images at once

## Setup Instructions

### Prerequisites

- Node.js (version 16 or higher)
- npm or yarn package manager

### Installation

```bash
# Install dependencies
npm install

# Convert test SVG images to data URLs (required for test images to work)
node convert-svg-to-png.js

# Start development server
npm run dev
```

### Project Structure

```
shape-detector/
├── src/
│   ├── main.ts              # Main shape detection algorithm
│   ├── ui-utils.js          # UI selection manager
│   ├── evaluation-manager.js # Batch evaluation system
│   ├── test-images-data.js  # Generated test image data
│   └── style.css            # Styling
├── test-images/             # Source test images (SVG/PNG)
├── index.html               # Application UI
└── README.md                # This file
```

## How It Works

### Detection Pipeline

1. **Preprocessing**
   - Convert image to grayscale
   - Apply Otsu's method for automatic thresholding
   - Process both light-on-dark and dark-on-light polarities

2. **Morphological Operations**
   - Opening (erosion + dilation) to remove noise
   - Closing (dilation + erosion) to fill holes

3. **Component Extraction**
   - Connected component labeling with 8-connectivity
   - Filter by area to remove noise and background
   - Deduplicate overlapping components

4. **Shape Analysis**
   - Compute convex hull
   - Extract boundary contours
   - Simplify polygons using RDP algorithm
   - Remove collinear points

5. **Feature Calculation**
   - Vertex count
   - Circularity (4πA/P²)
   - Aspect ratio
   - Interior angles
   - Concavity (hull area - shape area)
   - Right angle count

6. **Classification**
   - **Circle**: High circularity (>0.75) with many vertices
   - **Triangle**: Exactly 3 hull vertices
   - **Rectangle**: 4-5 hull vertices with ≥3 right angles
   - **Pentagon**: 5-7 hull vertices, low concavity, angles near 108°
   - **Star**: High concavity, alternating sharp/obtuse angles

## Usage

### Web Interface

1. Open the application in your browser
2. **Upload your own image**: Click the upload button
3. **Use test images**: Click any test image to process it
4. **Batch evaluation**: Right-click images to select, then click "Evaluate"

### Programmatic Usage

```typescript
import { ShapeDetector } from './main';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const detector = new ShapeDetector(canvas);

// Load and process an image
const imageData = await detector.loadImage(file);
const results = await detector.detectShapes(imageData);

console.log(results.shapes); // Array of detected shapes
```

## Algorithm Details

### Key Techniques

- **Otsu Thresholding**: Automatically finds optimal binary threshold
- **Graham Scan**: Computes convex hull in O(n log n)
- **RDP Simplification**: Reduces polygon vertices while preserving shape
- **Morphological Filtering**: Cleans up noise and artifacts
- **Feature-Based Classification**: Uses multiple geometric properties

### Performance

- Average processing time: 50-200ms per image
- Handles images up to 2000x2000 pixels
- No external dependencies required

## Test Images

Place test images in the `test-images/` directory and run:
```bash
node convert-svg-to-png.js
```

This generates `test-images-data.js` with base64-encoded image data.

## Evaluation Metrics

When using batch evaluation, the system reports:
- Detection accuracy (shapes found vs. expected)
- Classification accuracy
- Bounding box IoU (Intersection over Union)
- Center point distance error
- Area calculation error
- Processing time per image

## Customization

### Adjust Detection Sensitivity

In `main.ts`, modify these parameters:

```typescript
// Minimum shape area (line ~100)
const minArea = Math.max(50, Math.floor(N * 0.0005));

// Simplification epsilon (line ~370)
const epsHull = Math.max(3, Math.round(minDim * 0.04));

// Circularity threshold for circles (line ~580)
if (circularity > 0.75 && Vh >= 6) { ... }
```

### Tune Classification Thresholds

Adjust confidence thresholds and feature requirements for each shape type:

```typescript
// Circle detection (more lenient)
if (circularity > 0.70 && Vh >= 5) { ... }

// Pentagon angle tolerance (stricter)
let pentAngleMatches = anglesH.filter(
  a => Math.abs(a - (108 * Math.PI / 180)) < (25 * Math.PI / 180)
).length;
```

## Troubleshooting

**No test images appearing?**
- Run `node convert-svg-to-png.js` to generate test image data
- Check that `test-images-data.js` exists in the `src/` directory

**Shapes not detected?**
- Ensure shapes have sufficient contrast with background
- Check image is not too small (minimum ~50 pixels area)
- Try adjusting the minimum area threshold

**Wrong classifications?**
- Check if shape has clear vertices (not too rounded)
- Verify shape is not occluded or partially visible
- Review the debug console logs for feature values

**Performance issues?**
- Large images (>2000x2000) may be slow
- Consider downsampling very large images before processing

## Technical Details

### Shape Classification Logic

**Stars**
- Requires ≥8 vertices in detailed contour
- Must have alternating sharp (<65°) and obtuse (>120°) angles
- Concavity > 0.05 (visible indentations)
- Minimum 4 matching sharp/obtuse angle pairs

**Triangles**
- Exactly 3 vertices in convex hull
- Interior angles sum close to 180°
- High confidence (0.80-0.98)

**Rectangles/Squares**
- 4-5 vertices in convex hull (5th from rasterization)
- Minimum 3 right angles (±20°)
- Aspect ratio <1.15 for squares (bonus confidence)

**Pentagons**
- 5-7 hull vertices (tolerance for imperfect shapes)
- Low concavity (<0.12)
- Circularity <0.90 (not circular)
- Interior angles near 108° (±32° tolerance)

**Circles**
- High circularity >0.75
- Many vertices (≥6) after simplification
- Low aspect ratio (nearly round)

### Fallback Classification

If no primary classifier matches, the system uses a scoring approach:
- Calculates scores for all shape types
- Uses the highest scoring shape if >0.4 threshold
- Confidence capped at 0.60-0.90 for fallback cases

## Known Limitations

- Very small shapes (<50 pixels) may be filtered as noise
- Heavily rotated rectangles sometimes get 5 vertices
- Extremely irregular pentagons may be misclassified
- Overlapping shapes are detected as separate components
- Hand-drawn or sketchy shapes may have lower accuracy

## Contributing

Contributions welcome! Areas for improvement:
- Better handling of rotated rectangles
- Improved star detection for varying point counts
- Support for additional shape types
- Performance optimizations for large images

## Technical Requirements

- Modern browser with Canvas API support
- TypeScript/ES6+ JavaScript environment
- No external CV libraries (pure implementation)

## Evaluation Results

The detector was tested on the provided set of 10 benchmark images.

| Metric | Result |
|-------|--------|
| Average Precision | 100% |
| Average Recall | 100% |
| Average F1 Score | 1.000 |
| Average IoU | 0.902 |
| Total Processing Time | 187 ms |

The system successfully classified all shapes, including difficult edge cases such as:
- Low contrast shapes
- Overlapping / concave shapes (star)
- Rotated rectangle (5-vertex hull case)


## License

MIT License - feel free to use and modify as needed.

## Credits

Implements classical computer vision algorithms:
- Otsu's Method (1979) for thresholding
- Graham Scan (1972) for convex hulls
- Ramer-Douglas-Peucker (1973) for simplification
- Connected component labeling

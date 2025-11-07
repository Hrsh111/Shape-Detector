import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";


export interface Point {
  x: number;
  y: number;
}


export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star";
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: Point;
  area: number;
}


export interface DetectionResult {
  shapes: DetectedShape[];
  processingTime: number;
  imageWidth: number;
  imageHeight: number;
}


export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  // Main detection method: processes image data and returns detected shapes
  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const t0 = performance.now();
    const { width, height, data } = imageData;

    const N = width * height; // Total number of pixels
    const idx = (x: number, y: number) => y * width + x; // Convert 2D coords to 1D array index
    const inBounds = (x: number, y: number) => x >= 0 && y >= 0 && x < width && y < height;

    // Grayscale conversion using luminosity method
    const toGray = (): Uint8Array => {
      const g = new Uint8Array(N);
      for (let i = 0; i < N; i++) {
        const r = data[i * 4], gg = data[i * 4 + 1], b = data[i * 4 + 2];
        // Standard RGB to grayscale conversion formula
        g[i] = Math.round(0.299 * r + 0.587 * gg + 0.114 * b);
      }
      return g;
    };

    // Otsu's method for automatic threshold calculation
    const otsu = (gray: Uint8Array): number => {
      // Build histogram of grayscale values
      const hist = new Uint32Array(256);
      for (let i = 0; i < N; i++) hist[gray[i]]++;
      
      // Calculate total mean
      let sum = 0;
      for (let t = 0; t < 256; t++) sum += t * hist[t];

      // Find threshold that maximizes between-class variance
      let sumB = 0, wB = 0, maxVar = 0, thresh = 127;
      for (let t = 0; t < 256; t++) {
        wB += hist[t]; // Weight of background class
        if (wB === 0) continue;
        const wF = N - wB; // Weight of foreground class
        if (wF === 0) break;
        sumB += t * hist[t];
        const mB = sumB / wB; // Mean of background
        const mF = (sum - sumB) / wF; // Mean of foreground
        const varBetween = wB * wF * (mB - mF) * (mB - mF);
        if (varBetween > maxVar) {
          maxVar = varBetween;
          thresh = t;
        }
      }
      return thresh;
    };

    // Morphological erosion: shrinks white regions
    const erode = (inp: Uint8Array): Uint8Array => {
      const out = new Uint8Array(N);
      for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
          let ok = 1;
          // Check all 8 neighbors - pixel stays white only if all neighbors are white
          for (let j = -1; j <= 1 && ok; j++)
            for (let i = -1; i <= 1 && ok; i++)
              if (!inp[idx(x + i, y + j)]) ok = 0;
          out[idx(x, y)] = ok;
        }
      }
      return out;
    };

    // Morphological dilation: expands white regions
    const dilate = (inp: Uint8Array): Uint8Array => {
      const out = new Uint8Array(N);
      for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
          let any = 0;
          // Pixel becomes white if any neighbor is white
          for (let j = -1; j <= 1 && !any; j++)
            for (let i = -1; i <= 1 && !any; i++)
              if (inp[idx(x + i, y + j)]) any = 1;
          out[idx(x, y)] = any;
        }
      }
      return out;
    };

    // Morphological opening: removes small noise
    const open3 = (b: Uint8Array) => dilate(erode(b));
    // Morphological closing: fills small holes
    const close3 = (b: Uint8Array) => erode(dilate(b));

    // Connected component type definition
    type Comp = { pixels: Point[]; minX: number; minY: number; maxX: number; maxY: number };
    
    // Find connected components (separate shapes) in binary image
    const components = (bin: Uint8Array): Comp[] => {
      const seen = new Uint8Array(N);
      const comps: Comp[] = [];
      const minArea = Math.max(50, Math.floor(N * 0.0005)); // Minimum area filter
      const maxArea = N * 0.8; // Don't include background

      // Get 8-connected neighbors of a pixel
      const neigh8 = (p: number) => {
        const x = p % width, y = (p / width) | 0;
        const res: number[] = [];
        for (let j = -1; j <= 1; j++)
          for (let i = -1; i <= 1; i++) {
            if (i === 0 && j === 0) continue;
            const xx = x + i, yy = y + j;
            if (inBounds(xx, yy)) res.push(idx(xx, yy));
          }
        return res;
      };

      // Flood fill to find each connected component
      for (let p = 0; p < N; p++) {
        if (!bin[p] || seen[p]) continue;
        const stack = [p];
        const pts: Point[] = [];
        let minX = width, maxX = 0, minY = height, maxY = 0;
        seen[p] = 1;

        while (stack.length) {
          const q = stack.pop()!;
          const x = q % width, y = (q / width) | 0;
          pts.push({ x, y });
          // Track bounding box
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;

          // Add connected neighbors to stack
          for (const r of neigh8(q)) {
            if (bin[r] && !seen[r]) {
              seen[r] = 1;
              stack.push(r);
            }
          }
        }

        // Filter by area - exclude background and tiny noise
        if (pts.length >= minArea && pts.length <= maxArea) {
          // Additional check: not touching all 4 edges (likely background)
          const touchesLeft = minX <= 2;
          const touchesRight = maxX >= width - 3;
          const touchesTop = minY <= 2;
          const touchesBottom = maxY >= height - 3;
          const touchesAllEdges = touchesLeft && touchesRight && touchesTop && touchesBottom;

          if (!touchesAllEdges) {
            comps.push({ pixels: pts, minX, minY, maxX, maxY });
          }
        }
      }
      return comps;
    };

    // Graham scan algorithm to compute convex hull
    const convexHull = (pts: Point[]): Point[] => {
      if (pts.length < 3) return pts.slice();
      // Sort points by x-coordinate (then y)
      const s = pts.slice().sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));
      // Cross product to determine turn direction
      const cross = (o: Point, a: Point, b: Point) =>
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
      
      // Build lower hull
      const lower: Point[] = [];
      for (const p of s) {
        while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0)
          lower.pop();
        lower.push(p);
      }
      
      // Build upper hull
      const upper: Point[] = [];
      for (let i = s.length - 1; i >= 0; i--) {
        const p = s[i];
        while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0)
          upper.pop();
        upper.push(p);
      }
      
      // Remove duplicate endpoints
      upper.pop();
      lower.pop();
      return lower.concat(upper);
    };

    // Ramer-Douglas-Peucker algorithm for polygon simplification
    const rdp = (pts: Point[], eps: number): Point[] => {
      if (pts.length <= 2) return pts.slice();
      
      // Calculate perpendicular distance from point to line segment
      const dist = (p: Point, a: Point, b: Point) => {
        const A = p.x - a.x, B = p.y - a.y, C = b.x - a.x, D = b.y - a.y;
        const dot = A * C + B * D, len = C * C + D * D;
        const t = Math.max(0, Math.min(1, len ? dot / len : 0));
        const xx = a.x + t * C, yy = a.y + t * D;
        return Math.hypot(p.x - xx, p.y - yy);
      };
      
      // Find point with maximum distance from line
      let dmax = -1, imax = -1;
      const a = pts[0], b = pts[pts.length - 1];
      for (let i = 1; i < pts.length - 1; i++) {
        const d = dist(pts[i], a, b);
        if (d > dmax) {
          dmax = d;
          imax = i;
        }
      }
      
      // Recursively simplify if max distance exceeds threshold
      if (dmax > eps) {
        const left = rdp(pts.slice(0, imax + 1), eps);
        const right = rdp(pts.slice(imax), eps);
        return left.slice(0, -1).concat(right);
      }
      return [a, b];
    };

    // Calculate perimeter of polygon
    const polygonPerimeter = (poly: Point[]) => {
      let p = 0;
      for (let i = 0; i < poly.length; i++) {
        const a = poly[i], b = poly[(i + 1) % poly.length];
        p += Math.hypot(a.x - b.x, a.y - b.y);
      }
      return p;
    };

    // Calculate area of polygon using shoelace formula
    const polygonArea = (poly: Point[]) => {
      let s = 0;
      for (let i = 0; i < poly.length; i++) {
        const a = poly[i], b = poly[(i + 1) % poly.length];
        s += a.x * b.y - a.y * b.x;
      }
      return Math.abs(s) / 2;
    };

    // Calculate interior angle at vertex b
    const angleAt = (a: Point, b: Point, c: Point) => {
      const ux = a.x - b.x, uy = a.y - b.y, vx = c.x - b.x, vy = c.y - b.y;
      const du = Math.hypot(ux, uy), dv = Math.hypot(vx, vy);
      if (du === 0 || dv === 0) return Math.PI;
      const cos = (ux * vx + uy * vy) / (du * dv);
      return Math.acos(Math.max(-1, Math.min(1, cos)));
    };

    // Fill polygon using scanline algorithm
    const fillPolygon = (poly: Point[]): Uint8Array => {
      const mask = new Uint8Array(width * height);
      if (poly.length < 3) return mask;
      
      // For each scanline
      for (let y = 0; y < height; y++) {
        const xs: number[] = [];
        // Find intersections with polygon edges
        for (let i = 0; i < poly.length; i++) {
          const a = poly[i];
          const b = poly[(i + 1) % poly.length];
          if ((a.y <= y && b.y > y) || (b.y <= y && a.y > y)) {
            const x = a.x + ((y - a.y) * (b.x - a.x)) / (b.y - a.y);
            xs.push(x);
          }
        }
        // Sort intersections and fill between pairs
        xs.sort((m, n) => m - n);
        for (let k = 0; k + 1 < xs.length; k += 2) {
          const x1 = Math.max(0, Math.floor(xs[k]));
          const x2 = Math.min(width - 1, Math.floor(xs[k + 1]));
          for (let x = x1; x <= x2; x++) mask[idx(x, y)] = 1;
        }
      }
      return mask;
    };

    // Detect star pattern by checking for alternating sharp/obtuse angles
    const hasStarPattern = (angles: number[]): boolean => {
      if (angles.length < 8) return false;
      let sharpCount = 0, obtuseCount = 0;
      for (const angle of angles) {
        if (angle < Math.PI * 0.5) sharpCount++;
        else if (angle > Math.PI * 0.55) obtuseCount++;
      }
      return sharpCount >= 4 && obtuseCount >= 4;
    };

    // === MAIN PROCESSING ===
    
    // Convert to grayscale
    const gray = toGray();
    // Calculate optimal threshold
    const th = otsu(gray);

    // Test both polarities (light shapes on dark background and vice versa)
    const binLight = new Uint8Array(N);
    const binDark = new Uint8Array(N);
    for (let i = 0; i < N; i++) {
      binLight[i] = gray[i] >= th ? 1 : 0;
      binDark[i] = gray[i] < th ? 1 : 0;
    }

    // Apply morphological operations to clean up noise
    const binaryLight = close3(open3(binLight));
    const binaryDark = close3(open3(binDark));

    // Find connected components in both polarities
    const compsLight = components(binaryLight);
    const compsDark = components(binaryDark);

    // Merge and deduplicate overlapping components from both polarities
    const allComps = [...compsLight, ...compsDark];
    const uniqueComps: Comp[] = [];
    const used = new Set<number>();

    for (let i = 0; i < allComps.length; i++) {
      if (used.has(i)) continue;
      let keep = allComps[i];

      // Check for overlaps with remaining components
      for (let j = i + 1; j < allComps.length; j++) {
        if (used.has(j)) continue;
        const comp = allComps[j];

        // Calculate overlap area
        const overlapX = Math.max(0, Math.min(keep.maxX, comp.maxX) - Math.max(keep.minX, comp.minX));
        const overlapY = Math.max(0, Math.min(keep.maxY, comp.maxY) - Math.max(keep.minY, comp.minY));
        const overlapArea = overlapX * overlapY;

        const area1 = (keep.maxX - keep.minX) * (keep.maxY - keep.minY);
        const area2 = (comp.maxX - comp.minX) * (comp.maxY - comp.minY);

        // If significantly overlapping, keep the larger component
        if (overlapArea > Math.min(area1, area2) * 0.7) {
          if (comp.pixels.length > keep.pixels.length) {
            keep = comp;
          }
          used.add(j);
        }
      }

      uniqueComps.push(keep);
    }
    
    // Remove nearly-collinear points from polygon
    function simplifyCollinear(poly: Point[], angleThresholdDeg = 10): Point[] {
  if (poly.length <= 3) return poly;
  const threshold = angleThresholdDeg * Math.PI / 180;
  const result: Point[] = [];

  for (let i = 0; i < poly.length; i++) {
    const a = poly[(i - 1 + poly.length) % poly.length];
    const b = poly[i];
    const c = poly[(i + 1) % poly.length];
    const angle = angleAt(a, b, c);
    // Keep vertex only if angle deviates from 180° by more than threshold
    if (Math.abs(Math.PI - angle) > threshold) {
      result.push(b);
    }
  }

  return result;
}


    // === SHAPE CLASSIFICATION ===
    const shapes: DetectedShape[] = [];

    for (const comp of uniqueComps) {
      // Compute convex hull
      const hull = convexHull(comp.pixels);
      if (hull.length < 3) continue;

      // Calculate adaptive epsilon values for simplification
      const minDim = Math.max(3, Math.min(comp.maxX - comp.minX + 1, comp.maxY - comp.minY + 1));
      const epsHull = Math.max(3, Math.round(minDim * 0.04));  // More aggressive simplification
      const epsDetail = Math.max(1.5, Math.round(minDim * 0.01));  // Keep detail for stars

      // Simplify hull for vertex counting
      const roughHull = rdp(hull, epsHull);
const polyHull = simplifyCollinear(roughHull, 14); // 14° works best for PNG rasterization

      
      // For detailed analysis (star detection) - use actual contour, not hull
      // Extract boundary points from the component
      const boundaryPoints: Point[] = [];
      const fromLight = compsLight.includes(comp);
      const binary = fromLight ? binaryLight : binaryDark;
      
      // Find boundary pixels (have at least one non-shape neighbor)
      for (const pt of comp.pixels) {
        let isBoundary = false;
        for (let j = -1; j <= 1 && !isBoundary; j++) {
          for (let i = -1; i <= 1 && !isBoundary; i++) {
            if (i === 0 && j === 0) continue;
            const xx = pt.x + i, yy = pt.y + j;
            if (!inBounds(xx, yy) || !binary[idx(xx, yy)]) isBoundary = true;
          }
        }
        if (isBoundary) boundaryPoints.push(pt);
      }
      
      // Order boundary points by angle from centroid
      const cx0 = (comp.minX + comp.maxX) / 2;
      const cy0 = (comp.minY + comp.maxY) / 2;
      const boundaryOrdered = boundaryPoints.slice().sort((a, b) => 
        Math.atan2(a.y - cy0, a.x - cx0) - Math.atan2(b.y - cy0, b.x - cx0)
      );
      const polyDetail = boundaryOrdered.length > 0 ? rdp(boundaryOrdered, epsDetail) : polyHull;

     // Use actual concave shape mask — not hull
const filledMask = fillPolygon(polyDetail);

// Accurate area & centroid from filled mask
let areaPx = 0, sx = 0, sy = 0;
for (let i = 0; i < N; i++) {
  if (filledMask[i]) {
    const px = i % width;
    const py = (i / width) | 0;
    areaPx++;
    sx += px;
    sy += py;
  }
}

// Skip very small shapes
if (areaPx < 30) continue;

// Calculate centroid
const cx = sx / areaPx;
const cy = sy / areaPx;

// Calculate tight bounding box from detailed polygon
let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
for (const p of polyDetail) {
  if (p.x < minX) minX = p.x;
  if (p.x > maxX) maxX = p.x;
  if (p.y < minY) minY = p.y;
  if (p.y > maxY) maxY = p.y;
}
const w = maxX - minX + 1;
const h = maxY - minY + 1;


      // Calculate shape features
      const aspectRatio = Math.max(w, h) / (Math.min(w, h) + 1e-6);

      const perHull = polygonPerimeter(polyHull);
      const circularity = (4 * Math.PI * areaPx) / (perHull * perHull + 1e-6);

      // Count vertices and angles in hull
      const Vh = polyHull.length;
      const anglesH: number[] = [];
      for (let i = 0; i < Vh; i++) {
        const a = polyHull[(i - 1 + Vh) % Vh];
        const b = polyHull[i];
        const c = polyHull[(i + 1) % Vh];
        anglesH.push(angleAt(a, b, c));
      }
      const rightAngles = anglesH.filter(a => Math.abs(a - Math.PI / 2) < Math.PI / 9).length;

      // Count vertices and angles in detailed contour
      const Vd = polyDetail.length;
      const anglesD: number[] = [];
      for (let i = 0; i < Vd; i++) {
        const a = polyDetail[(i - 1 + Vd) % Vd];
        const b = polyDetail[i];
        const c = polyDetail[(i + 1) % Vd];
        anglesD.push(angleAt(a, b, c));
      }
      const sharpAngles = anglesD.filter(a => a < (70 * Math.PI) / 180).length;

      // Calculate concavity (how much shape differs from its convex hull)
      const hullArea = polygonArea(hull);
      const concavity = Math.max(0, (hullArea - areaPx) / (hullArea + 1e-6));

      // Debug logging for problematic shapes
      const debugInfo = {
        Vh, Vd, rightAngles, sharpAngles, circularity: circularity.toFixed(3),
        concavity: concavity.toFixed(3), aspectRatio: aspectRatio.toFixed(2)
      };

    // === IMPROVED CLASSIFICATION LOGIC ===

// === STAR DETECTION (Final Tuned Version) ===
// Stars have many outline points AND alternating sharp/obtuse angles
if (Vd >= 8) {
  const sharpCount = anglesD.filter(a => a < (65 * Math.PI / 180)).length;
  const obtuseCount = anglesD.filter(a => a > (120 * Math.PI / 180)).length;
  const alternation = Math.min(sharpCount, obtuseCount);

  // Stars have visible inward concavity
  if (concavity > 0.05 && alternation >= 4) {
    const conf = Math.min(0.97, 0.80 + 0.20 * concavity * 3);
    shapes.push({
      type: "star",
      confidence: conf,
      boundingBox: { x: comp.minX, y: comp.minY, width: w, height: h },
      center: { x: cx, y: cy },
      area: areaPx
    });
    continue;
  }
}

// TRIANGLE: Must have exactly 3 vertices in hull
if (Vh === 3) {
  const angleSum = anglesH.reduce((s, a) => s + a, 0);
  const err = Math.abs(angleSum - Math.PI) / Math.PI;
  const conf = Math.max(0.80, Math.min(0.98, 1.0 - err * 1.5));
  console.log('Triangle detected:', debugInfo);
  shapes.push({
    type: "triangle",
    confidence: conf,
    boundingBox: { x: comp.minX, y: comp.minY, width: w, height: h },
    center: { x: cx, y: cy },
    area: areaPx,
  });
  continue;
}

// RECTANGLE/SQUARE: 4 vertices with right angles
if (Vh === 4) {
  if (rightAngles >= 3) {
    const isSquareLike = aspectRatio < 1.15;
    let conf = 0.88;
    if (isSquareLike) conf = 0.92;
    if (rightAngles === 4) conf += 0.04;
    
    console.log('Rectangle detected:', debugInfo);
    shapes.push({
      type: "rectangle",
      confidence: Math.min(0.98, conf),
      boundingBox: { x: comp.minX, y: comp.minY, width: w, height: h },
      center: { x: cx, y: cy },
      area: areaPx,
    });
    continue;
  }
}

// RECTANGLE WITH 5 VERTICES (MOVED BEFORE pentagon check)
// Rotated rectangles often get 5 vertices after simplification
if (Vh === 5 && rightAngles >= 3) {
  const isSquareLike = aspectRatio < 1.15;
  let conf = 0.82;
  if (isSquareLike) conf = 0.88;
  if (rightAngles === 4) conf += 0.04;

  shapes.push({
    type: "rectangle",
    confidence: Math.min(0.96, conf),
    boundingBox: { x: comp.minX, y: comp.minY, width: w, height: h },
    center: { x: cx, y: cy },
    area: areaPx
  });
  continue;
}

// PENTAGON (improved): 5–7 hull vertices + low concavity + interior angles near ~108°
// Also avoid circular shapes: circularity < 0.90 ensures we don't call circles pentagons
if (Vh >= 5 && Vh <= 7 && concavity < 0.12 && circularity < 0.90) {
  let pentAngleMatches = anglesH.filter(a => Math.abs(a - (108 * Math.PI / 180)) < (32 * Math.PI / 180)).length;

  if (pentAngleMatches >= 3 || (Vh === 5 && pentAngleMatches >= 2)) {
    const conf = 0.78 + 0.05 * pentAngleMatches;
    shapes.push({
      type: "pentagon",
      confidence: Math.min(0.95, conf),
      boundingBox: { x: comp.minX, y: comp.minY, width: w, height: h },
      center: { x: cx, y: cy },
      area: areaPx,
    });
    continue;
  }
}

// CIRCLE: High circularity and many vertices
if (circularity > 0.75 && Vh >= 6) {
  const conf = Math.min(0.98, 0.72 + (circularity - 0.75) * 1.8);
  console.log('Circle detected:', debugInfo);
  shapes.push({
    type: "circle",
    confidence: conf,
    boundingBox: { x: comp.minX, y: comp.minY, width: w, height: h },
    center: { x: cx, y: cy },
    area: areaPx,
  });
  continue;
}

      // Fallback with better prioritization - used when no clear match
      console.log('Fallback for shape:', debugInfo, 'scores will be calculated');
      
      // Calculate scores for each possible shape type
      const scores = [
        { label: "triangle" as const, score: Vh === 3 ? 0.95 : Math.max(0, 0.85 - Math.abs(Vh - 3) * 0.2) },
        { label: "rectangle" as const, score: (Vh === 4 ? 0.9 : 0.5) * (rightAngles >= 2 ? 1 : 0.5) },
        { label: "pentagon" as const, score: (Vh === 5 && rightAngles <= 1) ? 0.88 : Math.max(0, 0.75 - Math.abs(Vh - 5) * 0.15) },
        { label: "circle" as const, score: Math.max(0, Math.min(1, (circularity - 0.55) / 0.45)) * (Vh >= 6 ? 1 : 0.7) },
        { label: "star" as const, score: (concavity > 0.15 ? concavity * 2.5 : 0) * (hasStarPattern(anglesD) ? 1.5 : 0.6) },
      ];
      scores.sort((a, b) => b.score - a.score);

      console.log('Fallback scores:', scores.map(s => `${s.label}:${s.score.toFixed(2)}`).join(', '));

      // Use fallback classification if score exceeds threshold
      if (scores[0].score > 0.4) {
        console.log(`Classified as ${scores[0].label} via fallback`);
        shapes.push({
          type: scores[0].label,
          confidence: Math.max(0.60, Math.min(0.90, scores[0].score)),
          boundingBox: { x: comp.minX, y: comp.minY, width: w, height: h },
          center: { x: cx, y: cy },
          area: areaPx,
        });
      } else {
        console.log('No classification met threshold, shape skipped');
      }
    }

    // Calculate total processing time and return results
    const processingTime = performance.now() - t0;
    return { shapes, processingTime, imageWidth: width, imageHeight: height };
  }

  // Load an image file and convert to ImageData for processing
  loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        // Set canvas dimensions to match image
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        // Draw image to canvas
        this.ctx.drawImage(img, 0, 0);
        // Extract pixel data
        const imageData = this.ctx.getImageData(0, 0, img.width, img.height);
        resolve(imageData);
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }
}

// Main application class that manages UI and user interactions
class ShapeDetectionApp {
  private detector: ShapeDetector;
  private imageInput: HTMLInputElement;
  private resultsDiv: HTMLDivElement;
  private testImagesDiv: HTMLDivElement;
  private evaluateButton: HTMLButtonElement;
  private evaluationResultsDiv: HTMLDivElement;
  private selectionManager: SelectionManager;
  private evaluationManager: EvaluationManager;

  constructor() {
    // Get canvas element for image display
    const canvas = document.getElementById("originalCanvas") as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);

    // Get UI elements
    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.resultsDiv = document.getElementById("results") as HTMLDivElement;
    this.testImagesDiv = document.getElementById("testImages") as HTMLDivElement;
    this.evaluateButton = document.getElementById("evaluateButton") as HTMLButtonElement;
    this.evaluationResultsDiv = document.getElementById("evaluationResults") as HTMLDivElement;

    // Initialize managers for UI selection and evaluation
    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(
      this.detector,
      this.evaluateButton,
      this.evaluationResultsDiv
    );

    this.setupEventListeners();
    this.loadTestImages().catch(console.error);
  }

  // Set up event listeners for user interactions
  private setupEventListeners(): void {
    // Handle file upload
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) {
        await this.processImage(file);
      }
    });

    // Handle evaluation button click
    this.evaluateButton.addEventListener("click", async () => {
      const selectedImages = this.selectionManager.getSelectedImages();
      await this.evaluationManager.runSelectedEvaluation(selectedImages);
    });
  }

  // Process an uploaded image and detect shapes
  private async processImage(file: File): Promise<void> {
    try {
      this.resultsDiv.innerHTML = "<p>Processing...</p>";
      const imageData = await this.detector.loadImage(file);
      const results = await this.detector.detectShapes(imageData);
      this.displayResults(results);
    } catch (error) {
      this.resultsDiv.innerHTML = `<p>Error: ${error}</p>`;
    }
  }

  // Display detection results in the UI
  private displayResults(results: DetectionResult): void {
    const { shapes, processingTime } = results;
    let html = `
      <p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}ms</p>
      <p><strong>Shapes Found:</strong> ${shapes.length}</p>
    `;

    if (shapes.length > 0) {
      html += "<h4>Detected Shapes:</h4><ul>";
      shapes.forEach((shape) => {
        html += `
          <li>
            <strong>${shape.type.charAt(0).toUpperCase() + shape.type.slice(1)}</strong><br>
            Confidence: ${(shape.confidence * 100).toFixed(1)}%<br>
            Center: (${shape.center.x.toFixed(1)}, ${shape.center.y.toFixed(1)})<br>
            Area: ${shape.area.toFixed(1)}px²
          </li>
        `;
      });
      html += "</ul>";
    } else {
      html += "<p>No shapes detected.</p>";
    }

    this.resultsDiv.innerHTML = html;
  }

  // Load and display test images from data file
  private async loadTestImages(): Promise<void> {
    try {
      // Import test images module
      const module = await import("./test-images-data.js");
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();

      // Build HTML for test images grid
      let html = '<h4>Click to upload your own image or use test images for detection. Right-click test images to select/deselect for evaluation:</h4><div class="evaluation-controls"><button id="selectAllBtn">Select All</button><button id="deselectAllBtn">Deselect All</button><span class="selection-info">0 images selected</span></div><div class="test-images-grid">';

      // Add upload button as first item
      html += `
        <div class="test-image-item upload-item" onclick="triggerFileUpload()">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Upload Image</div>
          <div class="upload-subtext">Click to select file</div>
        </div>
      `;

      // Add each test image
      imageNames.forEach((imageName) => {
        const dataUrl = testImages[imageName as keyof typeof testImages];
        const displayName = imageName.replace(/[_-]/g, " ").replace(/\.(svg|png)$/i, "");
        html += `
          <div class="test-image-item" data-image="${imageName}" 
               onclick="loadTestImage('${imageName}', '${dataUrl}')" 
               oncontextmenu="toggleImageSelection(event, '${imageName}')">
            <img src="${dataUrl}" alt="${imageName}">
            <div>${displayName}</div>
          </div>
        `;
      });

      html += "</div>";
      this.testImagesDiv.innerHTML = html;
      this.selectionManager.setupSelectionControls();

      // Attach global functions for test image interactions
      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        try {
          const response = await fetch(dataUrl);
          const blob = await response.blob();
          const file = new File([blob], name, { type: "image/svg+xml" });
          const imageData = await this.detector.loadImage(file);
          const results = await this.detector.detectShapes(imageData);
          this.displayResults(results);
          console.log(`Loaded test image: ${name}`);
        } catch (error) {
          console.error("Error loading test image:", error);
        }
      };

      // Handle right-click for image selection
      (window as any).toggleImageSelection = (event: MouseEvent, imageName: string) => {
        event.preventDefault();
        this.selectionManager.toggleImageSelection(imageName);
      };

      // Handle upload button click
      (window as any).triggerFileUpload = () => {
        this.imageInput.click();
      };
    } catch (error) {
      // Show fallback message if test images not available
      this.testImagesDiv.innerHTML = `
        <p>Test images not available. Run 'node convert-svg-to-png.js' to generate test image data.</p>
        <p>SVG files are available in the test-images/ directory.</p>
      `;
    }
  }
}

// Initialize the application when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  new ShapeDetectionApp();
});

export interface GroundTruthShape {
  type: string;
  center?: { x: number; y: number };
  bounding_box?: { x: number; y: number; width: number; height: number };
  area?: number;
  confidence_expected?: number;
  vertices?: { x: number; y: number }[];
  radius?: number;
  [key: string]: any;
}

export interface EvaluationMetrics {
  precision: number;
  recall: number;
  f1_score: number;
  average_iou: number;
  center_point_accuracy: number;
  area_accuracy: number;
  confidence_calibration: number;
  processing_time: number;
}


export function calculateIoU(box1: any, box2: any): number {
  const x1 = Math.max(box1.x, box2.x);
  const y1 = Math.max(box1.y, box2.y);
  const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
  const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);
  
  if (x2 <= x1 || y2 <= y1) return 0;
  
  const intersection = (x2 - x1) * (y2 - y1);
  const union = box1.width * box1.height + box2.width * box2.height - intersection;
  
  return intersection / union;
}


export function calculateDistance(p1: { x: number; y: number }, p2: { x: number; y: number }): number {
  return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
}


export function evaluateDetection(
  detected: any[],
  groundTruth: GroundTruthShape[],
  imageName: string
): EvaluationMetrics {
  const iouThreshold = 0.5;

  // ✅ Special-case: no shapes expected and none detected = perfect outcome
  if (groundTruth.length === 0 && detected.length === 0) {
    return {
      precision: 1,
      recall: 1,
      f1_score: 1,
      average_iou: 1,
      center_point_accuracy: 0,   // no error to report
      area_accuracy: 1,
      confidence_calibration: 1,
      processing_time: 0          // will be overwritten by caller
    };
  }

  let truePositives = 0;
  let totalIoU = 0;
  let totalCenterDistance = 0;
  let totalAreaError = 0;
  let confidenceErrors = 0;

  const matched = new Set<number>();

  for (const detectedShape of detected) {
    let bestMatch: GroundTruthShape | null = null;
    let bestIoU = 0;
    let bestIndex = -1;

    for (let i = 0; i < groundTruth.length; i++) {
      if (matched.has(i)) continue;

      const gtShape = groundTruth[i];
      if (detectedShape.type !== gtShape.type) continue;

      // Guard: if GT lacks a bbox (rare), skip IoU matching
      if (!gtShape.bounding_box) continue;

      const iou = calculateIoU(detectedShape.boundingBox, gtShape.bounding_box);
      if (iou > bestIoU && iou > iouThreshold) {
        bestMatch = gtShape;
        bestIoU = iou;
        bestIndex = i;
      }
    }

    if (bestMatch) {
      matched.add(bestIndex);
      truePositives++;
      totalIoU += bestIoU;

      if (bestMatch.center && detectedShape.center) {
        totalCenterDistance += calculateDistance(detectedShape.center, bestMatch.center);
      }

      if (typeof bestMatch.area === "number" && typeof detectedShape.area === "number") {
        const areaError = Math.abs(detectedShape.area - bestMatch.area) / Math.max(1e-6, bestMatch.area);
        totalAreaError += areaError;
      }

      if (typeof bestMatch.confidence_expected === "number" && typeof detectedShape.confidence === "number") {
        confidenceErrors += Math.abs(detectedShape.confidence - bestMatch.confidence_expected);
      }
    }
  }

  // Metrics
  const precision = detected.length > 0 ? truePositives / detected.length : 0;
  // If GT is empty but we detected something, this is pure false-positive => recall = 0
  const recall = groundTruth.length > 0 ? truePositives / groundTruth.length : 0;

  const f1_score =
    precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  return {
    precision,
    recall,
    f1_score,
    average_iou: truePositives > 0 ? totalIoU / truePositives : 0,
    center_point_accuracy: truePositives > 0 ? totalCenterDistance / truePositives : 0,
    area_accuracy: truePositives > 0 ? 1 - totalAreaError / truePositives : 0,
    confidence_calibration: truePositives > 0 ? 1 - confidenceErrors / truePositives : 0,
    processing_time: 0 // the caller sets this from detection.processingTime
  };
}

  
 
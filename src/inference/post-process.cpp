#include "post-process.hpp"

#include <opencv2/dnn.hpp>

std::vector<Detection> postprocess(const float* outputData,
                                   const std::vector<int64_t>& outputShape,
                                   float confThreshold, float nmsThreshold,
                                   int origWidth, int origHeight,
                                   int inputWidth, int inputHeight) {
  // Validate output shape before indexing it
  if (outputShape.size() < 3) return {};

  // YOLOv8 output shape: [1, 84, 8400]
  //   84 = 4 bbox coords (cx, cy, w, h) + 80 class scores
  //   8400 = number of candidate detections
  const int numFeatures = static_cast<int>(outputShape[1]);  // 84
  const int numCandidates = static_cast<int>(outputShape[2]);  // 8400
  const int numClasses = numFeatures - 4;  // 80

  // Scale factors to map model coordinates back to original image
  float scaleX = static_cast<float>(origWidth) / inputWidth;
  float scaleY = static_cast<float>(origHeight) / inputHeight;

  // Collect candidates that pass confidence threshold
  std::vector<int> classIds;
  std::vector<float> confidences;
  std::vector<cv::Rect2f> bboxes;  // float precision for accurate NMS
  std::vector<cv::Rect> nmsBoxes;  // int version required by cv::dnn::NMSBoxes

  for (int i = 0; i < numCandidates; i++) {
    // YOLOv8 output is [1, 84, 8400] — data layout is feature-major:
    //   outputData[feature * 8400 + candidate]
    // First 4 features = bbox (cx, cy, w, h), remaining = class scores
    float cx = outputData[0 * numCandidates + i];
    float cy = outputData[1 * numCandidates + i];
    float w = outputData[2 * numCandidates + i];
    float h = outputData[3 * numCandidates + i];

    // Find best class score for this candidate
    int bestClassId = 0;
    float bestScore = 0.0f;
    for (int c = 0; c < numClasses; c++) {
      float score = outputData[(4 + c) * numCandidates + i];
      if (score > bestScore) {
        bestScore = score;
        bestClassId = c;
      }
    }

    if (bestScore < confThreshold) continue;

    // Convert center-format to top-left corner and scale to original image
    // Clamp to image boundaries to avoid out-of-bounds coords
    float x = std::max(0.0f, (cx - w / 2.0f) * scaleX);
    float y = std::max(0.0f, (cy - h / 2.0f) * scaleY);
    float bw = std::min(w * scaleX, static_cast<float>(origWidth) - x);
    float bh = std::min(h * scaleY, static_cast<float>(origHeight) - y);

    bboxes.emplace_back(x, y, bw, bh);
    // NMSBoxes requires cv::Rect (int) — round for NMS, keep float for result
    nmsBoxes.emplace_back(cv::Rect(static_cast<int>(x), static_cast<int>(y),
                                    std::max(1, static_cast<int>(bw)),
                                    std::max(1, static_cast<int>(bh))));
    confidences.push_back(bestScore);
    classIds.push_back(bestClassId);
  }

  // Apply Non-Maximum Suppression to remove duplicate detections
  std::vector<int> nmsIndices;
  cv::dnn::NMSBoxes(nmsBoxes, confidences, confThreshold, nmsThreshold,
                    nmsIndices);

  // Build final detection list with float-precision bboxes
  std::vector<Detection> results;
  results.reserve(nmsIndices.size());

  for (int idx : nmsIndices) {
    Detection det;
    det.classId = classIds[idx];
    det.confidence = confidences[idx];
    det.bbox = bboxes[idx];
    results.push_back(det);
  }

  return results;
}

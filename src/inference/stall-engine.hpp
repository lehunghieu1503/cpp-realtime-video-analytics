#pragma once

#include "backends/inference-backend.hpp"

#include <array>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct StallEngineConfig {
  std::string modelPath;
  std::string backend = "cpu";
  std::string precision = "fp16";
  int numThreads = 2;
  float confidence = 0.5f;
  float nms = 0.5f;
};

// One parking stall in a frame, in pixels of that frame
struct StallDetection {
  bool occupied = false;  // a vehicle stands in it
  float confidence = 0.0f;
  cv::Rect2f bbox;  // of the part of the stall that is in the image
  // The two corners of the mouth, then the two of the back, around the
  // stall; each with how sure the model is that it is in the image. A corner
  // it does not see still has a position, which is a guess.
  std::array<cv::Point2f, 4> corners;
  std::array<float, 4> cornerConfidence{};
};

// Parking stall detector: an Ultralytics pose model with two classes (empty,
// occupied) and four keypoints, trained by scripts/train-stall-model.py on
// the images of simulation/stall-dataset.py. A model of its own next to the
// YOLO detector, with its own backend instance:
//
//   input   [1, 3, H, W]      RGB, [0, 1]; the frame is stretched to it
//   output  [1, 18, N]        per candidate: cx, cy, w, h, 2 class scores,
//                             4 x (x, y, confidence), in pixels of the input
class StallEngine {
public:
  static constexpr int kClasses = 2;
  static constexpr int kCorners = 4;
  static constexpr int kFeatures = 4 + kClasses + 3 * kCorners;

  explicit StallEngine(const StallEngineConfig& config);
  bool isLoaded() const { return backend_ != nullptr; }

  // Throws when inference fails
  std::vector<StallDetection> detect(const cv::Mat& bgr,
                                     double* preMs = nullptr,
                                     double* inferMs = nullptr,
                                     double* postMs = nullptr);

  // The output tensor to stalls of a width x height frame, after
  // class-agnostic NMS: a stall is found once, empty or occupied. Throws on
  // a shape that is not [1, 18, N].
  static std::vector<StallDetection> decode(
      const float* output, const std::vector<int64_t>& shape,
      float confidence, float nms, int width, int height, int inputWidth,
      int inputHeight);

private:
  StallEngineConfig config_;
  std::unique_ptr<IInferenceBackend> backend_;
  int inputWidth_ = 0;
  int inputHeight_ = 0;
};

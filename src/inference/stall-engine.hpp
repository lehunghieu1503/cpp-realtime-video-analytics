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

// Where a divider between two stalls ends, in pixels of the frame
struct StallPoint {
  // false: its end at the mouth of the stalls; true: where it meets the
  // line across at their back
  bool back = false;
  float confidence = 0.0f;
  cv::Point2f at;
};

struct StallResult {
  std::vector<StallDetection> stalls;
  std::vector<StallPoint> points;
};

// Parking stall detector: an Ultralytics pose model with four classes and
// four keypoints, trained by scripts/train-stall-model.py on the images of
// simulation/stall-dataset.py. A model of its own next to the YOLO
// detector, with its own backend instance:
//
//   input   [1, 3, H, W]      RGB, [0, 1]; the frame is stretched to it
//   output  [1, 20, N]        per candidate: cx, cy, w, h, 4 class scores,
//                             4 x (x, y, confidence), in pixels of the input
//
// Classes 0 and 1 are a stall, empty or occupied, with its four corners.
// Classes 2 and 3 are the end of a divider, at the mouth or at the back; its
// four keypoints are all that one place. A stall is large and its corners
// are far from its middle: the model puts them roughly. An end is small and
// is put sharply: stalls are measured by the ends.
class StallEngine {
public:
  static constexpr int kClasses = 4;
  static constexpr int kCorners = 4;
  static constexpr int kFeatures = 4 + kClasses + 3 * kCorners;

  explicit StallEngine(const StallEngineConfig& config);
  bool isLoaded() const { return backend_ != nullptr; }

  // Throws when inference fails
  StallResult detect(const cv::Mat& bgr, double* preMs = nullptr,
                     double* inferMs = nullptr, double* postMs = nullptr);

  // The output tensor to the stalls and divider ends of a width x height
  // frame. NMS runs among the stalls, whatever their class: a stall is found
  // once, empty or occupied; and among the ends likewise. Throws on a shape
  // that is not [1, 20, N].
  static StallResult decode(const float* output,
                            const std::vector<int64_t>& shape,
                            float confidence, float nms, int width, int height,
                            int inputWidth, int inputHeight);

private:
  StallEngineConfig config_;
  std::unique_ptr<IInferenceBackend> backend_;
  int inputWidth_ = 0;
  int inputHeight_ = 0;
};

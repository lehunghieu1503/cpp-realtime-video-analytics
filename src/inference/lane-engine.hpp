#pragma once

#include "backends/inference-backend.hpp"

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct LaneEngineConfig {
  std::string modelPath;
  std::string backend = "cpu";
  std::string precision = "fp16";
  int numThreads = 2;
};

// Lane detector: the network of Ultra Fast Lane Detection, ResNet-18, as
// scripts/train-lane-model.py trains it on the simulator's images or as
// scripts/export-lane-model.py exports the authors' TuSimple weights; both
// have this input and output. A model of its own next to the YOLO
// detector, with its own backend instance:
//
//   input   [1, 3, 288, 800]  RGB, [0, 1], then (x - mean) / std
//   output  [1, 101, 56, 4]   per lane and row anchor: 100 column bins and
//                             "no lane"
//
// The input is not the detector's: this one is normalised with the ImageNet
// mean and deviation, so it always goes through the tensor path of the
// backend, never through a backend's own frame pre-processing.
class LaneEngine {
public:
  static constexpr int kInputWidth = 800;
  static constexpr int kInputHeight = 288;
  static constexpr int kGrid = 100;  // column bins; one more means no lane
  static constexpr int kRows = 56;   // row anchors: 64, 68, ..., 284 of 288
  static constexpr int kLanes = 4;

  explicit LaneEngine(const LaneEngineConfig& config);
  bool isLoaded() const { return backend_ != nullptr; }

  // Lanes in a BGR frame, as pixels of that frame: one list per lane, empty
  // when the lane is not there. Throws when inference fails.
  std::vector<std::vector<cv::Point>> detect(const cv::Mat& bgr,
                                             double* preMs = nullptr,
                                             double* inferMs = nullptr,
                                             double* postMs = nullptr);

  // The model's input for a frame: NCHW float32, 3 x 288 x 800
  static std::vector<float> preprocess(const cv::Mat& bgr);
  // Logits [101][56][4] to pixels of a width x height image, as the model's
  // authors decode them (demo.py): rows reversed, columns counted from 1, a
  // lane with fewer than 3 points dropped. Throws on a value that is not
  // finite.
  static std::vector<std::vector<cv::Point>> decode(const float* logits,
                                                    int width, int height);

private:
  std::unique_ptr<IInferenceBackend> backend_;
};

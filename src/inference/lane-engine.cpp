#include "lane-engine.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace {
const std::vector<int64_t> kInputShape = {1, 3, LaneEngine::kInputHeight,
                                          LaneEngine::kInputWidth};
const std::vector<int64_t> kOutputShape = {1, LaneEngine::kGrid + 1,
                                           LaneEngine::kRows,
                                           LaneEngine::kLanes};
// ImageNet statistics, in R, G, B order
const float kMean[3] = {0.485f, 0.456f, 0.406f};
const float kStd[3] = {0.229f, 0.224f, 0.225f};

int rowAnchor(int index) { return 64 + 4 * index; }
}  // namespace

LaneEngine::LaneEngine(const LaneEngineConfig& config) {
  BackendConfig backendConfig;
  backendConfig.modelPath = config.modelPath;
  backendConfig.backend = config.backend;
  backendConfig.precision = config.precision;
  backendConfig.numThreads = config.numThreads;
  auto backend = createBackend(backendConfig);
  if (!backend) return;

  if (backend->inputShape() != kInputShape) {
    std::cerr << "Lane model: " << config.modelPath
              << " does not take a 1x3x288x800 input\n";
    return;
  }
  // One run on an empty image shows the output is the one decode() expects
  try {
    std::vector<float> zeros(3 * kInputHeight * kInputWidth, 0.0f);
    std::vector<int64_t> outputShape;
    backend->run(zeros.data(), kInputShape, outputShape);
    if (outputShape != kOutputShape) {
      std::cerr << "Lane model: " << config.modelPath
                << " does not give a 1x101x56x4 output\n";
      return;
    }
  } catch (const std::exception& e) {
    std::cerr << "Lane model: " << e.what() << "\n";
    return;
  }
  backend_ = std::move(backend);
  std::cout << "Lane model loaded: " << config.modelPath << " (backend "
            << config.backend << ")\n";
}

std::vector<float> LaneEngine::preprocess(const cv::Mat& bgr) {
  // The authors resize with PIL, which averages when it shrinks; INTER_AREA
  // is the closest OpenCV has
  cv::Mat resized;
  bool shrinks = bgr.cols > kInputWidth || bgr.rows > kInputHeight;
  cv::resize(bgr, resized, cv::Size(kInputWidth, kInputHeight), 0, 0,
             shrinks ? cv::INTER_AREA : cv::INTER_LINEAR);

  const size_t plane = static_cast<size_t>(kInputWidth) * kInputHeight;
  std::vector<float> tensor(3 * plane);
  for (int y = 0; y < kInputHeight; y++) {
    const uchar* row = resized.ptr<uchar>(y);
    size_t at = static_cast<size_t>(y) * kInputWidth;
    for (int x = 0; x < kInputWidth; x++, at++) {
      // BGR in the frame, RGB planes in the tensor
      for (int c = 0; c < 3; c++) {
        tensor[c * plane + at] =
            (row[x * 3 + (2 - c)] / 255.0f - kMean[c]) / kStd[c];
      }
    }
  }
  return tensor;
}

std::vector<std::vector<cv::Point>> LaneEngine::decode(const float* logits,
                                                       int width, int height) {
  const double step = (kInputWidth - 1) / static_cast<double>(kGrid - 1);
  std::vector<std::vector<cv::Point>> lanes(kLanes);
  // logits[g][r][l]
  auto at = [logits](int g, int r, int l) {
    return static_cast<double>(logits[(g * kRows + r) * kLanes + l]);
  };

  for (int l = 0; l < kLanes; l++) {
    // k counts rows from the bottom of the image: the model's last row first
    double column[kRows];
    int found = 0;
    for (int k = 0; k < kRows; k++) {
      int r = kRows - 1 - k;
      double best = at(0, r, l);
      int bestIndex = 0;
      for (int g = 0; g <= kGrid; g++) {
        double value = at(g, r, l);
        if (!std::isfinite(value)) {
          throw std::runtime_error("lane logits are not finite");
        }
        if (value > best) {
          best = value;
          bestIndex = g;
        }
      }
      column[k] = 0.0;
      if (bestIndex == kGrid) continue;  // no lane on this row
      // Expected column over the 100 bins, counted from 1
      double highest = at(0, r, l);
      for (int g = 1; g < kGrid; g++) highest = std::max(highest, at(g, r, l));
      double sum = 0.0, weighted = 0.0;
      for (int g = 0; g < kGrid; g++) {
        double p = std::exp(at(g, r, l) - highest);
        sum += p;
        weighted += p * (g + 1);
      }
      column[k] = weighted / sum;
      if (column[k] != 0.0) found++;
    }
    if (found <= 2) continue;
    for (int k = 0; k < kRows; k++) {
      if (!(column[k] > 0)) continue;
      int x = static_cast<int>(column[k] * step * width / kInputWidth) - 1;
      int y = static_cast<int>(height * (rowAnchor(kRows - 1 - k) /
                                         static_cast<double>(kInputHeight))) -
              1;
      if (x >= 0 && x < width && y >= 0 && y < height) {
        lanes[l].emplace_back(x, y);
      }
    }
  }
  return lanes;
}

std::vector<std::vector<cv::Point>> LaneEngine::detect(const cv::Mat& bgr,
                                                       double* preMs,
                                                       double* inferMs,
                                                       double* postMs) {
  if (!backend_) throw std::runtime_error("the lane model is not loaded");
  using Clock = std::chrono::steady_clock;
  auto ms = [](Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
  };
  auto start = Clock::now();
  std::vector<float> tensor = preprocess(bgr);
  auto preDone = Clock::now();

  std::vector<int64_t> outputShape;
  const float* logits = backend_->run(tensor.data(), kInputShape, outputShape);
  if (outputShape != kOutputShape) {
    throw std::runtime_error("the lane model changed its output shape");
  }
  auto inferDone = Clock::now();
  // decode() reads the backend's buffer before the next run() reuses it
  auto lanes = decode(logits, bgr.cols, bgr.rows);
  auto done = Clock::now();

  if (preMs) *preMs = ms(start, preDone);
  if (inferMs) *inferMs = ms(preDone, inferDone);
  if (postMs) *postMs = ms(inferDone, done);
  return lanes;
}

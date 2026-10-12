#include "stall-engine.hpp"

#include "pre-process.hpp"

#include <chrono>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <stdexcept>

StallEngine::StallEngine(const StallEngineConfig& config) : config_(config) {
  BackendConfig backendConfig;
  backendConfig.modelPath = config.modelPath;
  backendConfig.backend = config.backend;
  backendConfig.precision = config.precision;
  backendConfig.numThreads = config.numThreads;
  auto backend = createBackend(backendConfig);
  if (!backend) return;

  const auto& input = backend->inputShape();
  if (input.size() != 4 || input[1] != 3 || input[2] <= 0 || input[3] <= 0) {
    std::cerr << "Stall model: " << config.modelPath
              << " does not take a fixed 1x3xHxW input\n";
    return;
  }
  inputHeight_ = static_cast<int>(input[2]);
  inputWidth_ = static_cast<int>(input[3]);
  // One run on an empty image shows the output is the one decode() expects
  try {
    std::vector<float> zeros(3 * static_cast<size_t>(inputHeight_) * inputWidth_,
                             0.0f);
    std::vector<int64_t> outputShape;
    backend->run(zeros.data(), {1, 3, inputHeight_, inputWidth_}, outputShape);
    if (outputShape.size() != 3 || outputShape[1] != kFeatures) {
      std::cerr << "Stall model: " << config.modelPath
                << " does not give a 1x18xN output\n";
      return;
    }
  } catch (const std::exception& e) {
    std::cerr << "Stall model: " << e.what() << "\n";
    return;
  }
  backend_ = std::move(backend);
  std::cout << "Stall model loaded: " << config.modelPath << " (backend "
            << config.backend << ", input " << inputWidth_ << "x"
            << inputHeight_ << ")\n";
}

std::vector<StallDetection> StallEngine::decode(
    const float* output, const std::vector<int64_t>& shape, float confidence,
    float nms, int width, int height, int inputWidth, int inputHeight) {
  if (shape.size() != 3 || shape[1] != kFeatures) {
    throw std::runtime_error("the stall model's output is not [1, 18, N]");
  }
  const int count = static_cast<int>(shape[2]);
  // Feature-major, as the detector's: output[feature * count + candidate]
  auto at = [output, count](int feature, int i) {
    return output[static_cast<size_t>(feature) * count + i];
  };
  const float scaleX = static_cast<float>(width) / inputWidth;
  const float scaleY = static_cast<float>(height) / inputHeight;

  std::vector<StallDetection> candidates;
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  for (int i = 0; i < count; i++) {
    float empty = at(4, i), occupied = at(5, i);
    float score = std::max(empty, occupied);
    if (!(score >= confidence)) continue;  // also drops a score that is NaN
    float cx = at(0, i), cy = at(1, i), w = at(2, i), h = at(3, i);
    float x = std::max(0.0f, (cx - w / 2.0f) * scaleX);
    float y = std::max(0.0f, (cy - h / 2.0f) * scaleY);
    float x2 = std::min(static_cast<float>(width), (cx + w / 2.0f) * scaleX);
    float y2 = std::min(static_cast<float>(height), (cy + h / 2.0f) * scaleY);
    if (!(x2 > x && y2 > y)) continue;

    StallDetection stall;
    stall.occupied = occupied > empty;
    stall.confidence = score;
    stall.bbox = cv::Rect2f(x, y, x2 - x, y2 - y);
    for (int k = 0; k < kCorners; k++) {
      int feature = 4 + kClasses + 3 * k;
      // Not clamped to the image: a corner outside it is still where the
      // model puts it, and its confidence says it is not seen
      stall.corners[k] =
          cv::Point2f(at(feature, i) * scaleX, at(feature + 1, i) * scaleY);
      stall.cornerConfidence[k] = at(feature + 2, i);
    }
    candidates.push_back(stall);
    boxes.emplace_back(static_cast<int>(x), static_cast<int>(y),
                       std::max(1, static_cast<int>(x2 - x)),
                       std::max(1, static_cast<int>(y2 - y)));
    scores.push_back(score);
  }

  std::vector<int> kept;
  cv::dnn::NMSBoxes(boxes, scores, confidence, nms, kept);
  std::vector<StallDetection> stalls;
  stalls.reserve(kept.size());
  for (int index : kept) stalls.push_back(candidates[index]);
  return stalls;
}

std::vector<StallDetection> StallEngine::detect(const cv::Mat& bgr,
                                                double* preMs, double* inferMs,
                                                double* postMs) {
  if (!backend_) throw std::runtime_error("the stall model is not loaded");
  using Clock = std::chrono::steady_clock;
  auto ms = [](Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
  };
  auto start = Clock::now();
  cv::Mat blob = preprocess(bgr, inputWidth_, inputHeight_);
  auto preDone = Clock::now();

  std::vector<int64_t> outputShape;
  const float* output = backend_->run(
      blob.ptr<float>(), {1, 3, inputHeight_, inputWidth_}, outputShape);
  auto inferDone = Clock::now();
  // decode() reads the backend's buffer before the next run() reuses it
  auto stalls = decode(output, outputShape, config_.confidence, config_.nms,
                       bgr.cols, bgr.rows, inputWidth_, inputHeight_);
  auto done = Clock::now();

  if (preMs) *preMs = ms(start, preDone);
  if (inferMs) *inferMs = ms(preDone, inferDone);
  if (postMs) *postMs = ms(inferDone, done);
  return stalls;
}

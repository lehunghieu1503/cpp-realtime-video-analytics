#include "inference-engine.hpp"
#include "post-process.hpp"
#include "pre-process.hpp"

#include <chrono>
#include <iostream>

InferenceEngine::InferenceEngine(const InferenceConfig& config)
    : config_(config) {
  BackendConfig backendConfig;
  backendConfig.modelPath = config_.modelPath;
  backendConfig.backend = config_.backend;
  backendConfig.precision = config_.precision;
  backendConfig.numThreads = config_.numThreads;
  backend_ = createBackend(backendConfig);
  if (!backend_) return;

  // Use the model's own input size when it is fixed (e.g. exported at 320 or
  // 1280). Dynamic dimensions come back as -1 and keep the configured size.
  const auto& inputShape = backend_->inputShape();
  if (inputShape.size() == 4 && inputShape[2] > 0 && inputShape[3] > 0) {
    config_.inputHeight = static_cast<int>(inputShape[2]);
    config_.inputWidth = static_cast<int>(inputShape[3]);
  }

  std::cout << "Model loaded: " << config_.modelPath << " (backend "
            << config_.backend << ", input " << config_.inputWidth << "x"
            << config_.inputHeight << ")\n";
}

bool InferenceEngine::isLoaded() const { return backend_ != nullptr; }

std::vector<Detection> InferenceEngine::detect(const cv::Mat& frame,
                                               StageTimes* times) {
  if (!isLoaded()) return {};

  // Validate input frame
  if (frame.empty() || frame.channels() != 3) {
    std::cerr << "detect(): invalid frame (empty or not 3-channel BGR)\n";
    return {};
  }

  using Clock = std::chrono::steady_clock;
  auto ms = [](Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
  };
  auto start = Clock::now();

  // Preprocess: resize + normalize + HWC→CHW
  cv::Mat inputData =
      preprocess(frame, config_.inputWidth, config_.inputHeight);
  auto preDone = Clock::now();

  std::vector<int64_t> inputDims = {1, 3, config_.inputHeight,
                                    config_.inputWidth};
  std::vector<int64_t> outputShape;
  const float* outputData =
      backend_->run(inputData.ptr<float>(), inputDims, outputShape);
  auto inferDone = Clock::now();

  // Postprocess: parse detections + NMS
  auto detections =
      postprocess(outputData, outputShape, config_.confidenceThreshold,
                  config_.nmsThreshold, frame.cols, frame.rows,
                  config_.inputWidth, config_.inputHeight);
  if (times) {
    times->preMs = ms(start, preDone);
    times->inferMs = ms(preDone, inferDone);
    times->postMs = ms(inferDone, Clock::now());
  }
  return detections;
}

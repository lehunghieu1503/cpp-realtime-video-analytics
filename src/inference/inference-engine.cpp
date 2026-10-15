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
  backendConfig.profileLayers = config_.profileLayers;
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
  if (!backend_->describe().empty()) {
    std::cout << "  " << backend_->describe() << "\n";
  }
}

std::string InferenceEngine::layerProfile() const {
  return backend_ ? backend_->layerProfile() : "";
}

bool InferenceEngine::isLoaded() const { return backend_ != nullptr; }

std::vector<Detection> InferenceEngine::detect(const cv::Mat& input,
                                               StageTimes* times,
                                               DeviceFrame* device,
                                               std::shared_ptr<DeviceFrame>* uploaded) {
  if (!isLoaded()) return {};

  using Clock = std::chrono::steady_clock;
  auto ms = [](Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
  };
  auto start = Clock::now();

  std::vector<int64_t> outputShape;
  const float* outputData = nullptr;
  Clock::time_point preDone;
  int frameWidth = 0;
  int frameHeight = 0;

  // Best case: the frame is on the device and the backend reads it there
  if (device && config_.devicePreprocess) {
    double preMs = 0.0;
    outputData = backend_->runDeviceFrame(*device, outputShape, &preMs);
    if (outputData) {
      preDone = start + std::chrono::duration_cast<Clock::duration>(
                            std::chrono::duration<double, std::milli>(preMs));
      frameWidth = device->width();
      frameHeight = device->height();
    }
  }

  cv::Mat frame = input;
  if (!outputData) {
    // The backend needs the pixels in system memory
    if (frame.empty() && device) frame = device->toBgr();
    if (frame.empty() || frame.channels() != 3) {
      std::cerr << "detect(): invalid frame (empty or not 3-channel BGR)\n";
      return {};
    }
    frameWidth = frame.cols;
    frameHeight = frame.rows;
  }

  if (outputData) {
    // Done above, on the device
  } else if (config_.devicePreprocess && backend_->preprocessesFrames()) {
    // The backend takes the frame as captured and prepares it on its device
    double preMs = 0.0;
    outputData = backend_->runFrame(frame, outputShape, &preMs);
    if (uploaded) *uploaded = backend_->uploadedFrame();
    preDone = start + std::chrono::duration_cast<Clock::duration>(
                          std::chrono::duration<double, std::milli>(preMs));
  } else {
    // Preprocess: resize + normalize + HWC→CHW
    cv::Mat inputData =
        preprocess(frame, config_.inputWidth, config_.inputHeight);
    preDone = Clock::now();

    std::vector<int64_t> inputDims = {1, 3, config_.inputHeight,
                                      config_.inputWidth};
    outputData = backend_->run(inputData.ptr<float>(), inputDims, outputShape);
  }
  auto inferDone = Clock::now();

  // Postprocess: parse detections + NMS
  auto detections =
      postprocess(outputData, outputShape, config_.confidenceThreshold,
                  config_.nmsThreshold, frameWidth, frameHeight,
                  config_.inputWidth, config_.inputHeight);
  if (times) {
    times->preMs = ms(start, preDone);
    times->inferMs = ms(preDone, inferDone);
    times->postMs = ms(inferDone, Clock::now());
  }
  return detections;
}

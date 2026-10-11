#pragma once

#include "backends/inference-backend.hpp"
#include "detection.hpp"

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Configuration for the inference engine
struct InferenceConfig {
  std::string modelPath;
  int inputWidth = 640;
  int inputHeight = 640;
  float confidenceThreshold = 0.5f;
  float nmsThreshold = 0.45f;
  int numThreads = 4;
  // "cpu", "cuda", "tensorrt" or "openvino:CPU|GPU|NPU"; the engine stays
  // unloaded if the backend is unavailable. tensorrt caches its engine in
  // trt-cache/ next to the model, so that directory must be writable.
  std::string backend = "cpu";
  // "fp32", "fp16" or "int8"; see BackendConfig
  std::string precision = "fp16";
  // Let a backend that can pre-process on its own device do so; false keeps
  // the CPU path, to compare the two
  bool devicePreprocess = true;
  // Time every layer (tensorrt-native); slows inference
  bool profileLayers = false;
};

// Time spent in each stage of one detect() call, in milliseconds
struct StageTimes {
  double preMs = 0.0;
  double inferMs = 0.0;
  double postMs = 0.0;
};

// YOLO detector: shared pre-process and post-process around a backend that
// only runs the tensor.
// Usage:
//   InferenceConfig cfg;
//   cfg.modelPath = "models/yolov8n.onnx";
//   InferenceEngine engine(cfg);
//   auto detections = engine.detect(frame);
class InferenceEngine {
public:
  explicit InferenceEngine(const InferenceConfig& config);

  // Run object detection on a BGR frame, returns detected objects
  std::vector<Detection> detect(const cv::Mat& frame,
                                StageTimes* times = nullptr);

  // Check if model was loaded successfully
  bool isLoaded() const;

  // Time per layer since the engine was created, as a Markdown table; empty
  // unless InferenceConfig::profileLayers was set and the backend supports it
  std::string layerProfile() const;

private:
  InferenceConfig config_;
  std::unique_ptr<IInferenceBackend> backend_;
};

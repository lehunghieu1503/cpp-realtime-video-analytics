#pragma once

#include "detection.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

// Configuration for the ONNX inference engine
struct InferenceConfig {
  std::string modelPath;
  int inputWidth = 640;
  int inputHeight = 640;
  float confidenceThreshold = 0.5f;
  float nmsThreshold = 0.45f;
  int numThreads = 4;
  // "cpu", "cuda" or "tensorrt"; the engine stays unloaded if the provider is
  // unavailable. tensorrt builds an FP16 engine and caches it in trt-cache/
  // next to the model, so that directory must be writable.
  std::string provider = "cpu";
};

// ONNX Runtime-based YOLOv8 inference engine
// Usage:
//   InferenceConfig cfg;
//   cfg.modelPath = "models/yolov8n.onnx";
//   InferenceEngine engine(cfg);
//   auto detections = engine.detect(frame);
class InferenceEngine {
public:
  explicit InferenceEngine(const InferenceConfig& config);

  // Run object detection on a BGR frame, returns detected objects
  std::vector<Detection> detect(const cv::Mat& frame);

  // Check if model was loaded successfully
  bool isLoaded() const;

private:
  InferenceConfig config_;
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;

  // Cached model metadata (queried once at construction)
  std::vector<std::string> inputNames_;
  std::vector<std::string> outputNames_;
  std::vector<int64_t> inputShape_;   // e.g., [1, 3, 640, 640]
  std::vector<int64_t> outputShape_;  // e.g., [1, 84, 8400]
};

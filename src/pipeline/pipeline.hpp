#pragma once

#include "bounded-queue.hpp"
#include "frame-data.hpp"
#include "capture/video-capture.hpp"
#include "inference/inference-engine.hpp"

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

// Pipeline configuration
struct PipelineConfig {
  int numWorkers = 2;
  size_t frameQueueSize = 8;
  size_t resultQueueSize = 16;
  // false: capture waits for room in the frame queue instead of dropping the
  // oldest frame (benchmark mode — every frame is processed)
  bool dropWhenFull = true;
  CaptureConfig captureConfig;
  InferenceConfig inferenceConfig;
};

// Orchestrates capture → inference → results pipeline.
// Usage:
//   PipelineConfig cfg;
//   cfg.inferenceConfig.modelPath = "models/yolov8n.onnx";
//   Pipeline pipeline(cfg);
//   pipeline.start();
//   // consume pipeline.resultQueue() in your loop
//   pipeline.stop();
class Pipeline {
public:
  explicit Pipeline(const PipelineConfig& config);
  ~Pipeline();

  void start();
  void stop();

  // Access result queue for consuming detection results
  BoundedQueue<AnalyticsResult>& resultQueue() { return resultQueue_; }

  bool isRunning() const { return running_; }

private:
  void captureLoop();
  void inferenceWorker(int workerId);

  PipelineConfig config_;
  std::unique_ptr<IVideoCapture> capture_;
  std::vector<std::unique_ptr<InferenceEngine>> engines_;
  BoundedQueue<FrameData> frameQueue_;
  BoundedQueue<AnalyticsResult> resultQueue_;
  std::thread captureThread_;
  std::vector<std::thread> workerThreads_;
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> frameCounter_{0};
};

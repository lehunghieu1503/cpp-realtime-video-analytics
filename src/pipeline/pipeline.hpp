#pragma once

#include "bounded-queue.hpp"
#include "frame-data.hpp"
#include "stream-queue.hpp"
#include "capture/video-capture.hpp"
#include "inference/inference-engine.hpp"

#include <atomic>
#include <deque>
#include <memory>
#include <thread>
#include <vector>

// Pipeline configuration
struct PipelineConfig {
  int numWorkers = 2;
  // Frames each stream may have waiting for inference; beyond that its
  // oldest frame is dropped
  size_t frameQueueSize = 8;
  size_t resultQueueSize = 16;
  // false: capture waits for room instead of dropping the oldest frame
  // (benchmark mode — every frame is processed)
  bool dropWhenFull = true;
  // One capture thread per source; frames carry the index as streamId
  std::vector<CaptureConfig> sources;
  InferenceConfig inferenceConfig;
};

// Capture-side counters of one stream, safe to read from any thread
struct StreamStats {
  std::atomic<uint64_t> framesCaptured{0};
  std::atomic<uint64_t> reconnects{0};
  std::atomic<bool> connected{false};
};

// Orchestrates capture → inference → results pipeline.
// Usage:
//   PipelineConfig cfg;
//   cfg.sources.emplace_back();
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

  // Access result queue for consuming detection results. It is stopped once
  // every source has ended and the workers have drained the frames.
  BoundedQueue<AnalyticsResult>& resultQueue() { return resultQueue_; }

  // False if the pipeline did not start, was stopped, or a worker failed
  bool isRunning() const { return running_; }

  size_t numStreams() const { return config_.sources.size(); }
  const StreamStats& streamStats(size_t stream) const {
    return stats_[stream];
  }

private:
  void captureLoop(int streamId);
  void inferenceWorker(int workerId);

  PipelineConfig config_;
  std::vector<std::unique_ptr<InferenceEngine>> engines_;
  StreamQueue frameQueue_;
  BoundedQueue<AnalyticsResult> resultQueue_;
  std::vector<std::thread> captureThreads_;
  std::vector<std::thread> workerThreads_;
  std::deque<StreamStats> stats_;  // deque: atomics cannot be moved
  std::atomic<bool> running_{false};
  std::atomic<int> activeCaptures_{0};
  std::atomic<int> activeWorkers_{0};
};

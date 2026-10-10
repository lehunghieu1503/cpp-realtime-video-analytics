#include "pipeline.hpp"
#include "capture/opencv-capture.hpp"

#include <iostream>

Pipeline::Pipeline(const PipelineConfig& config)
    : config_(config),
      frameQueue_(config.frameQueueSize),
      resultQueue_(config.resultQueueSize) {}

Pipeline::~Pipeline() { stop(); }

void Pipeline::start() {
  if (running_) return;

  // Initialize capture
  capture_ = std::make_unique<OpenCvCapture>();
  if (!capture_->open(config_.captureConfig)) {
    std::cerr << "Pipeline: failed to open camera "
              << config_.captureConfig.deviceIndex << "\n";
    capture_.reset();
    return;
  }
  std::cout << "Pipeline: " << capture_->getInfo() << "\n";

  // Create one InferenceEngine per worker (session-per-worker for thread safety)
  engines_.clear();
  for (int i = 0; i < config_.numWorkers; i++) {
    auto engine = std::make_unique<InferenceEngine>(config_.inferenceConfig);
    if (!engine->isLoaded()) {
      std::cerr << "Pipeline: failed to load model for worker " << i << "\n";
      // Cleanup already-created resources on failure
      engines_.clear();
      capture_->release();
      capture_.reset();
      return;
    }
    engines_.push_back(std::move(engine));
  }

  running_ = true;
  frameCounter_ = 0;

  // Launch capture thread
  captureThread_ = std::thread(&Pipeline::captureLoop, this);

  // Launch inference workers
  for (int i = 0; i < config_.numWorkers; i++) {
    workerThreads_.emplace_back(&Pipeline::inferenceWorker, this, i);
  }

  std::cout << "Pipeline: started with " << config_.numWorkers
            << " inference worker(s)\n";
}

void Pipeline::stop() {
  // Not gated on running_: the capture thread clears it when the camera is
  // lost, and the threads still have to be joined.
  if (!captureThread_.joinable() && workerThreads_.empty()) return;

  running_ = false;

  // Stop queues to unblock waiting threads
  frameQueue_.stop();
  resultQueue_.stop();

  // Join all threads
  if (captureThread_.joinable()) captureThread_.join();
  for (auto& t : workerThreads_) {
    if (t.joinable()) t.join();
  }
  workerThreads_.clear();

  // Release resources
  if (capture_) capture_->release();
  engines_.clear();

  std::cout << "Pipeline: stopped after " << frameCounter_.load()
            << " frames captured\n";
}

void Pipeline::captureLoop() {
  bool isVideoFile = !config_.captureConfig.videoPath.empty();
  int failCount = 0;
  const int kMaxFails = 30;

  while (running_) {
    // Only this thread pushes, so once there is room the push cannot drop.
    // shortcut: polls every 1 ms, use a blocking push if capture rate matters
    if (!config_.dropWhenFull &&
        frameQueue_.size() >= config_.frameQueueSize) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    cv::Mat frame;
    if (!capture_->read(frame)) {
      if (isVideoFile) {
        // Video file ended — stop capture, let inference drain the queue
        std::cout << "Pipeline: video ended after " << frameCounter_.load()
                  << " frames\n";
        break;
      }
      failCount++;
      if (failCount >= kMaxFails) {
        std::cerr << "Pipeline: camera disconnected after " << kMaxFails
                  << " consecutive failures\n";
        running_ = false;
        frameQueue_.stop();
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    failCount = 0;

    FrameData fd;
    fd.frameId = frameCounter_++;
    fd.captureTime = std::chrono::steady_clock::now();
    fd.frame = std::move(frame);

    frameQueue_.push(std::move(fd));
  }

  // For video files: signal queue stop so inference workers can drain and exit
  if (isVideoFile) {
    frameQueue_.stop();
  }
}

void Pipeline::inferenceWorker(int workerId) {
  auto& engine = engines_[workerId];

  while (running_) {
    auto frameOpt = frameQueue_.pop();
    if (!frameOpt) break;  // queue stopped

    auto& fd = *frameOpt;
    auto detectStart = std::chrono::steady_clock::now();
    auto detections = engine->detect(fd.frame);
    auto now = std::chrono::steady_clock::now();

    AnalyticsResult result;
    result.frameId = fd.frameId;
    result.captureTime = fd.captureTime;
    result.inferenceDoneTime = now;
    result.frame = std::move(fd.frame);
    result.detections = std::move(detections);
    result.inferenceLatencyMs =
        std::chrono::duration<double, std::milli>(now - fd.captureTime).count();
    result.detectMs =
        std::chrono::duration<double, std::milli>(now - detectStart).count();

    resultQueue_.push(std::move(result));
  }
}

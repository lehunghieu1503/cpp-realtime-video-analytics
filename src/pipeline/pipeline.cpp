#include "pipeline.hpp"

#include <sched.h>

#include <algorithm>
#include <iostream>

Pipeline::Pipeline(const PipelineConfig& config)
    : config_(config),
      frameQueue_(config.sources.size(), config.frameQueueSize),
      resultQueue_(config.resultQueueSize),
      stats_(config.sources.size()) {}

Pipeline::~Pipeline() { stop(); }

void Pipeline::start() {
  if (running_ || config_.sources.empty()) return;

  // A video file that does not open is a mistake to report now. A live
  // source (camera, RTSP, GStreamer) may just not be up yet: its capture
  // thread keeps retrying.
  for (const auto& source : config_.sources) {
    if (source.isLive()) continue;
    if (!createCapture(source)->open(source)) {
      std::cerr << "Pipeline: failed to open " << source.id << "\n";
      return;
    }
  }

  // Keep workers x intra-op threads within the cores, leaving one for capture
  // and the consumer of the results. The affinity mask, unlike
  // hardware_concurrency(), follows taskset and docker --cpuset-cpus.
  cpu_set_t cpus;
  if (sched_getaffinity(0, sizeof(cpus), &cpus) == 0) {
    int cores = CPU_COUNT(&cpus);
    config_.inferenceConfig.numThreads =
        std::min(config_.inferenceConfig.numThreads,
                 std::max(1, (cores - 1) / config_.numWorkers));
  }

  // Create one InferenceEngine per worker (session-per-worker for thread safety)
  engines_.clear();
  for (int i = 0; i < config_.numWorkers; i++) {
    auto engine = std::make_unique<InferenceEngine>(config_.inferenceConfig);
    if (!engine->isLoaded()) {
      std::cerr << "Pipeline: failed to load model for worker " << i << "\n";
      engines_.clear();
      return;
    }
    engines_.push_back(std::move(engine));
  }

  running_ = true;
  activeCaptures_ = static_cast<int>(config_.sources.size());
  activeWorkers_ = config_.numWorkers;

  for (size_t i = 0; i < config_.sources.size(); i++) {
    captureThreads_.emplace_back(&Pipeline::captureLoop, this,
                                 static_cast<int>(i));
  }
  for (int i = 0; i < config_.numWorkers; i++) {
    workerThreads_.emplace_back(&Pipeline::inferenceWorker, this, i);
  }

  std::cout << "Pipeline: started with " << config_.sources.size()
            << " source(s), " << config_.numWorkers
            << " inference worker(s), "
            << config_.inferenceConfig.numThreads << " thread(s) each\n";
}

void Pipeline::stop() {
  // Not gated on running_: it is cleared when the sources end or a worker
  // fails, and the threads still have to be joined.
  if (captureThreads_.empty() && workerThreads_.empty()) return;

  running_ = false;

  // Stop queues to unblock waiting threads
  frameQueue_.stop();
  resultQueue_.stop();

  for (auto& t : captureThreads_) {
    if (t.joinable()) t.join();
  }
  captureThreads_.clear();
  for (auto& t : workerThreads_) {
    if (t.joinable()) t.join();
  }
  workerThreads_.clear();
  engines_.clear();

  uint64_t frames = 0;
  for (const auto& s : stats_) frames += s.framesCaptured;
  std::cout << "Pipeline: stopped after " << frames << " frames captured\n";
}

void Pipeline::captureLoop(int streamId) {
  const CaptureConfig& source = config_.sources[streamId];
  StreamStats& stats = stats_[streamId];
  std::unique_ptr<IVideoCapture> capturePtr = createCapture(source);
  IVideoCapture& capture = *capturePtr;
  uint64_t frameId = 0;
  // Reopen delay after a failure: doubles from 1 s up to 30 s, and resets
  // once a frame arrives
  int backoffSec = 1;
  int failCount = 0;
  const int kMaxFails = 30;

  // Sleeps in short steps so stop() is not held up by a long backoff
  auto waitBackoff = [&] {
    for (int i = 0; i < backoffSec * 10 && running_; i++) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    backoffSec = std::min(backoffSec * 2, 30);
  };

  while (running_) {
    if (!capture.isOpened()) {
      if (!capture.open(source)) {
        std::cerr << "Pipeline: " << source.id << " not available, retry in "
                  << backoffSec << " s\n";
        waitBackoff();
        continue;
      }
      std::cout << "Pipeline: " << capture.getInfo() << "\n";
      stats.connected = true;
      failCount = 0;
    }

    // Only this thread pushes to this stream's slot, so once there is room
    // the push cannot drop.
    // shortcut: polls every 1 ms, use a blocking push if capture rate matters
    if (!config_.dropWhenFull &&
        frameQueue_.size(streamId) >= config_.frameQueueSize) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    cv::Mat frame;
    auto readStart = std::chrono::steady_clock::now();
    if (!capture.read(frame)) {
      if (!source.isLive()) {
        std::cout << "Pipeline: " << source.id << " ended after " << frameId
                  << " frames\n";
        break;
      }
      if (++failCount < kMaxFails) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      std::cerr << "Pipeline: " << source.id << " lost, reconnecting in "
                << backoffSec << " s\n";
      capture.release();
      stats.connected = false;
      stats.reconnects++;
      waitBackoff();
      continue;
    }
    auto now = std::chrono::steady_clock::now();
    failCount = 0;
    backoffSec = 1;

    FrameData fd;
    fd.frameId = frameId++;
    fd.streamId = streamId;
    fd.captureTime = now;
    fd.decodeMs =
        std::chrono::duration<double, std::milli>(now - readStart).count();
    fd.frame = std::move(frame);
    stats.framesCaptured++;

    frameQueue_.push(std::move(fd));
  }

  capture.release();
  stats.connected = false;
  // After the last source ends, let the workers drain the queue and exit
  if (--activeCaptures_ == 0) frameQueue_.stop();
}

void Pipeline::inferenceWorker(int workerId) {
  auto& engine = engines_[workerId];

  while (true) {
    auto frameOpt = frameQueue_.pop();
    if (!frameOpt) break;  // queue stopped and drained

    auto& fd = *frameOpt;
    auto detectStart = std::chrono::steady_clock::now();
    std::vector<Detection> detections;
    StageTimes times;
    try {
      detections = engine->detect(fd.frame, &times);
    } catch (const std::exception& e) {
      // An exception leaving a std::thread would terminate the process
      std::cerr << "Pipeline: inference failed, stopping: " << e.what()
                << "\n";
      running_ = false;
      frameQueue_.stop();
      break;
    }
    auto now = std::chrono::steady_clock::now();

    AnalyticsResult result;
    result.frameId = fd.frameId;
    result.streamId = fd.streamId;
    result.captureTime = fd.captureTime;
    result.inferenceDoneTime = now;
    result.frame = std::move(fd.frame);
    result.detections = std::move(detections);
    result.inferenceLatencyMs =
        std::chrono::duration<double, std::milli>(now - fd.captureTime).count();
    result.detectMs =
        std::chrono::duration<double, std::milli>(now - detectStart).count();
    result.decodeMs = fd.decodeMs;
    result.preMs = times.preMs;
    result.inferMs = times.inferMs;
    result.postMs = times.postMs;

    resultQueue_.push(std::move(result));
  }

  // Once the last worker is done no more results can arrive: the consumer
  // drains what is queued and then sees the queue stopped
  if (--activeWorkers_ == 0) resultQueue_.stop();
}

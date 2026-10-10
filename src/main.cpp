#include "pipeline/pipeline.hpp"
#include "server/sse-server.hpp"
#include "server/result-serializer.hpp"
#include "benchmark/latency-tracker.hpp"
#include "benchmark/benchmark-report.hpp"
#include "inference/detection.hpp"
#include "utils/fps-tracker.hpp"

#include <csignal>
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>

// Global shutdown flag for SIGINT handling
static std::atomic<bool> gShutdown{false};

void signalHandler(int) { gShutdown = true; }

// Draw detection bounding boxes and labels on frame
void drawDetections(cv::Mat& frame,
                    const std::vector<Detection>& detections) {
  for (const auto& det : detections) {
    cv::Rect box(static_cast<int>(det.bbox.x), static_cast<int>(det.bbox.y),
                 static_cast<int>(det.bbox.width),
                 static_cast<int>(det.bbox.height));

    cv::Scalar color(
        static_cast<int>(det.classId * 50) % 256,
        static_cast<int>(det.classId * 80 + 100) % 256,
        static_cast<int>(det.classId * 120 + 50) % 256);

    cv::rectangle(frame, box, color, 2);

    std::string label = std::string(getClassName(det.classId)) + " " +
                        cv::format("%.2f", det.confidence);

    int baseline = 0;
    cv::Size textSize =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    cv::rectangle(frame, cv::Point(box.x, box.y - textSize.height - 6),
                  cv::Point(box.x + textSize.width + 4, box.y), color, -1);
    cv::putText(frame, label, cv::Point(box.x + 2, box.y - 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  }
}

// Draw pipeline stats overlay
void drawOverlay(cv::Mat& frame, double fps, double latencyMs,
                 int detectionCount, uint64_t frameId) {
  cv::rectangle(frame, cv::Point(5, 5), cv::Point(280, 85),
                cv::Scalar(0, 0, 0), -1);
  cv::rectangle(frame, cv::Point(5, 5), cv::Point(280, 85),
                cv::Scalar(0, 255, 0), 1);

  cv::Scalar green(0, 255, 0);
  cv::putText(frame, cv::format("FPS: %.1f | Latency: %.0fms", fps, latencyMs),
              cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.55, green, 1);
  cv::putText(frame, cv::format("Detections: %d", detectionCount),
              cv::Point(12, 50), cv::FONT_HERSHEY_SIMPLEX, 0.55, green, 1);
  cv::putText(frame,
              cv::format("Frame: %llu",
                         static_cast<unsigned long long>(frameId)),
              cv::Point(12, 72), cv::FONT_HERSHEY_SIMPLEX, 0.55, green, 1);
}

// ── Benchmark mode ──────────────────────────────────────────────────────────
int runBenchmark(const std::string& inputVideo, int maxFrames,
                 const std::string& outputReport, const std::string& modelPath,
                 int numWorkers) {
  std::cout << "Benchmark mode: " << inputVideo << " (" << maxFrames
            << " frames, model: " << modelPath << ", workers: " << numWorkers
            << ")\n";

  PipelineConfig config;
  config.captureConfig.videoPath = inputVideo;
  config.inferenceConfig.modelPath = modelPath;
  config.inferenceConfig.confidenceThreshold = 0.5f;
  config.inferenceConfig.nmsThreshold = 0.45f;
  config.numWorkers = numWorkers;
  // Use larger queues for benchmark — no frame dropping, capped at 2000
  size_t benchQueueSize = std::min(static_cast<size_t>(maxFrames + 10),
                                   static_cast<size_t>(2000));
  config.frameQueueSize = benchQueueSize;
  config.resultQueueSize = benchQueueSize;

  Pipeline pipeline(config);
  pipeline.start();

  if (!pipeline.isRunning()) {
    std::cerr << "Failed to start pipeline\n";
    return -1;
  }

  LatencyTracker latency;
  FpsTracker fpsTracker;
  int frameCount = 0;

  while (frameCount < maxFrames && !gShutdown) {
    auto resultOpt =
        pipeline.resultQueue().tryPopFor(std::chrono::milliseconds(1000));
    if (!resultOpt) {
      // Video ended or pipeline stopped — try once more then exit
      if (!pipeline.isRunning()) break;
      auto retry = pipeline.resultQueue().tryPopFor(std::chrono::milliseconds(500));
      if (!retry) break;
      resultOpt = std::move(retry);
    }

    frameCount++;
    latency.record(resultOpt->inferenceLatencyMs);
    fpsTracker.update();

    // Progress every 100 frames
    if (frameCount % 100 == 0) {
      std::cout << "  " << frameCount << "/" << maxFrames
                << " frames (FPS: " << cv::format("%.1f", fpsTracker.fps)
                << ")\n";
    }
  }

  pipeline.stop();

  // Print results
  BenchmarkResults results;
  results.fpsMean = fpsTracker.fps;
  results.totalFrames = frameCount;
  results.latency = latency;

  std::cout << "\n" << generateReport(results);

  if (!outputReport.empty()) {
    writeReport(results, outputReport);
  }

  return 0;
}

// ── Live mode (camera + dashboard) ──────────────────────────────────────────
int runLive(int deviceIndex, const std::string& modelPath, int port,
            int numWorkers, const std::string& bindAddress, bool headless) {
  PipelineConfig config;
  config.captureConfig.deviceIndex = deviceIndex;
  config.captureConfig.width = 640;
  config.captureConfig.height = 480;
  config.captureConfig.codec = "MJPG";
  config.inferenceConfig.modelPath = modelPath;
  config.inferenceConfig.confidenceThreshold = 0.5f;
  config.inferenceConfig.nmsThreshold = 0.45f;
  config.numWorkers = numWorkers;

  Pipeline pipeline(config);
  pipeline.start();

  if (!pipeline.isRunning()) {
    std::cerr << "Failed to start pipeline\n";
    return -1;
  }

  SseServerConfig sseConfig;
  sseConfig.port = port;
  sseConfig.bindAddress = bindAddress;
  sseConfig.staticDir = "web/";
  SseServer sseServer(sseConfig);
  sseServer.start();

  std::cout << "Pipeline running. Press "
            << (headless ? "Ctrl+C" : "'q' or Ctrl+C") << " to stop.\n";
  std::cout << "Dashboard: http://localhost:" << port << "\n";

  FpsTracker fpsTracker;

  while (!gShutdown) {
    auto resultOpt =
        pipeline.resultQueue().tryPopFor(std::chrono::milliseconds(10));
    if (!resultOpt) continue;

    auto& result = *resultOpt;
    fpsTracker.update();

    drawDetections(result.frame, result.detections);
    drawOverlay(result.frame, fpsTracker.fps, result.inferenceLatencyMs,
                static_cast<int>(result.detections.size()), result.frameId);

    if (!headless) cv::imshow("Video Analytics", result.frame);

    auto json = serializeResult(result, fpsTracker.fps);
    sseServer.broadcast(json.dump());
    sseServer.updateFrame(result.frame);

    if (!headless && cv::waitKey(1) == 'q') break;
  }

  sseServer.stop();
  pipeline.stop();
  if (!headless) cv::destroyAllWindows();

  return 0;
}

// ── CLI parsing ─────────────────────────────────────────────────────────────
void printUsage(const char* prog) {
  std::cout
      << "Usage:\n"
      << "  " << prog << " [OPTIONS]                 Live camera mode\n"
      << "  " << prog << " --benchmark OPTIONS        Benchmark mode\n\n"
      << "General options:\n"
      << "  --device N       Camera device index (default: 0)\n"
      << "  --model PATH     ONNX model path (default: models/yolov8n.onnx)\n"
      << "  --port N         Dashboard port (default: 9001)\n"
      << "  --bind ADDR      Bind address (default: 127.0.0.1)\n"
      << "  --workers N      Inference worker threads (default: 2)\n"
      << "  --headless       No local window; dashboard only\n\n"
      << "Benchmark options:\n"
      << "  --input FILE     Input video file (required)\n"
      << "  --frames N       Number of frames (default: 300)\n"
      << "  --output FILE    Write report to file\n";
}

int main(int argc, char* argv[]) {
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  // Parse CLI arguments
  bool benchmarkMode = false;
  std::string inputVideo;
  int maxFrames = 300;
  std::string outputReport;
  int deviceIndex = 0;
  std::string modelPath = "models/yolov8n.onnx";
  int port = 9001;
  std::string bindAddress = "127.0.0.1";
  int numWorkers = 2;
  bool headless = false;

  auto parseIntArg = [&](const char* val, const char* name) -> int {
    try {
      return std::stoi(val);
    } catch (const std::exception&) {
      std::cerr << "Error: invalid value for " << name << ": " << val << "\n";
      std::exit(1);
    }
  };

  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "--benchmark") == 0) {
      benchmarkMode = true;
    } else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
      inputVideo = argv[++i];
    } else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
      maxFrames = parseIntArg(argv[++i], "--frames");
    } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
      outputReport = argv[++i];
    } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      deviceIndex = parseIntArg(argv[++i], "--device");
    } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      modelPath = argv[++i];
    } else if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
      port = parseIntArg(argv[++i], "--port");
    } else if (std::strcmp(argv[i], "--bind") == 0 && i + 1 < argc) {
      bindAddress = argv[++i];
    } else if (std::strcmp(argv[i], "--workers") == 0 && i + 1 < argc) {
      numWorkers = parseIntArg(argv[++i], "--workers");
    } else if (std::strcmp(argv[i], "--headless") == 0) {
      headless = true;
    } else if (std::strcmp(argv[i], "--help") == 0) {
      printUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Warning: unrecognized option: " << argv[i] << "\n";
    }
  }

  // Validate numeric ranges
  if (port < 1 || port > 65535) {
    std::cerr << "Error: --port must be between 1 and 65535\n";
    return -1;
  }
  if (numWorkers < 1 || numWorkers > 16) {
    std::cerr << "Error: --workers must be between 1 and 16\n";
    return -1;
  }

  if (benchmarkMode) {
    if (inputVideo.empty()) {
      std::cerr << "Error: --benchmark requires --input VIDEO_FILE\n";
      printUsage(argv[0]);
      return -1;
    }
    if (maxFrames <= 0) {
      std::cerr << "Error: --frames must be positive\n";
      return -1;
    }
    return runBenchmark(inputVideo, maxFrames, outputReport, modelPath,
                        numWorkers);
  }

  return runLive(deviceIndex, modelPath, port, numWorkers, bindAddress,
                 headless);
}

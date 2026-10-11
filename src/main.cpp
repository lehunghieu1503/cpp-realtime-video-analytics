#include "analytics/rules.hpp"
#include "analytics/tracker.hpp"
#include "benchmark/benchmark-report.hpp"
#include "benchmark/latency-tracker.hpp"
#include "benchmark/power-meter.hpp"
#include "config/app-config.hpp"
#include "inference/detection.hpp"
#include "output/mqtt-publisher.hpp"
#include "pipeline/pipeline.hpp"
#include "server/result-serializer.hpp"
#include "server/sse-server.hpp"
#include "utils/fps-tracker.hpp"

#include <algorithm>
#include <csignal>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
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
    if (det.trackId >= 0) label = cv::format("#%d ", det.trackId) + label;

    int baseline = 0;
    cv::Size textSize =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    cv::rectangle(frame, cv::Point(box.x, box.y - textSize.height - 6),
                  cv::Point(box.x + textSize.width + 4, box.y), color, -1);
    cv::putText(frame, label, cv::Point(box.x + 2, box.y - 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  }
}

// Draw the zones and lines of a stream with their live counts
void drawRules(cv::Mat& frame, const Rules& rules) {
  const cv::Scalar zoneColor(0, 200, 255);
  const cv::Scalar lineColor(255, 120, 0);
  const auto& config = rules.config();

  auto zoneCounts = rules.zoneCounts();
  for (size_t i = 0; i < config.zones.size(); i++) {
    std::vector<cv::Point> polygon(config.zones[i].points.begin(),
                                   config.zones[i].points.end());
    cv::polylines(frame, polygon, true, zoneColor, 2);
    cv::putText(frame,
                cv::format("%s: %d", config.zones[i].name.c_str(),
                           zoneCounts[i]),
                polygon[0] + cv::Point(4, 18), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                zoneColor, 2);
  }
  for (size_t i = 0; i < config.lines.size(); i++) {
    cv::Point a = config.lines[i].a;
    cv::Point b = config.lines[i].b;
    cv::line(frame, a, b, lineColor, 2);
    cv::putText(frame,
                cv::format("%s in %d out %d", config.lines[i].name.c_str(),
                           rules.lineIn()[i], rules.lineOut()[i]),
                a + cv::Point(4, -8), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                lineColor, 2);
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

// Shared by every mode: the inference part of the pipeline configuration
PipelineConfig makePipelineConfig(const AppConfig& app) {
  PipelineConfig config;
  config.inferenceConfig.modelPath = app.model;
  config.inferenceConfig.backend = app.backend;
  config.inferenceConfig.precision = app.precision;
  config.inferenceConfig.confidenceThreshold = app.confidence;
  config.inferenceConfig.nmsThreshold = 0.45f;
  config.numWorkers = app.workers;
  return config;
}

// ── Benchmark mode ──────────────────────────────────────────────────────────
int runBenchmark(const AppConfig& app, const std::string& inputVideo,
                 int maxFrames, int warmupFrames,
                 const std::string& outputReport, const std::string& csvPath) {
  std::cout << "Benchmark mode: " << inputVideo << " (" << maxFrames
            << " frames after " << warmupFrames << " warm-up, model: "
            << app.model << ", workers: " << app.workers << ", backend: "
            << app.backend << ")\n";

  PipelineConfig config = makePipelineConfig(app);
  CaptureConfig source;
  source.videoPath = inputVideo;
  config.sources.push_back(source);
  // Process every frame: capture waits for room instead of dropping, so a
  // short queue is enough and memory stays bounded
  config.dropWhenFull = false;
  config.frameQueueSize = static_cast<size_t>(app.workers) * 2;
  config.resultQueueSize = 64;

  Pipeline pipeline(config);
  pipeline.start();

  if (!pipeline.isRunning()) {
    std::cerr << "Failed to start pipeline\n";
    return -1;
  }

  PowerMeter power(app.backend == "cuda" || app.backend == "tensorrt");
  LatencyTracker latency;
  FpsTracker fpsTracker;
  BenchmarkResults results;
  int frameCount = 0;  // measured frames, warm-up excluded
  int seen = 0;
  auto startTime = std::chrono::steady_clock::now();
  auto lastResultTime = startTime;

  while (frameCount < maxFrames && !gShutdown) {
    auto resultOpt =
        pipeline.resultQueue().tryPopFor(std::chrono::milliseconds(1000));
    if (!resultOpt) {
      // Stopped and drained: the video ended or a worker failed
      if (pipeline.resultQueue().isStopped()) break;
      continue;
    }

    // The first frames pay for lazy initialisation and cold caches
    if (++seen <= warmupFrames) {
      if (seen == warmupFrames) {
        startTime = std::chrono::steady_clock::now();
        power.reset();
      }
      continue;
    }

    frameCount++;
    lastResultTime = std::chrono::steady_clock::now();
    // detect() time only; capture → done would mostly measure queue wait
    latency.record(resultOpt->detectMs);
    results.decodeMs += resultOpt->decodeMs;
    results.preMs += resultOpt->preMs;
    results.inferMs += resultOpt->inferMs;
    results.postMs += resultOpt->postMs;
    fpsTracker.update();

    // Progress every 100 frames
    if (frameCount % 100 == 0) {
      std::cout << "  " << frameCount << "/" << maxFrames
                << " frames (FPS: " << cv::format("%.1f", fpsTracker.fps)
                << ")\n";
    }
  }

  // False only if a worker gave up; a finished video keeps it
  bool pipelineFailed = !pipeline.isRunning();
  pipeline.stop();

  // Mean over the whole run; FpsTracker only knows its last 1 s window
  double elapsedSec =
      std::chrono::duration<double>(lastResultTime - startTime).count();
  results.fpsMean = elapsedSec > 0.0 ? frameCount / elapsedSec : 0.0;
  results.totalFrames = frameCount;
  results.warmupFrames = std::min(seen, warmupFrames);
  results.latency = latency;
  results.device = app.backend;
  results.precision = app.backend == "tensorrt" ? app.precision : "model";
  results.modelName = std::filesystem::path(app.model).stem().string();
  if (frameCount > 0) {
    results.decodeMs /= frameCount;
    results.preMs /= frameCount;
    results.inferMs /= frameCount;
    results.postMs /= frameCount;
  }
  results.watts = power.watts();
  results.powerSource = power.source();

  std::cout << "\n" << generateReport(results);

  if (!outputReport.empty()) writeReport(results, outputReport);
  if (!csvPath.empty()) appendCsv(results, csvPath);

  return pipelineFailed ? -1 : 0;
}

// ── Evaluation mode (detections for scripts/eval-map.py) ────────────────────
// Runs the detector over every image in a directory and writes the results
// as JSON: [{"file": "000000000139.jpg", "class": 0, "score": 0.91,
//            "bbox": [x, y, w, h]}, ...]
int runEval(const AppConfig& app, const std::string& imageDir,
            const std::string& outputPath) {
  InferenceConfig config = makePipelineConfig(app).inferenceConfig;
  // mAP integrates over the whole precision-recall curve, so keep the
  // low-confidence detections too
  config.confidenceThreshold = 0.001f;
  InferenceEngine engine(config);
  if (!engine.isLoaded()) return -1;

  std::vector<std::filesystem::path> images;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(imageDir, ec)) {
    if (entry.path().extension() == ".jpg") images.push_back(entry.path());
  }
  if (images.empty()) {
    std::cerr << "No .jpg images in " << imageDir << "\n";
    return -1;
  }
  std::sort(images.begin(), images.end());

  nlohmann::json results = nlohmann::json::array();
  for (size_t i = 0; i < images.size() && !gShutdown; i++) {
    cv::Mat image = cv::imread(images[i].string());
    if (image.empty()) continue;
    for (const auto& det : engine.detect(image)) {
      results.push_back(
          {{"file", images[i].filename().string()},
           {"class", det.classId},
           {"score", det.confidence},
           {"bbox",
            {det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height}}});
    }
    if ((i + 1) % 100 == 0) {
      std::cout << "  " << i + 1 << "/" << images.size() << " images\n";
    }
  }

  std::ofstream file(outputPath);
  if (!file) {
    std::cerr << "Failed to write " << outputPath << "\n";
    return -1;
  }
  file << results.dump();
  std::cout << "Wrote " << results.size() << " detections to " << outputPath
            << "\n";
  return 0;
}

// ── Live mode (cameras + dashboard) ─────────────────────────────────────────
namespace {
// What the consumer loop knows about one stream
struct StreamState {
  std::string id;
  uint64_t nextFrameId = 0;
  FpsTracker fps;
  std::unique_ptr<Tracker> tracker;
  std::unique_ptr<Rules> rules;
};

// Numbers shared between the consumer loop and the /metrics handler
struct LiveMetrics {
  struct Stream {
    double fps = 0.0;
    double latencyMs = 0.0;
    uint64_t frames = 0;
    int detections = 0;
    std::vector<int> zoneCounts;
    std::vector<int> lineIn;
    std::vector<int> lineOut;
    std::map<std::string, uint64_t> events;  // by type
  };
  std::mutex mutex;
  std::vector<Stream> streams;
  // Smoothed time per stage over all streams, milliseconds
  double decodeMs = 0.0;
  double preMs = 0.0;
  double inferMs = 0.0;
  double postMs = 0.0;
};

double unixNow() {
  return std::chrono::duration<double>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

long residentBytes() {
  long pages = 0, resident = 0;
  std::ifstream("/proc/self/statm") >> pages >> resident;
  return resident * 4096;
}

std::string renderMetrics(LiveMetrics& metrics, const Pipeline& pipeline,
                          const std::vector<StreamState>& streams,
                          const PowerMeter& power) {
  std::ostringstream out;
  std::lock_guard<std::mutex> lock(metrics.mutex);

  auto gauge = [&out](const char* name, const char* help) {
    out << "# HELP " << name << " " << help << "\n# TYPE " << name
        << " gauge\n";
  };
  auto counter = [&out](const char* name, const char* help) {
    out << "# HELP " << name << " " << help << "\n# TYPE " << name
        << " counter\n";
  };
  size_t n = streams.size();

  gauge("va_stream_fps", "Frames processed per second");
  for (size_t i = 0; i < n; i++) {
    // The last measured rate would otherwise linger while the source is away
    out << "va_stream_fps{stream=\"" << streams[i].id << "\"} "
        << (pipeline.streamStats(i).connected ? metrics.streams[i].fps : 0.0)
        << "\n";
  }
  gauge("va_stream_latency_ms", "Capture to inference done");
  for (size_t i = 0; i < n; i++) {
    out << "va_stream_latency_ms{stream=\"" << streams[i].id << "\"} "
        << metrics.streams[i].latencyMs << "\n";
  }
  gauge("va_stream_detections", "Objects in the latest frame");
  for (size_t i = 0; i < n; i++) {
    out << "va_stream_detections{stream=\"" << streams[i].id << "\"} "
        << metrics.streams[i].detections << "\n";
  }
  gauge("va_stream_connected", "1 while the source delivers frames");
  for (size_t i = 0; i < n; i++) {
    out << "va_stream_connected{stream=\"" << streams[i].id << "\"} "
        << (pipeline.streamStats(i).connected ? 1 : 0) << "\n";
  }
  counter("va_stream_frames_processed_total", "Frames through inference");
  for (size_t i = 0; i < n; i++) {
    out << "va_stream_frames_processed_total{stream=\"" << streams[i].id
        << "\"} " << metrics.streams[i].frames << "\n";
  }
  counter("va_stream_frames_captured_total", "Frames read from the source");
  for (size_t i = 0; i < n; i++) {
    out << "va_stream_frames_captured_total{stream=\"" << streams[i].id
        << "\"} " << pipeline.streamStats(i).framesCaptured << "\n";
  }
  counter("va_stream_reconnects_total", "Times the source was lost");
  for (size_t i = 0; i < n; i++) {
    out << "va_stream_reconnects_total{stream=\"" << streams[i].id << "\"} "
        << pipeline.streamStats(i).reconnects << "\n";
  }
  counter("va_events_total", "Events raised by the rules");
  for (size_t i = 0; i < n; i++) {
    for (const auto& [type, count] : metrics.streams[i].events) {
      out << "va_events_total{stream=\"" << streams[i].id << "\",type=\""
          << type << "\"} " << count << "\n";
    }
  }
  gauge("va_zone_occupancy", "Tracked objects inside the zone");
  for (size_t i = 0; i < n; i++) {
    if (!streams[i].rules) continue;
    const auto& zones = streams[i].rules->config().zones;
    for (size_t z = 0; z < metrics.streams[i].zoneCounts.size(); z++) {
      out << "va_zone_occupancy{stream=\"" << streams[i].id << "\",zone=\""
          << zones[z].name << "\"} " << metrics.streams[i].zoneCounts[z]
          << "\n";
    }
  }
  counter("va_line_crossings_total", "Objects that crossed the line");
  for (size_t i = 0; i < n; i++) {
    if (!streams[i].rules) continue;
    const auto& lines = streams[i].rules->config().lines;
    for (size_t l = 0; l < metrics.streams[i].lineIn.size(); l++) {
      out << "va_line_crossings_total{stream=\"" << streams[i].id
          << "\",line=\"" << lines[l].name << "\",direction=\"in\"} "
          << metrics.streams[i].lineIn[l] << "\n";
      out << "va_line_crossings_total{stream=\"" << streams[i].id
          << "\",line=\"" << lines[l].name << "\",direction=\"out\"} "
          << metrics.streams[i].lineOut[l] << "\n";
    }
  }
  // decode is the time blocked in the capture read: on a live source that
  // includes waiting for the next frame
  gauge("va_stage_ms", "Smoothed time per frame in each stage");
  out << "va_stage_ms{stage=\"decode\"} " << metrics.decodeMs << "\n"
      << "va_stage_ms{stage=\"preprocess\"} " << metrics.preMs << "\n"
      << "va_stage_ms{stage=\"inference\"} " << metrics.inferMs << "\n"
      << "va_stage_ms{stage=\"postprocess\"} " << metrics.postMs << "\n";
  if (power.lastWatts() >= 0.0) {
    gauge("va_power_watts", "Power drawn, see the benchmark report for scope");
    out << "va_power_watts " << power.lastWatts() << "\n";
  }
  gauge("va_process_resident_memory_bytes", "Resident set size");
  out << "va_process_resident_memory_bytes " << residentBytes() << "\n";
  return out.str();
}

// Saves event frames and deletes the oldest beyond `keep`
class SnapshotWriter {
public:
  SnapshotWriter(const std::string& dir, int keep) : dir_(dir), keep_(keep) {
    if (dir_.empty()) return;
    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    // Count what earlier runs left, oldest first, so the cap holds across
    // restarts
    std::vector<std::filesystem::path> existing;
    for (const auto& entry : std::filesystem::directory_iterator(dir_, ec)) {
      if (entry.path().extension() == ".jpg") existing.push_back(entry.path());
    }
    std::sort(existing.begin(), existing.end());
    files_.assign(existing.begin(), existing.end());
  }

  // Returns the path written, or an empty string
  std::string save(const cv::Mat& frame, const Event& event) {
    if (dir_.empty()) return "";
    // Millisecond timestamp first: names sort by age
    std::string name =
        cv::format("%013lld_%s_%s_%d.jpg",
                   static_cast<long long>(event.timestamp * 1000),
                   event.stream.c_str(), event.type.c_str(), event.trackId);
    auto path = std::filesystem::path(dir_) / name;
    if (!cv::imwrite(path.string(), frame)) return "";
    files_.push_back(path);
    while (static_cast<int>(files_.size()) > keep_) {
      std::error_code ec;
      std::filesystem::remove(files_.front(), ec);
      files_.pop_front();
    }
    return path.string();
  }

private:
  std::string dir_;
  int keep_;
  std::deque<std::filesystem::path> files_;
};

// What the dashboard needs to lay itself out: GET /streams
std::string streamsJson(const AppConfig& app) {
  nlohmann::json streams = nlohmann::json::array();
  for (const auto& source : app.sources) {
    nlohmann::json zones = nlohmann::json::array();
    for (const auto& zone : source.zones) zones.push_back(zone.name);
    nlohmann::json lines = nlohmann::json::array();
    for (const auto& line : source.lines) lines.push_back(line.name);
    streams.push_back(
        {{"id", source.capture.id}, {"zones", zones}, {"lines", lines}});
  }
  return streams.dump();
}
}  // namespace

int runLive(const AppConfig& app) {
  PipelineConfig config = makePipelineConfig(app);
  for (const auto& source : app.sources) {
    config.sources.push_back(source.capture);
  }
  // When inference is slower than the cameras the queue stays full, and every
  // queued frame adds one inference time of delay to what is shown. Keep
  // just enough per stream to hand every worker a frame.
  config.frameQueueSize = std::max<size_t>(
      1, static_cast<size_t>(app.workers) / app.sources.size());
  config.resultQueueSize = std::max<size_t>(16, app.sources.size() * 4);

  Pipeline pipeline(config);
  pipeline.start();

  if (!pipeline.isRunning()) {
    std::cerr << "Failed to start pipeline\n";
    return -1;
  }

  std::vector<StreamState> streams(app.sources.size());
  LiveMetrics metrics;
  metrics.streams.resize(app.sources.size());
  for (size_t i = 0; i < app.sources.size(); i++) {
    streams[i].id = app.sources[i].capture.id;
    if (!app.tracking) continue;
    streams[i].tracker = std::make_unique<Tracker>(app.sources[i].capture.fps);
    RulesConfig rules;
    rules.zones = app.sources[i].zones;
    rules.lines = app.sources[i].lines;
    rules.dwellSec = app.dwellSeconds;
    streams[i].rules = std::make_unique<Rules>(streams[i].id, rules);
  }

  PowerMeter power(app.backend == "cuda" || app.backend == "tensorrt");
  SnapshotWriter snapshots(app.snapshotDir, app.snapshotKeep);
  MqttPublisher mqtt;
  bool mqttOn = !app.mqtt.host.empty() &&
                mqtt.connect(app.mqtt.host, app.mqtt.port);

  // Steady-clock milliseconds of the latest result, read by /health
  using Clock = std::chrono::steady_clock;
  auto nowMs = [] {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               Clock::now().time_since_epoch())
        .count();
  };
  std::atomic<long long> lastResultMs{nowMs()};
  auto anyConnected = [&pipeline] {
    for (size_t i = 0; i < pipeline.numStreams(); i++) {
      if (pipeline.streamStats(i).connected) return true;
    }
    return false;
  };

  SseServerConfig sseConfig;
  sseConfig.port = app.port;
  sseConfig.bindAddress = app.bind;
  sseConfig.staticDir = "web/";
  sseConfig.numStreams = static_cast<int>(app.sources.size());
  sseConfig.streamsJson = streamsJson(app);
  sseConfig.metrics = [&] {
    return renderMetrics(metrics, pipeline, streams, power);
  };
  // Healthy: results are flowing. A source that is away is reported by
  // va_stream_connected, not here, unless every source is away.
  sseConfig.healthy = [&] {
    return pipeline.isRunning() && anyConnected() &&
           nowMs() - lastResultMs < 10000;
  };
  SseServer sseServer(sseConfig);
  sseServer.start();
  if (!sseServer.isRunning()) {
    pipeline.stop();
    return -1;
  }

  std::cout << "Pipeline running. Press "
            << (app.headless ? "Ctrl+C" : "'q' or Ctrl+C") << " to stop.\n";
  std::cout << "Dashboard: http://localhost:" << app.port << "\n"
            << std::flush;

  // Create the windows up front so waitKey() can pump events before the
  // first frame arrives
  if (!app.headless) {
    for (const auto& stream : streams) cv::namedWindow(stream.id);
  }

  bool failed = false;
  // Ends when the pipeline stops: a worker failed, or every source ended
  while (!gShutdown && pipeline.isRunning()) {
    // Every pass, not only when a frame arrived, so the window stays
    // responsive and 'q' works while the camera stalls
    if (!app.headless && cv::waitKey(1) == 'q') break;

    auto resultOpt =
        pipeline.resultQueue().tryPopFor(std::chrono::milliseconds(10));
    if (!resultOpt) {
      // Stopped and drained: every source has ended
      if (pipeline.resultQueue().isStopped()) break;
      // Watchdog: a source is connected but nothing comes out. Exit so the
      // supervisor (Docker restart policy, systemd) starts a fresh process.
      if (app.watchdogSeconds > 0 && anyConnected() &&
          nowMs() - lastResultMs > app.watchdogSeconds * 1000LL) {
        std::cerr << "Watchdog: no result for " << app.watchdogSeconds
                  << " s, exiting\n";
        failed = true;
        break;
      }
      continue;
    }
    lastResultMs = nowMs();

    auto& result = *resultOpt;
    size_t s = static_cast<size_t>(result.streamId);
    StreamState& stream = streams[s];
    // Workers finish out of order; never step back to an older frame. The
    // tracker relies on this too.
    if (result.frameId < stream.nextFrameId) continue;
    stream.nextFrameId = result.frameId + 1;
    stream.fps.update();

    if (!app.classes.empty()) {
      auto& dets = result.detections;
      dets.erase(std::remove_if(dets.begin(), dets.end(),
                                [&app](const Detection& d) {
                                  return std::find(app.classes.begin(),
                                                   app.classes.end(),
                                                   d.classId) ==
                                         app.classes.end();
                                }),
                 dets.end());
    }

    std::vector<Event> events;
    if (stream.tracker) {
      stream.tracker->update(result.detections);
      events = stream.rules->update(result.detections, unixNow());
    }

    // Drawing costs more than the rules; skip it when nobody will see it
    bool needFrame = !app.headless || sseServer.hasViewers(s) ||
                     (!events.empty() && !app.snapshotDir.empty());
    if (needFrame) {
      drawDetections(result.frame, result.detections);
      if (stream.rules) drawRules(result.frame, *stream.rules);
      drawOverlay(result.frame, stream.fps.fps, result.inferenceLatencyMs,
                  static_cast<int>(result.detections.size()), result.frameId);
    }

    auto json = serializeResult(result, stream.fps.fps);
    json["stream"] = stream.id;
    if (!events.empty()) {
      // One snapshot per frame, shared by the events it raised
      std::string snapshot = snapshots.save(result.frame, events.front());
      nlohmann::json eventList = nlohmann::json::array();
      for (auto& event : events) {
        event.snapshot = snapshot;
        auto eventJson = serializeEvent(event);
        if (mqttOn) {
          mqtt.publish(app.mqtt.topic + "/" + event.stream + "/" + event.type,
                       eventJson.dump());
        }
        eventList.push_back(std::move(eventJson));
      }
      json["events"] = std::move(eventList);
    }
    if (stream.rules) {
      json["zones"] = stream.rules->zoneCounts();
      json["lines_in"] = stream.rules->lineIn();
      json["lines_out"] = stream.rules->lineOut();
    }

    {
      std::lock_guard<std::mutex> lock(metrics.mutex);
      auto& m = metrics.streams[s];
      m.fps = stream.fps.fps;
      m.latencyMs = result.inferenceLatencyMs;
      m.frames++;
      m.detections = static_cast<int>(result.detections.size());
      for (const auto& event : events) m.events[event.type]++;
      if (stream.rules) {
        m.zoneCounts = stream.rules->zoneCounts();
        m.lineIn = stream.rules->lineIn();
        m.lineOut = stream.rules->lineOut();
      }
      auto smooth = [](double& avg, double value) {
        avg = avg == 0.0 ? value : avg * 0.95 + value * 0.05;
      };
      smooth(metrics.decodeMs, result.decodeMs);
      smooth(metrics.preMs, result.preMs);
      smooth(metrics.inferMs, result.inferMs);
      smooth(metrics.postMs, result.postMs);
    }

    if (!app.headless) cv::imshow(stream.id, result.frame);
    sseServer.broadcast(json.dump(), s);
    sseServer.updateFrame(result.frame, s);
  }

  // A worker that gave up clears isRunning(); sources that simply ended
  // (video files) leave it set
  failed = failed || !pipeline.isRunning();

  sseServer.stop();
  pipeline.stop();
  if (!app.headless) cv::destroyAllWindows();

  return failed ? -1 : 0;
}

// ── CLI parsing ─────────────────────────────────────────────────────────────
void printUsage(const char* prog) {
  std::cout
      << "Usage:\n"
      << "  " << prog << " [OPTIONS]                 Live mode\n"
      << "  " << prog << " --benchmark OPTIONS        Benchmark mode\n"
      << "  " << prog << " --eval-coco DIR OPTIONS    Detections for mAP\n\n"
      << "General options (flags override the config file):\n"
      << "  --config FILE    YAML config: sources, zones, lines, MQTT, ...\n"
      << "  --device N       Camera device index (default: 0)\n"
      << "  --rtsp URL       RTSP or HTTP stream, decoded by FFmpeg\n"
      << "  --gst PIPELINE   GStreamer pipeline ending in appsink (CSI\n"
      << "                   camera, hardware decode, ...)\n"
      << "  --streams N      Open the source given on the command line N\n"
      << "                   times, to measure how many cameras fit\n"
      << "  --model PATH     Model path (default: models/yolov8n.onnx)\n"
      << "  --backend NAME   cpu, cuda, tensorrt, openvino:CPU,\n"
      << "                   openvino:GPU or openvino:NPU (default: cpu)\n"
      << "  --precision P    tensorrt only: fp32, fp16 or int8 (default:\n"
      << "                   fp16)\n"
      << "  --conf X         Confidence threshold (default: 0.5)\n"
      << "  --port N         Dashboard port (default: 9001)\n"
      << "  --bind ADDR      Bind address (default: 127.0.0.1)\n"
      << "  --workers N      Inference worker threads (default: 2, or 1 with\n"
      << "                   an accelerator)\n"
      << "  --headless       No local window; dashboard only\n\n"
      << "Benchmark options:\n"
      << "  --input FILE     Input video file (required)\n"
      << "  --frames N       Number of frames to measure (default: 300)\n"
      << "  --warmup N       Frames to skip first (default: 30)\n"
      << "  --output FILE    Write the report to a Markdown file\n"
      << "  --csv FILE       Append one line with the results\n\n"
      << "Evaluation options:\n"
      << "  --eval-coco DIR  Directory of .jpg images\n"
      << "  --eval-out FILE  Detections JSON (default: detections.json)\n";
}

int main(int argc, char* argv[]) {
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  auto parseIntArg = [&](const char* val, const char* name) -> int {
    try {
      return std::stoi(val);
    } catch (const std::exception&) {
      std::cerr << "Error: invalid value for " << name << ": " << val << "\n";
      std::exit(1);
    }
  };

  // The config file is read first so that flags, in any order, override it
  AppConfig app;
  for (int i = 1; i + 1 < argc; i++) {
    if (std::strcmp(argv[i], "--config") == 0 && !loadConfig(argv[i + 1], app)) {
      return -1;
    }
  }

  bool benchmarkMode = false;
  std::string inputVideo;
  int maxFrames = 300;
  int warmupFrames = 30;
  std::string outputReport;
  std::string csvPath;
  std::string evalDir;
  std::string evalOut = "detections.json";
  // A source given on the command line replaces the ones from the config
  CaptureConfig cliSource;
  bool cliSourceGiven = false;
  int streamCopies = 1;

  for (int i = 1; i < argc; i++) {
    auto is = [&](const char* flag) { return std::strcmp(argv[i], flag) == 0; };
    bool hasValue = i + 1 < argc;
    if (is("--benchmark")) {
      benchmarkMode = true;
    } else if (is("--config") && hasValue) {
      i++;  // loaded above
    } else if (is("--input") && hasValue) {
      inputVideo = argv[++i];
    } else if (is("--frames") && hasValue) {
      maxFrames = parseIntArg(argv[++i], "--frames");
    } else if (is("--warmup") && hasValue) {
      warmupFrames = parseIntArg(argv[++i], "--warmup");
    } else if (is("--output") && hasValue) {
      outputReport = argv[++i];
    } else if (is("--csv") && hasValue) {
      csvPath = argv[++i];
    } else if (is("--eval-coco") && hasValue) {
      evalDir = argv[++i];
    } else if (is("--eval-out") && hasValue) {
      evalOut = argv[++i];
    } else if (is("--device") && hasValue) {
      cliSource.deviceIndex = parseIntArg(argv[++i], "--device");
      cliSourceGiven = true;
    } else if (is("--rtsp") && hasValue) {
      cliSource.videoPath = argv[++i];
      cliSourceGiven = true;
    } else if (is("--gst") && hasValue) {
      cliSource.gstPipeline = argv[++i];
      cliSourceGiven = true;
    } else if (is("--streams") && hasValue) {
      streamCopies = parseIntArg(argv[++i], "--streams");
    } else if (is("--model") && hasValue) {
      app.model = argv[++i];
    } else if (is("--port") && hasValue) {
      app.port = parseIntArg(argv[++i], "--port");
    } else if (is("--bind") && hasValue) {
      app.bind = argv[++i];
    } else if (is("--workers") && hasValue) {
      app.workers = parseIntArg(argv[++i], "--workers");
    } else if ((is("--backend") || is("--provider")) && hasValue) {
      // --provider is the old name of --backend
      app.backend = argv[++i];
    } else if (is("--precision") && hasValue) {
      app.precision = argv[++i];
    } else if (is("--conf") && hasValue) {
      try {
        app.confidence = std::stof(argv[++i]);
      } catch (const std::exception&) {
        std::cerr << "Error: invalid value for --conf: " << argv[i] << "\n";
        return 1;
      }
    } else if (is("--headless")) {
      app.headless = true;
    } else if (is("--help")) {
      printUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Warning: unrecognized option: " << argv[i] << "\n";
    }
  }

  // Validate
  if (app.port < 1 || app.port > 65535) {
    std::cerr << "Error: --port must be between 1 and 65535\n";
    return -1;
  }
  if (!isKnownBackend(app.backend)) {
    std::cerr << "Error: --backend must be cpu, cuda, tensorrt or "
                 "openvino:<DEVICE>\n";
    return -1;
  }
  if (app.precision != "fp32" && app.precision != "fp16" &&
      app.precision != "int8") {
    std::cerr << "Error: --precision must be fp32, fp16 or int8\n";
    return -1;
  }
  // Each worker owns a session: on an accelerator a second one duplicates
  // the model in memory while the device still runs them one after the other
  if (app.workers == 0) app.workers = app.backend == "cpu" ? 2 : 1;
  if (app.workers < 1 || app.workers > 16) {
    std::cerr << "Error: --workers must be between 1 and 16\n";
    return -1;
  }
  if (streamCopies < 1 || streamCopies > 64) {
    std::cerr << "Error: --streams must be between 1 and 64\n";
    return -1;
  }

  if (!evalDir.empty()) return runEval(app, evalDir, evalOut);

  if (benchmarkMode) {
    if (inputVideo.empty()) {
      std::cerr << "Error: --benchmark requires --input VIDEO_FILE\n";
      printUsage(argv[0]);
      return -1;
    }
    if (maxFrames <= 0 || warmupFrames < 0) {
      std::cerr << "Error: --frames must be positive and --warmup not "
                   "negative\n";
      return -1;
    }
    return runBenchmark(app, inputVideo, maxFrames, warmupFrames,
                        outputReport, csvPath);
  }

  // No source anywhere means camera 0, as before
  if (cliSourceGiven || app.sources.empty()) {
    app.sources.clear();
    for (int i = 0; i < streamCopies; i++) {
      SourceConfig source;
      source.capture = cliSource;
      source.capture.id = "cam" + std::to_string(i);
      app.sources.push_back(source);
    }
  }

  return runLive(app);
}

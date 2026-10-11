#pragma once

#include "latency-tracker.hpp"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// Generate a Markdown benchmark report from collected metrics
struct BenchmarkResults {
  double fpsMean = 0.0;
  uint64_t totalFrames = 0;
  LatencyTracker latency;
  std::string modelName = "yolov8n";
  int inputWidth = 640;
  int inputHeight = 640;
  std::string device = "CPU";
  std::string precision;
  // Mean time per frame in each stage, milliseconds
  double decodeMs = 0.0;
  double preMs = 0.0;
  double inferMs = 0.0;
  double postMs = 0.0;
  // Mean power over the measured frames; negative when it cannot be read
  double watts = -1.0;
  std::string powerSource = "unavailable";
  uint64_t warmupFrames = 0;
};

inline std::string generateReport(const BenchmarkResults& r) {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(1);

  ss << "# Benchmark Results\n\n";

  ss << "## Configuration\n";
  ss << "| Setting | Value |\n";
  ss << "|---------|-------|\n";
  ss << "| Model | " << r.modelName << " |\n";
  ss << "| Input Size | " << r.inputWidth << "x" << r.inputHeight << " |\n";
  ss << "| Backend | " << r.device << " |\n";
  ss << "| Precision | " << r.precision << " |\n";
  ss << "| Frames Measured | " << r.totalFrames << " |\n";
  ss << "| Warm-up Frames (not measured) | " << r.warmupFrames << " |\n\n";

  ss << "## Performance\n";
  ss << "| Metric | Value |\n";
  ss << "|--------|-------|\n";
  ss << "| FPS (mean) | " << r.fpsMean << " |\n";
  ss << "| Latency p50 | " << r.latency.p50() << " ms |\n";
  ss << "| Latency p95 | " << r.latency.p95() << " ms |\n";
  ss << "| Latency p99 | " << r.latency.p99() << " ms |\n";
  ss << "| Latency max | " << r.latency.max() << " ms |\n";
  ss << "| Latency min | " << r.latency.min() << " ms |\n";
  ss << "| Latency mean | " << r.latency.mean() << " ms |\n";
  ss << "| Latency stddev | " << r.latency.stdDev() << " ms |\n";
  ss << "| Total samples | " << r.latency.count() << " |\n\n";

  ss << std::setprecision(2);
  ss << "## Time per Stage (mean)\n";
  ss << "| Stage | ms |\n";
  ss << "|-------|----|\n";
  ss << "| Decode | " << r.decodeMs << " |\n";
  ss << "| Pre-process | " << r.preMs << " |\n";
  ss << "| Inference | " << r.inferMs << " |\n";
  ss << "| Post-process | " << r.postMs << " |\n\n";

  ss << "## Power\n";
  ss << "| Metric | Value |\n";
  ss << "|--------|-------|\n";
  ss << "| Measured | " << r.powerSource << " |\n";
  if (r.watts > 0.0) {
    ss << "| Power (mean) | " << r.watts << " W |\n";
    ss << "| FPS per watt | " << r.fpsMean / r.watts << " |\n";
  } else {
    ss << "| Power (mean) | n/a |\n";
  }

  return ss.str();
}

// One CSV line per run, for tables that compare backends:
// model,backend,precision,frames,fps,p50,p95,p99,decode,pre,infer,post,
// watts,fps_per_watt
inline bool appendCsv(const BenchmarkResults& r, const std::string& csvPath) {
  bool isNew = !std::ifstream(csvPath).good();
  std::ofstream file(csvPath, std::ios::app);
  if (!file) {
    std::cerr << "Failed to write " << csvPath << "\n";
    return false;
  }
  if (isNew) {
    file << "model,backend,precision,frames,fps,p50_ms,p95_ms,p99_ms,"
            "decode_ms,pre_ms,infer_ms,post_ms,watts,fps_per_watt\n";
  }
  file << std::fixed << std::setprecision(2) << r.modelName << ","
       << r.device << "," << r.precision << "," << r.totalFrames << ","
       << r.fpsMean << "," << r.latency.p50() << "," << r.latency.p95() << ","
       << r.latency.p99() << "," << r.decodeMs << "," << r.preMs << ","
       << r.inferMs << "," << r.postMs << ",";
  if (r.watts > 0.0) {
    file << r.watts << "," << r.fpsMean / r.watts << "\n";
  } else {
    file << ",\n";
  }
  return true;
}

inline bool writeReport(const BenchmarkResults& r,
                         const std::string& outputPath) {
  std::ofstream file(outputPath);
  if (!file) {
    std::cerr << "Failed to write report to: " << outputPath << "\n";
    return false;
  }
  file << generateReport(r);
  std::cout << "Benchmark report written to: " << outputPath << "\n";
  return true;
}

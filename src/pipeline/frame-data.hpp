#pragma once

#include "inference/detection.hpp"

#include <chrono>
#include <opencv2/core.hpp>
#include <vector>

// Data passed from capture thread to inference workers
struct FrameData {
  uint64_t frameId;
  std::chrono::steady_clock::time_point captureTime;
  cv::Mat frame;
};

// Result from inference worker, consumed by output/display
struct AnalyticsResult {
  uint64_t frameId;
  std::chrono::steady_clock::time_point captureTime;
  std::chrono::steady_clock::time_point inferenceDoneTime;
  cv::Mat frame;  // original frame for overlay drawing
  std::vector<Detection> detections;
  double inferenceLatencyMs;  // capture → inference done, includes queue wait
  double detectMs = 0.0;      // time inside detect() only
};

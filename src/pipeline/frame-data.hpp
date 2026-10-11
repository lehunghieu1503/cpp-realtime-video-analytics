#pragma once

#include "capture/device-frame.hpp"
#include "inference/detection.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>

// Data passed from a capture thread to the inference workers
struct FrameData {
  uint64_t frameId;  // counts per stream
  std::chrono::steady_clock::time_point captureTime;
  cv::Mat frame;  // empty when the frame is in device memory
  // Set instead of `frame` by a capture that keeps frames on the device
  std::shared_ptr<DeviceFrame> device;
  int streamId = 0;       // index into PipelineConfig::sources
  double decodeMs = 0.0;  // time inside the capture read (decode included)
  // The source's own timestamp of this frame, see
  // IVideoCapture::sourceTimestampNs()
  std::optional<uint64_t> sourcePtsNs;
};

// Result from inference worker, consumed by output/display
struct AnalyticsResult {
  uint64_t frameId;
  std::chrono::steady_clock::time_point captureTime;
  std::chrono::steady_clock::time_point inferenceDoneTime;
  // Original frame for overlay drawing; empty when the frame stayed in
  // device memory, see cpuFrame()
  cv::Mat frame;
  std::shared_ptr<DeviceFrame> device;
  std::vector<Detection> detections;
  double inferenceLatencyMs;  // capture → inference done, includes queue wait
  double detectMs = 0.0;      // time inside detect() only
  int streamId = 0;
  // Per-stage times; detectMs = preMs + inferMs + postMs
  double decodeMs = 0.0;
  double preMs = 0.0;
  double inferMs = 0.0;
  double postMs = 0.0;
  std::optional<uint64_t> sourcePtsNs;
  // Size of the source frame, known without copying a device frame to the CPU
  int imageWidth = 0;
  int imageHeight = 0;

  // The frame in system memory, copied from the device the first time it is
  // asked for. Most frames never are: nobody draws on them.
  cv::Mat& cpuFrame() {
    if (frame.empty() && device) frame = device->toBgr();
    return frame;
  }
};

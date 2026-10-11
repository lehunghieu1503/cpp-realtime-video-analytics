#pragma once

#include "device-frame.hpp"

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

struct CaptureConfig {
  std::string id = "cam0";  // stream name in events, metrics and the dashboard
  int deviceIndex = 0;
  int width = 640;
  int height = 480;
  int fps = 30;
  std::string codec = "MJPG";
  int bufferSize = 1;
  // if set, open this video file or network URL (rtsp://, http://) instead of
  // the camera
  std::string videoPath;
  // if set, open this GStreamer pipeline (ending in appsink) as the camera,
  // e.g. a Jetson CSI sensor through nvarguscamerasrc
  std::string gstPipeline;

  // A live source is reopened when it fails; a video file just ends
  bool isLive() const {
    return videoPath.empty() || videoPath.find("://") != std::string::npos;
  }
};

class IVideoCapture {
public:
  virtual ~IVideoCapture() = default;
  virtual bool open(const CaptureConfig& config) = 0;
  // A capture that keeps frames in device memory returns true with `frame`
  // empty; the frame is then fetched with takeDeviceFrame()
  virtual bool read(cv::Mat& frame) = 0;
  virtual std::shared_ptr<DeviceFrame> takeDeviceFrame() { return nullptr; }
  virtual void release() = 0;
  virtual bool isOpened() const = 0;
  virtual double getActualFps() const = 0;
  virtual std::string getInfo() const = 0;
};

// The capture for a source: the native GStreamer one for a GStreamer
// pipeline when the build has it, OpenCV's otherwise (cameras, files, RTSP
// through FFmpeg)
std::unique_ptr<IVideoCapture> createCapture(const CaptureConfig& config);

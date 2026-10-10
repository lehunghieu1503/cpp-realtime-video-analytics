#pragma once

#include <opencv2/opencv.hpp>
#include <string>

struct CaptureConfig {
  int deviceIndex = 0;
  int width = 640;
  int height = 480;
  int fps = 30;
  std::string codec = "MJPG";
  int bufferSize = 1;
  std::string videoPath;  // if set, open video file instead of camera
  // if set, open this GStreamer pipeline (ending in appsink) as the camera,
  // e.g. a Jetson CSI sensor through nvarguscamerasrc
  std::string gstPipeline;
};

class IVideoCapture {
public:
  virtual ~IVideoCapture() = default;
  virtual bool open(const CaptureConfig& config) = 0;
  virtual bool read(cv::Mat& frame) = 0;
  virtual void release() = 0;
  virtual bool isOpened() const = 0;
  virtual double getActualFps() const = 0;
  virtual std::string getInfo() const = 0;
};

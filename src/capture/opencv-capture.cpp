#include "opencv-capture.hpp"

#include <cstring>
#include <iostream>
#include <sstream>

bool OpenCvCapture::open(const CaptureConfig& config) {
  config_ = config;

  // Open video file or camera device
  if (!config.videoPath.empty()) {
    cap_.open(config.videoPath);
    if (!cap_.isOpened()) {
      std::cerr << "Failed to open video: " << config.videoPath << "\n";
      return false;
    }
    return true;  // skip codec/resolution settings for video files
  }

  if (!config.gstPipeline.empty()) {
    cap_.open(config.gstPipeline, cv::CAP_GSTREAMER);
    if (!cap_.isOpened()) {
      std::cerr << "Failed to open GStreamer pipeline: " << config.gstPipeline
                << "\n";
      return false;
    }
    return true;  // size, rate and format are set inside the pipeline
  }

  cap_.open(config.deviceIndex);
  if (!cap_.isOpened()) {
    std::cerr << "Failed to open camera " << config.deviceIndex << "\n";
    return false;
  }

  // Set codec first — resolution/FPS limits depend on codec
  if (config.codec.size() == 4) {
    cap_.set(cv::CAP_PROP_FOURCC,
             cv::VideoWriter::fourcc(config.codec[0], config.codec[1],
                                     config.codec[2], config.codec[3]));
  }

  cap_.set(cv::CAP_PROP_FRAME_WIDTH, config.width);
  cap_.set(cv::CAP_PROP_FRAME_HEIGHT, config.height);
  cap_.set(cv::CAP_PROP_FPS, config.fps);
  cap_.set(cv::CAP_PROP_BUFFERSIZE, config.bufferSize);

  return true;
}

bool OpenCvCapture::read(cv::Mat& frame) {
  cap_ >> frame;
  return !frame.empty();
}

void OpenCvCapture::release() {
  if (cap_.isOpened()) {
    cap_.release();
  }
}

bool OpenCvCapture::isOpened() const {
  return cap_.isOpened();
}

double OpenCvCapture::getActualFps() const {
  double fps = cap_.get(cv::CAP_PROP_FPS);
  return (fps > 0) ? fps : 30.0;
}

std::string OpenCvCapture::getInfo() const {
  int fourcc = static_cast<int>(cap_.get(cv::CAP_PROP_FOURCC));
  char fmt[5] = {0};
  std::memcpy(fmt, &fourcc, 4);

  std::ostringstream ss;
  ss << "Camera " << config_.deviceIndex << ": "
     << static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)) << "x"
     << static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)) << " @ "
     << cap_.get(cv::CAP_PROP_FPS) << " FPS"
     << " format=" << fmt;
  return ss.str();
}

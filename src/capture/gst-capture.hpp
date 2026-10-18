#pragma once

#include "video-capture.hpp"

typedef struct _GstElement GstElement;

// Runs a GStreamer pipeline directly, through its appsink, instead of going
// through OpenCV's wrapper. The pipeline description must contain an
// appsink that delivers video/x-raw in BGR, BGRx, RGBA or RGBx. On a Jetson
// it may instead deliver video/x-raw(memory:NVMM),format=RGBA: the frames
// then stay in the hardware buffers (see NvmmFrame).
//
// Owning the pipeline gives what the wrapper hides: the reason a pipeline
// failed (from its bus), end-of-stream as a distinct result, and access to
// the buffers themselves, which a later step uses to keep frames on the GPU.
class GstCapture : public IVideoCapture {
public:
  GstCapture() = default;
  ~GstCapture() override { release(); }
  GstCapture(const GstCapture&) = delete;
  GstCapture& operator=(const GstCapture&) = delete;

  bool open(const CaptureConfig& config) override;
  bool read(cv::Mat& frame) override;
  std::shared_ptr<DeviceFrame> takeDeviceFrame() override {
    return std::move(deviceFrame_);
  }
  std::optional<uint64_t> sourceTimestampNs() const override { return pts_; }
  void release() override;
  bool isOpened() const override { return pipeline_ != nullptr; }
  double getActualFps() const override { return fps_ > 0 ? fps_ : 30.0; }
  std::string getInfo() const override;

private:
  // Prints an error waiting on the pipeline's bus; true if there was one
  bool reportBusError();

  CaptureConfig config_;
  GstElement* pipeline_ = nullptr;
  GstElement* sink_ = nullptr;
  int width_ = 0;
  int height_ = 0;
  double fps_ = 0.0;
  std::string format_;
  std::shared_ptr<DeviceFrame> deviceFrame_;
  std::optional<uint64_t> pts_;  // of the last frame read
};

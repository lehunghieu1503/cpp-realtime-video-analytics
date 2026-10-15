#pragma once

#include "capture/device-frame.hpp"

#include <cstddef>
#include <memory>

// A packed 8-bit BGR image in GPU memory: a camera frame a backend uploaded,
// or the bird's-eye view stitched there. Whoever reads it next on the GPU
// reads it where it is; toBgr() is the one copy back, for whoever draws.
// Its memory comes from a pool and goes back there when the frame is
// released, from any thread: cudaMalloc and cudaFree synchronise the whole
// device.
class CudaFrame : public DeviceFrame {
public:
  // Throws std::runtime_error when the memory cannot be had
  static std::shared_ptr<CudaFrame> create(int width, int height);
  // A frame holding a copy of `bgr` (CV_8UC3); returns once it is copied
  static std::shared_ptr<CudaFrame> upload(const cv::Mat& bgr);
  ~CudaFrame() override;
  CudaFrame(const CudaFrame&) = delete;
  CudaFrame& operator=(const CudaFrame&) = delete;

  int width() const override { return width_; }
  int height() const override { return height_; }
  cv::Mat toBgr() override;

  unsigned char* data() const { return data_; }
  int pitch() const { return width_ * 3; }
  size_t bytes() const { return static_cast<size_t>(pitch()) * height_; }

private:
  CudaFrame(unsigned char* data, int width, int height)
      : data_(data), width_(width), height_(height) {}

  unsigned char* data_;
  int width_;
  int height_;
};

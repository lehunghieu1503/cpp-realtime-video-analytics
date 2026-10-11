#pragma once

#include "device-frame.hpp"

typedef struct _GstSample GstSample;
struct NvBufSurface;

// An RGBA frame in a Jetson NVMM buffer (the memory hardware decoders,
// cameras and nvvidconv work in), kept alive by its GStreamer sample. The
// buffer returns to its pool when the frame is destroyed.
class NvmmFrame : public DeviceFrame {
public:
  // Takes over the caller's reference to `sample`. Throws std::runtime_error
  // when the buffer is not a single RGBA surface.
  explicit NvmmFrame(GstSample* sample);
  ~NvmmFrame() override;
  NvmmFrame(const NvmmFrame&) = delete;
  NvmmFrame& operator=(const NvmmFrame&) = delete;

  int width() const override { return width_; }
  int height() const override { return height_; }
  cv::Mat toBgr() override;

  // The buffer as an EGLImage, which CUDA can register and read in place.
  // Valid until unmapEglImage() or destruction. Throws on failure.
  void* mapEglImage();
  void unmapEglImage();

private:
  GstSample* sample_;
  NvBufSurface* surface_ = nullptr;
  void* mapInfo_ = nullptr;  // GstMapInfo, kept mapped for the frame's life
  int width_ = 0;
  int height_ = 0;
  bool eglMapped_ = false;
};

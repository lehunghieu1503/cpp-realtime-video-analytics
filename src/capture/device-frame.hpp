#pragma once

#include <opencv2/core.hpp>

// A captured frame that lives in device memory, such as a hardware
// decoder's buffer. A backend that knows the concrete type reads it where it
// is; everything else asks for a CPU copy, which is made only then.
class DeviceFrame {
public:
  virtual ~DeviceFrame() = default;
  virtual int width() const = 0;
  virtual int height() const = 0;
  // BGR copy in system memory
  virtual cv::Mat toBgr() = 0;
};

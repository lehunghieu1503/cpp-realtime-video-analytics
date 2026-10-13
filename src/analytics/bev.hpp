#pragma once

#include "lane-geometry.hpp"

#include <cstdint>
#include <map>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

// The bird's-eye view: the frames of the cameras around the vehicle, put on
// the ground as one image. Forward is up, the vehicle's origin is in the
// middle, and a pixel is metresPerPixel of ground: column u and row v show
//
//   x = (height / 2 - v) * metresPerPixel   (forward)
//   y = (u - width / 2) * metresPerPixel    (right)
//
// A pixel is taken from the camera nearest to that place on the ground,
// among those that have it in their image; where none has, it stays black.
// The ground is taken as flat. The reference is
// simulation/video_analytics_simulation/bev.py; both are tested against
// tests/fixtures/bev.json.
struct BevConfig {
  struct Camera {
    std::string source;  // id of the source whose frames it is
    CameraRig rig;
  };
  std::string id;  // the view's results come under it; empty: no view
  int width = 0;
  int height = 0;
  double metresPerPixel = 0.0;
  std::vector<Camera> cameras;

  bool isSet() const { return !id.empty(); }
};

class BevStitcher {
public:
  // Moments that wait for a camera's frame; older ones are dropped
  static constexpr size_t kPending = 8;

  BevStitcher(int width, int height, double metresPerPixel,
              const std::vector<CameraRig>& rigs);

  int width() const { return source_.cols; }
  int height() const { return source_.rows; }
  // The camera a pixel is taken from, -1 for none
  int source(int column, int row) const;
  // Where in that camera's image, counted from the centre of its first pixel
  cv::Point2f place(int camera, int column, int row) const;

  // A camera's frame of the moment `stamp`, as captured. Returns the view
  // of that moment once every camera has given its frame of it, and an
  // empty image until then. Frames need not come in order; the frame's
  // pixels are taken at once, so it may be drawn on afterwards. Throws when
  // the frame is not of the size the camera's calibration says.
  cv::Mat offer(size_t camera, uint64_t stamp, const cv::Mat& bgr);

  // Views given out, and moments dropped because a camera's frame of them
  // never came: when the device does not keep up, each camera loses other
  // frames and few moments are whole
  uint64_t made() const { return made_; }
  uint64_t dropped() const { return dropped_; }

private:
  struct Pending {
    cv::Mat view;
    unsigned have = 0;  // bit per camera
  };
  std::vector<CameraRig> rigs_;
  cv::Mat source_;  // CV_8S
  std::vector<cv::Mat> mapX_, mapY_;
  std::map<uint64_t, Pending> pending_;
  uint64_t made_ = 0;
  uint64_t dropped_ = 0;
};

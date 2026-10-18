#pragma once

#include "lane-result.hpp"

#include <string>
#include <vector>

// A forward camera fixed to the vehicle, over a flat road. No lens
// distortion. Frames, signs and the id are those of
// simulation/video_analytics_simulation/calibration.py;
// both are tested against tests/fixtures/lane-calibration.json.
struct CameraRig {
  int width = 0;  // 0: no calibration
  int height = 0;
  double fovDeg = 90.0;  // horizontal
  // Camera position in the ego frame (x forward, y right, z up), metres
  double x = 0.0;
  double y = 0.0;
  double z = 1.5;
  double pitchDeg = 0.0;  // positive looks up
  double yawDeg = 0.0;    // positive looks right
  double rollDeg = 0.0;
  double groundZ = 0.0;  // height of the road in the ego frame

  bool isSet() const { return width > 0 && height > 0; }
  // Empty when usable; the reason otherwise
  std::string check() const;
  // Names these exact numbers, e.g. rig_5a5f398f89b3b347
  std::string calibrationId() const;
  double focal() const;
  // Where the ray through a pixel meets the road. False for a pixel outside
  // the image, at or above the horizon, or farther than maxRange metres.
  bool uvToGround(double u, double v, double& groundX, double& groundY,
                  double maxRange = 150.0) const;
  // The pixel of a point of the ego frame, and its depth along the optical
  // axis; a depth that is not positive is behind the camera, and u, v are
  // then not set
  void project(double px, double py, double pz, double& u, double& v,
               double& depth) const;
};

struct LaneGeometryConfig {
  double roiXMin = 6.0;  // metres ahead of the vehicle's origin
  double roiXMax = 30.0;
  double referenceX = 8.0;
  int minPoints = 5;         // per boundary, inside the region
  double minSupport = 5.0;   // metres both boundaries must share
  double maxResidual = 0.25; // RMS distance of points to the curve, metres
  double widthMin = 2.5;
  double widthMax = 4.6;
};

// Fits a boundary to ground points; false when they do not support one
bool fitBoundary(const std::vector<cv::Point2d>& ground,
                 const LaneGeometryConfig& config, LaneCurve& out);

// Lane pixels of one frame to the lane in metres: the boundaries either
// side of the vehicle, the centreline between them, and the errors at the
// reference distance. `lanes` is kept in the result as `pixels`.
LaneResult estimateLane(const std::vector<std::vector<cv::Point>>& lanes,
                        const CameraRig& rig, const LaneGeometryConfig& config);

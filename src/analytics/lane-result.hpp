#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

// y(x) = a x^2 + b x + c on the road, in the ego frame: x forward, y right,
// metres; valid for xMin <= x <= xMax
struct LaneCurve {
  double a = 0.0;
  double b = 0.0;
  double c = 0.0;
  double xMin = 0.0;
  double xMax = 0.0;
  int points = 0;    // ground points the fit kept
  double rms = 0.0;  // their RMS distance to the curve, metres

  double at(double x) const { return a * x * x + b * x + c; }
};

// The lane the vehicle is in, as seen in one frame
struct LaneResult {
  bool valid = false;
  double confidence = 0.0;  // a quality score, not a probability
  std::string reason;       // why it is not valid
  // What the model found, in pixels of the source frame: one list per lane
  std::vector<std::vector<cv::Point>> pixels;
  LaneCurve left;
  LaneCurve right;
  LaneCurve centerline;
  // Where the three values below are taken; inside the centreline's range
  double referenceX = 0.0;
  double lateralError = 0.0;  // of the lane centre from the vehicle; + right
  double heading = 0.0;       // radians; + turns right
  double curvature = 0.0;     // 1/m; + curves right
  double preMs = 0.0;
  double inferMs = 0.0;
  double postMs = 0.0;
};

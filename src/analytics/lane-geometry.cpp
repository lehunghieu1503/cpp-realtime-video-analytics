#include "lane-geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <opencv2/core.hpp>

namespace {
const double kPi = 3.14159265358979323846;

double radians(double degrees) { return degrees * kPi / 180.0; }

// Rotation of a CARLA transform: columns are its forward, right, up axes
cv::Matx33d rotation(double rollDeg, double pitchDeg, double yawDeg) {
  double cy = std::cos(radians(yawDeg)), sy = std::sin(radians(yawDeg));
  double cr = std::cos(radians(rollDeg)), sr = std::sin(radians(rollDeg));
  double cp = std::cos(radians(pitchDeg)), sp = std::sin(radians(pitchDeg));
  return {cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr,
          cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr,
          sp,      -cp * sr,               cp * cr};
}

bool solveQuadratic(const std::vector<cv::Point2d>& points, LaneCurve& curve,
                    std::vector<double>& residual) {
  cv::Mat A(static_cast<int>(points.size()), 3, CV_64F);
  cv::Mat y(static_cast<int>(points.size()), 1, CV_64F);
  for (int i = 0; i < A.rows; i++) {
    A.at<double>(i, 0) = points[i].x * points[i].x;
    A.at<double>(i, 1) = points[i].x;
    A.at<double>(i, 2) = 1.0;
    y.at<double>(i) = points[i].y;
  }
  cv::Mat coefficients;
  if (!cv::solve(A, y, coefficients, cv::DECOMP_SVD)) return false;
  curve.a = coefficients.at<double>(0);
  curve.b = coefficients.at<double>(1);
  curve.c = coefficients.at<double>(2);
  residual.resize(points.size());
  for (size_t i = 0; i < points.size(); i++) {
    residual[i] = curve.at(points[i].x) - points[i].y;
  }
  return true;
}

double span(const std::vector<cv::Point2d>& points) {
  auto [low, high] = std::minmax_element(
      points.begin(), points.end(),
      [](const cv::Point2d& a, const cv::Point2d& b) { return a.x < b.x; });
  return high->x - low->x;
}

LaneResult invalid(LaneResult result, const char* reason,
                   const LaneGeometryConfig& config) {
  result.valid = false;
  result.confidence = 0.0;
  result.reason = reason;
  result.referenceX = config.referenceX;
  return result;
}
}  // namespace

std::string CameraRig::check() const {
  for (double v : {fovDeg, x, y, z, pitchDeg, yawDeg, rollDeg, groundZ}) {
    if (!std::isfinite(v)) return "calibration has a value that is not finite";
  }
  if (width <= 0 || height <= 0) return "calibration needs width and height";
  if (fovDeg < 1.0 || fovDeg > 170.0) {
    return "calibration needs a horizontal fov_deg of 1..170";
  }
  if (z <= groundZ) return "calibration: the camera must be above the road";
  return "";
}

std::string CameraRig::calibrationId() const {
  char text[320];
  std::snprintf(text, sizeof(text),
                "w=%d;h=%d;fov=%.6f;x=%.6f;y=%.6f;z=%.6f;pitch=%.6f;yaw=%.6f;"
                "roll=%.6f;ground=%.6f",
                width, height, fovDeg, x, y, z, pitchDeg, yawDeg, rollDeg,
                groundZ);
  // FNV-1a, 64 bit
  uint64_t hash = 0xCBF29CE484222325ULL;
  for (const char* c = text; *c; c++) {
    hash = (hash ^ static_cast<unsigned char>(*c)) * 0x100000001B3ULL;
  }
  char id[32];
  std::snprintf(id, sizeof(id), "rig_%016llx",
                static_cast<unsigned long long>(hash));
  return id;
}

double CameraRig::focal() const {
  return width / (2.0 * std::tan(radians(fovDeg) / 2.0));
}

bool CameraRig::uvToGround(double u, double v, double& groundX,
                           double& groundY, double maxRange) const {
  if (!(u >= 0 && u < width && v >= 0 && v < height)) return false;
  double f = focal();
  // The ray in the camera's frame (x forward, y right, z up), then the ego's
  cv::Vec3d ray = rotation(rollDeg, pitchDeg, yawDeg) *
                  cv::Vec3d(1.0, (u - width / 2.0) / f, -(v - height / 2.0) / f);
  if (!(ray[2] < -1e-6)) return false;  // at or above the horizon
  double t = (groundZ - z) / ray[2];
  double dx = ray[0] * t, dy = ray[1] * t;
  if (!(t > 0) || std::hypot(dx, dy) > maxRange) return false;
  groundX = x + dx;
  groundY = y + dy;
  return true;
}

void CameraRig::project(double px, double py, double pz, double& u, double& v,
                        double& depth) const {
  // Into the camera's frame: x forward, y right, z up
  cv::Vec3d camera = rotation(rollDeg, pitchDeg, yawDeg).t() *
                     cv::Vec3d(px - x, py - y, pz - z);
  depth = camera[0];
  if (!(depth > 0)) return;
  double f = focal();
  u = f * camera[1] / depth + width / 2.0;
  v = -f * camera[2] / depth + height / 2.0;
}

bool fitBoundary(const std::vector<cv::Point2d>& ground,
                 const LaneGeometryConfig& config, LaneCurve& out) {
  std::vector<cv::Point2d> points;
  for (const auto& p : ground) {
    if (p.x >= config.roiXMin && p.x <= config.roiXMax) points.push_back(p);
  }
  if (static_cast<int>(points.size()) < config.minPoints ||
      span(points) < config.minSupport) {
    return false;
  }

  LaneCurve curve;
  std::vector<double> residual;
  if (!solveQuadratic(points, curve, residual)) return false;

  // One pass of outlier rejection: a point of another marking, a kerb
  std::vector<double> magnitude(residual.size());
  for (size_t i = 0; i < residual.size(); i++) {
    magnitude[i] = std::abs(residual[i]);
  }
  std::sort(magnitude.begin(), magnitude.end());
  size_t n = magnitude.size();
  double median = n % 2 ? magnitude[n / 2]
                        : (magnitude[n / 2 - 1] + magnitude[n / 2]) / 2;
  double limit = std::max(0.15, 2.5 * median);
  std::vector<cv::Point2d> inliers;
  for (size_t i = 0; i < points.size(); i++) {
    if (std::abs(residual[i]) <= limit) inliers.push_back(points[i]);
  }
  if (inliers.size() < points.size() &&
      static_cast<int>(inliers.size()) >= config.minPoints) {
    points = std::move(inliers);
    if (!solveQuadratic(points, curve, residual)) return false;
  }

  double sum = 0.0;
  for (double r : residual) sum += r * r;
  curve.rms = std::sqrt(sum / residual.size());
  if (curve.rms > config.maxResidual || span(points) < config.minSupport) {
    return false;
  }
  curve.points = static_cast<int>(points.size());
  curve.xMin = curve.xMax = points[0].x;
  for (const auto& p : points) {
    curve.xMin = std::min(curve.xMin, p.x);
    curve.xMax = std::max(curve.xMax, p.x);
  }
  out = curve;
  return true;
}

LaneResult estimateLane(const std::vector<std::vector<cv::Point>>& lanes,
                        const CameraRig& rig,
                        const LaneGeometryConfig& config) {
  LaneResult result;
  result.pixels = lanes;

  // The boundary nearest to the vehicle on each side
  bool haveLeft = false, haveRight = false;
  double leftY = 0.0, rightY = 0.0;
  for (const auto& lane : lanes) {
    std::vector<cv::Point2d> ground;
    for (const auto& pixel : lane) {
      double gx, gy;
      if (rig.uvToGround(pixel.x, pixel.y, gx, gy)) ground.emplace_back(gx, gy);
    }
    LaneCurve curve;
    if (!fitBoundary(ground, config, curve)) continue;
    double x = std::clamp(config.referenceX, curve.xMin, curve.xMax);
    double y = curve.at(x);
    if (y < 0 && (!haveLeft || y > leftY)) {
      haveLeft = true;
      leftY = y;
      result.left = curve;
    }
    if (y >= 0 && (!haveRight || y < rightY)) {
      haveRight = true;
      rightY = y;
      result.right = curve;
    }
  }
  // One boundary is not a lane: no width is assumed to make a second
  if (!haveLeft || !haveRight) {
    return invalid(std::move(result), "missing_boundary", config);
  }

  double xMin = std::max(result.left.xMin, result.right.xMin);
  double xMax = std::min(result.left.xMax, result.right.xMax);
  if (xMax - xMin < config.minSupport) {
    return invalid(std::move(result), "insufficient_support", config);
  }
  double narrowest = 1e9, widest = -1e9;
  for (double x : {xMin, (xMin + xMax) / 2, xMax}) {
    double width = result.right.at(x) - result.left.at(x);
    narrowest = std::min(narrowest, width);
    widest = std::max(widest, width);
  }
  if (narrowest <= 0) return invalid(std::move(result), "crossing", config);
  if (narrowest < config.widthMin || widest > config.widthMax) {
    return invalid(std::move(result), "width", config);
  }

  LaneCurve& centre = result.centerline;
  centre.a = (result.left.a + result.right.a) / 2;
  centre.b = (result.left.b + result.right.b) / 2;
  centre.c = (result.left.c + result.right.c) / 2;
  centre.xMin = xMin;
  centre.xMax = xMax;

  double x = std::clamp(config.referenceX, xMin, xMax);
  double slope = 2 * centre.a * x + centre.b;
  result.valid = true;
  result.referenceX = x;
  result.lateralError = centre.at(x);
  result.heading = std::atan(slope);
  result.curvature = 2 * centre.a / std::pow(1 + slope * slope, 1.5);
  double worst = std::max(result.left.rms, result.right.rms);
  result.confidence =
      std::min(1.0, std::min(result.left.points, result.right.points) / 15.0) *
      (1.0 - worst / config.maxResidual);
  return result;
}

#include "bev.hpp"

#include <cmath>
#include <iterator>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace {
// Nearer to a camera's plane than this, the ground is not its
const double kNear = 0.15;
// Outside every image: remap leaves such a pixel as it is
const float kNowhere = -10.0f;
}  // namespace

BevStitcher::BevStitcher(int width, int height, double metresPerPixel,
                         const std::vector<CameraRig>& rigs)
    : rigs_(rigs), source_(height, width, CV_8S, cv::Scalar(-1)) {
  if (rigs.empty() || rigs.size() > 32) {
    throw std::invalid_argument("the bird's-eye view needs 1 to 32 cameras");
  }
  for (size_t i = 0; i < rigs.size(); i++) {
    mapX_.emplace_back(height, width, CV_32F, cv::Scalar(kNowhere));
    mapY_.emplace_back(height, width, CV_32F, cv::Scalar(kNowhere));
  }
  for (int row = 0; row < height; row++) {
    for (int column = 0; column < width; column++) {
      // The ground under the pixel's centre
      double x = (height / 2.0 - (row + 0.5)) * metresPerPixel;
      double y = ((column + 0.5) - width / 2.0) * metresPerPixel;
      double nearest = std::numeric_limits<double>::infinity();
      int taken = -1;
      double takenU = 0.0, takenV = 0.0;
      for (size_t i = 0; i < rigs.size(); i++) {
        const CameraRig& rig = rigs[i];
        double u = 0.0, v = 0.0, depth = 0.0;
        // The ground is the ego frame's z = 0, as the reference has it
        rig.project(x, y, 0.0, u, v, depth);
        if (!(depth > kNear) || !(u >= 0.5 && u <= rig.width - 0.5) ||
            !(v >= 0.5 && v <= rig.height - 0.5)) {
          continue;
        }
        double dx = x - rig.x, dy = y - rig.y, dz = -rig.z;
        double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (distance < nearest) {
          nearest = distance;
          taken = static_cast<int>(i);
          takenU = u;
          takenV = v;
        }
      }
      if (taken < 0) continue;
      source_.at<schar>(row, column) = static_cast<schar>(taken);
      mapX_[taken].at<float>(row, column) = static_cast<float>(takenU - 0.5);
      mapY_[taken].at<float>(row, column) = static_cast<float>(takenV - 0.5);
    }
  }
}

int BevStitcher::source(int column, int row) const {
  return source_.at<schar>(row, column);
}

cv::Point2f BevStitcher::place(int camera, int column, int row) const {
  return {mapX_[camera].at<float>(row, column),
          mapY_[camera].at<float>(row, column)};
}

cv::Mat BevStitcher::offer(size_t camera, uint64_t stamp, const cv::Mat& bgr) {
  const CameraRig& rig = rigs_.at(camera);
  if (bgr.cols != rig.width || bgr.rows != rig.height || bgr.type() != CV_8UC3) {
    throw std::runtime_error(
        "bird's-eye view: camera " + std::to_string(camera) + " gives " +
        std::to_string(bgr.cols) + "x" + std::to_string(bgr.rows) +
        ", its calibration says " + std::to_string(rig.width) + "x" +
        std::to_string(rig.height));
  }
  Pending& moment = pending_[stamp];
  if (moment.view.empty()) {
    moment.view = cv::Mat::zeros(source_.rows, source_.cols, CV_8UC3);
  }
  cv::remap(bgr, moment.view, mapX_[camera], mapY_[camera], cv::INTER_LINEAR,
            cv::BORDER_TRANSPARENT);
  moment.have |= 1u << camera;
  if (moment.have != (rigs_.size() == 32 ? ~0u : (1u << rigs_.size()) - 1)) {
    while (pending_.size() > kPending) {
      pending_.erase(pending_.begin());
      dropped_++;
    }
    return {};
  }
  cv::Mat view = moment.view;
  // Nothing older can still be wanted: results go out in order
  auto end = pending_.upper_bound(stamp);
  dropped_ += std::distance(pending_.begin(), end) - 1;
  pending_.erase(pending_.begin(), end);
  made_++;
  return view;
}

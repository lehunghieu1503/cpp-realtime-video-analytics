#include "bev.hpp"

#ifdef WITH_TENSORRT
#include "bev-remap.hpp"
#include "inference/backends/cuda-frame.hpp"

#include <cuda_runtime.h>
#endif

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

void BevStitcher::checkSize(size_t camera, int width, int height) const {
  const CameraRig& rig = rigs_.at(camera);
  if (width != rig.width || height != rig.height) {
    throw std::runtime_error(
        "bird's-eye view: camera " + std::to_string(camera) + " gives " +
        std::to_string(width) + "x" + std::to_string(height) +
        ", its calibration says " + std::to_string(rig.width) + "x" +
        std::to_string(rig.height));
  }
}

BevStitcher::Pending BevStitcher::finish(size_t camera, uint64_t stamp) {
  Pending& moment = pending_[stamp];
  moment.have |= 1u << camera;
  if (moment.have != allCameras()) {
    while (pending_.size() > kPending) {
      pending_.erase(pending_.begin());
      dropped_++;
    }
    return {};
  }
  Pending whole = std::move(moment);
  // Nothing older can still be wanted: results go out in order
  auto end = pending_.upper_bound(stamp);
  dropped_ += std::distance(pending_.begin(), end) - 1;
  pending_.erase(pending_.begin(), end);
  made_++;
  return whole;
}

cv::Mat BevStitcher::offer(size_t camera, uint64_t stamp, const cv::Mat& bgr) {
  checkSize(camera, bgr.cols, bgr.rows);
  if (bgr.type() != CV_8UC3) {
    throw std::runtime_error("bird's-eye view: a frame that is not 8-bit BGR");
  }
  Pending& moment = pending_[stamp];
  if (moment.view.empty()) {
    moment.view = cv::Mat::zeros(source_.rows, source_.cols, CV_8UC3);
  }
  cv::remap(bgr, moment.view, mapX_[camera], mapY_[camera], cv::INTER_LINEAR,
            cv::BORDER_TRANSPARENT);
  return finish(camera, stamp).view;
}

#ifdef WITH_TENSORRT
namespace {
void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("bird's-eye view: ") + what + ": " +
                             cudaGetErrorString(status));
  }
}
}  // namespace

struct BevStitcher::GpuTables {
  float* map = nullptr;  // x, y in its camera's image, per pixel of the view
  signed char* source = nullptr;  // as source_
  cudaStream_t stream = nullptr;

  ~GpuTables() {
    cudaFree(map);
    cudaFree(source);
    if (stream) cudaStreamDestroy(stream);
  }
};

bool BevStitcher::canUseGpu() {
  int devices = 0;
  return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
}

std::shared_ptr<DeviceFrame> BevStitcher::offerOnGpu(
    size_t camera, uint64_t stamp, std::shared_ptr<DeviceFrame> device,
    const cv::Mat& bgr) {
  if (!gpu_) {
    // The tables of the constructor, as one map of every camera's pixels
    cv::Mat map(source_.rows, source_.cols, CV_32FC2, cv::Scalar(0, 0));
    for (int row = 0; row < source_.rows; row++) {
      for (int column = 0; column < source_.cols; column++) {
        int taken = source_.at<schar>(row, column);
        if (taken >= 0) map.at<cv::Vec2f>(row, column) = place(taken, column, row);
      }
    }
    auto tables = std::make_shared<GpuTables>();
    check(cudaStreamCreate(&tables->stream), "cudaStreamCreate");
    check(cudaMalloc(&tables->map, map.total() * map.elemSize()), "cudaMalloc");
    check(cudaMalloc(&tables->source, source_.total()), "cudaMalloc");
    check(cudaMemcpy(tables->map, map.data, map.total() * map.elemSize(),
                     cudaMemcpyHostToDevice),
          "upload map");
    check(cudaMemcpy(tables->source, source_.data, source_.total(),
                     cudaMemcpyHostToDevice),
          "upload map");
    gpu_ = std::move(tables);
  }

  // The frame where the detector left it, or uploaded now
  auto* frame = dynamic_cast<CudaFrame*>(device.get());
  if (!frame) {
    checkSize(camera, bgr.cols, bgr.rows);
    device = CudaFrame::upload(bgr);
    frame = static_cast<CudaFrame*>(device.get());
  }
  checkSize(camera, frame->width(), frame->height());

  Pending& moment = pending_[stamp];
  if (!moment.device) {
    auto view = CudaFrame::create(source_.cols, source_.rows);
    check(cudaMemsetAsync(view->data(), 0, view->bytes(), gpu_->stream),
          "clear the view");
    moment.device = std::move(view);
  }
  auto* view = static_cast<CudaFrame*>(moment.device.get());
  cudaBevRemap({frame->data(), frame->width(), frame->height(), frame->pitch(),
                /*channels=*/3, /*redIndex=*/2},
               gpu_->map, gpu_->source, static_cast<int>(camera), view->data(),
               source_.cols, source_.rows, gpu_->stream);
  moment.frames.push_back(std::move(device));
  // Waited for once a moment: when it is whole, and before finish() drops
  // moments, whose frames and views go back to the pool where the next
  // user must not find the GPU still reading or writing them
  if ((moment.have | 1u << camera) == allCameras() ||
      pending_.size() > kPending) {
    check(cudaStreamSynchronize(gpu_->stream), "stitch");
  }
  return finish(camera, stamp).device;
}
#else
bool BevStitcher::canUseGpu() { return false; }

std::shared_ptr<DeviceFrame> BevStitcher::offerOnGpu(
    size_t, uint64_t, std::shared_ptr<DeviceFrame>, const cv::Mat&) {
  throw std::runtime_error("bird's-eye view: built without CUDA");
}
#endif

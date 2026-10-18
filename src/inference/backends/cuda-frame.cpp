#include "cuda-frame.hpp"

#include <cuda_runtime.h>

#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// Released buffers, with their sizes. Frames of a run have few sizes (one
// per camera resolution, one for the view), so a linear search is enough.
// shortcut: never shrinks, nor frees at exit; add a cap if sizes vary a lot
struct Pool {
  std::mutex mutex;
  std::vector<std::pair<unsigned char*, size_t>> free;
};

Pool& pool() {
  // Never destroyed: a frame released during static destruction would
  // otherwise find it gone
  static Pool* instance = new Pool;
  return *instance;
}

}  // namespace

std::shared_ptr<CudaFrame> CudaFrame::create(int width, int height) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("CudaFrame: empty size");
  }
  size_t need = static_cast<size_t>(width) * 3 * height;
  {
    std::lock_guard<std::mutex> lock(pool().mutex);
    auto& free = pool().free;
    for (auto it = free.begin(); it != free.end(); ++it) {
      if (it->second == need) {
        unsigned char* data = it->first;
        free.erase(it);
        return std::shared_ptr<CudaFrame>(
            new CudaFrame(data, width, height));
      }
    }
  }
  void* data = nullptr;
  cudaError_t status = cudaMalloc(&data, need);
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CudaFrame: cudaMalloc: ") +
                             cudaGetErrorString(status));
  }
  return std::shared_ptr<CudaFrame>(
      new CudaFrame(static_cast<unsigned char*>(data), width, height));
}

std::shared_ptr<CudaFrame> CudaFrame::upload(const cv::Mat& bgr) {
  if (bgr.type() != CV_8UC3) {
    throw std::invalid_argument("CudaFrame: not an 8-bit BGR image");
  }
  // Rows of a cropped Mat are not adjacent in memory
  cv::Mat packed = bgr.isContinuous() ? bgr : bgr.clone();
  auto frame = create(packed.cols, packed.rows);
  cudaError_t status =
      cudaMemcpyAsync(frame->data(), packed.data, frame->bytes(),
                      cudaMemcpyHostToDevice, cudaStreamPerThread);
  if (status == cudaSuccess) status = cudaStreamSynchronize(cudaStreamPerThread);
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CudaFrame: upload: ") +
                             cudaGetErrorString(status));
  }
  return frame;
}

CudaFrame::~CudaFrame() {
  std::lock_guard<std::mutex> lock(pool().mutex);
  pool().free.emplace_back(data_, bytes());
}

cv::Mat CudaFrame::toBgr() {
  cv::Mat bgr(height_, width_, CV_8UC3);
  // On this thread's own stream: the legacy default stream would wait for
  // every inference running on the device
  cudaError_t status = cudaMemcpyAsync(bgr.data, data_, bytes(),
                                       cudaMemcpyDeviceToHost,
                                       cudaStreamPerThread);
  if (status == cudaSuccess) status = cudaStreamSynchronize(cudaStreamPerThread);
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CudaFrame: download: ") +
                             cudaGetErrorString(status));
  }
  return bgr;
}

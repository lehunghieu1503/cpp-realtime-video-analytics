#include "cuda-preprocess.hpp"

#include <stdexcept>
#include <string>

namespace {

__global__ void resizeNormalizeKernel(const unsigned char* src, int srcWidth,
                                      int srcHeight, int srcPitch,
                                      int channels, int redIndex, float* dst,
                                      int dstWidth, int dstHeight) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dstWidth || y >= dstHeight) return;

  // Pixel centres map onto pixel centres, as in cv::resize
  float sx = (x + 0.5f) * srcWidth / dstWidth - 0.5f;
  float sy = (y + 0.5f) * srcHeight / dstHeight - 0.5f;
  int x0 = static_cast<int>(floorf(sx));
  int y0 = static_cast<int>(floorf(sy));
  float fx = sx - x0;
  float fy = sy - y0;
  int x1 = min(max(x0 + 1, 0), srcWidth - 1);
  int y1 = min(max(y0 + 1, 0), srcHeight - 1);
  x0 = min(max(x0, 0), srcWidth - 1);
  y0 = min(max(y0, 0), srcHeight - 1);

  const unsigned char* p00 = src + y0 * srcPitch + x0 * channels;
  const unsigned char* p01 = src + y0 * srcPitch + x1 * channels;
  const unsigned char* p10 = src + y1 * srcPitch + x0 * channels;
  const unsigned char* p11 = src + y1 * srcPitch + x1 * channels;

  int plane = dstWidth * dstHeight;
  int index = y * dstWidth + x;
  // Output plane 0 is red, 1 green, 2 blue
  const int byteOf[3] = {redIndex, 1, 2 - redIndex};
  for (int c = 0; c < 3; c++) {
    int b = byteOf[c];
    float top = p00[b] + fx * (p01[b] - p00[b]);
    float bottom = p10[b] + fx * (p11[b] - p10[b]);
    dst[c * plane + index] = (top + fy * (bottom - top)) * (1.0f / 255.0f);
  }
}

}  // namespace

void cudaPreprocess(const GpuImage& src, float* dst, int dstWidth,
                    int dstHeight, cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((dstWidth + block.x - 1) / block.x,
            (dstHeight + block.y - 1) / block.y);
  resizeNormalizeKernel<<<grid, block, 0, stream>>>(
      src.data, src.width, src.height, src.pitch, src.channels, src.redIndex,
      dst, dstWidth, dstHeight);
  // A kernel that cannot be launched (for instance one compiled by a
  // toolkit newer than the driver) fails here, not at the next synchronise:
  // without this check the engine would silently run on an empty input
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA pre-process kernel: ") +
                             cudaGetErrorString(status));
  }
}

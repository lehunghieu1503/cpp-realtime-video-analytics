#include "bev-remap.hpp"

#include <stdexcept>
#include <string>

namespace {

__global__ void bevRemapKernel(const unsigned char* src, int srcWidth,
                               int srcHeight, int srcPitch, int channels,
                               const float2* map, const signed char* source,
                               int camera, unsigned char* view, int width,
                               int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  int index = y * width + x;
  if (source[index] != camera) return;

  float2 at = map[index];
  int x0 = static_cast<int>(floorf(at.x));
  int y0 = static_cast<int>(floorf(at.y));
  float fx = at.x - x0;
  float fy = at.y - y0;
  // The map keeps inside the image; only a neighbour of weight 0 can fall
  // off its last row or column
  int x1 = min(x0 + 1, srcWidth - 1);
  int y1 = min(y0 + 1, srcHeight - 1);
  x0 = min(max(x0, 0), srcWidth - 1);
  y0 = min(max(y0, 0), srcHeight - 1);

  const unsigned char* p00 = src + y0 * srcPitch + x0 * channels;
  const unsigned char* p01 = src + y0 * srcPitch + x1 * channels;
  const unsigned char* p10 = src + y1 * srcPitch + x0 * channels;
  const unsigned char* p11 = src + y1 * srcPitch + x1 * channels;
  unsigned char* out = view + index * 3;
  for (int c = 0; c < 3; c++) {
    float top = p00[c] + fx * (p01[c] - p00[c]);
    float bottom = p10[c] + fx * (p11[c] - p10[c]);
    out[c] = static_cast<unsigned char>(
        fminf(fmaxf(top + fy * (bottom - top) + 0.5f, 0.0f), 255.0f));
  }
}

}  // namespace

void cudaBevRemap(const GpuImage& src, const float* map,
                  const signed char* source, int camera, unsigned char* view,
                  int width, int height, cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  bevRemapKernel<<<grid, block, 0, stream>>>(
      src.data, src.width, src.height, src.pitch, src.channels,
      reinterpret_cast<const float2*>(map), source, camera, view, width,
      height);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA bird's-eye view kernel: ") +
                             cudaGetErrorString(status));
  }
}

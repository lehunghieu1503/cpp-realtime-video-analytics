#pragma once

#include <cuda_runtime.h>

// Describes a packed 8-bit image in GPU memory
struct GpuImage {
  const unsigned char* data;
  int width;
  int height;
  int pitch;     // bytes per row
  int channels;  // bytes per pixel: 3 (BGR) or 4 (RGBA, BGRx, ...)
  int redIndex;  // position of the red byte within a pixel; blue is
                 // channels-agnostic: 2 - redIndex
};

// Resizes `src` to dstWidth x dstHeight (bilinear, stretched like cv::resize),
// converts to RGB, scales to [0, 1] and writes planar float32 (CHW) into
// `dst`, all in one pass on the GPU. Asynchronous on `stream`. Throws
// std::runtime_error if the kernel cannot be launched.
void cudaPreprocess(const GpuImage& src, float* dst, int dstWidth,
                    int dstHeight, cudaStream_t stream);

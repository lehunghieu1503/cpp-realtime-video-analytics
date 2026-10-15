#pragma once

#include "inference/backends/cuda-preprocess.hpp"

// The GPU half of BevStitcher::offerOnGpu(): writes the pixels of the view
// that `camera` is the source of, sampled bilinearly from `src` at the
// places `map` gives (x, y per pixel, as cv::remap's maps). Pixels of other
// cameras are left as they are. `view` is packed BGR, width x height.
// Asynchronous on `stream`. Throws std::runtime_error if the kernel cannot
// be launched.
void cudaBevRemap(const GpuImage& src, const float* map,
                  const signed char* source, int camera, unsigned char* view,
                  int width, int height, cudaStream_t stream);

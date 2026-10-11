#pragma once

#include "inference-backend.hpp"

#include <map>
#include <memory>

// Native TensorRT backend. Unlike the "tensorrt" execution provider of ONNX
// Runtime it owns the engine, so it can
//   - keep the input on the GPU: runFrame() uploads the 8-bit frame once and
//     resizes, reorders and normalises it in a CUDA kernel that writes
//     straight into the engine's input
//   - report what the engine is made of and where the time goes, layer by
//     layer
// The engine is built from the ONNX model on first use and cached in
// trt-cache/ next to it, named after the precision and the TensorRT version.
class TensorRtBackend : public IInferenceBackend {
public:
  // Throws std::runtime_error when the model cannot be parsed or built
  explicit TensorRtBackend(const BackendConfig& config);
  ~TensorRtBackend() override;

  const std::vector<int64_t>& inputShape() const override {
    return inputShape_;
  }
  const float* run(const float* input, const std::vector<int64_t>& shape,
                   std::vector<int64_t>& outputShape) override;

  bool preprocessesFrames() const override { return true; }
  const float* runFrame(const cv::Mat& bgr, std::vector<int64_t>& outputShape,
                        double* preprocessMs) override;

  std::string describe() const override { return description_; }
  std::string layerProfile() const override;

private:
  struct Impl;  // keeps the TensorRT and CUDA headers out of this one
  std::unique_ptr<Impl> impl_;
  std::vector<int64_t> inputShape_;
  std::vector<int64_t> outputShape_;
  std::string description_;
};

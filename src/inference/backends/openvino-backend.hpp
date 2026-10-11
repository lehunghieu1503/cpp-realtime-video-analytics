#pragma once

#include "inference-backend.hpp"

#include <openvino/openvino.hpp>

// OpenVINO backend: Intel CPU, integrated GPU or NPU. Reads ONNX and
// OpenVINO IR (.xml) models.
class OpenVinoBackend : public IInferenceBackend {
public:
  // device: "CPU", "GPU" or "NPU". Throws ov::Exception when the device is
  // missing or the model cannot be compiled for it.
  OpenVinoBackend(const BackendConfig& config, const std::string& device);

  const std::vector<int64_t>& inputShape() const override {
    return inputShape_;
  }
  const float* run(const float* input, const std::vector<int64_t>& shape,
                   std::vector<int64_t>& outputShape) override;

private:
  ov::InferRequest request_;
  std::vector<int64_t> inputShape_;
};

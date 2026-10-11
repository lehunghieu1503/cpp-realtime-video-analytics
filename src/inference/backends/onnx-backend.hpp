#pragma once

#include "inference-backend.hpp"

#include <onnxruntime_cxx_api.h>

// ONNX Runtime backend: CPU, CUDA or TensorRT execution provider
class OnnxBackend : public IInferenceBackend {
public:
  // Throws Ort::Exception when the provider or the model cannot be loaded
  explicit OnnxBackend(const BackendConfig& config);

  const std::vector<int64_t>& inputShape() const override {
    return inputShape_;
  }
  const float* run(const float* input, const std::vector<int64_t>& shape,
                   std::vector<int64_t>& outputShape) override;

private:
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> inputNames_;
  std::vector<std::string> outputNames_;
  std::vector<int64_t> inputShape_;
  // Owns the buffer run() returns a pointer into
  std::vector<Ort::Value> outputs_;
};

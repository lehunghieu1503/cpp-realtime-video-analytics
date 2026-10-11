#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// What a backend needs to load a model. Pre-process, post-process and NMS are
// shared by every backend (see InferenceEngine), so benchmarks compare only
// the tensor execution.
struct BackendConfig {
  std::string modelPath;
  // "cpu", "cuda", "tensorrt" (ONNX Runtime) or "openvino:DEVICE" where
  // DEVICE is CPU, GPU or NPU
  std::string backend = "cpu";
  // "fp32", "fp16" or "int8". Only tensorrt acts on it: it builds the engine
  // at that precision. int8 needs a calibration table next to the model (see
  // scripts/calibrate-int8.py). OpenVINO takes its precision from the model
  // file, so pass an INT8 IR as the model instead.
  std::string precision = "fp16";
  int numThreads = 4;
};

// Runs one NCHW float32 tensor through a loaded model
class IInferenceBackend {
public:
  virtual ~IInferenceBackend() = default;

  // Model input shape, e.g. [1, 3, 640, 640]; dynamic dimensions are -1
  virtual const std::vector<int64_t>& inputShape() const = 0;

  // input holds the tensor for `shape`. The returned pointer stays valid
  // until the next run() on this backend. Throws std::exception on failure.
  virtual const float* run(const float* input,
                           const std::vector<int64_t>& shape,
                           std::vector<int64_t>& outputShape) = 0;
};

// Returns nullptr, after printing the reason, when the backend is unknown,
// not compiled in, or fails to load the model. There is no fallback to CPU:
// it would look like a working accelerator that is several times slower.
std::unique_ptr<IInferenceBackend> createBackend(const BackendConfig& config);

// True for "cpu", "cuda", "tensorrt" and "openvino:<DEVICE>"
bool isKnownBackend(const std::string& backend);

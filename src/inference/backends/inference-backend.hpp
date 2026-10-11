#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cv {
class Mat;
}
class DeviceFrame;

// What a backend needs to load a model. Pre-process, post-process and NMS are
// shared by every backend (see InferenceEngine), so benchmarks compare only
// the tensor execution.
struct BackendConfig {
  std::string modelPath;
  // "cpu", "cuda", "tensorrt" (ONNX Runtime), "tensorrt-native" (TensorRT
  // itself) or "openvino:DEVICE" where DEVICE is CPU, GPU or NPU
  std::string backend = "cpu";
  // "fp32", "fp16" or "int8". Only the tensorrt backends act on it: they
  // build the engine at that precision. int8 needs a calibration table next
  // to the model (see scripts/calibrate-int8.py). OpenVINO takes its precision from the model
  // file, so pass an INT8 IR as the model instead.
  std::string precision = "fp16";
  int numThreads = 4;
  // tensorrt-native: time every layer (slows inference; for benchmarks)
  bool profileLayers = false;
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

  // A backend that pre-processes on its own device returns true here and
  // implements runFrame(): it takes the 8-bit BGR frame as captured, and
  // does the resize, BGR→RGB, scaling to [0, 1] and HWC→CHW itself.
  // preprocessMs receives the time that took.
  virtual bool preprocessesFrames() const { return false; }
  virtual const float* runFrame(const cv::Mat& /*bgr*/,
                                std::vector<int64_t>& /*outputShape*/,
                                double* /*preprocessMs*/) {
    return nullptr;
  }

  // Same, for a frame that is already in device memory. Returns nullptr
  // when the backend cannot read that kind of frame in place; the caller
  // then copies it to the CPU.
  virtual const float* runDeviceFrame(DeviceFrame& /*frame*/,
                                      std::vector<int64_t>& /*outputShape*/,
                                      double* /*preprocessMs*/) {
    return nullptr;
  }

  // One line about the loaded model for the log, e.g. layer and precision
  // counts; empty when the backend has nothing to say
  virtual std::string describe() const { return ""; }
  // Markdown table of the time per layer, when BackendConfig::profileLayers
  // was set and the backend supports it; empty otherwise
  virtual std::string layerProfile() const { return ""; }
};

// Returns nullptr, after printing the reason, when the backend is unknown,
// not compiled in, or fails to load the model. There is no fallback to CPU:
// it would look like a working accelerator that is several times slower.
std::unique_ptr<IInferenceBackend> createBackend(const BackendConfig& config);

// True for "cpu", "cuda", "tensorrt", "tensorrt-native" and
// "openvino:<DEVICE>"
bool isKnownBackend(const std::string& backend);

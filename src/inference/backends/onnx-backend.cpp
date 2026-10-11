#include "onnx-backend.hpp"

#include <filesystem>

OnnxBackend::OnnxBackend(const BackendConfig& config)
    : env_(ORT_LOGGING_LEVEL_WARNING, "video-analytics") {
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetIntraOpNumThreads(config.numThreads);
  sessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  if (config.backend == "tensorrt") {
    const OrtApi& api = Ort::GetApi();
    OrtTensorRTProviderOptionsV2* raw = nullptr;
    Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&raw));
    std::unique_ptr<OrtTensorRTProviderOptionsV2,
                    decltype(api.ReleaseTensorRTProviderOptions)>
        trtOptions(raw, api.ReleaseTensorRTProviderOptions);

    // Building the engine takes minutes, so keep it across runs
    auto modelPath = std::filesystem::path(config.modelPath);
    auto modelDir = modelPath.parent_path();
    std::string cacheDir =
        ((modelDir.empty() ? "." : modelDir) / "trt-cache").string();
    // Written by scripts/calibrate-int8.py; TensorRT looks for it in the
    // cache directory
    std::string calibTable = modelPath.stem().string() + ".calibration";

    std::vector<const char*> keys = {"trt_engine_cache_enable",
                                     "trt_engine_cache_path"};
    std::vector<const char*> values = {"1", cacheDir.c_str()};
    if (config.precision != "fp32") {
      // INT8 layers fall back to FP16 where that is faster
      keys.push_back("trt_fp16_enable");
      values.push_back("1");
    }
    if (config.precision == "int8") {
      keys.push_back("trt_int8_enable");
      values.push_back("1");
      keys.push_back("trt_int8_calibration_table_name");
      values.push_back(calibTable.c_str());
    }
    Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(
        trtOptions.get(), keys.data(), values.data(), keys.size()));
    sessionOptions.AppendExecutionProvider_TensorRT_V2(*trtOptions);
  }
  // After TensorRT, CUDA runs the nodes TensorRT does not support
  if (config.backend != "cpu") {
    OrtCUDAProviderOptions cudaOptions;
    cudaOptions.device_id = 0;
    sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
  }

  session_ = std::make_unique<Ort::Session>(env_, config.modelPath.c_str(),
                                            sessionOptions);

  Ort::AllocatorWithDefaultOptions allocator;
  for (size_t i = 0; i < session_->GetInputCount(); i++) {
    inputNames_.emplace_back(
        session_->GetInputNameAllocated(i, allocator).get());
    inputShape_ =
        session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
  }
  for (size_t i = 0; i < session_->GetOutputCount(); i++) {
    outputNames_.emplace_back(
        session_->GetOutputNameAllocated(i, allocator).get());
  }
}

const float* OnnxBackend::run(const float* input,
                              const std::vector<int64_t>& shape,
                              std::vector<int64_t>& outputShape) {
  size_t count = 1;
  for (int64_t d : shape) count *= static_cast<size_t>(d);

  auto memoryInfo =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  // ONNX Runtime does not write to the input; the API just is not const
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      memoryInfo, const_cast<float*>(input), count, shape.data(),
      shape.size());

  std::vector<const char*> inputNamePtrs;
  for (const auto& name : inputNames_) inputNamePtrs.push_back(name.c_str());
  std::vector<const char*> outputNamePtrs;
  for (const auto& name : outputNames_) outputNamePtrs.push_back(name.c_str());

  outputs_ = session_->Run(Ort::RunOptions{nullptr}, inputNamePtrs.data(),
                           &inputTensor, inputNamePtrs.size(),
                           outputNamePtrs.data(), outputNamePtrs.size());
  outputShape = outputs_[0].GetTensorTypeAndShapeInfo().GetShape();
  return outputs_[0].GetTensorData<float>();
}

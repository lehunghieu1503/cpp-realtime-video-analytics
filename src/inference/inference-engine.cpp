#include "inference-engine.hpp"
#include "post-process.hpp"
#include "pre-process.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>

InferenceEngine::InferenceEngine(const InferenceConfig& config)
    : config_(config),
      env_(ORT_LOGGING_LEVEL_WARNING, "video-analytics") {
  // Configure session options
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetIntraOpNumThreads(config_.numThreads);
  sessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  // No silent fallback to CPU: it would look like a working GPU setup that is
  // several times slower
  try {
    if (config_.provider == "tensorrt") {
      const OrtApi& api = Ort::GetApi();
      OrtTensorRTProviderOptionsV2* raw = nullptr;
      Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&raw));
      std::unique_ptr<OrtTensorRTProviderOptionsV2,
                      decltype(api.ReleaseTensorRTProviderOptions)>
          trtOptions(raw, api.ReleaseTensorRTProviderOptions);

      // Building the engine takes minutes, so keep it across runs
      auto modelDir = std::filesystem::path(config_.modelPath).parent_path();
      std::string cacheDir =
          ((modelDir.empty() ? "." : modelDir) / "trt-cache").string();
      const char* keys[] = {"trt_fp16_enable", "trt_engine_cache_enable",
                            "trt_engine_cache_path"};
      const char* values[] = {"1", "1", cacheDir.c_str()};
      Ort::ThrowOnError(
          api.UpdateTensorRTProviderOptions(trtOptions.get(), keys, values, 3));
      sessionOptions.AppendExecutionProvider_TensorRT_V2(*trtOptions);
    }
    // After TensorRT, CUDA runs the nodes TensorRT does not support
    if (config_.provider != "cpu") {
      OrtCUDAProviderOptions cudaOptions;
      cudaOptions.device_id = 0;
      sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
    }
  } catch (const Ort::Exception& e) {
    std::cerr << "Execution provider '" << config_.provider
              << "' not available: " << e.what() << "\n";
    return;
  }

  // Load model
  try {
    session_ =
        std::make_unique<Ort::Session>(env_, config_.modelPath.c_str(),
                                       sessionOptions);
  } catch (const Ort::Exception& e) {
    std::cerr << "Failed to load model: " << e.what() << "\n";
    return;
  }

  Ort::AllocatorWithDefaultOptions allocator;

  // Query input metadata
  size_t numInputs = session_->GetInputCount();
  for (size_t i = 0; i < numInputs; i++) {
    auto name = session_->GetInputNameAllocated(i, allocator);
    inputNames_.emplace_back(name.get());

    auto typeInfo = session_->GetInputTypeInfo(i);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    inputShape_ = tensorInfo.GetShape();
  }

  // Use the model's own input size when it is fixed (e.g. exported at 320 or
  // 1280). Dynamic dimensions come back as -1 and keep the configured size.
  if (inputShape_.size() == 4 && inputShape_[2] > 0 && inputShape_[3] > 0) {
    config_.inputHeight = static_cast<int>(inputShape_[2]);
    config_.inputWidth = static_cast<int>(inputShape_[3]);
  }

  // Query output metadata
  size_t numOutputs = session_->GetOutputCount();
  for (size_t i = 0; i < numOutputs; i++) {
    auto name = session_->GetOutputNameAllocated(i, allocator);
    outputNames_.emplace_back(name.get());

    auto typeInfo = session_->GetOutputTypeInfo(i);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    outputShape_ = tensorInfo.GetShape();
  }

  std::cout << "Model loaded: " << config_.modelPath << "\n";
  std::cout << "  Input shape: [";
  for (size_t i = 0; i < inputShape_.size(); i++) {
    std::cout << inputShape_[i] << (i + 1 < inputShape_.size() ? ", " : "");
  }
  std::cout << "]\n";
  std::cout << "  Output shape: [";
  for (size_t i = 0; i < outputShape_.size(); i++) {
    std::cout << outputShape_[i] << (i + 1 < outputShape_.size() ? ", " : "");
  }
  std::cout << "]\n";
}

bool InferenceEngine::isLoaded() const { return session_ != nullptr; }

std::vector<Detection> InferenceEngine::detect(const cv::Mat& frame) {
  if (!isLoaded()) return {};

  // Validate input frame
  if (frame.empty() || frame.channels() != 3) {
    std::cerr << "detect(): invalid frame (empty or not 3-channel BGR)\n";
    return {};
  }

  // Preprocess: resize + normalize + HWC→CHW
  auto inputData = preprocess(frame, config_.inputWidth, config_.inputHeight);

  // Create input tensor
  auto memoryInfo =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<int64_t> inputDims = {1, 3, config_.inputHeight,
                                     config_.inputWidth};
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      memoryInfo, inputData.data(), inputData.size(), inputDims.data(),
      inputDims.size());

  // Build name arrays for Run() — needs const char* pointers
  std::vector<const char*> inputNamePtrs;
  for (const auto& name : inputNames_) inputNamePtrs.push_back(name.c_str());

  std::vector<const char*> outputNamePtrs;
  for (const auto& name : outputNames_) outputNamePtrs.push_back(name.c_str());

  // Run inference
  auto outputTensors = session_->Run(
      Ort::RunOptions{nullptr}, inputNamePtrs.data(), &inputTensor,
      inputNamePtrs.size(), outputNamePtrs.data(), outputNamePtrs.size());

  // Extract output tensor data
  const float* outputData = outputTensors[0].GetTensorData<float>();
  auto outputInfo =
      outputTensors[0].GetTensorTypeAndShapeInfo();
  auto outputShape = outputInfo.GetShape();

  // Postprocess: parse detections + NMS
  return postprocess(outputData, outputShape, config_.confidenceThreshold,
                     config_.nmsThreshold, frame.cols, frame.rows,
                     config_.inputWidth, config_.inputHeight);
}

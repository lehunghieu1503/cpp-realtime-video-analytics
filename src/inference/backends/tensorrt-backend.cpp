#include "tensorrt-backend.hpp"
#include "cuda-preprocess.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TensorRT] " << message << "\n";
    }
  }
};

// Sums the time TensorRT reports for every layer over all profiled runs
class LayerProfiler : public nvinfer1::IProfiler {
public:
  void reportLayerTime(const char* layerName, float ms) noexcept override {
    auto [it, isNew] = totalMs_.try_emplace(layerName, 0.0);
    if (isNew) order_.push_back(layerName);
    it->second += ms;
    // Layers are reported in execution order; the first one starts a run
    if (layerName == order_.front()) runs_++;
  }

  std::string table(size_t top) const {
    if (runs_ == 0) return "";
    std::vector<std::pair<std::string, double>> layers(totalMs_.begin(),
                                                       totalMs_.end());
    std::sort(layers.begin(), layers.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    double total = 0.0;
    for (const auto& layer : layers) total += layer.second;

    std::ostringstream out;
    out << std::fixed << std::setprecision(3);
    out << "| Layer | ms | Share |\n|-------|----|-------|\n";
    for (size_t i = 0; i < layers.size() && i < top; i++) {
      out << "| " << layers[i].first << " | " << layers[i].second / runs_
          << " | " << std::setprecision(1) << 100.0 * layers[i].second / total
          << "% |" << std::setprecision(3) << "\n";
    }
    out << "| All " << layers.size() << " layers | " << total / runs_
        << " | 100% |\n";
    return out.str();
  }

private:
  std::map<std::string, double> totalMs_;
  std::vector<std::string> order_;
  int runs_ = 0;
};

#if NV_TENSORRT_MAJOR < 11
// Hands TensorRT the value ranges measured by scripts/calibrate-int8.py, so
// no calibration images are needed on the device
class CacheCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
  explicit CacheCalibrator(std::string cache) : cache_(std::move(cache)) {}
  int32_t getBatchSize() const noexcept override { return 1; }
  bool getBatch(void*[], const char*[], int32_t) noexcept override {
    return false;  // nothing to calibrate with: the cache has to suffice
  }
  const void* readCalibrationCache(size_t& length) noexcept override {
    length = cache_.size();
    return cache_.empty() ? nullptr : cache_.data();
  }
  void writeCalibrationCache(const void*, size_t) noexcept override {}

private:
  std::string cache_;
};
#endif

std::string readFile(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  std::ostringstream content;
  content << file.rdbuf();
  return content.str();
}

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(status));
  }
}

std::vector<int64_t> toVector(const nvinfer1::Dims& dims) {
  return std::vector<int64_t>(dims.d, dims.d + dims.nbDims);
}

size_t elementCount(const std::vector<int64_t>& shape) {
  size_t count = 1;
  for (int64_t d : shape) count *= static_cast<size_t>(d);
  return count;
}

// "142 layers: 120 FP16, 22 FP32" from the engine inspector's JSON
std::string summarise(const std::string& layersJson) {
  auto json = nlohmann::json::parse(layersJson, nullptr, false);
  if (!json.is_object() || !json.contains("Layers")) return "";
  // TensorRT 10 writes "Format/Datatype": "... FP16 format"; TensorRT 11
  // writes "Datatype": "Half"
  const std::pair<const char*, const char*> kNames[] = {
      {"Int8", "INT8"},  {"FP16", "FP16"},   {"Half", "FP16"},
      {"FP32", "FP32"},  {"Float", "FP32"},  {"Int32", "INT32"},
      {"Int64", "INT64"}, {"Bool", "BOOL"},  {"UInt8", "UINT8"}};
  std::map<std::string, int> byPrecision;
  int count = 0;
  for (const auto& layer : json["Layers"]) {
    count++;
    // Without detailed profiling verbosity a layer is only its name
    if (!layer.is_object() || !layer.contains("Outputs") ||
        layer["Outputs"].empty()) {
      continue;
    }
    const auto& output = layer["Outputs"][0];
    std::string type = output.value("Datatype", std::string());
    if (type.empty()) type = output.value("Format/Datatype", std::string());
    std::string precision = "other";
    for (const auto& [text, name] : kNames) {
      if (type.find(text) != std::string::npos) {
        precision = name;
        break;
      }
    }
    byPrecision[precision]++;
  }
  std::ostringstream out;
  out << count << " layers";
  const char* separator = " (output types: ";
  for (const auto& [precision, n] : byPrecision) {
    out << separator << n << " " << precision;
    separator = ", ";
  }
  if (!byPrecision.empty()) out << ")";
  return out.str();
}

}  // namespace

struct TensorRtBackend::Impl {
  Logger logger;
  LayerProfiler profiler;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream = nullptr;
  float* inputDevice = nullptr;
  float* outputDevice = nullptr;
  float* outputHost = nullptr;  // page-locked, so the copy back is direct
  unsigned char* frameDevice = nullptr;
  size_t frameCapacity = 0;
  size_t inputBytes = 0;
  size_t outputBytes = 0;

  ~Impl() {
    // The context and engine go before the buffers they point into
    context.reset();
    engine.reset();
    runtime.reset();
    cudaFree(inputDevice);
    cudaFree(outputDevice);
    cudaFree(frameDevice);
    cudaFreeHost(outputHost);
    if (stream) cudaStreamDestroy(stream);
  }

  // Builds the engine from the ONNX file; returns the serialized plan
  std::string build(const BackendConfig& config,
                    const std::filesystem::path& cacheDir) {
    using namespace nvinfer1;
    std::cout << "TensorRT: building the " << config.precision
              << " engine for " << config.modelPath
              << ", this takes minutes\n"
              << std::flush;

    std::unique_ptr<IBuilder> builder(createInferBuilder(logger));
    std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0));
    std::unique_ptr<nvonnxparser::IParser> parser(
        nvonnxparser::createParser(*network, logger));
    if (!parser->parseFromFile(config.modelPath.c_str(),
                               static_cast<int>(ILogger::Severity::kERROR))) {
      // Name the operators TensorRT cannot run: that is what to fix in the
      // model, or what would have to fall back to another device
      std::ostringstream reason;
      reason << "TensorRT cannot parse the model:";
      for (int i = 0; i < parser->getNbErrors(); i++) {
        const auto* error = parser->getError(i);
        reason << "\n  node '" << (error->nodeName() ? error->nodeName() : "")
               << "' (" << (error->nodeOperator() ? error->nodeOperator() : "?")
               << "): " << error->desc();
      }
      throw std::runtime_error(reason.str());
    }

    std::unique_ptr<IBuilderConfig> builderConfig(
        builder->createBuilderConfig());
    // Keeps layer names, types and precisions in the engine for the report
    builderConfig->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
#if NV_TENSORRT_MAJOR < 11
    if (config.precision != "fp32") builderConfig->setFlag(BuilderFlag::kFP16);
#else
    // TensorRT 11 builds every layer at the precision the model gives it.
    // For FP16 convert the model first: scripts/quantize-fp16.py
    if (config.precision == "fp16") {
      std::cout << "TensorRT " << NV_TENSORRT_MAJOR
                << " takes the precision from the model file\n";
    }
#endif

#if NV_TENSORRT_MAJOR < 11
    std::unique_ptr<CacheCalibrator> calibrator;
#endif
    if (config.precision == "int8") {
#if NV_TENSORRT_MAJOR < 11
      auto cachePath =
          cacheDir / (std::filesystem::path(config.modelPath).stem().string() +
                      ".calibration.cache");
      std::string cache = readFile(cachePath);
      if (cache.empty()) {
        throw std::runtime_error(
            "int8 needs " + cachePath.string() +
            "; create it with scripts/calibrate-int8.py");
      }
      calibrator = std::make_unique<CacheCalibrator>(std::move(cache));
      builderConfig->setFlag(BuilderFlag::kINT8);
      builderConfig->setInt8Calibrator(calibrator.get());
#else
      throw std::runtime_error(
          "this TensorRT version no longer calibrates: int8 needs a model "
          "with quantize/dequantize nodes, built with precision fp16");
#endif
    }

    std::unique_ptr<IHostMemory> plan(
        builder->buildSerializedNetwork(*network, *builderConfig));
    if (!plan) throw std::runtime_error("TensorRT failed to build the engine");
    (void)cacheDir;  // only the calibration path above uses it
    return std::string(static_cast<const char*>(plan->data()), plan->size());
  }
};

TensorRtBackend::TensorRtBackend(const BackendConfig& config)
    : impl_(std::make_unique<Impl>()) {
  namespace fs = std::filesystem;
  fs::path modelPath(config.modelPath);
  fs::path modelDir = modelPath.parent_path();
  fs::path cacheDir = (modelDir.empty() ? "." : modelDir) / "trt-cache";
  // An engine only loads with the TensorRT version that built it
  std::ostringstream name;
#if NV_TENSORRT_MAJOR < 11
  const std::string& precision = config.precision;
#else
  const std::string precision = "model";  // see build()
#endif
  name << modelPath.stem().string() << "." << precision << ".trt"
       << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "."
       << NV_TENSORRT_PATCH;
  fs::path enginePath = cacheDir / (name.str() + ".engine");
  fs::path layersPath = cacheDir / (name.str() + ".layers.json");

  if (!fs::exists(modelPath)) {
    throw std::runtime_error("model file does not exist");
  }
  std::string plan = readFile(enginePath);
  if (plan.empty()) {
    plan = impl_->build(config, cacheDir);
    std::error_code ec;
    fs::create_directories(cacheDir, ec);
    std::ofstream(enginePath, std::ios::binary) << plan;
  }

  impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
  impl_->engine.reset(
      impl_->runtime->deserializeCudaEngine(plan.data(), plan.size()));
  if (!impl_->engine) {
    throw std::runtime_error("cannot load the engine " + enginePath.string() +
                             "; delete it to rebuild");
  }
  impl_->context.reset(impl_->engine->createExecutionContext());

  // One input and one output, as the YOLO detectors export
  const char* inputName = nullptr;
  const char* outputName = nullptr;
  for (int i = 0; i < impl_->engine->getNbIOTensors(); i++) {
    const char* tensor = impl_->engine->getIOTensorName(i);
    bool isInput = impl_->engine->getTensorIOMode(tensor) ==
                   nvinfer1::TensorIOMode::kINPUT;
    (isInput ? inputName : outputName) = tensor;
  }
  if (!inputName || !outputName) {
    throw std::runtime_error("the engine needs one input and one output");
  }
  inputShape_ = toVector(impl_->engine->getTensorShape(inputName));
  outputShape_ = toVector(impl_->engine->getTensorShape(outputName));
  for (int64_t d : inputShape_) {
    if (d <= 0) {
      throw std::runtime_error(
          "the model has a dynamic input size; export it with a fixed one");
    }
  }

  impl_->inputBytes = elementCount(inputShape_) * sizeof(float);
  impl_->outputBytes = elementCount(outputShape_) * sizeof(float);
  check(cudaStreamCreate(&impl_->stream), "cudaStreamCreate");
  check(cudaMalloc(&impl_->inputDevice, impl_->inputBytes), "cudaMalloc");
  check(cudaMalloc(&impl_->outputDevice, impl_->outputBytes), "cudaMalloc");
  check(cudaHostAlloc(&impl_->outputHost, impl_->outputBytes,
                      cudaHostAllocDefault),
        "cudaHostAlloc");
  impl_->context->setTensorAddress(inputName, impl_->inputDevice);
  impl_->context->setTensorAddress(outputName, impl_->outputDevice);
  if (config.profileLayers) impl_->context->setProfiler(&impl_->profiler);

  // What the engine is made of: saved for inspection, summarised for the log
  std::unique_ptr<nvinfer1::IEngineInspector> inspector(
      impl_->engine->createEngineInspector());
  inspector->setExecutionContext(impl_->context.get());
  std::string layers = inspector->getEngineInformation(
      nvinfer1::LayerInformationFormat::kJSON);
  if (!fs::exists(layersPath)) std::ofstream(layersPath) << layers;
  description_ = "TensorRT " + std::to_string(NV_TENSORRT_MAJOR) + "." +
                 std::to_string(NV_TENSORRT_MINOR) + " engine, " +
                 summarise(layers) + " (details in " + layersPath.string() +
                 ")";
}

TensorRtBackend::~TensorRtBackend() = default;

const float* TensorRtBackend::run(const float* input,
                                  const std::vector<int64_t>& shape,
                                  std::vector<int64_t>& outputShape) {
  if (shape != inputShape_) {
    throw std::runtime_error("input shape does not match the engine");
  }
  check(cudaMemcpyAsync(impl_->inputDevice, input, impl_->inputBytes,
                        cudaMemcpyHostToDevice, impl_->stream),
        "upload input");
  if (!impl_->context->enqueueV3(impl_->stream)) {
    throw std::runtime_error("TensorRT inference failed");
  }
  check(cudaMemcpyAsync(impl_->outputHost, impl_->outputDevice,
                        impl_->outputBytes, cudaMemcpyDeviceToHost,
                        impl_->stream),
        "download output");
  check(cudaStreamSynchronize(impl_->stream), "inference");
  outputShape = outputShape_;
  return impl_->outputHost;
}

const float* TensorRtBackend::runFrame(const cv::Mat& bgr,
                                       std::vector<int64_t>& outputShape,
                                       double* preprocessMs) {
  auto start = std::chrono::steady_clock::now();
  // Rows of a cropped Mat are not adjacent in memory
  cv::Mat frame = bgr.isContinuous() ? bgr : bgr.clone();
  size_t frameBytes = frame.step * frame.rows;
  if (frameBytes > impl_->frameCapacity) {
    cudaFree(impl_->frameDevice);
    impl_->frameDevice = nullptr;
    check(cudaMalloc(&impl_->frameDevice, frameBytes), "cudaMalloc");
    impl_->frameCapacity = frameBytes;
  }
  // The only copy of the image: 8-bit, as captured. Resize, colour order,
  // scaling and layout all happen on the GPU, into the engine's input.
  check(cudaMemcpyAsync(impl_->frameDevice, frame.data, frameBytes,
                        cudaMemcpyHostToDevice, impl_->stream),
        "upload frame");
  GpuImage image{impl_->frameDevice, frame.cols, frame.rows,
                 static_cast<int>(frame.step), frame.channels(),
                 /*redIndex=*/2};
  cudaPreprocess(image, impl_->inputDevice, static_cast<int>(inputShape_[3]),
                 static_cast<int>(inputShape_[2]), impl_->stream);
  if (preprocessMs) {
    // Waits for the kernel only to time it; the cost is a few microseconds
    check(cudaStreamSynchronize(impl_->stream), "pre-process");
    *preprocessMs = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count();
  }

  if (!impl_->context->enqueueV3(impl_->stream)) {
    throw std::runtime_error("TensorRT inference failed");
  }
  check(cudaMemcpyAsync(impl_->outputHost, impl_->outputDevice,
                        impl_->outputBytes, cudaMemcpyDeviceToHost,
                        impl_->stream),
        "download output");
  check(cudaStreamSynchronize(impl_->stream), "inference");
  outputShape = outputShape_;
  return impl_->outputHost;
}

std::string TensorRtBackend::layerProfile() const {
  return impl_->profiler.table(15);
}

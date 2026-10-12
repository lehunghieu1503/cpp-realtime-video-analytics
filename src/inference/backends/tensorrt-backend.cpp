#include "tensorrt-backend.hpp"
#include "cuda-preprocess.hpp"

#ifdef WITH_NVMM
#include "capture/nvmm-frame.hpp"

#include <cuda.h>
#include <cudaEGL.h>
#endif

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
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

#if NV_TENSORRT_MAJOR < 11
// scripts/calibrate-int8.py writes "tensor max_abs" lines; TensorRT reads
// "tensor: <scale as float32 bits in hex>" under a header naming the
// calibrator, where scale = max_abs / 127. A cache that already has the
// header is passed through.
std::string toTensorRtCache(const std::string& cache) {
  if (cache.rfind("TRT-", 0) == 0) return cache;
  std::ostringstream out;
  out << "TRT-" << NV_TENSORRT_MAJOR << std::setfill('0') << std::setw(2)
      << NV_TENSORRT_MINOR << std::setw(2) << NV_TENSORRT_PATCH
      << "-EntropyCalibration2\n";
  std::istringstream lines(cache);
  std::string name;
  float maxAbs = 0.0f;
  while (lines >> name >> maxAbs) {
    float scale = maxAbs / 127.0f;
    uint32_t bits = 0;
    std::memcpy(&bits, &scale, sizeof(bits));
    out << name << ": " << std::hex << std::setw(8) << bits << std::dec
        << "\n";
  }
  return out.str();
}
#endif

// Megabytes from /proc/meminfo, 0 if the key is missing
long meminfoMb(const char* key) {
  std::ifstream file("/proc/meminfo");
  std::string name;
  long kb = 0;
  while (file >> name >> kb) {
    if (name == key) return kb / 1024;
    file.ignore(256, '\n');
  }
  return 0;
}

// On a Jetson the GPU shares the system memory, and CUDA counts only free
// pages as available: memory held by the page cache looks taken. With a few
// megabytes "available" the builder rejects every tactic and fails with
// "Could not find any implementation for node". Root would drop the cache;
// without it, allocating and touching memory makes the kernel evict it.
void reclaimPageCache() {
  long freeMb = meminfoMb("MemFree:");
  long availableMb = meminfoMb("MemAvailable:");
  if (freeMb >= 2048 || availableMb - freeMb < 512) return;
  // Leave a margin: MemAvailable is an estimate
  long wantMb = std::min(availableMb - 768, 4096L) - freeMb;
  std::cout << "TensorRT: " << freeMb << " MB free of " << availableMb
            << " MB available; evicting page cache before the build\n";
  const size_t kChunk = 64u << 20;
  std::vector<std::unique_ptr<char[]>> chunks;
  for (long mb = 0; mb < wantMb; mb += 64) {
    chunks.emplace_back(new char[kChunk]);
    // Touch every page so that it is really allocated
    for (size_t offset = 0; offset < kChunk; offset += 4096) {
      chunks.back()[offset] = 1;
    }
  }
}

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

    reclaimPageCache();
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
      calibrator = std::make_unique<CacheCalibrator>(toTensorRtCache(cache));
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
  // An engine older than its model file was built from the file that one
  // replaced (a model trained again under the same name): built again
  std::error_code stale;
  bool current = fs::exists(enginePath, stale) &&
                 fs::last_write_time(enginePath, stale) >=
                     fs::last_write_time(modelPath, stale);
  std::string plan = current ? readFile(enginePath) : std::string();
  const bool built = plan.empty();
  if (built) {
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

  // One float32 input and one float32 output, as the YOLO detectors and the
  // lane model export. Anything else is refused: picking one tensor of
  // several would run, and return another tensor's numbers.
  const char* inputName = nullptr;
  const char* outputName = nullptr;
  int inputs = 0, outputs = 0;
  for (int i = 0; i < impl_->engine->getNbIOTensors(); i++) {
    const char* tensor = impl_->engine->getIOTensorName(i);
    bool isInput = impl_->engine->getTensorIOMode(tensor) ==
                   nvinfer1::TensorIOMode::kINPUT;
    (isInput ? inputName : outputName) = tensor;
    (isInput ? inputs : outputs)++;
    if (impl_->engine->getTensorDataType(tensor) !=
        nvinfer1::DataType::kFLOAT) {
      throw std::runtime_error(std::string("tensor ") + tensor +
                               " is not float32");
    }
  }
  if (inputs != 1 || outputs != 1) {
    throw std::runtime_error("the engine has " + std::to_string(inputs) +
                             " input(s) and " + std::to_string(outputs) +
                             " output(s); one of each is supported");
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
  // Of an engine just built it is written again: the file there may be of
  // the engine this one replaced
  if (built || !fs::exists(layersPath)) std::ofstream(layersPath) << layers;
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
  return infer(outputShape);
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

  return infer(outputShape);
}

const float* TensorRtBackend::infer(std::vector<int64_t>& outputShape) {
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

const float* TensorRtBackend::runDeviceFrame(
    DeviceFrame& frame, std::vector<int64_t>& outputShape,
    double* preprocessMs) {
#ifdef WITH_NVMM
  auto* nvmm = dynamic_cast<NvmmFrame*>(&frame);
  if (!nvmm) return nullptr;

  auto start = std::chrono::steady_clock::now();
  // The driver API below needs this thread's CUDA context, which the
  // runtime API sets up on its first call from a thread
  check(cudaStreamSynchronize(impl_->stream), "stream");

  // Hand the decoder's buffer to CUDA as it is: registering the EGLImage
  // maps the same memory, it does not copy it
  CUgraphicsResource resource = nullptr;
  if (cuGraphicsEGLRegisterImage(&resource, nvmm->mapEglImage(),
                                 CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY) !=
      CUDA_SUCCESS) {
    nvmm->unmapEglImage();
    throw std::runtime_error("CUDA cannot register the NVMM buffer");
  }
  CUeglFrame mapped;
  bool usable =
      cuGraphicsResourceGetMappedEglFrame(&mapped, resource, 0, 0) ==
          CUDA_SUCCESS &&
      mapped.frameType == CU_EGL_FRAME_TYPE_PITCH;
  if (usable) {
    GpuImage image{static_cast<const unsigned char*>(mapped.frame.pPitch[0]),
                   static_cast<int>(mapped.width),
                   static_cast<int>(mapped.height),
                   static_cast<int>(mapped.pitch), /*channels=*/4,
                   /*redIndex=*/0};
    cudaPreprocess(image, impl_->inputDevice, static_cast<int>(inputShape_[3]),
                   static_cast<int>(inputShape_[2]), impl_->stream);
    // The kernel must be done reading before the buffer is released
    check(cudaStreamSynchronize(impl_->stream), "pre-process");
  }
  cuGraphicsUnregisterResource(resource);
  nvmm->unmapEglImage();
  if (!usable) {
    throw std::runtime_error("the NVMM buffer is not a pitch-linear image");
  }
  if (preprocessMs) {
    *preprocessMs = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count();
  }
  return infer(outputShape);
#else
  (void)frame;
  (void)outputShape;
  (void)preprocessMs;
  return nullptr;
#endif
}

std::string TensorRtBackend::layerProfile() const {
  return impl_->profiler.table(15);
}

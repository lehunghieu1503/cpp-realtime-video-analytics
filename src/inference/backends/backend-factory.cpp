#include "inference-backend.hpp"
#include "onnx-backend.hpp"
#ifdef WITH_OPENVINO
#include "openvino-backend.hpp"
#endif

#include <iostream>

namespace {
const std::string kOpenVinoPrefix = "openvino:";

bool isOpenVino(const std::string& backend) {
  return backend.rfind(kOpenVinoPrefix, 0) == 0 &&
         backend.size() > kOpenVinoPrefix.size();
}
}  // namespace

bool isKnownBackend(const std::string& backend) {
  return backend == "cpu" || backend == "cuda" || backend == "tensorrt" ||
         isOpenVino(backend);
}

std::unique_ptr<IInferenceBackend> createBackend(const BackendConfig& config) {
  try {
    if (isOpenVino(config.backend)) {
#ifdef WITH_OPENVINO
      return std::make_unique<OpenVinoBackend>(
          config, config.backend.substr(kOpenVinoPrefix.size()));
#else
      std::cerr << "Backend '" << config.backend
                << "' needs a build with -DWITH_OPENVINO=ON\n";
      return nullptr;
#endif
    }
    if (isKnownBackend(config.backend)) {
      return std::make_unique<OnnxBackend>(config);
    }
    std::cerr << "Unknown backend '" << config.backend << "'\n";
  } catch (const std::exception& e) {
    std::cerr << "Backend '" << config.backend
              << "' failed to load " << config.modelPath << ": " << e.what()
              << "\n";
  }
  return nullptr;
}

#include "openvino-backend.hpp"

OpenVinoBackend::OpenVinoBackend(const BackendConfig& config,
                                 const std::string& device) {
  ov::Core core;
  auto model = core.read_model(config.modelPath);

  // The NPU compiles only static shapes; a model exported with dynamic axes
  // has to be re-exported with a fixed input size
  auto partial = model->input().get_partial_shape();
  for (const auto& dim : partial) {
    inputShape_.push_back(dim.is_static() ? dim.get_length() : -1);
  }

  ov::AnyMap properties;
  if (device == "CPU") {
    properties[ov::inference_num_threads.name()] = config.numThreads;
  }
  request_ =
      core.compile_model(model, device, properties).create_infer_request();
}

const float* OpenVinoBackend::run(const float* input,
                                  const std::vector<int64_t>& shape,
                                  std::vector<int64_t>& outputShape) {
  ov::Shape ovShape(shape.begin(), shape.end());
  // Wraps the caller's buffer without copying; OpenVINO does not write to it
  request_.set_input_tensor(
      ov::Tensor(ov::element::f32, ovShape, const_cast<float*>(input)));
  request_.infer();

  ov::Tensor output = request_.get_output_tensor();
  outputShape.assign(output.get_shape().begin(), output.get_shape().end());
  return output.data<float>();
}

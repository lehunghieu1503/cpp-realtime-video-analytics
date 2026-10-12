// The lane model's decoder and the lane geometry, against the numbers the
// Python reference wrote to tests/fixtures/lane-calibration.json
// (simulation/test/fixtures.py).

#include "analytics/lane-geometry.hpp"
#include "config/app-config.hpp"
#include "inference/lane-engine.hpp"
#include "server/result-serializer.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

using nlohmann::json;

static int testsRun = 0;
static int testsFailed = 0;

static void check(bool condition, const std::string& name) {
  testsRun++;
  if (!condition) {
    testsFailed++;
    std::cout << "  [FAIL] " << name << "\n";
  }
}

static bool near(double a, double b, double tolerance = 1e-5) {
  return std::abs(a - b) <= tolerance;
}

static CameraRig rigFrom(const json& c) {
  CameraRig rig;
  rig.width = c["width"];
  rig.height = c["height"];
  rig.fovDeg = c["fov_deg"];
  rig.x = c["camera"]["x"];
  rig.y = c["camera"]["y"];
  rig.z = c["camera"]["z"];
  rig.pitchDeg = c["camera"]["pitch_deg"];
  rig.groundZ = c["ground_z"];
  return rig;
}

static void testCalibration(const json& fixture) {
  std::cout << "=== Calibration ===\n";
  CameraRig rig = rigFrom(fixture["calibration"]);
  check(rig.check().empty(), "The fixture's rig is usable");
  check(rig.calibrationId() == fixture["calibration_id"].get<std::string>(),
        "Calibration id equals the Python one");
  check(near(rig.focal(), fixture["focal_px"], 1e-9), "Focal length");

  for (const auto& item : fixture["ground"]) {
    double x = 0, y = 0;
    bool valid = rig.uvToGround(item["uv"][0], item["uv"][1], x, y);
    std::string name = "Pixel " + item["uv"].dump() + " on the road";
    if (item["xy"].is_null()) {
      check(!valid, name + " is rejected");
    } else {
      check(valid && near(x, item["xy"][0], 1e-6) && near(y, item["xy"][1], 1e-6),
            name);
    }
  }

  CameraRig other = rig;
  other.fovDeg += 0.001;
  check(other.calibrationId() != rig.calibrationId(),
        "Another field of view is another calibration");
  other = rig;
  other.z = -1.0;
  check(!other.check().empty(), "A camera under the road is rejected");
  other = rig;
  other.pitchDeg = std::numeric_limits<double>::quiet_NaN();
  check(!other.check().empty(), "A value that is not a number is rejected");
  other = rig;
  other.width = 0;
  check(!other.isSet() && !other.check().empty(), "No image size, no rig");
}

// The same formula as synthetic_logits() in the Python fixture
static std::vector<float> syntheticLogits(const json& lanes) {
  std::vector<float> logits((LaneEngine::kGrid + 1) * LaneEngine::kRows *
                            LaneEngine::kLanes);
  auto at = [&](int g, int r, int l) -> float& {
    return logits[(g * LaneEngine::kRows + r) * LaneEngine::kLanes + l];
  };
  for (int l = 0; l < LaneEngine::kLanes; l++) {
    double c0 = lanes[l][0], slope = lanes[l][1], sharp = lanes[l][2];
    int noneFrom = lanes[l][3], noneTo = lanes[l][4];
    for (int r = 0; r < LaneEngine::kRows; r++) {
      double centre = c0 + slope * r;
      for (int g = 0; g < LaneEngine::kGrid; g++) {
        at(g, r, l) = static_cast<float>(-sharp * (g - centre) * (g - centre));
      }
      at(LaneEngine::kGrid, r, l) = (noneFrom <= r && r <= noneTo) ? 5.0f : -50.0f;
    }
  }
  return logits;
}

static void testDecoder(const json& fixture) {
  std::cout << "=== Decoder ===\n";
  int number = 0;
  for (const auto& item : fixture["decoder"]) {
    auto logits = syntheticLogits(item["lanes"]);
    auto lanes = LaneEngine::decode(logits.data(), item["width"], item["height"]);
    std::string name = "Decoder case " + std::to_string(number++);
    check(lanes.size() == 4, name + ": four lanes");
    for (size_t l = 0; l < lanes.size(); l++) {
      const auto& expected = item["points"][l];
      bool same = lanes[l].size() == expected.size();
      for (size_t i = 0; same && i < lanes[l].size(); i++) {
        // Rows exactly; a column may fall either side of a truncation
        same = lanes[l][i].y == expected[i][1].get<int>() &&
               std::abs(lanes[l][i].x - expected[i][0].get<int>()) <= 1;
      }
      check(same, name + ": lane " + std::to_string(l) + " (" +
                      std::to_string(lanes[l].size()) + " points)");
    }
  }

  auto logits = syntheticLogits(fixture["decoder"][0]["lanes"]);
  logits[1234] = std::numeric_limits<float>::quiet_NaN();
  bool threw = false;
  try {
    LaneEngine::decode(logits.data(), 1280, 720);
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "Logits that are not finite are refused");
}

static void testGeometry(const json& fixture) {
  std::cout << "=== Lane geometry ===\n";
  CameraRig rig = rigFrom(fixture["calibration"]);
  LaneGeometryConfig config;
  const json& c = fixture["lane_config"];
  check(near(config.roiXMin, c["roi_x_min_m"]) &&
            near(config.roiXMax, c["roi_x_max_m"]) &&
            near(config.referenceX, c["reference_x_m"]) &&
            config.minPoints == c["min_points"].get<int>() &&
            near(config.minSupport, c["min_support_m"]) &&
            near(config.maxResidual, c["max_residual_m"]) &&
            near(config.widthMin, c["width_min_m"]) &&
            near(config.widthMax, c["width_max_m"]),
        "The C++ defaults are the Python defaults");

  for (const auto& item : fixture["lanes"]) {
    std::vector<std::vector<cv::Point>> lanes;
    for (const auto& lane : item["pixels"]) {
      lanes.emplace_back();
      for (const auto& p : lane) {
        lanes.back().emplace_back(p[0].get<int>(), p[1].get<int>());
      }
    }
    LaneResult result = estimateLane(lanes, rig, config);
    const json& e = item["expected"];
    std::string name = item["name"];
    check(result.valid == e["valid"].get<bool>(), name + ": valid");
    if (!e["valid"].get<bool>()) {
      check(result.reason == e["reason"].get<std::string>(),
            name + ": reason " + result.reason);
      json out = serializeLane(result, "rig");
      check(out["centerline"].is_null() && out["lateral_error_m"].is_null() &&
                out["heading_rad"].is_null() && out["confidence"] == 0.0,
            name + ": serialized without numbers");
      continue;
    }
    check(near(result.lateralError, e["lateral_error_m"]) &&
              near(result.heading, e["heading_rad"]) &&
              near(result.curvature, e["curvature_inv_m"], 1e-7) &&
              near(result.referenceX, e["reference_x_m"]),
          name + ": errors");
    check(near(result.confidence, e["confidence"], 1e-3), name + ": confidence");
    auto sameCurve = [](const LaneCurve& curve, const json& expected) {
      return near(curve.a, expected["coefficients"][0], 1e-7) &&
             near(curve.b, expected["coefficients"][1], 1e-6) &&
             near(curve.c, expected["coefficients"][2]) &&
             near(curve.xMin, expected["x_min_m"]) &&
             near(curve.xMax, expected["x_max_m"]);
    };
    check(sameCurve(result.left, e["left"]) &&
              sameCurve(result.right, e["right"]) &&
              sameCurve(result.centerline, e["centerline"]),
          name + ": curves");
    json out = serializeLane(result, "rig_1");
    check(out["valid"] == true && out["calibration_id"] == "rig_1" &&
              out["pixels"].size() == lanes.size() &&
              out["centerline"]["coefficients"].size() == 3,
          name + ": serialized");
  }

  LaneResult broken;
  broken.valid = true;
  broken.heading = std::numeric_limits<double>::quiet_NaN();
  json out = serializeLane(broken, "rig");
  check(out["valid"] == false && out["reason"] == "not_finite" &&
            out["heading_rad"].is_null() &&
            out.dump().find("nan") == std::string::npos,
        "A value that is not finite is never serialized as a number");
}

static void testPreprocess() {
  std::cout << "=== Pre-processing ===\n";
  // Blue in BGR: the tensor's planes are R, G, B
  cv::Mat blue(360, 640, CV_8UC3, cv::Scalar(255, 0, 0));
  auto tensor = LaneEngine::preprocess(blue);
  size_t plane = static_cast<size_t>(LaneEngine::kInputWidth) *
                 LaneEngine::kInputHeight;
  check(tensor.size() == 3 * plane, "Tensor is 3 x 288 x 800");
  check(near(tensor[0], (0 - 0.485) / 0.229, 1e-5) &&
            near(tensor[plane], (0 - 0.456) / 0.224, 1e-5) &&
            near(tensor[2 * plane], (1 - 0.406) / 0.225, 1e-5),
        "RGB order with the ImageNet mean and deviation");
  // Left half white: it must land in the left columns of each plane
  cv::Mat half(720, 1280, CV_8UC3, cv::Scalar(0, 0, 0));
  half(cv::Rect(0, 0, 640, 720)) = cv::Scalar(255, 255, 255);
  tensor = LaneEngine::preprocess(half);
  check(tensor[100] > 2.0f && tensor[700] < -2.0f, "Columns are not mirrored");
}

static void testConfig() {
  std::cout << "=== Config ===\n";
  auto path = std::filesystem::temp_directory_path() / "va-test-lane.yaml";
  std::ofstream(path) << R"(lane:
  model: models/ufld-tusimple18.onnx
  backend: tensorrt-native
  reference_x_m: 9.5
calibration:
  width: 1280
  height: 720
  fov_deg: 52
  camera: {x: 1.5, y: 0, z: 1.6, pitch_deg: 3}
sources:
  - id: front
    gst: "x ! appsink"
    lane: true
  - id: back
    gst: "x ! appsink"
)";
  AppConfig config;
  check(loadConfig(path.string(), config), "Lane config loads");
  check(config.sources[0].lane && !config.sources[1].lane, "Lane is per source");
  check(config.lane.backend == "tensorrt-native" &&
            near(config.lane.geometry.referenceX, 9.5),
        "Lane settings are read");
  check(config.calibration.calibrationId() == "rig_5a5f398f89b3b347",
        "Calibration from YAML has the fixture's id");
  check(checkLane(config).empty(), "A complete lane config is accepted");
  AppConfig incomplete = config;
  incomplete.lane.model.clear();
  check(!checkLane(incomplete).empty(), "Lane without a model is rejected");
  incomplete = config;
  incomplete.calibration = CameraRig();
  check(!checkLane(incomplete).empty(), "Lane without calibration is rejected");
  incomplete.sources[0].lane = false;
  check(checkLane(incomplete).empty(), "No lane source, nothing to check");

  std::ofstream(path) << "calibration:\n  distortion: [0.1, 0, 0, 0, 0]\n";
  AppConfig distorted;
  check(!loadConfig(path.string(), distorted), "Lens distortion is refused");
  std::filesystem::remove(path);
}

static void testModel() {
  std::cout << "=== Model ===\n";
  const char* model = "models/ufld-tusimple18.onnx";
  if (!std::filesystem::exists(model)) {
    std::cout << "SKIP: " << model
              << " not found (scripts/export-lane-model.py)\n";
    return;
  }
  LaneEngineConfig config;
  config.modelPath = model;
  LaneEngine engine(config);
  check(engine.isLoaded(), "The lane model loads on the CPU backend");
  if (!engine.isLoaded()) return;
  cv::Mat image = cv::imread("tests/test-image.jpg");
  double pre = 0, infer = 0, post = 0;
  auto lanes = engine.detect(image, &pre, &infer, &post);
  check(lanes.size() == 4 && infer > 0, "It runs on an image");
  for (const auto& lane : lanes) {
    for (const auto& p : lane) {
      check(p.x >= 0 && p.x < image.cols && p.y >= 0 && p.y < image.rows,
            "Lane pixels are inside the image");
    }
  }

  if (std::filesystem::exists("models/yolov8n.onnx")) {
    config.modelPath = "models/yolov8n.onnx";
    check(!LaneEngine(config).isLoaded(),
          "A model with another input is not accepted as a lane model");
  }
}

int main() {
  std::ifstream file(LANE_FIXTURE);
  if (!file) {
    std::cerr << "cannot open " << LANE_FIXTURE << "\n";
    return 1;
  }
  json fixture = json::parse(file);
  testCalibration(fixture);
  testDecoder(fixture);
  testGeometry(fixture);
  testPreprocess();
  testConfig();
  testModel();
  std::cout << "\nTotal: " << testsRun << ", failed: " << testsFailed << "\n";
  return testsFailed == 0 ? 0 : 1;
}

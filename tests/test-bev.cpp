// The bird's-eye view, against the numbers the Python reference wrote to
// tests/fixtures/bev.json (simulation/test/test_bev.py write).

#include "analytics/bev.hpp"
#include "config/app-config.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using nlohmann::json;

namespace {
int failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << "\n";
    failures++;
  }
}

CameraRig rigFrom(const json& c) {
  CameraRig rig;
  rig.width = c["width"];
  rig.height = c["height"];
  rig.fovDeg = c["fov_deg"];
  rig.x = c["camera"]["x"];
  rig.y = c["camera"]["y"];
  rig.z = c["camera"]["z"];
  rig.pitchDeg = c["camera"]["pitch_deg"];
  rig.yawDeg = c["camera"]["yaw_deg"];
  return rig;
}

void testFixture(const json& fixture) {
  std::cout << "=== Fixture ===\n";
  std::vector<CameraRig> rigs;
  for (const auto& camera : fixture["cameras"]) rigs.push_back(rigFrom(camera));
  const json& view = fixture["view"];
  BevStitcher stitcher(view["width"], view["height"],
                       view["metres_per_pixel"], rigs);
  check(stitcher.width() == 800 && stitcher.height() == 800, "Size of the view");
  int taken = 0;
  for (const auto& pixel : fixture["pixels"]) {
    int column = pixel[0], row = pixel[1], source = pixel[2];
    check(stitcher.source(column, row) == source, "The camera a pixel is taken from");
    if (source < 0 || stitcher.source(column, row) != source) continue;
    taken++;
    cv::Point2f place = stitcher.place(source, column, row);
    check(std::abs(place.x - pixel[3].get<double>()) < 0.01 &&
              std::abs(place.y - pixel[4].get<double>()) < 0.01,
          "Where in the camera's image a pixel is");
  }
  check(taken >= 10, "The fixture has pixels of every camera");

  // Each camera's frame in one colour: the view has that colour where the
  // camera is the pixel's, and black where nobody sees the ground
  cv::Mat frames[3] = {
      cv::Mat(720, 1280, CV_8UC3, cv::Scalar(255, 0, 0)),
      cv::Mat(720, 1280, CV_8UC3, cv::Scalar(0, 255, 0)),
      cv::Mat(720, 1280, CV_8UC3, cv::Scalar(0, 0, 255))};
  check(stitcher.offer(0, 1000, frames[0]).empty(), "No view from one camera");
  check(stitcher.offer(2, 1000, frames[2]).empty(), "No view from two cameras");
  // Another moment in between does not complete this one
  check(stitcher.offer(1, 2000, frames[1]).empty(), "Moments are kept apart");
  cv::Mat whole = stitcher.offer(1, 1000, frames[1]);
  check(!whole.empty() && whole.cols == 800 && whole.rows == 800,
        "The view once every camera has given its frame");
  if (!whole.empty()) {
    bool colours = true;
    for (const auto& pixel : fixture["pixels"]) {
      int column = pixel[0], row = pixel[1], source = pixel[2];
      cv::Vec3b expected(0, 0, 0);
      if (source >= 0) expected[source] = 255;
      colours = colours && whole.at<cv::Vec3b>(row, column) == expected;
    }
    check(colours, "Each pixel comes from its camera");
  }
  // Drawing on a frame after it was offered does not reach the view
  cv::Mat drawn = frames[0].clone();
  stitcher.offer(0, 3000, drawn);
  drawn.setTo(cv::Scalar(9, 9, 9));
  stitcher.offer(1, 3000, frames[1]);
  cv::Mat later = stitcher.offer(2, 3000, frames[2]);
  check(!later.empty() && later.at<cv::Vec3b>(700, 400) == cv::Vec3b(255, 0, 0),
        "The frame's pixels are taken when it is offered");
  // The moment 2000 was older than a view that went out: it is gone
  check(stitcher.offer(0, 2000, frames[0]).empty() &&
            stitcher.offer(2, 2000, frames[2]).empty(),
        "A moment older than a finished one is forgotten");

  // A camera that stops: its moments do not pile up
  for (uint64_t stamp = 10000; stamp < 10000 + 3 * BevStitcher::kPending; stamp++) {
    stitcher.offer(0, stamp, frames[0]);
    stitcher.offer(1, stamp, frames[1]);
  }
  uint64_t oldest = 10000, newest = 10000 + 3 * BevStitcher::kPending - 1;
  check(stitcher.offer(2, oldest, frames[2]).empty(), "Old moments are dropped");
  check(!stitcher.offer(2, newest, frames[2]).empty(), "The newest is still whole");
  // Views: the moments 1000, 3000 and the newest. Dropped: 2000 twice (it
  // was offered again after it was forgotten), every moment of the run
  // above but the newest, and `oldest` once more for the same reason
  check(stitcher.made() == 3 &&
            stitcher.dropped() == 2 + (3 * BevStitcher::kPending - 1) + 1,
        "Views and dropped moments are counted");

  bool refused = false;
  try {
    stitcher.offer(0, 99999, cv::Mat(360, 640, CV_8UC3));
  } catch (const std::runtime_error&) {
    refused = true;
  }
  check(refused, "A frame of another size is refused");
}

void testProject() {
  std::cout << "=== Project ===\n";
  // A pixel put on the ground and brought back is the same pixel
  CameraRig rig;
  rig.width = 1280;
  rig.height = 720;
  rig.fovDeg = 120.0;
  rig.x = 0.55;
  rig.y = -1.05;
  rig.z = 1.0;
  rig.pitchDeg = -45.0;
  rig.yawDeg = -105.0;
  double gx = 0, gy = 0, u = 0, v = 0, depth = 0;
  check(rig.uvToGround(300.5, 500.25, gx, gy), "A pixel below the horizon");
  rig.project(gx, gy, 0.0, u, v, depth);
  check(depth > 0 && std::abs(u - 300.5) < 1e-6 && std::abs(v - 500.25) < 1e-6,
        "project is the inverse of uvToGround");
  rig.project(0.55, 5.0, 0.0, u, v, depth);
  check(!(depth > 0), "A point behind the camera has no positive depth");
}

void testConfig() {
  std::cout << "=== Config ===\n";
  auto path = std::filesystem::temp_directory_path() / "va-test-bev.yaml";
  const char* yaml = R"(stall:
  model: models/stall.onnx
rear_calibration:
  width: 1280
  height: 720
  fov_deg: 120
  camera: {x: -2.45, y: 0, z: 1.0, pitch_deg: -30, yaw_deg: 180}
left_calibration:
  width: 1280
  height: 720
  fov_deg: 120
  camera: {x: 0.55, y: -1.05, z: 1.0, pitch_deg: -45, yaw_deg: -105}
bev:
  id: view
  width: 800
  height: 800
  metres_per_pixel: 0.025
  cameras:
    - {source: rear, calibration: rear_calibration, port: 5001}
    - {source: left, calibration: left_calibration, port: 5002}
sources:
  - id: rear
    gst: "x ! appsink"
  - id: left
    gst: "x ! appsink"
)";
  std::ofstream(path) << yaml;
  AppConfig config;
  check(loadConfig(path.string(), config), "A config with a view loads");
  check(config.bev.isSet() && config.bev.cameras.size() == 2 &&
            config.bev.cameras[1].source == "left" &&
            std::abs(config.bev.cameras[1].rig.yawDeg + 105.0) < 1e-9 &&
            config.bev.cameras[0].rig.width == 1280,
        "The view's cameras have their calibration");
  check(checkBev(config).empty(), "A complete view is accepted");
  check(checkBev(AppConfig()).empty(), "No view, nothing to check");
  AppConfig broken = config;
  broken.stall.model.clear();
  check(!checkBev(broken).empty(), "A view without a stall model is rejected");
  broken = config;
  broken.bev.cameras[0].source = "front";
  check(!checkBev(broken).empty(), "A camera that is not a source is rejected");
  broken = config;
  broken.bev.cameras[1].rig = CameraRig();
  check(!checkBev(broken).empty(), "A camera without calibration is rejected");
  broken = config;
  broken.bev.id = "rear";
  check(!checkBev(broken).empty(), "The view cannot have a source's id");
  broken = config;
  broken.bev.metresPerPixel = 0.0;
  check(!checkBev(broken).empty(), "A view without a scale is rejected");
  std::filesystem::remove(path);
}
}  // namespace

int main() {
  std::ifstream file(BEV_FIXTURE);
  if (!file) {
    std::cerr << "cannot open " << BEV_FIXTURE << "\n";
    return 1;
  }
  testFixture(json::parse(file));
  testProject();
  testConfig();
  std::cout << (failures ? "FAILED" : "All bird's-eye view tests passed") << "\n";
  return failures ? 1 : 0;
}

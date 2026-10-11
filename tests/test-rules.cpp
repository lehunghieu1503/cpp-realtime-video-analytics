// Tests for the zone and line rules, the tracker wrapper and the YAML config
#include "analytics/rules.hpp"
#include "analytics/tracker.hpp"
#include "config/app-config.hpp"

#include <fstream>
#include <iostream>

namespace test {
int total = 0;
int failed = 0;

void assert_true(bool condition, const std::string& name) {
  total++;
  if (condition) {
    std::cout << "  [PASS] " << name << "\n";
  } else {
    std::cout << "  [FAIL] " << name << "\n";
    failed++;
  }
}
}  // namespace test

// A 20x40 person whose feet (bottom centre of the box) are at (x, y)
Detection personAt(float x, float y, int trackId) {
  Detection det;
  det.classId = 0;
  det.confidence = 0.9f;
  det.bbox = cv::Rect2f(x - 10, y - 40, 20, 40);
  det.trackId = trackId;
  return det;
}

int countType(const std::vector<Event>& events, const std::string& type) {
  int n = 0;
  for (const auto& e : events) n += e.type == type;
  return n;
}

void testLineCrossing() {
  std::cout << "\n=== Line crossing ===\n";
  RulesConfig config;
  // Horizontal line from left to right at y = 100; "left" of a → b is the
  // side with larger y in image coordinates
  config.lines.push_back({"gate", {0, 100}, {200, 100}});
  Rules rules("cam0", config);

  auto events = rules.update({personAt(50, 80, 1)}, 0.0);
  test::assert_true(events.empty(), "First sighting raises nothing");

  events = rules.update({personAt(50, 120, 1)}, 0.1);
  test::assert_true(events.size() == 1 && events[0].type == "line_cross",
                    "Crossing raises line_cross");
  test::assert_true(!events.empty() && events[0].direction == "in" &&
                        events[0].name == "gate" && events[0].trackId == 1,
                    "Event carries direction, line name and track id");
  test::assert_true(rules.lineIn()[0] == 1 && rules.lineOut()[0] == 0,
                    "In counter incremented");

  events = rules.update({personAt(50, 80, 1)}, 0.2);
  test::assert_true(events.size() == 1 && events[0].direction == "out" &&
                        rules.lineOut()[0] == 1,
                    "Crossing back counts as out");

  // Moving along one side, or crossing beyond the end of the segment
  events = rules.update({personAt(150, 90, 1)}, 0.3);
  test::assert_true(events.empty(), "Moving on one side raises nothing");
  rules.update({personAt(300, 90, 2)}, 0.4);
  events = rules.update({personAt(300, 120, 2)}, 0.5);
  test::assert_true(events.empty(), "Crossing past the segment end is ignored");

  events = rules.update({personAt(50, 300, -1)}, 0.6);
  test::assert_true(events.empty(), "Untracked detections are ignored");
}

void testZones() {
  std::cout << "\n=== Zones ===\n";
  RulesConfig config;
  config.zones.push_back({"door", {{0, 0}, {100, 0}, {100, 100}, {0, 100}}});
  config.dwellSec = 5.0;
  config.lostSec = 2.0;
  Rules rules("cam0", config);

  auto events = rules.update({personAt(200, 50, 1)}, 0.0);
  test::assert_true(events.empty() && rules.zoneCounts()[0] == 0,
                    "Outside the zone: nothing");

  events = rules.update({personAt(50, 50, 1)}, 1.0);
  test::assert_true(countType(events, "zone_enter") == 1 &&
                        rules.zoneCounts()[0] == 1,
                    "Entering raises zone_enter and counts the object");

  events = rules.update({personAt(55, 50, 1)}, 3.0);
  test::assert_true(events.empty(), "Staying below the dwell time: nothing");

  events = rules.update({personAt(60, 50, 1)}, 6.5);
  test::assert_true(countType(events, "dwell_time") == 1 &&
                        events[0].dwellSec >= 5.0,
                    "dwell_time raised after the threshold");
  events = rules.update({personAt(60, 50, 1)}, 7.0);
  test::assert_true(events.empty(), "dwell_time raised only once per visit");

  events = rules.update({personAt(200, 50, 1)}, 8.0);
  test::assert_true(countType(events, "zone_exit") == 1 &&
                        events[0].dwellSec == 7.0 &&
                        rules.zoneCounts()[0] == 0,
                    "Leaving raises zone_exit with the time spent inside");

  // A track that vanishes inside the zone leaves it once it is forgotten
  rules.update({personAt(50, 50, 2)}, 10.0);
  events = rules.update({}, 11.0);
  test::assert_true(events.empty() && rules.zoneCounts()[0] == 1,
                    "A briefly missing track stays in the zone");
  events = rules.update({}, 12.5);
  test::assert_true(countType(events, "zone_exit") == 1 &&
                        events[0].trackId == 2 && rules.zoneCounts()[0] == 0,
                    "A lost track raises zone_exit");
}

void testTracker() {
  std::cout << "\n=== Tracker ===\n";
  Tracker tracker(30);
  int idA = -1;
  int idB = -1;
  bool stable = true;
  // Two people walking in opposite directions on separate rows
  for (int frame = 0; frame < 30; frame++) {
    std::vector<Detection> dets = {personAt(100 + frame * 3.0f, 200, -1),
                                   personAt(500 - frame * 3.0f, 400, -1)};
    // Large enough for the tracker's IoU matching between frames
    for (auto& det : dets) det.bbox = cv::Rect2f(det.bbox.x - 20, det.bbox.y - 60, 60, 120);
    tracker.update(dets);
    if (frame < 2) continue;  // a track is confirmed on its second frame
    if (idA < 0) {
      idA = dets[0].trackId;
      idB = dets[1].trackId;
    }
    stable = stable && dets[0].trackId == idA && dets[1].trackId == idB;
  }
  test::assert_true(idA >= 0 && idB >= 0, "Both objects get a track id");
  test::assert_true(idA != idB, "The two objects get different ids");
  test::assert_true(stable, "Ids stay the same from frame to frame");
}

void testConfig() {
  std::cout << "\n=== YAML config ===\n";
  const char* path = "/tmp/va-test-config.yaml";
  std::ofstream(path) << R"(# comment
model: models/yolo11s.onnx
backend: tensorrt
port: 9100
bind: 0.0.0.0
headless: true
confidence: 0.35
classes: [person, car]
dwell_seconds: 4.5
sources:
  - id: gate
    gst: "videotestsrc ! videoconvert ! appsink"
    zones:
      - name: door
        points: [[10, 20], [110, 20], [110, 120.5]]
    lines:
      - name: entry
        a: [0, 300]
        b: [960, 300]
  - rtsp: rtsp://camera/stream
  - device: 2
mqtt:
  host: mosquitto
  topic: site-a
snapshots:
  dir: /data/snapshots
  keep: 50
)";

  AppConfig config;
  test::assert_true(loadConfig(path, config), "Config file loads");
  test::assert_true(config.model == "models/yolo11s.onnx" &&
                        config.backend == "tensorrt" && config.port == 9100 &&
                        config.bind == "0.0.0.0" && config.headless,
                    "Scalar settings are read");
  test::assert_true(config.confidence > 0.34f && config.confidence < 0.36f &&
                        config.dwellSeconds == 4.5,
                    "Floating point settings are read");
  test::assert_true(config.classes == std::vector<int>({0, 2}),
                    "Class names map to COCO ids");
  test::assert_true(config.precision == "fp16" && config.workers == 0,
                    "Absent keys keep their defaults");
  test::assert_true(config.sources.size() == 3, "Three sources");
  if (config.sources.size() == 3) {
    const auto& gate = config.sources[0];
    test::assert_true(gate.capture.id == "gate" &&
                          gate.capture.gstPipeline.find("videotestsrc") == 0 &&
                          gate.capture.isLive(),
                      "GStreamer source");
    test::assert_true(gate.zones.size() == 1 && gate.zones[0].name == "door" &&
                          gate.zones[0].points.size() == 3 &&
                          gate.zones[0].points[2].y == 120.5f,
                      "Zone polygon");
    test::assert_true(gate.lines.size() == 1 && gate.lines[0].b.x == 960.0f,
                      "Line end points");
    test::assert_true(config.sources[1].capture.id == "cam1" &&
                          config.sources[1].capture.videoPath ==
                              "rtsp://camera/stream" &&
                          config.sources[1].capture.isLive(),
                      "RTSP source gets a default id and is live");
    test::assert_true(config.sources[2].capture.deviceIndex == 2,
                      "Camera device source");
  }
  test::assert_true(config.mqtt.host == "mosquitto" &&
                        config.mqtt.port == 1883 &&
                        config.mqtt.topic == "site-a",
                    "MQTT settings");
  test::assert_true(config.snapshotDir == "/data/snapshots" &&
                        config.snapshotKeep == 50,
                    "Snapshot settings");

  AppConfig other;
  test::assert_true(!loadConfig("/nonexistent/config.yaml", other),
                    "Missing file is reported");
  std::ofstream(path) << "classes: [unicorn]\n";
  test::assert_true(!loadConfig(path, other), "Unknown class is rejected");
  std::ofstream(path) << "sources:\n  - zones:\n      - name: z\n"
                         "        points: [[1, 2]]\n";
  test::assert_true(!loadConfig(path, other),
                    "Zone with fewer than 3 points is rejected");
  CaptureConfig file;
  file.videoPath = "video.mp4";
  test::assert_true(!file.isLive(), "A video file is not a live source");
}

int main() {
  testLineCrossing();
  testZones();
  testTracker();
  testConfig();

  std::cout << "\nTotal: " << test::total << ", failed: " << test::failed
            << "\n";
  return test::failed == 0 ? 0 : 1;
}

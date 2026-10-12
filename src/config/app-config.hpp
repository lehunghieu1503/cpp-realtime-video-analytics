#pragma once

#include "analytics/lane-geometry.hpp"
#include "analytics/rules.hpp"
#include "capture/video-capture.hpp"

#include <string>
#include <vector>

struct SourceConfig {
  CaptureConfig capture;
  std::vector<ZoneConfig> zones;
  std::vector<LineConfig> lines;
  bool lane = false;  // run the lane model on this source
  bool stall = false;  // run the parking stall model on this source
};

// The lane model; used by the sources that set `lane: true`
struct LaneConfig {
  std::string model;
  std::string backend;  // empty: the detector's backend
  std::string precision = "fp16";
  LaneGeometryConfig geometry;
};

// The parking stall model; used by the sources that set `stall: true`
struct StallConfig {
  std::string model;
  std::string backend;  // empty: the detector's backend
  std::string precision = "fp16";
  float confidence = 0.5f;
};

struct MqttConfig {
  std::string host;  // empty: MQTT is off
  int port = 1883;
  // Events go to <topic>/<stream>/<event type>
  std::string topic = "video-analytics";
  std::string username;  // empty: connect without credentials
  std::string password;
  // CA certificate that signed the broker's certificate; set: connect over
  // TLS and verify the broker
  std::string caFile;
};

// Frames come from a simulator: each result carries the episode's run id and
// the frame's own timestamp, which together identify the simulator's frame
struct SimulationConfig {
  bool enabled = false;
  // New for every episode; set here or through VA_RUN_ID
  std::string runId;
};

// Everything the application can be told. Loaded from a YAML file, then
// overridden by command-line flags.
struct AppConfig {
  std::string model = "models/yolov8n.onnx";
  std::string backend = "cpu";
  std::string precision = "fp16";
  // false: pre-process on the CPU even when the backend could do it on its
  // device (to measure the difference)
  bool devicePreprocess = true;
  bool profileLayers = false;  // benchmark: time every layer
  int workers = 0;  // 0: 2 on CPU, 1 on an accelerator
  int port = 9001;
  std::string bind = "127.0.0.1";
  bool headless = false;
  // false: the dashboard's page and video are not served; results still are
  bool dashboard = true;
  float confidence = 0.5f;
  // Keep only these COCO classes; empty keeps all of them
  std::vector<int> classes;
  bool tracking = true;
  double dwellSeconds = 10.0;
  std::vector<SourceConfig> sources;
  MqttConfig mqtt;
  std::string snapshotDir;  // empty: no snapshots
  int snapshotKeep = 200;   // older snapshots are deleted
  // Exit, so the supervisor restarts the process, when a connected source
  // has produced no result for this long. 0 disables the watchdog.
  int watchdogSeconds = 60;
  // HTTP Basic credentials for the dashboard and its endpoints; an empty
  // password leaves them open
  std::string httpUser = "admin";
  std::string httpPassword;
  // PEM certificate and key; both set: the dashboard is served over HTTPS
  std::string tlsCert;
  std::string tlsKey;
  SimulationConfig simulation;
  LaneConfig lane;
  StallConfig stall;
  // The camera of the sources that run the lane model
  CameraRig calibration;
};

// Empty when no source asks for lanes or when everything they need is
// there: a model and a usable calibration. The reason otherwise.
std::string checkLane(const AppConfig& config);

// Empty when no source asks for stalls or the stall model is named; the
// reason otherwise
std::string checkStall(const AppConfig& config);

// Empty when the simulation settings can be used; the reason otherwise.
// A simulation needs a run id and, for every source, a GStreamer pipeline
// read by the native capture: nothing else delivers the frame's timestamp.
std::string checkSimulation(const AppConfig& config);

// Fills `config` from a YAML file; keys that are absent keep their value.
// Returns false after printing the reason when the file cannot be used.
bool loadConfig(const std::string& path, AppConfig& config);

// Overrides the security settings from the environment, so that secrets need
// not be written into the config file:
//   VA_HTTP_USER, VA_HTTP_PASSWORD, VA_TLS_CERT, VA_TLS_KEY,
//   VA_MQTT_HOST, VA_MQTT_PORT, VA_MQTT_USER, VA_MQTT_PASSWORD, VA_MQTT_CA
// and the simulation run id from VA_RUN_ID
void applyEnvironment(AppConfig& config);

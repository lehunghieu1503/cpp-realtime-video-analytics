#pragma once

#include "analytics/rules.hpp"
#include "capture/video-capture.hpp"

#include <string>
#include <vector>

struct SourceConfig {
  CaptureConfig capture;
  std::vector<ZoneConfig> zones;
  std::vector<LineConfig> lines;
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
};

// Fills `config` from a YAML file; keys that are absent keep their value.
// Returns false after printing the reason when the file cannot be used.
bool loadConfig(const std::string& path, AppConfig& config);

// Overrides the security settings from the environment, so that secrets need
// not be written into the config file:
//   VA_HTTP_USER, VA_HTTP_PASSWORD, VA_TLS_CERT, VA_TLS_KEY,
//   VA_MQTT_HOST, VA_MQTT_PORT, VA_MQTT_USER, VA_MQTT_PASSWORD, VA_MQTT_CA
void applyEnvironment(AppConfig& config);

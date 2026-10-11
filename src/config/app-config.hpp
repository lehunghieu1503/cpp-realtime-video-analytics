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
};

// Everything the application can be told. Loaded from a YAML file, then
// overridden by command-line flags.
struct AppConfig {
  std::string model = "models/yolov8n.onnx";
  std::string backend = "cpu";
  std::string precision = "fp16";
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
};

// Fills `config` from a YAML file; keys that are absent keep their value.
// Returns false after printing the reason when the file cannot be used.
bool loadConfig(const std::string& path, AppConfig& config);

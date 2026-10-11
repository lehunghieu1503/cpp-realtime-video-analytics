#include "app-config.hpp"

#include "inference/detection.hpp"

#include <yaml-cpp/yaml.h>

#include <cctype>
#include <cstdlib>
#include <iostream>

namespace {
// Leaves `out` alone when the key is absent; a value of the wrong type
// throws YAML::Exception, reported by loadConfig()
template <typename T>
void read(const YAML::Node& node, T& out) {
  if (node) out = node.as<T>();
}

bool readPoint(const YAML::Node& node, cv::Point2f& out) {
  if (!node.IsSequence() || node.size() != 2) return false;
  out = cv::Point2f(node[0].as<float>(), node[1].as<float>());
  return true;
}

int classIdByName(const std::string& name) {
  for (size_t i = 0; i < kCocoClassNames.size(); i++) {
    if (name == kCocoClassNames[i]) return static_cast<int>(i);
  }
  return -1;
}

bool readSource(const YAML::Node& node, size_t index, SourceConfig& out) {
  out.capture.id = "cam" + std::to_string(index);
  read(node["id"], out.capture.id);
  read(node["device"], out.capture.deviceIndex);
  read(node["file"], out.capture.videoPath);
  read(node["rtsp"], out.capture.videoPath);
  read(node["gst"], out.capture.gstPipeline);
  read(node["width"], out.capture.width);
  read(node["height"], out.capture.height);
  read(node["fps"], out.capture.fps);
  read(node["lane"], out.lane);

  for (const auto& zoneNode : node["zones"]) {
    ZoneConfig zone;
    read(zoneNode["name"], zone.name);
    for (const auto& pointNode : zoneNode["points"]) {
      cv::Point2f point;
      if (!readPoint(pointNode, point)) {
        std::cerr << "Config: zone '" << zone.name
                  << "' has a point that is not [x, y]\n";
        return false;
      }
      zone.points.push_back(point);
    }
    if (zone.name.empty() || zone.points.size() < 3) {
      std::cerr << "Config: a zone needs a name and at least 3 points\n";
      return false;
    }
    out.zones.push_back(std::move(zone));
  }

  for (const auto& lineNode : node["lines"]) {
    LineConfig line;
    read(lineNode["name"], line.name);
    if (line.name.empty() || !readPoint(lineNode["a"], line.a) ||
        !readPoint(lineNode["b"], line.b)) {
      std::cerr << "Config: a line needs a name and points a and b\n";
      return false;
    }
    out.lines.push_back(std::move(line));
  }
  return true;
}
}  // namespace

bool loadConfig(const std::string& path, AppConfig& config) {
  try {
    YAML::Node root = YAML::LoadFile(path);

    read(root["model"], config.model);
    read(root["backend"], config.backend);
    read(root["precision"], config.precision);
    read(root["workers"], config.workers);
    read(root["port"], config.port);
    read(root["bind"], config.bind);
    read(root["headless"], config.headless);
    read(root["confidence"], config.confidence);
    read(root["tracking"], config.tracking);
    read(root["dwell_seconds"], config.dwellSeconds);
    read(root["watchdog_seconds"], config.watchdogSeconds);

    for (const auto& classNode : root["classes"]) {
      int id = classIdByName(classNode.as<std::string>());
      if (id < 0) {
        std::cerr << "Config: unknown class '" << classNode.as<std::string>()
                  << "'\n";
        return false;
      }
      config.classes.push_back(id);
    }

    size_t index = 0;
    for (const auto& sourceNode : root["sources"]) {
      SourceConfig source;
      if (!readSource(sourceNode, index++, source)) return false;
      config.sources.push_back(std::move(source));
    }

    read(root["mqtt"]["host"], config.mqtt.host);
    read(root["mqtt"]["port"], config.mqtt.port);
    read(root["mqtt"]["topic"], config.mqtt.topic);
    read(root["mqtt"]["username"], config.mqtt.username);
    read(root["mqtt"]["password"], config.mqtt.password);
    read(root["mqtt"]["ca_file"], config.mqtt.caFile);
    read(root["auth"]["username"], config.httpUser);
    read(root["auth"]["password"], config.httpPassword);
    read(root["tls"]["cert"], config.tlsCert);
    read(root["tls"]["key"], config.tlsKey);
    read(root["snapshots"]["dir"], config.snapshotDir);
    read(root["snapshots"]["keep"], config.snapshotKeep);
    const YAML::Node lane = root["lane"];
    read(lane["model"], config.lane.model);
    read(lane["backend"], config.lane.backend);
    read(lane["precision"], config.lane.precision);
    read(lane["roi_x_min_m"], config.lane.geometry.roiXMin);
    read(lane["roi_x_max_m"], config.lane.geometry.roiXMax);
    read(lane["reference_x_m"], config.lane.geometry.referenceX);
    read(lane["min_points"], config.lane.geometry.minPoints);
    read(lane["min_support_m"], config.lane.geometry.minSupport);
    read(lane["max_residual_m"], config.lane.geometry.maxResidual);
    read(lane["width_min_m"], config.lane.geometry.widthMin);
    read(lane["width_max_m"], config.lane.geometry.widthMax);

    if (const YAML::Node rig = root["calibration"]) {
      read(rig["width"], config.calibration.width);
      read(rig["height"], config.calibration.height);
      read(rig["fov_deg"], config.calibration.fovDeg);
      read(rig["ground_z"], config.calibration.groundZ);
      if (const YAML::Node camera = rig["camera"]) {
        read(camera["x"], config.calibration.x);
        read(camera["y"], config.calibration.y);
        read(camera["z"], config.calibration.z);
        read(camera["pitch_deg"], config.calibration.pitchDeg);
        read(camera["yaw_deg"], config.calibration.yawDeg);
        read(camera["roll_deg"], config.calibration.rollDeg);
      }
      for (const auto& k : rig["distortion"]) {
        if (k.as<double>() != 0.0) {
          std::cerr << "Config: lens distortion is not supported\n";
          return false;
        }
      }
    }

    read(root["simulation"]["enabled"], config.simulation.enabled);
    read(root["simulation"]["run_id"], config.simulation.runId);
  } catch (const YAML::Exception& e) {
    // e.what() carries the line and column
    std::cerr << "Config: " << path << ": " << e.what() << "\n";
    return false;
  }
  return true;
}

void applyEnvironment(AppConfig& config) {
  auto env = [](const char* name, std::string& out) {
    const char* value = std::getenv(name);
    if (value && *value) out = value;
  };
  env("VA_HTTP_USER", config.httpUser);
  env("VA_HTTP_PASSWORD", config.httpPassword);
  env("VA_TLS_CERT", config.tlsCert);
  env("VA_TLS_KEY", config.tlsKey);
  env("VA_MQTT_HOST", config.mqtt.host);
  env("VA_MQTT_USER", config.mqtt.username);
  env("VA_MQTT_PASSWORD", config.mqtt.password);
  env("VA_MQTT_CA", config.mqtt.caFile);
  env("VA_RUN_ID", config.simulation.runId);
  std::string port;
  env("VA_MQTT_PORT", port);
  if (!port.empty()) config.mqtt.port = std::atoi(port.c_str());
}

std::string checkSimulation(const AppConfig& config) {
  if (!config.simulation.enabled) return "";
  const std::string& id = config.simulation.runId;
  // The run id becomes part of a ROS topic name
  bool valid = !id.empty() && id.size() <= 64;
  for (char c : id) {
    valid = valid && (std::isalnum(static_cast<unsigned char>(c)) || c == '_');
  }
  if (!valid) {
    return "simulation needs a run_id of letters, digits and '_' "
           "(simulation.run_id or VA_RUN_ID)";
  }
#ifndef WITH_GSTREAMER
  return "simulation needs a build with the native GStreamer capture";
#else
  for (const auto& source : config.sources) {
    if (source.capture.gstPipeline.empty()) {
      return "simulation source '" + source.capture.id +
             "' needs a gst pipeline: only it delivers frame timestamps";
    }
  }
  return "";
#endif
}

std::string checkLane(const AppConfig& config) {
  bool wanted = false;
  for (const auto& source : config.sources) wanted = wanted || source.lane;
  if (!wanted) return "";
  if (config.lane.model.empty()) return "a source has lane: true but lane.model is not set";
  if (!config.calibration.isSet()) {
    return "a source has lane: true but there is no calibration section";
  }
  return config.calibration.check();
}

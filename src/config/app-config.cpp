#include "app-config.hpp"

#include "inference/detection.hpp"

#include <yaml-cpp/yaml.h>

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
  std::string port;
  env("VA_MQTT_PORT", port);
  if (!port.empty()) config.mqtt.port = std::atoi(port.c_str());
}

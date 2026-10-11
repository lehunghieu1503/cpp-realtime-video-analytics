#pragma once

#include "analytics/event.hpp"
#include "pipeline/frame-data.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <string>

// Serialize AnalyticsResult to compact JSON for SSE streaming
// Example output:
// {"frame_id":42,"detections":[{"class":"person","confidence":0.94,
//   "bbox":{"x":100,"y":50,"w":200,"h":300}}],"latency_ms":35.2,"fps":21.5}
inline nlohmann::json serializeResult(const AnalyticsResult& result,
                                       double currentFps) {
  nlohmann::json detections = nlohmann::json::array();
  for (const auto& det : result.detections) {
    nlohmann::json item = {
        {"class", getClassName(det.classId)},
        // 2 decimals; divided as double so the JSON reads 0.94, not
        // 0.9399999976158142
        {"confidence", std::round(det.confidence * 100) / 100.0},
        {"bbox",
         {{"x", static_cast<int>(det.bbox.x)},
          {"y", static_cast<int>(det.bbox.y)},
          {"w", static_cast<int>(det.bbox.width)},
          {"h", static_cast<int>(det.bbox.height)}}}
    };
    if (det.trackId >= 0) item["track_id"] = det.trackId;
    detections.push_back(std::move(item));
  }

  return {
      {"frame_id", result.frameId},
      {"detections", detections},
      {"latency_ms", std::round(result.inferenceLatencyMs * 10) / 10},
      {"fps", std::round(currentFps * 10) / 10}
  };
}

// Serialize an event for MQTT and SSE
// Example output:
// {"type":"line_cross","stream":"cam0","name":"gate","track_id":7,
//   "class":"person","direction":"in","timestamp":1760000000.25}
inline nlohmann::json serializeEvent(const Event& event) {
  nlohmann::json json = {
      {"type", event.type},
      {"stream", event.stream},
      {"name", event.name},
      {"track_id", event.trackId},
      {"class", getClassName(event.classId)},
      {"timestamp", std::round(event.timestamp * 1000) / 1000}
  };
  if (!event.direction.empty()) json["direction"] = event.direction;
  if (event.type == "zone_exit" || event.type == "dwell_time") {
    json["dwell_s"] = std::round(event.dwellSec * 10) / 10;
  }
  if (!event.snapshot.empty()) json["snapshot"] = event.snapshot;
  return json;
}

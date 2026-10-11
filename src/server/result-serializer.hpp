#pragma once

#include "analytics/event.hpp"
#include "analytics/lane-result.hpp"
#include "pipeline/frame-data.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <string>

// Serialize AnalyticsResult to compact JSON for SSE streaming
// Example output:
// {"frame_id":42,"detections":[{"class":"person","confidence":0.94,
//   "bbox":{"x":100,"y":50,"w":200,"h":300}}],"latency_ms":35.2,"fps":21.5,
//   "image":{"width":1280,"height":720}}
// With a simulation run id, and only for a frame that has a source
// timestamp, it also carries the identity of the simulator's frame:
//   "simulation":{"run_id":"r_20261011_ab12","sim_ns":1533333333}
inline nlohmann::json serializeResult(const AnalyticsResult& result,
                                       double currentFps,
                                       const std::string& runId = "") {
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

  nlohmann::json json = {
      {"frame_id", result.frameId},
      {"detections", detections},
      {"latency_ms", std::round(result.inferenceLatencyMs * 10) / 10},
      {"fps", std::round(currentFps * 10) / 10}
  };
  if (result.imageWidth > 0 && result.imageHeight > 0) {
    json["image"] = {{"width", result.imageWidth},
                     {"height", result.imageHeight}};
  }
  if (!runId.empty() && result.sourcePtsNs) {
    // An integer: nanoseconds must come back exactly as the source sent them
    json["simulation"] = {{"run_id", runId}, {"sim_ns", *result.sourcePtsNs}};
  }
  return json;
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

// The lane of a frame, added to the result as "lane". Lengths are metres in
// the ego frame of the capture (x forward, y right); a lane that is not
// valid has null instead of numbers, never a made-up value.
// Example output:
// {"valid":true,"confidence":0.91,"reason":"","calibration_id":"rig_5a5f...",
//   "pixels":[[[589,709],...],...],
//   "left":{"coefficients":[a,b,c],"x_min_m":10.0,"x_max_m":29.6},
//   "right":{...},"centerline":{...},"reference_x_m":10.0,
//   "lateral_error_m":0.4,"heading_rad":0.0,"curvature_inv_m":0.0,
//   "timing_ms":{"pre":1.2,"infer":6.5,"post":0.3}}
inline nlohmann::json serializeLane(const LaneResult& lane,
                                     const std::string& calibrationId) {
  auto curve = [](const LaneCurve& c) {
    return nlohmann::json{{"coefficients", {c.a, c.b, c.c}},
                          {"x_min_m", c.xMin},
                          {"x_max_m", c.xMax}};
  };
  nlohmann::json pixels = nlohmann::json::array();
  for (const auto& points : lane.pixels) {
    nlohmann::json list = nlohmann::json::array();
    for (const auto& p : points) list.push_back({p.x, p.y});
    pixels.push_back(std::move(list));
  }
  bool finite = std::isfinite(lane.lateralError) &&
                std::isfinite(lane.heading) && std::isfinite(lane.curvature);
  bool valid = lane.valid && finite;
  nlohmann::json json = {
      {"valid", valid},
      {"confidence", valid ? std::round(lane.confidence * 1000) / 1000 : 0.0},
      {"reason", valid ? "" : (lane.valid ? "not_finite" : lane.reason)},
      {"calibration_id", calibrationId},
      {"pixels", std::move(pixels)},
      {"reference_x_m", lane.referenceX},
      {"timing_ms",
       {{"pre", std::round(lane.preMs * 100) / 100},
        {"infer", std::round(lane.inferMs * 100) / 100},
        {"post", std::round(lane.postMs * 100) / 100}}}};
  json["left"] = valid ? curve(lane.left) : nlohmann::json();
  json["right"] = valid ? curve(lane.right) : nlohmann::json();
  json["centerline"] = valid ? curve(lane.centerline) : nlohmann::json();
  json["lateral_error_m"] = valid ? nlohmann::json(lane.lateralError) : nlohmann::json();
  json["heading_rad"] = valid ? nlohmann::json(lane.heading) : nlohmann::json();
  json["curvature_inv_m"] = valid ? nlohmann::json(lane.curvature) : nlohmann::json();
  return json;
}

#pragma once

#include "event.hpp"
#include "inference/detection.hpp"

#include <map>
#include <string>
#include <vector>

struct ZoneConfig {
  std::string name;
  std::vector<cv::Point2f> points;  // polygon, pixel coordinates
};

struct LineConfig {
  std::string name;
  cv::Point2f a;
  cv::Point2f b;
};

struct RulesConfig {
  std::vector<ZoneConfig> zones;
  std::vector<LineConfig> lines;
  // An object inside a zone for this long raises one dwell_time event
  double dwellSec = 10.0;
  // A track not seen for this long is forgotten (and leaves its zones)
  double lostSec = 2.0;
};

// Zone and line rules over the tracked detections of one stream. An object
// is located by the bottom centre of its box (its feet).
class Rules {
public:
  Rules(const std::string& stream, const RulesConfig& config);

  // now: Unix time in seconds. Only detections with a trackId take part.
  std::vector<Event> update(const std::vector<Detection>& detections,
                            double now);

  const RulesConfig& config() const { return config_; }
  // Objects currently inside each zone, in config order
  std::vector<int> zoneCounts() const;
  // Crossings per line so far, in config order
  const std::vector<int>& lineIn() const { return lineIn_; }
  const std::vector<int>& lineOut() const { return lineOut_; }

private:
  struct TrackState {
    cv::Point2f anchor;
    int classId = -1;
    double lastSeen = 0.0;
    // Per zone: time the object entered, or a negative value when outside
    std::vector<double> enteredAt;
    std::vector<bool> dwellRaised;
  };

  Event makeEvent(const char* type, const std::string& name, int trackId,
                  const TrackState& state, double now) const;

  std::string stream_;
  RulesConfig config_;
  std::map<int, TrackState> tracks_;
  std::vector<int> lineIn_;
  std::vector<int> lineOut_;
};

#pragma once

#include <string>

// Something a tracked object did, published over MQTT and SSE
struct Event {
  // "line_cross", "zone_enter", "zone_exit" or "dwell_time"
  std::string type;
  std::string stream;  // source id
  std::string name;    // name of the zone or line
  int trackId = -1;
  int classId = -1;
  // line_cross: "in" when the object moves to the left of the line's
  // direction a → b, "out" otherwise
  std::string direction;
  // zone_exit and dwell_time: seconds the object has been in the zone
  double dwellSec = 0.0;
  double timestamp = 0.0;  // Unix time, seconds
  std::string snapshot;    // path of the saved frame, empty if none
};

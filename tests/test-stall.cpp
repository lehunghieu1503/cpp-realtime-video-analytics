// StallEngine::decode on tensors written here: no model needed
#include "inference/stall-engine.hpp"
#include "server/result-serializer.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
int failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << "\n";
    failures++;
  }
}

// Feature-major [1, 20, count]: cx, cy, w, h, the four class scores (empty,
// occupied, mouth, back), then four times x, y, confidence
struct Tensor {
  int count;
  std::vector<float> data;
  explicit Tensor(int n) : count(n), data(StallEngine::kFeatures * n, 0.0f) {}
  void set(int i, std::initializer_list<float> values) {
    int feature = 0;
    for (float v : values) data[feature++ * count + i] = v;
  }
  StallResult decode(float confidence = 0.5f) const {
    // A 1280x720 frame into a 960x544 input
    return StallEngine::decode(data.data(), {1, StallEngine::kFeatures, count},
                               confidence, 0.5f, 1280, 720, 960, 544);
  }
};
}  // namespace

int main() {
  const float sx = 1280.0f / 960.0f, sy = 720.0f / 544.0f;
  Tensor tensor(7);
  // 0: an empty stall; 1: the same stall again, less sure (NMS drops it);
  // 2: an occupied one elsewhere; 3: below the threshold
  tensor.set(0, {300, 300, 200, 100, 0.9f, 0.05f, 0, 0, 210, 345, 0.95f, 390,
                 350, 0.9f, 380, 255, 0.2f, 220, 250, 0.8f});
  tensor.set(1, {302, 301, 200, 100, 0.7f, 0.05f});
  tensor.set(2, {700, 200, 100, 60, 0.1f, 0.8f, 0, 0, 650, 230, 0.9f, 750, 230,
                 0.9f, 740, 170, 0.9f, 660, 170, 0.9f});
  tensor.set(3, {100, 100, 50, 50, 0.3f, 0.2f});
  // 4: the end of a divider at the mouth, inside stall 0's box (it is not
  // dropped for that); 5: the same end again; 6: one at the back
  tensor.set(4, {210, 345, 30, 16, 0.0f, 0.0f, 0.85f, 0.1f, 209, 344, 0.9f, 211,
                 346, 0.9f, 210, 345, 0.9f, 210, 345, 0.9f});
  tensor.set(5, {211, 346, 30, 16, 0.0f, 0.0f, 0.6f, 0.1f, 215, 350, 0.9f, 215,
                 350, 0.9f, 215, 350, 0.9f, 215, 350, 0.9f});
  tensor.set(6, {380, 255, 24, 10, 0.0f, 0.0f, 0.2f, 0.7f, 380, 255, 0.9f, 380,
                 255, 0.9f, 380, 255, 0.9f, 380, 255, 0.9f});
  auto found = tensor.decode();
  const auto& stalls = found.stalls;
  check(stalls.size() == 2, "two stalls after the threshold and NMS");
  if (stalls.size() == 2) {
    const auto& empty = stalls[0].occupied ? stalls[1] : stalls[0];
    const auto& taken = stalls[0].occupied ? stalls[0] : stalls[1];
    check(!empty.occupied && taken.occupied, "one empty, one occupied");
    check(std::abs(empty.confidence - 0.9f) < 1e-6, "the surer of the two is kept");
    check(std::abs(empty.bbox.x - 200 * sx) < 1e-3 &&
              std::abs(empty.bbox.y - 250 * sy) < 1e-3 &&
              std::abs(empty.bbox.width - 200 * sx) < 1e-3 &&
              std::abs(empty.bbox.height - 100 * sy) < 1e-3,
          "the box in pixels of the frame");
    check(std::abs(empty.corners[1].x - 390 * sx) < 1e-3 &&
              std::abs(empty.corners[1].y - 350 * sy) < 1e-3,
          "a corner in pixels of the frame");
    check(std::abs(empty.cornerConfidence[2] - 0.2f) < 1e-6,
          "a corner keeps its own confidence");

    auto json = serializeStalls({empty});
    check(json.size() == 1 && json[0]["class"] == "empty" &&
              json[0]["corners"].size() == 4 && json[0]["corners"][2][2] == 0.2,
          "the JSON has the class and four corners with their confidence");
  }
  const auto& points = found.points;
  check(points.size() == 2, "two ends of dividers, the repeated one once");
  if (points.size() == 2) {
    const auto& mouth = points[0].back ? points[1] : points[0];
    const auto& back = points[0].back ? points[0] : points[1];
    check(!mouth.back && back.back, "one at the mouth, one at the back");
    check(std::abs(mouth.at.x - 210 * sx) < 1e-3 &&
              std::abs(mouth.at.y - 345 * sy) < 1e-3,
          "an end is the mean of its four points, in pixels of the frame");
    check(std::abs(mouth.confidence - 0.85f) < 1e-6, "the surer end is kept");
    auto json = serializeStallPoints(points);
    check(json.size() == 2 && (json[0]["class"] == "mouth" || json[0]["class"] == "back") &&
              json[0].contains("x") && json[0].contains("y"),
          "the JSON of the ends");
  }

  // A corner outside the image stays where the model put it
  Tensor outside(1);
  outside.set(0, {480, 500, 400, 80, 0.9f, 0.0f, 0, 0, -40, 600, 0.1f, 1000, 600,
                  0.1f, 700, 470, 0.9f, 260, 470, 0.9f});
  auto partial = outside.decode().stalls;
  check(partial.size() == 1 && partial[0].corners[0].x < 0 &&
            partial[0].bbox.y + partial[0].bbox.height <= 720.0f,
        "the box is clipped to the frame, the corners are not");

  // A NaN score is neither a stall nor an end
  Tensor bad(1);
  bad.set(0, {300, 300, 200, 100, std::nanf(""), 0.1f, std::nanf(""), 0.1f});
  auto none = bad.decode();
  check(none.stalls.empty() && none.points.empty(),
        "a score that is not a number is dropped");

  bool thrown = false;
  try {
    std::vector<float> data(84 * 4, 0.0f);
    StallEngine::decode(data.data(), {1, 84, 4}, 0.5f, 0.5f, 1280, 720, 960, 544);
  } catch (const std::runtime_error&) {
    thrown = true;
  }
  check(thrown, "another model's output is refused");

  if (failures == 0) std::cout << "test_stall: all checks passed\n";
  return failures == 0 ? 0 : 1;
}

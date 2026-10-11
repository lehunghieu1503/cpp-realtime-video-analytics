#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>

#include "inference/detection.hpp"
#include "pipeline/frame-data.hpp"
#include "server/result-serializer.hpp"

// Test utilities
namespace test {

struct TestStats {
  int total = 0;
  int passed = 0;
  int failed = 0;

  void reset() {
    total = 0;
    passed = 0;
    failed = 0;
  }

  float passRate() const {
    if (total == 0) return 0.0f;
    return static_cast<float>(passed) / total * 100.0f;
  }
};

TestStats stats;

void assert_true(bool condition, const std::string& testName) {
  stats.total++;
  if (condition) {
    std::cout << "  [PASS] " << testName << "\n";
    stats.passed++;
  } else {
    std::cout << "  [FAIL] " << testName << "\n";
    stats.failed++;
  }
}

void assert_equal(double actual, double expected, double tolerance,
                  const std::string& testName) {
  stats.total++;
  bool pass = std::abs(actual - expected) < tolerance;
  if (pass) {
    std::cout << "  [PASS] " << testName << " (actual=" << actual << ")\n";
    stats.passed++;
  } else {
    std::cout << "  [FAIL] " << testName << " (expected=" << expected
              << ", actual=" << actual << ")\n";
    stats.failed++;
  }
}

void assert_equal_string(const std::string& actual, const std::string& expected,
                         const std::string& testName) {
  stats.total++;
  bool pass = actual == expected;
  if (pass) {
    std::cout << "  [PASS] " << testName << " (actual=\"" << actual << "\")\n";
    stats.passed++;
  } else {
    std::cout << "  [FAIL] " << testName << " (expected=\"" << expected
              << "\", actual=\"" << actual << "\")\n";
    stats.failed++;
  }
}

}  // namespace test

// Helper: Create AnalyticsResult with given detections
AnalyticsResult createResult(uint64_t frameId, double latencyMs,
                            const std::vector<Detection>& detections) {
  AnalyticsResult result;
  result.frameId = frameId;
  result.detections = detections;
  result.inferenceLatencyMs = latencyMs;
  result.captureTime = std::chrono::steady_clock::now();
  result.inferenceDoneTime = result.captureTime;
  result.frame = cv::Mat();  // Empty frame for test
  return result;
}

// Test 1: Serialize empty detections
void testEmptyDetections() {
  std::cout << "\n=== Test 1: Serialize Empty Detections ===\n";

  AnalyticsResult result = createResult(42, 35.26, {});
  nlohmann::json json = serializeResult(result, 21.5);

  test::assert_true(json.contains("frame_id"), "JSON has frame_id field");
  test::assert_true(json.contains("detections"), "JSON has detections field");
  test::assert_true(json.contains("latency_ms"), "JSON has latency_ms field");
  test::assert_true(json.contains("fps"), "JSON has fps field");

  test::assert_true(json["detections"].is_array(),
                    "detections is array");
  test::assert_true(json["detections"].size() == 0,
                    "detections array is empty");

  test::assert_true(json["frame_id"].is_number(), "frame_id is number");
  test::assert_true(json["frame_id"].get<uint64_t>() == 42,
                    "frame_id equals 42");
}

// Test 2: Serialize single detection
void testSingleDetection() {
  std::cout << "\n=== Test 2: Serialize Single Detection ===\n";

  Detection det;
  det.classId = 0;  // person
  det.confidence = 0.95f;
  det.bbox = cv::Rect2f(100.5f, 50.2f, 200.8f, 300.3f);

  AnalyticsResult result = createResult(100, 45.0, {det});
  nlohmann::json json = serializeResult(result, 25.0);

  test::assert_true(json["detections"].size() == 1,
                    "detections array has 1 element");

  auto detection = json["detections"][0];

  test::assert_true(detection.contains("class"), "detection has class field");
  test::assert_equal_string(detection["class"].get<std::string>(), "person",
                            "class name is 'person'");

  test::assert_true(detection.contains("confidence"),
                    "detection has confidence field");
  test::assert_true(detection["confidence"].is_number_float(),
                    "confidence is float");

  test::assert_true(detection.contains("bbox"), "detection has bbox field");
  test::assert_true(detection["bbox"].contains("x"), "bbox has x field");
  test::assert_true(detection["bbox"].contains("y"), "bbox has y field");
  test::assert_true(detection["bbox"].contains("w"), "bbox has w field");
  test::assert_true(detection["bbox"].contains("h"), "bbox has h field");

  // bbox fields should be integers (cast from float)
  test::assert_true(detection["bbox"]["x"].is_number_integer(),
                    "bbox.x is integer");
  test::assert_true(detection["bbox"]["y"].is_number_integer(),
                    "bbox.y is integer");
  test::assert_true(detection["bbox"]["w"].is_number_integer(),
                    "bbox.w is integer");
  test::assert_true(detection["bbox"]["h"].is_number_integer(),
                    "bbox.h is integer");

  test::assert_true(detection["bbox"]["x"].get<int>() == 100,
                    "bbox.x rounds to 100");
  test::assert_true(detection["bbox"]["y"].get<int>() == 50,
                    "bbox.y rounds to 50");
  test::assert_true(detection["bbox"]["w"].get<int>() == 200,
                    "bbox.w rounds to 200");
  test::assert_true(detection["bbox"]["h"].get<int>() == 300,
                    "bbox.h rounds to 300");
}

// Test 3: Multiple detections
void testMultipleDetections() {
  std::cout << "\n=== Test 3: Serialize Multiple Detections ===\n";

  Detection det1;
  det1.classId = 0;  // person
  det1.confidence = 0.92f;
  det1.bbox = cv::Rect2f(10.0f, 20.0f, 100.0f, 150.0f);

  Detection det2;
  det2.classId = 2;  // car
  det2.confidence = 0.87f;
  det2.bbox = cv::Rect2f(200.0f, 150.0f, 250.0f, 180.0f);

  Detection det3;
  det3.classId = 15;  // cat
  det3.confidence = 0.79f;
  det3.bbox = cv::Rect2f(500.0f, 300.0f, 80.0f, 90.0f);

  AnalyticsResult result = createResult(200, 32.5, {det1, det2, det3});
  nlohmann::json json = serializeResult(result, 30.2);

  test::assert_true(json["detections"].size() == 3,
                    "detections array has 3 elements");

  // Check first detection (person)
  test::assert_equal_string(json["detections"][0]["class"].get<std::string>(),
                            "person", "detection[0] class is 'person'");

  // Check second detection (car)
  test::assert_equal_string(json["detections"][1]["class"].get<std::string>(),
                            "car", "detection[1] class is 'car'");

  // Check third detection (cat)
  test::assert_equal_string(json["detections"][2]["class"].get<std::string>(),
                            "cat", "detection[2] class is 'cat'");
}

// Test 4: Verify JSON field types
void testJsonFieldTypes() {
  std::cout << "\n=== Test 4: Verify JSON Field Types ===\n";

  Detection det;
  det.classId = 1;  // bicycle
  det.confidence = 0.88f;
  det.bbox = cv::Rect2f(75.0f, 100.0f, 120.0f, 160.0f);

  AnalyticsResult result = createResult(50, 28.34, {det});
  nlohmann::json json = serializeResult(result, 24.8);

  // Top-level fields
  test::assert_true(json["frame_id"].is_number_unsigned(),
                    "frame_id is unsigned number");
  test::assert_true(json["fps"].is_number_float(), "fps is float");
  test::assert_true(json["latency_ms"].is_number_float(),
                    "latency_ms is float");
  test::assert_true(json["detections"].is_array(), "detections is array");

  // Detection fields
  auto det_json = json["detections"][0];
  test::assert_true(det_json["class"].is_string(),
                    "detection.class is string");
  test::assert_true(det_json["confidence"].is_number_float(),
                    "detection.confidence is float");
  test::assert_true(det_json["bbox"].is_object(), "detection.bbox is object");

  // Bbox fields
  test::assert_true(det_json["bbox"]["x"].is_number_integer(),
                    "bbox.x is integer");
  test::assert_true(det_json["bbox"]["y"].is_number_integer(),
                    "bbox.y is integer");
  test::assert_true(det_json["bbox"]["w"].is_number_integer(),
                    "bbox.w is integer");
  test::assert_true(det_json["bbox"]["h"].is_number_integer(),
                    "bbox.h is integer");
}

// Test 5: Verify class name mapping for various IDs
void testClassNameMapping() {
  std::cout << "\n=== Test 5: Verify Class Name Mapping ===\n";

  // Test various class IDs
  std::vector<std::pair<int, const char*>> classTests = {
      {0, "person"},      {1, "bicycle"},   {2, "car"},         {5, "bus"},
      {15, "cat"},        {16, "dog"},      {39, "bottle"},     {40, "wine glass"},
      {50, "broccoli"},   {79, "toothbrush"}};

  for (const auto& [classId, expectedName] : classTests) {
    Detection det;
    det.classId = classId;
    det.confidence = 0.85f;
    det.bbox = cv::Rect2f(0.0f, 0.0f, 100.0f, 100.0f);

    AnalyticsResult result = createResult(1, 10.0, {det});
    nlohmann::json json = serializeResult(result, 20.0);

    test::assert_equal_string(json["detections"][0]["class"].get<std::string>(),
                              expectedName,
                              std::string("classId ") + std::to_string(classId) +
                                  " maps to '" + expectedName + "'");
  }
}

// Test 6: Confidence rounding to 2 decimals
void testConfidenceRounding() {
  std::cout << "\n=== Test 6: Confidence Rounding (2 decimals) ===\n";

  std::vector<std::pair<float, double>> roundingTests = {
      {0.9181f, 0.92},  // rounds up
      {0.9145f, 0.91},  // rounds down
      {0.995f, 1.00},   // rounds up to 1.0
      {0.994f, 0.99},   // rounds down
      {0.505f, 0.51},   // .505 rounds up (round half away from zero)
      {0.515f, 0.52},   // .515 rounds up
      {0.1f, 0.10},     // single decimal
      {1.0f, 1.00},     // full confidence
      {0.0f, 0.00},     // no confidence
  };

  for (const auto& [inputConf, expectedConf] : roundingTests) {
    Detection det;
    det.classId = 0;
    det.confidence = inputConf;
    det.bbox = cv::Rect2f(0.0f, 0.0f, 100.0f, 100.0f);

    AnalyticsResult result = createResult(1, 10.0, {det});
    nlohmann::json json = serializeResult(result, 20.0);

    double actualConf = json["detections"][0]["confidence"].get<double>();
    test::assert_equal(
        actualConf, expectedConf, 0.001,
        std::string("confidence ") + std::to_string(inputConf) + " rounds to " +
            std::to_string(expectedConf));
  }

  // The serialized text must be short too, not a float widened to double
  Detection det;
  det.classId = 0;
  det.confidence = 0.94f;
  det.bbox = cv::Rect2f(0.0f, 0.0f, 100.0f, 100.0f);
  nlohmann::json json = serializeResult(createResult(1, 10.0, {det}), 20.0);
  test::assert_equal_string(json["detections"][0]["confidence"].dump(), "0.94",
                            "confidence 0.94f serializes as 0.94");
}

// Test 7: Latency rounding to 1 decimal
void testLatencyRounding() {
  std::cout << "\n=== Test 7: Latency Rounding (1 decimal) ===\n";

  std::vector<std::pair<double, double>> roundingTests = {
      {35.26, 35.3},   // rounds up
      {35.24, 35.2},   // rounds down
      {35.95, 36.0},   // rounds up to next integer
      {35.94, 35.9},   // rounds down
      {10.05, 10.1},   // .05 rounds up
      {10.04, 10.0},   // .04 rounds down
      {0.0, 0.0},      // zero latency
      {100.0, 100.0},  // whole number
  };

  for (const auto& [inputLatency, expectedLatency] : roundingTests) {
    Detection det;
    det.classId = 0;
    det.confidence = 0.9f;
    det.bbox = cv::Rect2f(0.0f, 0.0f, 100.0f, 100.0f);

    AnalyticsResult result = createResult(1, inputLatency, {det});
    nlohmann::json json = serializeResult(result, 20.0);

    double actualLatency = json["latency_ms"].get<double>();
    test::assert_equal(
        actualLatency, expectedLatency, 0.001,
        std::string("latency ") + std::to_string(inputLatency) + " rounds to " +
            std::to_string(expectedLatency));
  }
}

// Test 8: FPS rounding to 1 decimal
void testFpsRounding() {
  std::cout << "\n=== Test 8: FPS Rounding (1 decimal) ===\n";

  std::vector<std::pair<double, double>> roundingTests = {
      {21.54, 21.5},   // rounds down
      {21.56, 21.6},   // rounds up
      {29.95, 30.0},   // rounds up to next integer
      {29.94, 29.9},   // rounds down
      {60.0, 60.0},    // whole number
      {15.0, 15.0},    // whole number
  };

  AnalyticsResult result = createResult(1, 10.0, {});

  for (const auto& [inputFps, expectedFps] : roundingTests) {
    nlohmann::json json = serializeResult(result, inputFps);

    double actualFps = json["fps"].get<double>();
    test::assert_equal(
        actualFps, expectedFps, 0.001,
        std::string("fps ") + std::to_string(inputFps) + " rounds to " +
            std::to_string(expectedFps));
  }
}

// Test 9: BBox integer casting
void testBboxIntegerCasting() {
  std::cout << "\n=== Test 9: BBox Integer Casting ===\n";

  std::vector<std::tuple<float, float, float, float, int, int, int, int>>
      castingTests = {
          {10.2f, 20.8f, 100.5f, 200.3f, 10, 20, 100, 200},  // truncate decimals
          {0.1f, 0.9f, 1.1f, 1.9f, 0, 0, 1, 1},              // round down/up
          {99.99f, 99.99f, 99.99f, 99.99f, 99, 99, 99, 99},  // near integer
  };

  for (const auto& [x, y, w, h, ex, ey, ew, eh] : castingTests) {
    Detection det;
    det.classId = 0;
    det.confidence = 0.8f;
    det.bbox = cv::Rect2f(x, y, w, h);

    AnalyticsResult result = createResult(1, 10.0, {det});
    nlohmann::json json = serializeResult(result, 20.0);

    auto bbox = json["detections"][0]["bbox"];
    test::assert_true(bbox["x"].get<int>() == ex,
                      std::string("bbox.x: ") + std::to_string(x) + " -> " +
                          std::to_string(ex));
    test::assert_true(bbox["y"].get<int>() == ey,
                      std::string("bbox.y: ") + std::to_string(y) + " -> " +
                          std::to_string(ey));
    test::assert_true(bbox["w"].get<int>() == ew,
                      std::string("bbox.w: ") + std::to_string(w) + " -> " +
                          std::to_string(ew));
    test::assert_true(bbox["h"].get<int>() == eh,
                      std::string("bbox.h: ") + std::to_string(h) + " -> " +
                          std::to_string(eh));
  }
}

// Test 10: Large frame IDs
void testLargeFrameIds() {
  std::cout << "\n=== Test 10: Large Frame IDs ===\n";

  uint64_t largeFrameId = 18446744073709551615ULL;  // max uint64_t

  AnalyticsResult result = createResult(largeFrameId, 10.0, {});
  nlohmann::json json = serializeResult(result, 20.0);

  uint64_t actualFrameId = json["frame_id"].get<uint64_t>();
  test::assert_true(actualFrameId == largeFrameId,
                    "Large frame ID preserved correctly");
}

void testSimulationIdentity() {
  std::cout << "\n=== Simulation identity and image size ===\n";
  AnalyticsResult result = createResult(7, 10.0, {});

  auto plain = serializeResult(result, 30.0);
  test::assert_true(!plain.contains("simulation") && !plain.contains("image"),
                    "An ordinary source has neither simulation nor image");

  result.imageWidth = 1280;
  result.imageHeight = 720;
  // Past 2^53: a double would round it
  result.sourcePtsNs = 9007199254740993ULL;
  auto json = serializeResult(result, 30.0, "r_20261011_ab12");
  test::assert_true(json["image"]["width"] == 1280 &&
                        json["image"]["height"] == 720,
                    "Image size is that of the source frame");
  test::assert_equal_string(json["simulation"]["run_id"], "r_20261011_ab12",
                            "Run id");
  test::assert_true(json["simulation"]["sim_ns"].is_number_unsigned() &&
                        json["simulation"]["sim_ns"].get<uint64_t>() ==
                            9007199254740993ULL,
                    "sim_ns is the exact integer");
  test::assert_true(json.dump().find("9007199254740993") != std::string::npos,
                    "sim_ns is written without rounding");
  test::assert_true(json["frame_id"] == 7, "The local frame id stays");

  test::assert_true(!serializeResult(result, 30.0).contains("simulation"),
                    "No run id, no simulation block");
  result.sourcePtsNs.reset();
  test::assert_true(!serializeResult(result, 30.0, "r1").contains("simulation"),
                    "No timestamp, no simulation block: none is invented");
  result.sourcePtsNs = 0;
  test::assert_true(serializeResult(result, 30.0, "r1")["simulation"]["sim_ns"] == 0,
                    "A timestamp of 0 is a timestamp");
}

int main() {
  std::cout << "==================================================\n";
  std::cout << "Result Serializer Test Suite\n";
  std::cout << "SSE Streaming JSON Serialization\n";
  std::cout << "==================================================\n";

  test::stats.reset();

  // Run all tests
  testEmptyDetections();
  testSingleDetection();
  testMultipleDetections();
  testJsonFieldTypes();
  testClassNameMapping();
  testConfidenceRounding();
  testLatencyRounding();
  testFpsRounding();
  testBboxIntegerCasting();
  testLargeFrameIds();
  testSimulationIdentity();

  // Print summary
  std::cout << "\n==================================================\n";
  std::cout << "Test Summary\n";
  std::cout << "==================================================\n";
  std::cout << "Total assertions: " << test::stats.total << "\n";
  std::cout << "Passed: " << test::stats.passed << "\n";
  std::cout << "Failed: " << test::stats.failed << "\n";
  std::cout << "Pass rate: " << test::stats.passRate() << "%\n";
  std::cout << "==================================================\n";

  // Return success if all tests passed
  return test::stats.failed == 0 ? 0 : 1;
}

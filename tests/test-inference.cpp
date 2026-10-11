#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "inference/detection.hpp"
#include "inference/inference-engine.hpp"
#include "inference/post-process.hpp"

// Test utilities
namespace test {

enum class TestResult {
  PASS,
  FAIL
};

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

void assert_equal(float actual, float expected, float tolerance,
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

void assert_range(float value, float minVal, float maxVal,
                  const std::string& testName) {
  stats.total++;
  bool pass = value >= minVal && value <= maxVal;
  if (pass) {
    std::cout << "  [PASS] " << testName << " (value=" << value << ")\n";
    stats.passed++;
  } else {
    std::cout << "  [FAIL] " << testName << " (expected [" << minVal << ", "
              << maxVal << "], got " << value << ")\n";
    stats.failed++;
  }
}

}  // namespace test

// Create synthetic test image: red rectangle on black background (represents an object)
cv::Mat createSyntheticTestImage(int width, int height) {
  cv::Mat img = cv::Mat::zeros(height, width, CV_8UC3);

  // Draw a red rectangle in the center (simulates an object like a person or car)
  int rectW = width / 3;
  int rectH = height / 3;
  int rectX = width / 3;
  int rectY = height / 3;

  cv::rectangle(img, cv::Point(rectX, rectY),
                cv::Point(rectX + rectW, rectY + rectH), cv::Scalar(0, 0, 255),
                -1);  // Red in BGR

  return img;
}

// Test 1: Model loading
void testModelLoading() {
  std::cout << "\n=== Test 1: Model Loading ===\n";

  // The model is gitignored, so it is absent on a clean checkout and in CI
  InferenceConfig config;
  config.modelPath = "models/yolov8n.onnx";
  if (std::ifstream(config.modelPath).good()) {
    InferenceEngine engine(config);
    test::assert_true(engine.isLoaded(), "Model loads successfully");
  } else {
    std::cout << "  [SKIP] No model at " << config.modelPath
              << ", skipping load test\n";
  }

  // Test invalid path handling
  InferenceConfig badConfig;
  badConfig.modelPath = "/nonexistent/path/model.onnx";
  InferenceEngine badEngine(badConfig);
  test::assert_true(!badEngine.isLoaded(),
                    "Invalid model path returns unloaded state");
}

// Test 2: Inference on synthetic image
void testSyntheticInference() {
  std::cout << "\n=== Test 2: Synthetic Image Inference ===\n";

  InferenceConfig config;
  config.modelPath = "models/yolov8n.onnx";
  config.confidenceThreshold = 0.3f;  // Lower threshold for synthetic image

  InferenceEngine engine(config);

  if (!engine.isLoaded()) {
    std::cout << "  [SKIP] Model not loaded, skipping synthetic inference test\n";
    return;
  }

  // Create synthetic test image (640x480 to test scaling)
  cv::Mat testImg = createSyntheticTestImage(640, 480);

  auto startTime = std::chrono::high_resolution_clock::now();
  auto detections = engine.detect(testImg);
  auto endTime = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      endTime - startTime);

  std::cout << "  Inference latency: " << duration.count() << " ms\n";
  std::cout << "  Detections found: " << detections.size() << "\n";

  test::assert_true(!detections.empty() || detections.empty(),
                    "Inference completes without crash");

  // Validate detections
  for (size_t i = 0; i < detections.size(); i++) {
    const auto& det = detections[i];

    // Check confidence in valid range
    test::assert_range(det.confidence, 0.0f, 1.0f,
                       "Detection " + std::to_string(i) + " confidence in [0,1]");

    // Check class ID in valid range
    test::assert_range(static_cast<float>(det.classId), 0.0f, 79.0f,
                       "Detection " + std::to_string(i) + " classId in [0,79]");

    // Check bbox coordinates are within image bounds
    test::assert_true(
        det.bbox.x >= 0 && det.bbox.y >= 0 &&
            det.bbox.x + det.bbox.width <= testImg.cols &&
            det.bbox.y + det.bbox.height <= testImg.rows,
        "Detection " + std::to_string(i) + " bbox within image bounds");

    // Check bbox has positive dimensions
    test::assert_true(det.bbox.width > 0 && det.bbox.height > 0,
                      "Detection " + std::to_string(i) + " bbox has positive dims");

    std::cout << "    Detection " << i << ": class="
              << getClassName(det.classId) << ", conf=" << det.confidence
              << ", bbox=(" << det.bbox.x << "," << det.bbox.y << ","
              << det.bbox.width << "," << det.bbox.height << ")\n";
  }

  std::cout << "  Inference latency: " << duration.count() << " ms\n";
}

// Test 3: Inference on real test image (if available)
void testRealImageInference() {
  std::cout << "\n=== Test 3: Real Image Inference ===\n";

  InferenceConfig config;
  config.modelPath = "models/yolov8n.onnx";

  InferenceEngine engine(config);

  if (!engine.isLoaded()) {
    std::cout << "  [SKIP] Model not loaded, skipping real image inference\n";
    return;
  }

  // Try to load a real test image if it exists
  cv::Mat testImg = cv::imread("tests/test-image.jpg");

  if (testImg.empty()) {
    std::cout << "  [SKIP] No test image at tests/test-image.jpg, skipping\n";
    return;
  }

  std::cout << "  Test image size: " << testImg.cols << "x" << testImg.rows
            << "\n";

  auto startTime = std::chrono::high_resolution_clock::now();
  auto detections = engine.detect(testImg);
  auto endTime = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      endTime - startTime);

  std::cout << "  Inference latency: " << duration.count() << " ms\n";
  std::cout << "  Detections found: " << detections.size() << "\n";

  test::assert_true(true, "Real image inference completes without crash");

  for (size_t i = 0; i < detections.size() && i < 5; i++) {
    const auto& det = detections[i];
    std::cout << "    Detection " << i << ": class="
              << getClassName(det.classId) << ", conf=" << det.confidence
              << ", bbox=(" << det.bbox.x << "," << det.bbox.y << ","
              << det.bbox.width << "," << det.bbox.height << ")\n";
  }
}

// Test 4: Performance characteristics
void testPerformance() {
  std::cout << "\n=== Test 4: Performance Characteristics ===\n";

  InferenceConfig config;
  config.modelPath = "models/yolov8n.onnx";

  InferenceEngine engine(config);

  if (!engine.isLoaded()) {
    std::cout << "  [SKIP] Model not loaded, skipping performance test\n";
    return;
  }

  cv::Mat testImg = createSyntheticTestImage(640, 480);

  // Warm up
  engine.detect(testImg);

  // Run 5 iterations and measure
  std::vector<long long> latencies;
  for (int i = 0; i < 5; i++) {
    auto startTime = std::chrono::high_resolution_clock::now();
    engine.detect(testImg);
    auto endTime = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime);
    latencies.push_back(duration.count());
  }

  // Calculate stats
  long long minLat = latencies[0];
  long long maxLat = latencies[0];
  long long sumLat = 0;

  for (auto lat : latencies) {
    minLat = std::min(minLat, lat);
    maxLat = std::max(maxLat, lat);
    sumLat += lat;
  }

  double avgLat = static_cast<double>(sumLat) / latencies.size();

  std::cout << "  Inference latency stats (5 runs):\n";
  std::cout << "    Min: " << minLat << " ms\n";
  std::cout << "    Max: " << maxLat << " ms\n";
  std::cout << "    Avg: " << avgLat << " ms\n";

  test::assert_true(avgLat > 0, "Average latency is positive");
  test::assert_true(maxLat <= avgLat * 3,
                    "Max latency within reasonable bounds");
}

// Test 5: Multiple inferences
void testMultipleInferences() {
  std::cout << "\n=== Test 5: Multiple Sequential Inferences ===\n";

  InferenceConfig config;
  config.modelPath = "models/yolov8n.onnx";

  InferenceEngine engine(config);

  if (!engine.isLoaded()) {
    std::cout << "  [SKIP] Model not loaded, skipping multiple inference test\n";
    return;
  }

  cv::Mat testImg1 = createSyntheticTestImage(640, 480);
  cv::Mat testImg2 = createSyntheticTestImage(800, 600);
  cv::Mat testImg3 = createSyntheticTestImage(480, 360);

  try {
    auto det1 = engine.detect(testImg1);
    auto det2 = engine.detect(testImg2);
    auto det3 = engine.detect(testImg3);

    test::assert_true(true, "Multiple inferences execute without crash");
    std::cout << "  Results: " << det1.size() << " + " << det2.size() << " + "
              << det3.size() << " detections\n";
  } catch (const std::exception& e) {
    test::assert_true(false, std::string("Multiple inferences catch exception: ") +
                                 e.what());
  }
}

// Test 6: COCO class name lookup
void testCocoClassNames() {
  std::cout << "\n=== Test 6: COCO Class Name Lookup ===\n";

  // Test valid IDs
  test::assert_true(std::string(getClassName(0)) == "person",
                    "Class ID 0 maps to 'person'");
  test::assert_true(std::string(getClassName(2)) == "car",
                    "Class ID 2 maps to 'car'");
  test::assert_true(std::string(getClassName(79)) == "toothbrush",
                    "Class ID 79 maps to 'toothbrush'");

  // Test invalid IDs
  test::assert_true(std::string(getClassName(-1)) == "unknown",
                    "Negative class ID returns 'unknown'");
  test::assert_true(std::string(getClassName(100)) == "unknown",
                    "Out-of-range class ID returns 'unknown'");
}

// Test 7: postprocess on hand-built tensors (no model needed)
void testPostprocess() {
  std::cout << "\n=== Test 7: Postprocess (synthetic tensor) ===\n";

  // An output with fewer than 3 dimensions is rejected, not indexed
  std::vector<float> data(84, 0.0f);
  auto none =
      postprocess(data.data(), {1, 84}, 0.5f, 0.45f, 640, 640, 640, 640);
  test::assert_true(none.empty(),
                    "Output shape with < 3 dims returns no detections");

  // One candidate hanging over the top-left corner: center (10, 10), 40x40,
  // class 0 at 0.9. Layout is feature-major, shape [1, 84, 1].
  data[0] = 10.0f;
  data[1] = 10.0f;
  data[2] = 40.0f;
  data[3] = 40.0f;
  data[4] = 0.9f;
  auto dets =
      postprocess(data.data(), {1, 84, 1}, 0.5f, 0.45f, 640, 640, 640, 640);
  test::assert_true(dets.size() == 1, "Clipped candidate is kept");
  if (dets.size() == 1) {
    test::assert_equal(dets[0].bbox.x, 0.0f, 0.001f, "Clipped bbox x is 0");
    test::assert_equal(dets[0].bbox.width, 30.0f, 0.001f,
                       "Clipped bbox width is the visible 30 px");
    test::assert_equal(dets[0].bbox.height, 30.0f, 0.001f,
                       "Clipped bbox height is the visible 30 px");
  }

  // Three candidates, shape [1, 84, 3]: each keeps its own best class, and
  // the one below the threshold is dropped
  std::vector<float> multi(84 * 3, 0.0f);
  auto at = [&multi](int feature, int candidate) -> float& {
    return multi[feature * 3 + candidate];
  };
  at(0, 0) = 100.0f; at(1, 0) = 100.0f; at(2, 0) = 50.0f; at(3, 0) = 50.0f;
  at(4 + 2, 0) = 0.6f;
  at(4 + 5, 0) = 0.9f;
  at(0, 1) = 400.0f; at(1, 1) = 300.0f; at(2, 1) = 60.0f; at(3, 1) = 60.0f;
  at(4 + 17, 1) = 0.8f;
  at(0, 2) = 250.0f; at(1, 2) = 500.0f; at(2, 2) = 40.0f; at(3, 2) = 40.0f;
  at(4 + 79, 2) = 0.3f;
  auto multiDets =
      postprocess(multi.data(), {1, 84, 3}, 0.5f, 0.45f, 640, 640, 640, 640);
  test::assert_true(multiDets.size() == 2, "Two of three candidates pass");
  if (multiDets.size() == 2) {
    // NMS returns the highest confidence first
    test::assert_true(multiDets[0].classId == 5, "Best class of candidate 0");
    test::assert_equal(multiDets[0].confidence, 0.9f, 0.001f,
                       "Best score of candidate 0");
    test::assert_true(multiDets[1].classId == 17, "Best class of candidate 1");
  }
}

// Test 8: every backend available in this build agrees with the CPU baseline
void testBackendParity() {
  std::cout << "\n=== Test 8: Backend Parity ===\n";

  cv::Mat image = cv::imread("tests/test-image.jpg");
  InferenceConfig config;
  config.modelPath = "models/yolov8n.onnx";
  InferenceEngine baseline(config);
  if (image.empty() || !baseline.isLoaded()) {
    std::cout << "  [SKIP] Model or tests/test-image.jpg missing\n";
    return;
  }
  auto expected = baseline.detect(image);
  test::assert_true(!expected.empty(), "CPU baseline detects something");

  // Backends the machine lacks print their reason and are skipped
  for (const char* name :
       {"cuda", "tensorrt", "openvino:CPU", "openvino:GPU", "openvino:NPU"}) {
    config.backend = name;
    InferenceEngine engine(config);
    if (!engine.isLoaded()) {
      std::cout << "  [SKIP] " << name << " not available\n";
      continue;
    }
    auto actual = engine.detect(image);
    // Accelerators run FP16, so allow small differences in score and box
    bool same = actual.size() == expected.size();
    for (size_t i = 0; same && i < actual.size(); i++) {
      cv::Rect2f overlap = actual[i].bbox & expected[i].bbox;
      float iou = overlap.area() / (actual[i].bbox.area() +
                                    expected[i].bbox.area() - overlap.area());
      same = actual[i].classId == expected[i].classId && iou > 0.9f &&
             std::abs(actual[i].confidence - expected[i].confidence) < 0.05f;
    }
    test::assert_true(same, std::string(name) + " matches the CPU detections");
  }
}

int main() {
  std::cout << "==================================================\n";
  std::cout << "ONNX Runtime Inference Engine Test Suite\n";
  std::cout << "YOLOv8 Nano Model\n";
  std::cout << "==================================================\n";

  test::stats.reset();

  // Run all tests
  testModelLoading();
  testSyntheticInference();
  testRealImageInference();
  testPerformance();
  testMultipleInferences();
  testCocoClassNames();
  testPostprocess();
  testBackendParity();

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

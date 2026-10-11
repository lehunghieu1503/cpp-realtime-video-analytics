#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "pipeline/bounded-queue.hpp"
#include "pipeline/frame-data.hpp"
#include "pipeline/stream-queue.hpp"

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

template <typename T>
void assert_eq(T actual, T expected, const std::string& testName) {
  stats.total++;
  if (actual == expected) {
    std::cout << "  [PASS] " << testName << "\n";
    stats.passed++;
  } else {
    std::cout << "  [FAIL] " << testName << " (expected " << expected << ", got " << actual << ")\n";
    stats.failed++;
  }
}

}  // namespace test

// ============================================================================
// TEST SUITE 1: BoundedQueue Basic Operations
// ============================================================================

void test_bounded_queue_single_push_pop() {
  std::cout << "\n[TEST SUITE 1] BoundedQueue Basic Operations\n";

  BoundedQueue<int> q(5);

  // Test 1.1: push and pop single item
  q.push(42);
  auto result = q.pop();
  test::assert_true(result.has_value(), "Single push/pop returns value");
  test::assert_true(result.value() == 42, "Single push/pop returns correct value");
}

void test_bounded_queue_fifo_order() {
  BoundedQueue<int> q(10);

  q.push(1);
  q.push(2);
  q.push(3);

  auto r1 = q.pop();
  auto r2 = q.pop();
  auto r3 = q.pop();

  test::assert_true(r1.has_value() && r1.value() == 1, "FIFO order: first item is 1");
  test::assert_true(r2.has_value() && r2.value() == 2, "FIFO order: second item is 2");
  test::assert_true(r3.has_value() && r3.value() == 3, "FIFO order: third item is 3");
}

void test_bounded_queue_size_tracking() {
  BoundedQueue<int> q(5);

  test::assert_eq(q.size(), 0UL, "Initial queue is empty");

  q.push(1);
  test::assert_eq(q.size(), 1UL, "Size increases after push");

  q.push(2);
  q.push(3);
  test::assert_eq(q.size(), 3UL, "Size tracks multiple pushes");

  q.pop();
  test::assert_eq(q.size(), 2UL, "Size decreases after pop");
}

// ============================================================================
// TEST SUITE 2: BoundedQueue Overflow Behavior
// ============================================================================

void test_bounded_queue_overflow_drops_oldest() {
  std::cout << "\n[TEST SUITE 2] BoundedQueue Overflow (Drop-Oldest Policy)\n";

  BoundedQueue<int> q(3);

  q.push(10);
  q.push(20);
  q.push(30);
  test::assert_eq(q.size(), 3UL, "Queue fills to maxSize");

  q.push(40);  // overflow: should drop 10
  test::assert_eq(q.size(), 3UL, "Queue stays at maxSize after overflow");

  auto r1 = q.pop();
  auto r2 = q.pop();
  auto r3 = q.pop();

  test::assert_true(r1.has_value() && r1.value() == 20, "Overflow drops oldest (10), returns 20");
  test::assert_true(r2.has_value() && r2.value() == 30, "Second item is 30");
  test::assert_true(r3.has_value() && r3.value() == 40, "Newest item is 40");
}

void test_bounded_queue_multiple_overflows() {
  BoundedQueue<int> q(2);

  q.push(1);
  q.push(2);
  q.push(3);  // drops 1
  q.push(4);  // drops 2
  q.push(5);  // drops 3

  test::assert_eq(q.size(), 2UL, "Queue size never exceeds maxSize after multiple overflows");

  auto r1 = q.pop();
  auto r2 = q.pop();

  test::assert_true(r1.has_value() && r1.value() == 4, "Multiple overflows: first result is 4");
  test::assert_true(r2.has_value() && r2.value() == 5, "Multiple overflows: second result is 5");
}

void test_bounded_queue_max_size_one() {
  BoundedQueue<int> q(1);

  q.push(100);
  test::assert_eq(q.size(), 1UL, "Single-item queue stores one item");

  q.push(200);  // overflow: drops 100
  test::assert_eq(q.size(), 1UL, "Single-item queue never exceeds size 1");

  auto result = q.pop();
  test::assert_true(result.has_value() && result.value() == 200, "Single-item queue after overflow returns newest");
}

// ============================================================================
// TEST SUITE 3: BoundedQueue Stop Signal
// ============================================================================

void test_bounded_queue_stop_unblocks_pop() {
  std::cout << "\n[TEST SUITE 3] BoundedQueue Stop Signal\n";

  BoundedQueue<int> q(5);
  std::atomic<bool> pop_returned{false};
  std::optional<int> popResult;

  // Start a thread that tries to pop from empty queue (will block)
  std::thread popper([&q, &pop_returned, &popResult]() {
    popResult = q.pop();
    pop_returned = true;
  });

  // Give popper time to block on pop()
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  test::assert_true(!pop_returned, "Pop is blocking before stop");

  // Stop the queue to unblock pop()
  q.stop();

  // Wait for popper to finish
  popper.join();

  test::assert_true(pop_returned, "Pop unblocked after stop");
  test::assert_true(!popResult.has_value(), "Pop after stop returns nullopt");
}

void test_bounded_queue_stop_with_items() {
  BoundedQueue<int> q(5);

  q.push(1);
  q.push(2);

  q.stop();

  auto r1 = q.pop();
  auto r2 = q.pop();
  auto r3 = q.pop();

  test::assert_true(r1.has_value() && r1.value() == 1, "Can pop items after stop (first item)");
  test::assert_true(r2.has_value() && r2.value() == 2, "Can pop items after stop (second item)");
  test::assert_true(!r3.has_value(), "Pop after draining returns nullopt");
}

void test_bounded_queue_is_stopped() {
  BoundedQueue<int> q(5);

  test::assert_true(!q.isStopped(), "Queue is not stopped initially");

  q.stop();

  test::assert_true(q.isStopped(), "Queue is stopped after stop() call");
}

// ============================================================================
// TEST SUITE 4: BoundedQueue Multi-Threaded Operations
// ============================================================================

void test_bounded_queue_producer_consumer() {
  std::cout << "\n[TEST SUITE 4] BoundedQueue Multi-Threaded Operations\n";

  BoundedQueue<int> q(10);
  const int numItems = 50;
  std::atomic<int> consumed{0};

  // Producer thread - produce items with slight delay
  std::thread producer([&q, numItems]() {
    for (int i = 0; i < numItems; ++i) {
      q.push(i);
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });

  // Single consumer thread
  std::thread consumer([&q, &consumed]() {
    while (true) {
      auto result = q.pop();
      if (result.has_value()) {
        consumed++;
      } else {
        // nullopt returned after stop
        break;
      }
    }
  });

  producer.join();

  // Give consumer time to process remaining items, then stop queue
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  q.stop();

  consumer.join();

  test::assert_true(consumed.load() >= numItems / 2, "Producer-consumer stress test: consumed at least half items");
  test::assert_eq(q.size(), 0UL, "Producer-consumer stress test: queue is empty after completion");
}

void test_bounded_queue_concurrent_push_pop() {
  BoundedQueue<int> q(8);
  std::atomic<int> pushCount{0};
  std::atomic<int> popCount{0};
  const int operationsPerThread = 30;

  auto pusher = [&q, &pushCount]() {
    for (int i = 0; i < operationsPerThread; ++i) {
      q.push(i);
      pushCount++;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  };

  auto popper = [&q, &popCount]() {
    while (true) {
      auto result = q.pop();
      if (result.has_value()) {
        popCount++;
      } else {
        // Queue was stopped, break
        break;
      }
    }
  };

  std::thread p1(pusher);
  std::thread p2(pusher);
  std::thread c1(popper);
  std::thread c2(popper);

  p1.join();
  p2.join();

  // Allow consumers to finish
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  q.stop();

  c1.join();
  c2.join();

  test::assert_eq(pushCount.load(), 60, "Concurrent push: all pushes completed");
  test::assert_true(popCount.load() >= 30, "Concurrent pop: consumed at least half items");
}

void test_bounded_queue_overflow_under_load() {
  BoundedQueue<int> q(5);
  std::atomic<int> consumed{0};

  // Fast producer that will cause overflow
  std::thread producer([&q]() {
    for (int i = 0; i < 100; ++i) {
      q.push(i);
    }
  });

  // Consumer thread
  std::thread consumer([&q, &consumed]() {
    while (true) {
      auto result = q.pop();
      if (result.has_value()) {
        consumed++;
      } else {
        break;
      }
    }
  });

  producer.join();

  // Give consumer time to process, then stop
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  q.stop();
  consumer.join();

  test::assert_true(q.size() <= 5UL, "Queue never exceeds maxSize under load");
  test::assert_true(consumed.load() >= 1, "Consumer receives items from queue");
}

// ============================================================================
// TEST SUITE 5: FrameData and AnalyticsResult Structures
// ============================================================================

void test_frame_data_creation() {
  std::cout << "\n[TEST SUITE 5] FrameData and AnalyticsResult Structures\n";

  FrameData frame;
  frame.frameId = 42;
  frame.captureTime = std::chrono::steady_clock::now();

  test::assert_eq(frame.frameId, 42UL, "FrameData frameId assignment");
  test::assert_true(frame.frame.empty(), "FrameData frame is empty by default");
}

void test_analytics_result_creation() {
  AnalyticsResult result;
  result.frameId = 99;
  result.inferenceLatencyMs = 15.5;

  test::assert_eq(result.frameId, 99UL, "AnalyticsResult frameId assignment");
  test::assert_true(result.detections.empty(), "AnalyticsResult detections is empty by default");
}

void test_bounded_queue_with_frame_data() {
  BoundedQueue<FrameData> q(5);

  FrameData f1;
  f1.frameId = 1;

  FrameData f2;
  f2.frameId = 2;

  q.push(std::move(f1));
  q.push(std::move(f2));

  auto r1 = q.pop();
  auto r2 = q.pop();

  test::assert_true(r1.has_value() && r1.value().frameId == 1, "BoundedQueue<FrameData>: first frame");
  test::assert_true(r2.has_value() && r2.value().frameId == 2, "BoundedQueue<FrameData>: second frame");
}

void test_bounded_queue_with_analytics_result() {
  BoundedQueue<AnalyticsResult> q(5);

  AnalyticsResult r1;
  r1.frameId = 10;
  r1.inferenceLatencyMs = 12.5;

  AnalyticsResult r2;
  r2.frameId = 20;
  r2.inferenceLatencyMs = 14.2;

  q.push(std::move(r1));
  q.push(std::move(r2));

  auto result1 = q.pop();
  auto result2 = q.pop();

  test::assert_true(result1.has_value() && result1.value().frameId == 10,
                    "BoundedQueue<AnalyticsResult>: first frameId");
  test::assert_true(result1.has_value() && result1.value().inferenceLatencyMs > 12.0,
                    "BoundedQueue<AnalyticsResult>: first latency");
  test::assert_true(result2.has_value() && result2.value().frameId == 20,
                    "BoundedQueue<AnalyticsResult>: second frameId");
}

// ============================================================================
// Main Test Runner
// ============================================================================

// ============================================================================
// TEST SUITE 6: StreamQueue (per-stream slots, round-robin)
// ============================================================================

FrameData makeFrame(int streamId, uint64_t frameId) {
  FrameData frame;
  frame.streamId = streamId;
  frame.frameId = frameId;
  return frame;
}

void test_stream_queue_is_fair() {
  std::cout << "\n[TEST SUITE 6] StreamQueue\n";

  // Stream 0 floods the queue, stream 1 pushes once
  StreamQueue q(2, 2);
  for (uint64_t i = 0; i < 10; i++) q.push(makeFrame(0, i));
  q.push(makeFrame(1, 100));
  test::assert_eq(q.size(0), 2UL, "StreamQueue: a stream keeps `depth` frames");
  test::assert_eq(q.size(1), 1UL,
                  "StreamQueue: a busy stream does not evict another's frame");

  auto a = q.pop();
  auto b = q.pop();
  auto c = q.pop();
  test::assert_true(a && a->streamId == 0 && a->frameId == 8,
                    "StreamQueue: the oldest frames of a full stream are dropped");
  test::assert_true(b && b->streamId == 1,
                    "StreamQueue: streams are served in turn");
  test::assert_true(c && c->streamId == 0 && c->frameId == 9,
                    "StreamQueue: back to the first stream");
}

void test_stream_queue_stop() {
  StreamQueue q(2, 1);
  q.push(makeFrame(1, 7));
  q.stop();
  auto left = q.pop();
  test::assert_true(left && left->frameId == 7,
                    "StreamQueue: frames queued before stop() are drained");
  test::assert_true(!q.pop().has_value(),
                    "StreamQueue: pop() returns nullopt once stopped and empty");

  StreamQueue blocked(1, 1);
  std::atomic<bool> returned{false};
  std::thread consumer([&] {
    blocked.pop();
    returned = true;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  blocked.stop();
  consumer.join();
  test::assert_true(returned, "StreamQueue: stop() unblocks a waiting pop()");
}

int main() {
  std::cout << "========================================\n";
  std::cout << "  Real-Time Video Analytics Pipeline Tests\n";
  std::cout << "========================================\n";

  // Suite 1: Basic Operations
  test_bounded_queue_single_push_pop();
  test_bounded_queue_fifo_order();
  test_bounded_queue_size_tracking();

  // Suite 2: Overflow Behavior
  test_bounded_queue_overflow_drops_oldest();
  test_bounded_queue_multiple_overflows();
  test_bounded_queue_max_size_one();

  // Suite 3: Stop Signal
  test_bounded_queue_stop_unblocks_pop();
  test_bounded_queue_stop_with_items();
  test_bounded_queue_is_stopped();

  // Suite 4: Multi-Threaded
  test_bounded_queue_producer_consumer();
  test_bounded_queue_concurrent_push_pop();
  test_bounded_queue_overflow_under_load();

  // Suite 5: Data Structures
  test_frame_data_creation();
  test_analytics_result_creation();
  test_bounded_queue_with_frame_data();
  test_bounded_queue_with_analytics_result();

  // Suite 6: StreamQueue
  test_stream_queue_is_fair();
  test_stream_queue_stop();

  // Summary
  std::cout << "\n========================================\n";
  std::cout << "  Test Summary\n";
  std::cout << "========================================\n";
  std::cout << "Total:  " << test::stats.total << "\n";
  std::cout << "Passed: " << test::stats.passed << "\n";
  std::cout << "Failed: " << test::stats.failed << "\n";
  std::cout << "Rate:   " << test::stats.passRate() << "%\n";

  if (test::stats.failed == 0) {
    std::cout << "\nAll tests passed!\n";
    return 0;
  } else {
    std::cout << "\nSome tests failed!\n";
    return 1;
  }
}

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include <httplib.h>

#include "server/sse-server.hpp"

// Test utilities
namespace test {

struct TestStats {
  int total = 0;
  int passed = 0;
  int failed = 0;
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

}  // namespace test

const int kPort = 19001;

// Test 1: one broadcast reaches a connected client exactly once
void testBroadcastDeliveredOnce() {
  std::cout << "\n=== Test 1: Broadcast Delivered Once ===\n";

  SseServerConfig config;
  config.port = kPort;
  SseServer server(config);
  server.start();

  server.broadcast("{\"n\":1}");

  std::string body;
  httplib::Client client("127.0.0.1", kPort);
  client.set_read_timeout(0, 500000);  // stream never ends; stop after 500 ms
  client.Get("/events", [&](const char* data, size_t len) {
    body.append(data, len);
    return true;
  });

  test::assert_true(body == "data: {\"n\":1}\n\n",
                    "client receives the event exactly once");

  server.stop();
}

// Test 2: abandoned SSE connections release their worker thread.
// Each /events connection used to pin an httplib worker forever, so the server
// stopped answering once the pool (max(8, cores - 1) threads) was exhausted.
void testAbandonedClientsReleaseThreads() {
  std::cout << "\n=== Test 2: Abandoned SSE Clients Release Threads ===\n";

  SseServerConfig config;
  config.port = kPort + 1;
  SseServer server(config);
  server.start();

  server.broadcast("{\"n\":1}");

  const int clients =
      static_cast<int>(std::max(8u, std::thread::hardware_concurrency())) + 8;
  int connected = 0;
  for (int i = 0; i < clients; i++) {
    httplib::Client client("127.0.0.1", config.port);
    client.set_read_timeout(2);
    // Returning false drops the connection after the first chunk
    client.Get("/events", [&](const char*, size_t) {
      connected++;
      return false;
    });
  }
  test::assert_true(connected == clients,
                    "all " + std::to_string(clients) + " SSE clients served");

  httplib::Client client("127.0.0.1", config.port);
  client.set_read_timeout(2);
  auto res = client.Get("/health");
  test::assert_true(res && res->status == 200,
                    "/health still answers after clients disconnect");

  server.stop();
}

// Test 3: a busy port is reported, and stop() really closes the port
void testBindFailureAndStop() {
  std::cout << "\n=== Test 3: Bind Failure And Stop ===\n";

  SseServerConfig config;
  config.port = kPort + 2;
  SseServer first(config);
  first.start();
  test::assert_true(first.isRunning(), "server is running after start()");

  SseServer second(config);
  second.start();
  test::assert_true(!second.isRunning(),
                    "second server on a busy port does not start");

  // A connected client must not keep stop() waiting
  std::thread viewer([&] {
    httplib::Client client("127.0.0.1", config.port);
    client.Get("/events", [](const char*, size_t) { return true; });
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  auto begin = std::chrono::steady_clock::now();
  first.stop();
  double stopSec = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - begin)
                       .count();
  viewer.join();
  test::assert_true(stopSec < 2.0, "stop() returns with a client connected");

  httplib::Client client("127.0.0.1", config.port);
  client.set_connection_timeout(1);
  test::assert_true(!client.Get("/health"), "port is closed after stop()");
}

int main() {
  std::cout << "==================================================\n";
  std::cout << "SSE Server Test Suite\n";
  std::cout << "==================================================\n";

  testBroadcastDeliveredOnce();
  testAbandonedClientsReleaseThreads();
  testBindFailureAndStop();

  std::cout << "\n==================================================\n";
  std::cout << "Test Summary\n";
  std::cout << "==================================================\n";
  std::cout << "Total assertions: " << test::stats.total << "\n";
  std::cout << "Passed: " << test::stats.passed << "\n";
  std::cout << "Failed: " << test::stats.failed << "\n";
  std::cout << "==================================================\n";

  return test::stats.failed == 0 ? 0 : 1;
}

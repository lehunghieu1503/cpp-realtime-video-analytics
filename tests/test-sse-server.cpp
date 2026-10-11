#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
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

// Test 4: a /video viewer receives JPEG frames pushed after it connected
void testVideoStreamDeliversFrames() {
  std::cout << "\n=== Test 4: Video Stream Delivers Frames ===\n";

  SseServerConfig config;
  config.port = kPort + 3;
  SseServer server(config);
  server.start();

  std::atomic<bool> feeding{true};
  std::thread feeder([&] {
    cv::Mat frame(48, 64, CV_8UC3, cv::Scalar(0, 128, 255));
    while (feeding) {
      server.updateFrame(frame);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  });

  std::string body;
  httplib::Client client("127.0.0.1", config.port);
  client.set_read_timeout(2);
  client.Get("/video", [&](const char* data, size_t len) {
    body.append(data, len);
    // JPEG start-of-image marker: stop once a frame has arrived
    return body.find("\xFF\xD8") == std::string::npos;
  });

  feeding = false;
  feeder.join();
  test::assert_true(body.find("Content-Type: image/jpeg") != std::string::npos &&
                        body.find("\xFF\xD8") != std::string::npos,
                    "viewer receives a JPEG frame");

  server.stop();
}

// Test 5: with a password set, every route but /health needs credentials
void testBasicAuth() {
  std::cout << "\n=== Test 5: Basic Auth ===\n";

  SseServerConfig config;
  config.port = kPort + 4;
  config.username = "operator";
  config.password = "s3cret";
  config.metrics = [] { return std::string("va_test 1\n"); };
  SseServer server(config);
  server.start();

  httplib::Client anonymous("127.0.0.1", config.port);
  auto denied = anonymous.Get("/metrics");
  test::assert_true(denied && denied->status == 401,
                    "request without credentials gets 401");
  test::assert_true(denied && denied->has_header("WWW-Authenticate"),
                    "401 asks the browser for Basic credentials");
  auto health = anonymous.Get("/health");
  test::assert_true(health && health->status == 200,
                    "/health stays open for probes");

  httplib::Client wrong("127.0.0.1", config.port);
  wrong.set_basic_auth("operator", "guess");
  auto rejected = wrong.Get("/streams");
  test::assert_true(rejected && rejected->status == 401,
                    "wrong password gets 401");

  httplib::Client allowed("127.0.0.1", config.port);
  allowed.set_basic_auth("operator", "s3cret");
  auto ok = allowed.Get("/metrics");
  test::assert_true(ok && ok->status == 200 && ok->body == "va_test 1\n",
                    "right credentials reach the route");

  server.stop();
}

// Test 6: with a certificate and key the server speaks HTTPS only
void testTls() {
  std::cout << "\n=== Test 6: TLS ===\n";

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  const std::string cert = "/tmp/va-test-cert.pem";
  const std::string key = "/tmp/va-test-key.pem";
  std::string command =
      "openssl req -x509 -newkey rsa:2048 -nodes -days 1 -subj /CN=localhost "
      "-keyout " + key + " -out " + cert + " >/dev/null 2>&1";
  if (std::system(command.c_str()) != 0) {
    std::cout << "  [SKIP] openssl command not available\n";
    return;
  }

  SseServerConfig config;
  config.port = kPort + 5;
  config.tlsCertFile = cert;
  config.tlsKeyFile = key;
  SseServer server(config);
  server.start();
  test::assert_true(server.isRunning(), "server starts with a certificate");

  httplib::SSLClient secure("127.0.0.1", config.port);
  secure.enable_server_certificate_verification(false);  // self-signed
  auto ok = secure.Get("/health");
  test::assert_true(ok && ok->status == 200, "HTTPS request succeeds");

  httplib::Client plain("127.0.0.1", config.port);
  plain.set_read_timeout(2);
  test::assert_true(!plain.Get("/health"), "plain HTTP request is refused");
  server.stop();

  SseServerConfig broken;
  broken.port = kPort + 6;
  broken.tlsCertFile = "/nonexistent/cert.pem";
  broken.tlsKeyFile = "/nonexistent/key.pem";
  SseServer unusable(broken);
  unusable.start();
  test::assert_true(!unusable.isRunning(),
                    "a certificate that cannot be read stops the start");
#else
  std::cout << "  [SKIP] built without OpenSSL\n";
#endif
}

int main() {
  std::cout << "==================================================\n";
  std::cout << "SSE Server Test Suite\n";
  std::cout << "==================================================\n";

  testBroadcastDeliveredOnce();
  testAbandonedClientsReleaseThreads();
  testBindFailureAndStop();
  testVideoStreamDeliversFrames();
  testBasicAuth();
  testTls();

  std::cout << "\n==================================================\n";
  std::cout << "Test Summary\n";
  std::cout << "==================================================\n";
  std::cout << "Total assertions: " << test::stats.total << "\n";
  std::cout << "Passed: " << test::stats.passed << "\n";
  std::cout << "Failed: " << test::stats.failed << "\n";
  std::cout << "==================================================\n";

  return test::stats.failed == 0 ? 0 : 1;
}

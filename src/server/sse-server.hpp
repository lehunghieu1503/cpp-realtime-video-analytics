#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <string>
#include <thread>
#include <vector>

namespace httplib {
class Server;
}

// SSE + MJPEG server for streaming detection results and video to browsers.
// Main thread calls broadcast() and updateFrame() — no queue dependency.
struct SseServerConfig {
  int port = 9001;
  std::string bindAddress = "127.0.0.1";
  std::string staticDir = "web/";
};

class SseServer {
public:
  explicit SseServer(const SseServerConfig& config);
  ~SseServer();

  // Start HTTP server (SSE + MJPEG + static files) in background thread.
  // isRunning() stays false if the address cannot be bound.
  void start();
  // Close the port and wait for the HTTP thread and its handlers to finish
  void stop();

  // Called by main thread to push data to connected clients
  void broadcast(const std::string& json);    // → SSE /events
  void updateFrame(const cv::Mat& frame);     // → MJPEG /video

  bool isRunning() const { return running_; }

private:
  SseServerConfig config_;
  std::unique_ptr<httplib::Server> server_;
  std::thread httpThread_;
  std::atomic<bool> running_{false};

  // Latest SSE event; each /events handler sends it when the sequence advances
  std::mutex eventMutex_;
  std::condition_variable eventReady_;
  std::string latestEvent_;
  uint64_t eventSeq_ = 0;

  // Latest JPEG frame for MJPEG streaming
  std::mutex frameMutex_;
  std::condition_variable frameReady_;
  std::vector<unsigned char> jpegBuffer_;
  uint64_t frameSeq_ = 0;
};

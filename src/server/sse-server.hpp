#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
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
  int numStreams = 1;
  // Body of GET /streams: what the dashboard needs to lay itself out
  std::string streamsJson = "[]";
  // Body of GET /metrics in the Prometheus text format; unset: 404
  std::function<std::string()> metrics;
  // GET /health answers 200 while this returns true, 503 otherwise;
  // unset: always 200
  std::function<bool()> healthy;
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

  // Called by main thread to push data to connected clients. `stream` must
  // be below SseServerConfig::numStreams.
  void broadcast(const std::string& json, size_t stream = 0);  // → SSE /events
  // → MJPEG /video?stream=N
  void updateFrame(const cv::Mat& frame, size_t stream = 0);
  // True while a browser watches that stream; lets the caller skip drawing
  bool hasViewers(size_t stream = 0) const;

  bool isRunning() const { return running_; }

private:
  SseServerConfig config_;
  std::unique_ptr<httplib::Server> server_;
  std::thread httpThread_;
  std::atomic<bool> running_{false};

  // Latest SSE event of each stream; an /events handler sends the ones whose
  // sequence number is past what it sent last
  struct EventSlot {
    std::string json;
    uint64_t seq = 0;
  };
  std::mutex eventMutex_;
  std::condition_variable eventReady_;
  std::vector<EventSlot> events_;
  uint64_t eventSeq_ = 0;

  // Latest JPEG frame of each stream for MJPEG streaming
  struct VideoSlot {
    std::mutex mutex;
    std::condition_variable ready;
    std::vector<unsigned char> jpeg;
    uint64_t seq = 0;
    // Connected /video viewers; updateFrame() skips the JPEG encode at zero
    std::atomic<int> clients{0};
  };
  std::deque<VideoSlot> videos_;  // deque: the slots cannot be moved
};

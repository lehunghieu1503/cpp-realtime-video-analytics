#include "sse-server.hpp"

#include <httplib.h>
#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <iostream>
#include <sstream>

// ── Helpers ─────────────────────────────────────────────────────────────────
static bool endsWith(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() &&
         str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::string contentType(const std::string& path) {
  if (endsWith(path, ".html")) return "text/html";
  if (endsWith(path, ".js")) return "application/javascript";
  if (endsWith(path, ".css")) return "text/css";
  if (endsWith(path, ".json")) return "application/json";
  if (endsWith(path, ".png")) return "image/png";
  return "text/plain";
}

static std::string readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return "";
  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

// ── SseServer implementation ────────────────────────────────────────────────
SseServer::SseServer(const SseServerConfig& config) : config_(config) {}

SseServer::~SseServer() {
  if (running_) stop();
}

void SseServer::broadcast(const std::string& json) {
  {
    std::lock_guard<std::mutex> lock(eventMutex_);
    latestEvent_ = json;
    eventSeq_++;
  }
  eventReady_.notify_all();
}

void SseServer::updateFrame(const cv::Mat& frame) {
  std::vector<unsigned char> buf;
  cv::imencode(".jpg", frame, buf, {cv::IMWRITE_JPEG_QUALITY, 70});

  {
    std::lock_guard<std::mutex> lock(frameMutex_);
    jpegBuffer_ = std::move(buf);
    frameSeq_++;
  }
  frameReady_.notify_all();
}

void SseServer::start() {
  if (running_) return;

  server_ = std::make_unique<httplib::Server>();
  // httplib defaults to SO_REUSEPORT, which lets a second instance bind the
  // same port silently. SO_REUSEADDR still allows a quick restart.
  server_->set_socket_options([](socket_t sock) {
    int yes = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  });
  if (!server_->bind_to_port(config_.bindAddress, config_.port)) {
    std::cerr << "SSE server: cannot bind " << config_.bindAddress << ":"
              << config_.port << "\n";
    server_.reset();
    return;
  }
  std::cout << "SSE server listening on http://" << config_.bindAddress << ":"
            << config_.port << "\n";

  running_ = true;
  httpThread_ = std::thread([this]() {
    httplib::Server& svr = *server_;

    // Health check
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
      res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // SSE endpoint. The provider waits briefly for a new event, writes it and
    // returns, so httplib re-checks the socket and frees the worker thread
    // once the client is gone.
    svr.Get("/events", [this](const httplib::Request&, httplib::Response& res) {
      res.set_header("Cache-Control", "no-cache");
      res.set_chunked_content_provider(
          "text/event-stream",
          [this, lastSeq = uint64_t{0}](size_t,
                                         httplib::DataSink& sink) mutable {
            std::string message;
            {
              std::unique_lock<std::mutex> lock(eventMutex_);
              eventReady_.wait_for(lock, std::chrono::milliseconds(100),
                                   [this, lastSeq] {
                                     return eventSeq_ > lastSeq || !running_;
                                   });
              if (!running_) return false;
              if (eventSeq_ == lastSeq) return true;
              lastSeq = eventSeq_;
              message = "data: " + latestEvent_ + "\n\n";
            }
            return sink.write(message.data(), message.size());
          });
    });

    // MJPEG video stream endpoint
    svr.Get("/video", [this](const httplib::Request&, httplib::Response& res) {
      res.set_header("Cache-Control", "no-cache");
      res.set_chunked_content_provider(
          "multipart/x-mixed-replace; boundary=frame",
          [this](size_t, httplib::DataSink& sink) {
            uint64_t lastSeq = 0;
            while (running_) {
              // Wait for new frame
              std::unique_lock<std::mutex> lock(frameMutex_);
              frameReady_.wait_for(lock, std::chrono::milliseconds(100),
                                    [this, lastSeq] {
                                      return frameSeq_ > lastSeq || !running_;
                                    });

              if (!running_) break;
              if (frameSeq_ == lastSeq) continue;

              lastSeq = frameSeq_;
              // Copy JPEG buffer while holding lock
              std::vector<unsigned char> jpeg = jpegBuffer_;
              lock.unlock();

              // Write MJPEG frame boundary
              std::string header = "--frame\r\nContent-Type: image/jpeg\r\n"
                                   "Content-Length: " +
                                   std::to_string(jpeg.size()) + "\r\n\r\n";

              if (!sink.write(header.data(), header.size())) return false;
              if (!sink.write(reinterpret_cast<const char*>(jpeg.data()),
                              jpeg.size()))
                return false;
              if (!sink.write("\r\n", 2)) return false;
            }
            return false;
          },
          [](bool) {});
    });

    // Serve index.html at root
    svr.Get("/", [this](const httplib::Request&, httplib::Response& res) {
      auto content = readFile(config_.staticDir + "index.html");
      if (content.empty()) {
        res.status = 404;
        res.set_content("Not found", "text/plain");
      } else {
        res.set_content(content, "text/html");
      }
    });

    // Static files
    svr.Get("/static/(.*)",
            [this](const httplib::Request& req, httplib::Response& res) {
              std::string path = req.matches[1];
              if (path.find("..") != std::string::npos) {
                res.status = 403;
                res.set_content("Forbidden", "text/plain");
                return;
              }
              auto content = readFile(config_.staticDir + path);
              if (content.empty()) {
                res.status = 404;
                res.set_content("Not found", "text/plain");
              } else {
                res.set_content(content, contentType(path));
              }
            });

    svr.listen_after_bind();
  });
  // stop() only works once the listen loop is up
  server_->wait_until_ready();
}

void SseServer::stop() {
  if (!running_) return;
  running_ = false;
  frameReady_.notify_all();
  eventReady_.notify_all();

  // Closes the listening socket; handlers see !running_ and return, so the
  // join also guarantees nothing still uses this object.
  server_->stop();
  if (httpThread_.joinable()) httpThread_.join();
  server_.reset();
  std::cout << "SSE server stopped\n";
}

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
SseServer::SseServer(const SseServerConfig& config)
    : config_(config),
      events_(config.numStreams),
      videos_(config.numStreams) {}

SseServer::~SseServer() {
  if (running_) stop();
}

void SseServer::broadcast(const std::string& json, size_t stream) {
  {
    std::lock_guard<std::mutex> lock(eventMutex_);
    events_[stream].json = json;
    events_[stream].seq = ++eventSeq_;
  }
  eventReady_.notify_all();
}

bool SseServer::hasViewers(size_t stream) const {
  return videos_[stream].clients > 0;
}

void SseServer::updateFrame(const cv::Mat& frame, size_t stream) {
  VideoSlot& slot = videos_[stream];
  if (slot.clients == 0) return;

  std::vector<unsigned char> buf;
  cv::imencode(".jpg", frame, buf, {cv::IMWRITE_JPEG_QUALITY, 70});

  {
    std::lock_guard<std::mutex> lock(slot.mutex);
    slot.jpeg = std::move(buf);
    slot.seq++;
  }
  slot.ready.notify_all();
}

void SseServer::start() {
  if (running_) return;

  if (config_.tlsCertFile.empty() != config_.tlsKeyFile.empty()) {
    std::cerr << "SSE server: TLS needs both a certificate and a key\n";
    return;
  }
  bool tls = !config_.tlsCertFile.empty();
  if (tls) {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    server_ = std::make_unique<httplib::SSLServer>(
        config_.tlsCertFile.c_str(), config_.tlsKeyFile.c_str());
    if (!server_->is_valid()) {
      std::cerr << "SSE server: cannot load TLS certificate "
                << config_.tlsCertFile << " or key " << config_.tlsKeyFile
                << "\n";
      server_.reset();
      return;
    }
#else
    std::cerr << "SSE server: this build has no TLS support (libssl-dev was "
                 "missing at build time)\n";
    return;
#endif
  } else {
    server_ = std::make_unique<httplib::Server>();
  }

  if (!config_.password.empty()) {
    // /health stays open: a supervisor's probe carries no credentials and
    // the answer reveals nothing
    std::string expected =
        "Basic " + httplib::detail::base64_encode(config_.username + ":" +
                                                  config_.password);
    server_->set_pre_routing_handler(
        [expected](const httplib::Request& req, httplib::Response& res) {
          if (req.path == "/health") {
            return httplib::Server::HandlerResponse::Unhandled;
          }
          // Compare every byte, so the time taken does not tell an attacker
          // how much of a guess was right
          const std::string given = req.get_header_value("Authorization");
          unsigned char diff = given.size() == expected.size() ? 0 : 1;
          for (size_t i = 0; i < given.size() && i < expected.size(); i++) {
            diff |= static_cast<unsigned char>(given[i] ^ expected[i]);
          }
          if (diff == 0) return httplib::Server::HandlerResponse::Unhandled;
          res.status = 401;
          res.set_header("WWW-Authenticate",
                         "Basic realm=\"video-analytics\"");
          res.set_content("Unauthorized", "text/plain");
          return httplib::Server::HandlerResponse::Handled;
        });
  }
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
  std::cout << "SSE server listening on " << (tls ? "https" : "http") << "://"
            << config_.bindAddress << ":" << config_.port
            << (config_.password.empty() ? "" : " (password required)")
            << "\n";

  running_ = true;
  httpThread_ = std::thread([this]() {
    httplib::Server& svr = *server_;

    // Health check: 503 lets a supervisor see a stalled pipeline
    svr.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
      bool ok = !config_.healthy || config_.healthy();
      res.status = ok ? 200 : 503;
      res.set_content(ok ? "{\"status\":\"ok\"}"
                         : "{\"status\":\"unhealthy\"}",
                      "application/json");
    });

    // Prometheus scrape endpoint
    svr.Get("/metrics",
            [this](const httplib::Request&, httplib::Response& res) {
              if (!config_.metrics) {
                res.status = 404;
                return;
              }
              res.set_content(config_.metrics(),
                              "text/plain; version=0.0.4");
            });

    // Stream ids, zones and lines for the dashboard
    svr.Get("/streams",
            [this](const httplib::Request&, httplib::Response& res) {
              res.set_content(config_.streamsJson, "application/json");
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
              for (const auto& slot : events_) {
                if (slot.seq > lastSeq) message += "data: " + slot.json + "\n\n";
              }
              lastSeq = eventSeq_;
            }
            return sink.write(message.data(), message.size());
          });
    });

    // MJPEG video stream endpoint: /video?stream=N (default 0)
    svr.Get("/video", [this](const httplib::Request& req,
                             httplib::Response& res) {
      size_t stream = 0;
      if (req.has_param("stream")) {
        try {
          stream = std::stoul(req.get_param_value("stream"));
        } catch (const std::exception&) {
          stream = videos_.size();  // rejected below
        }
      }
      if (stream >= videos_.size()) {
        res.status = 404;
        res.set_content("No such stream", "text/plain");
        return;
      }
      VideoSlot& slot = videos_[stream];
      res.set_header("Cache-Control", "no-cache");
      slot.clients++;
      res.set_chunked_content_provider(
          "multipart/x-mixed-replace; boundary=frame",
          [this, &slot](size_t, httplib::DataSink& sink) {
            // Start after the buffered frame: with no viewer it is not
            // refreshed, so it can be arbitrarily old
            uint64_t lastSeq;
            {
              std::lock_guard<std::mutex> lock(slot.mutex);
              lastSeq = slot.seq;
            }
            while (running_) {
              // Wait for new frame
              std::unique_lock<std::mutex> lock(slot.mutex);
              slot.ready.wait_for(lock, std::chrono::milliseconds(100),
                                  [this, &slot, lastSeq] {
                                    return slot.seq > lastSeq || !running_;
                                  });

              if (!running_) break;
              if (slot.seq == lastSeq) continue;

              lastSeq = slot.seq;
              // Copy JPEG buffer while holding lock
              std::vector<unsigned char> jpeg = slot.jpeg;
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
          // Runs when the response is destroyed, however the stream ended
          [&slot](bool) { slot.clients--; });
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
  for (auto& slot : videos_) slot.ready.notify_all();
  eventReady_.notify_all();

  // Closes the listening socket; handlers see !running_ and return, so the
  // join also guarantees nothing still uses this object.
  server_->stop();
  if (httpThread_.joinable()) httpThread_.join();
  server_.reset();
  std::cout << "SSE server stopped\n";
}

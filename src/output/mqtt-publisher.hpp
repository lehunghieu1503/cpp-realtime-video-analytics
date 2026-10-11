#pragma once

#include <string>

struct mosquitto;

// Publishes messages to an MQTT broker. Connecting and reconnecting happen
// on a background thread, so publish() never blocks on the network; while
// the broker is away, QoS 1 messages are queued by the client library.
// Built without libmosquitto, the class exists but connect() fails.
class MqttPublisher {
public:
  MqttPublisher();
  ~MqttPublisher();
  MqttPublisher(const MqttPublisher&) = delete;
  MqttPublisher& operator=(const MqttPublisher&) = delete;

  // Returns false after printing the reason if the client cannot be set up.
  // An unreachable broker is not a failure: the client keeps retrying.
  // username empty: no credentials. caFile set: TLS, with the broker's
  // certificate verified against that CA.
  bool connect(const std::string& host, int port,
               const std::string& username = "",
               const std::string& password = "",
               const std::string& caFile = "");

  void publish(const std::string& topic, const std::string& payload);

private:
  mosquitto* client_ = nullptr;
};

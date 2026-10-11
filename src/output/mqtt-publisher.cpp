#include "mqtt-publisher.hpp"

#include <iostream>

#ifdef WITH_MQTT
#include <mosquitto.h>

MqttPublisher::MqttPublisher() { mosquitto_lib_init(); }

MqttPublisher::~MqttPublisher() {
  if (client_) {
    mosquitto_disconnect(client_);
    mosquitto_loop_stop(client_, false);
    mosquitto_destroy(client_);
  }
  mosquitto_lib_cleanup();
}

bool MqttPublisher::connect(const std::string& host, int port) {
  client_ = mosquitto_new(nullptr, true, nullptr);
  if (!client_) {
    std::cerr << "MQTT: cannot create the client\n";
    return false;
  }
  // Retry after 1 s, doubling up to 30 s
  mosquitto_reconnect_delay_set(client_, 1, 30, true);
  mosquitto_connect_callback_set(
      client_, [](mosquitto*, void*, int rc) {
        if (rc == 0) {
          std::cout << "MQTT: connected\n";
        } else {
          std::cerr << "MQTT: connection refused: "
                    << mosquitto_connack_string(rc) << "\n";
        }
      });
  mosquitto_disconnect_callback_set(client_, [](mosquitto*, void*, int rc) {
    if (rc != 0) std::cerr << "MQTT: connection lost, reconnecting\n";
  });

  // connect_async only records the address; the network thread started
  // next connects and keeps reconnecting
  int rc = mosquitto_connect_async(client_, host.c_str(), port, 30);
  if (rc == MOSQ_ERR_SUCCESS || rc == MOSQ_ERR_ERRNO || rc == MOSQ_ERR_EAI) {
    int loopRc = mosquitto_loop_start(client_);
    if (loopRc != MOSQ_ERR_SUCCESS) rc = loopRc;
  }
  // A broker that is down or a name that does not resolve yet is retried
  if (rc != MOSQ_ERR_SUCCESS && rc != MOSQ_ERR_ERRNO && rc != MOSQ_ERR_EAI) {
    std::cerr << "MQTT: " << mosquitto_strerror(rc) << "\n";
    return false;
  }
  std::cout << "MQTT: publishing to " << host << ":" << port << "\n";
  return true;
}

void MqttPublisher::publish(const std::string& topic,
                            const std::string& payload) {
  if (!client_) return;
  mosquitto_publish(client_, nullptr, topic.c_str(),
                    static_cast<int>(payload.size()), payload.data(),
                    /*qos=*/1, /*retain=*/false);
}

#else

MqttPublisher::MqttPublisher() = default;
MqttPublisher::~MqttPublisher() = default;

bool MqttPublisher::connect(const std::string&, int) {
  std::cerr << "MQTT: this build has no MQTT support (libmosquitto-dev was "
               "missing at build time)\n";
  return false;
}

void MqttPublisher::publish(const std::string&, const std::string&) {}

#endif

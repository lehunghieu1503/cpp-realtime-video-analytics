"""ROS 2 bridge for a video-analytics device.

Reads the device's /events stream (Server-Sent Events over HTTP or HTTPS)
and publishes:

  /video_analytics/<source>/detections   vision_msgs/Detection2DArray
      one message per frame; Detection2D.id carries the track id
  /video_analytics/events                std_msgs/String
      one JSON object per line_cross, zone_enter, zone_exit or dwell_time

The device keeps only its latest result per source: a slow reader misses
frames, and a reconnect receives the latest one again.

A frame of a simulation (it has a "simulation" block with the run id and the
frame's own time, sim_ns) is stamped with that time instead of the time it
was received, and is also published whole, as the JSON the device sent:

  /simulation/<run_id>/<source>/perception   std_msgs/String
  /simulation/<run_id>/<source>/lane         std_msgs/String, when it has a lane
      the same identity and image size, with the lane only

A consumer joins these with the simulator's own data by (run_id, source,
sim_ns), never by arrival order. A simulation frame that fails the checks in
envelope.py is dropped and counted, not repaired.

Parameters:
  url       base URL of the device (default http://localhost:9001)
  username  HTTP Basic user (default admin)
  password  HTTP Basic password; empty: no credentials
  ca_file   CA certificate to verify an https:// device against
"""
import base64
import json
import ssl
import threading
import time
import urllib.request

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from . import envelope


class Bridge(Node):
    def __init__(self):
        super().__init__("video_analytics_bridge")
        self.url = self.declare_parameter("url", "http://localhost:9001").value
        self.username = self.declare_parameter("username", "admin").value
        self.password = self.declare_parameter("password", "").value
        self.ca_file = self.declare_parameter("ca_file", "").value

        self.events = self.create_publisher(String, "video_analytics/events", 50)
        self.detections = {}  # source id -> publisher
        self.simulation = {}  # topic -> publisher
        self.rejected = 0
        self.running = True
        self.reader = threading.Thread(target=self.read_forever, daemon=True)
        self.reader.start()

    def open_stream(self):
        request = urllib.request.Request(self.url.rstrip("/") + "/events")
        if self.password:
            token = base64.b64encode(f"{self.username}:{self.password}".encode())
            request.add_header("Authorization", "Basic " + token.decode())
        context = ssl.create_default_context(cafile=self.ca_file) if self.ca_file else None
        return urllib.request.urlopen(request, timeout=10, context=context)

    def read_forever(self):
        while self.running and rclpy.ok():
            try:
                with self.open_stream() as stream:
                    self.get_logger().info(f"connected to {self.url}")
                    for line in stream:
                        if not self.running:
                            return
                        if line.startswith(b"data: "):
                            self.publish(json.loads(line[6:]))
            except Exception as error:  # the device restarting is routine
                self.get_logger().warning(f"{self.url}: {error}; retrying in 2 s")
            time.sleep(2)

    def publisher_for(self, source):
        if source not in self.detections:
            # Topic names allow only letters, digits and underscores
            name = envelope.topic_name(source)
            self.detections[source] = self.create_publisher(
                Detection2DArray, f"video_analytics/{name}/detections", 10
            )
        return self.detections[source]

    def string_publisher(self, topic):
        if topic not in self.simulation:
            self.simulation[topic] = self.create_publisher(String, topic, 10)
        return self.simulation[topic]

    def publish_simulation(self, frame):
        """The whole frame, and its lane, under the run's topics. False when
        the frame is not a usable simulation frame."""
        reason = envelope.check(frame)
        if reason:
            self.rejected += 1
            if self.rejected in (1, 10, 100) or self.rejected % 1000 == 0:
                self.get_logger().warning(
                    f"simulation frame dropped ({reason}); {self.rejected} so far")
            return False
        perception, lane = envelope.topics(frame)
        self.string_publisher(perception).publish(String(data=json.dumps(frame)))
        if "lane" in frame:
            only_lane = {k: frame[k] for k in ("simulation", "stream", "image", "lane")}
            self.string_publisher(lane).publish(String(data=json.dumps(only_lane)))
        return True

    def publish(self, frame):
        source = frame.get("stream", "cam0")
        message = Detection2DArray()
        if "simulation" in frame:
            if not self.publish_simulation(frame):
                return
            # The frame's own time: when it was received says nothing about
            # when it was captured
            sec, nanosec = envelope.stamp(frame["simulation"]["sim_ns"])
            message.header.stamp.sec = sec
            message.header.stamp.nanosec = nanosec
            message.header.frame_id = envelope.optical_frame(frame)
        else:
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = source
        for item in frame.get("detections", []):
            box = item["bbox"]
            detection = Detection2D()
            detection.header = message.header
            # The device reports the top-left corner; vision_msgs the centre
            detection.bbox.center.position.x = box["x"] + box["w"] / 2.0
            detection.bbox.center.position.y = box["y"] + box["h"] / 2.0
            detection.bbox.size_x = float(box["w"])
            detection.bbox.size_y = float(box["h"])
            if "track_id" in item:
                detection.id = str(item["track_id"])
            result = ObjectHypothesisWithPose()
            result.hypothesis.class_id = item["class"]
            result.hypothesis.score = float(item["confidence"])
            detection.results.append(result)
            message.detections.append(detection)
        self.publisher_for(source).publish(message)

        for event in frame.get("events", []):
            self.events.publish(String(data=json.dumps(event)))


def main():
    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

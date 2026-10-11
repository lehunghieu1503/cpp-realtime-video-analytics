"""Lane keeping and obstacle braking for the simulated vehicle.

Inputs, all of the same run:
  perception   the device's result for a camera frame: boxes, track ids, lane
  depth        the simulator's depth image of that same frame
  ego          the vehicle's own pose and velocity, every tick

Per perception frame, joined by (run_id, source, sim_ns):
  boxes + depth + ego pose at capture  ->  objects in metres (metric_tracker)

Per tick, with the ego pose of that tick:
  lane and objects are moved from the pose they were seen from to the
  current pose, then
    steering   pure pursuit on the lane's centreline
    speed      PI on the vehicle's own speed, at most 10 km/h
    brake      an object of a hazard class in the vehicle's path, closer than
               v * tau + v^2 / (2 * a_min) + margin

States:
  WAIT_DATA     brake. Leaves only on an explicit ARM for this run, and only
                when the gates pass (fresh frames with a valid lane, standing
                still, all hazard classes delivered). Nothing arms by itself.
  RUNNING       drives. Any doubt stops it: no fresh perception, no lane,
                no pose, an object in the path, an unreadable hazard.
  STOP_LATCHED  brake, until the episode is restarted with a new run id.

This module never reads another actor's true state or the map: it does not
import the simulator's API at all.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import yaml

from . import lane as lane_geometry
from . import protocol
from .calibration import Pose, Rig
from .metric_tracker import MetricTracker

WAIT_DATA, RUNNING, STOP_LATCHED = "WAIT_DATA", "RUNNING", "STOP_LATCHED"


def lane_config_from(section):
    config = dict(lane_geometry.DEFAULTS)
    for key in config:
        if key in (section or {}):
            config[key] = section[key]
    return config


def stopping_distance(speed, brake):
    """Distance covered from deciding to brake to standing still, plus the
    margin kept to the obstacle."""
    return (speed * brake["tau_s"] + speed * speed / (2.0 * brake["a_min_mps2"])
            + brake["margin_m"])


def pure_pursuit(target_x, target_y, wheelbase, rear_axle_x, max_steer_rad):
    """Normalised steer [-1, 1] to pass through a point of the ego frame.
    Positive steers right, as y is to the right."""
    dx = target_x - rear_axle_x
    distance = math.hypot(dx, target_y)
    if distance < 1e-6:
        return 0.0
    alpha = math.atan2(target_y, dx)
    wheel = math.atan2(2.0 * wheelbase * math.sin(alpha), distance)
    return max(-1.0, min(1.0, wheel / max_steer_rad))


class ControlCore:
    """The controller without ROS: fed with messages, asked for commands."""

    def __init__(self, config, vehicle, run_id, source="carla-front"):
        self.c = config["control"]
        self.rig = Rig.from_config(config["calibration"])
        self.lane_config = lane_config_from(config.get("lane"))
        self.vehicle = vehicle
        self.run_id = run_id
        self.source = source
        self.hazards = set(self.c.get("hazard_classes", protocol.HAZARD_CLASSES))
        # The device drops every class not listed in its `classes` key before
        # anything else sees it; a list without a hazard class hides it
        delivered = config.get("classes") or []
        self.missing_classes = sorted(self.hazards - set(delivered)) if delivered else []
        self.age_limit_ns = int(self.c["age_limit_ms"] * 1e6)
        # To arm, perception must be well inside the limit, not at its edge:
        # a run armed with 467 ms of 500 was stopped as stale one tick later
        self.arm_age_ns = int(self.age_limit_ns * self.c.get("arm_age_fraction", 0.6))
        self.tracker = MetricTracker(self.rig, config.get("metric"))
        self.depth = protocol.LiveCache(int(self.c.get("depth_cache_entries", 45)))
        self.ego = protocol.LiveCache(int(self.c.get("ego_cache_entries", 120)))
        self.state = WAIT_DATA
        self.stop_reason = ""
        self.basis = None
        self.fresh_frames = 0
        self.accepted = 0
        self.seq = 0
        self.integral = 0.0
        self.steer = 0.0
        self.rejected = {}
        self.last_step_ns = None
        self.debug = {}

    # ── Inputs ───────────────────────────────────────────────────────────────

    def on_ego(self, ego):
        if ego.get("run_id") != self.run_id:
            return "wrong_run"
        self.ego.put(int(ego["sim_ns"]), ego)
        return ""

    def on_depth(self, sim_ns, depth_m):
        self.depth.put(int(sim_ns), depth_m)

    def reject(self, reason):
        self.rejected[reason] = self.rejected.get(reason, 0) + 1
        # A frame that cannot be used breaks the run of fresh frames
        self.fresh_frames = 0
        return None, reason

    def on_perception(self, envelope, now_sim_ns=None):
        """Joins a device result with the depth and ego state of its frame.
        Returns (metric_perception, "") or (None, reason)."""
        simulation = envelope.get("simulation") or {}
        if simulation.get("run_id") != self.run_id:
            return self.reject("wrong_run")
        if envelope.get("stream") != self.source:
            return self.reject("wrong_source")
        sim_ns = simulation.get("sim_ns")
        if not isinstance(sim_ns, int) or isinstance(sim_ns, bool):
            return self.reject("bad_sim_ns")
        if self.basis is not None and sim_ns <= self.basis["sim_ns"]:
            # A repeat after a reconnect, or a slower worker's older frame
            self.rejected["not_newer"] = self.rejected.get("not_newer", 0) + 1
            return None, "not_newer"
        image = envelope.get("image") or {}
        if (image.get("width"), image.get("height")) != (self.rig.width, self.rig.height):
            return self.reject("image_size")
        lane = envelope.get("lane")
        if lane is None:
            return self.reject("no_lane_block")
        if lane.get("calibration_id") != self.rig.calibration_id:
            return self.reject("calibration_mismatch")
        ego = self.ego.get(sim_ns)
        if ego is None:
            return self.reject("no_ego_for_frame")
        depth = self.depth.get(sim_ns)
        if depth is None:
            return self.reject("no_depth_for_frame")
        capture = dict(ego, pose=Pose.from_dict(ego["pose"]))
        try:
            metric = self.tracker.update(envelope, depth, capture)
        except (KeyError, TypeError, ValueError):
            return self.reject("bad_envelope")
        if metric is None:
            return self.reject("not_newer")
        age_ok = now_sim_ns is None or now_sim_ns - sim_ns <= self.arm_age_ns
        self.fresh_frames = self.fresh_frames + 1 if (lane.get("valid") and age_ok) else 0
        self.accepted += 1
        self.basis = {"sim_ns": sim_ns, "lane": lane, "metric": metric,
                      "pose": capture["pose"], "detections": envelope.get("detections", [])}
        return metric, ""

    def gates(self, ego_now):
        """Why the run may not be armed yet; empty when it may."""
        reasons = []
        if self.missing_classes:
            reasons.append("classes_filter_drops:" + ",".join(self.missing_classes))
        if self.basis is None:
            reasons.append("no_perception")
        else:
            if ego_now["sim_ns"] - self.basis["sim_ns"] > self.arm_age_ns:
                reasons.append("stale_perception")
            if not self.basis["lane"].get("valid"):
                reasons.append("lane_invalid")
        if self.fresh_frames < int(self.c.get("ready_frames", 3)):
            reasons.append("fresh_frames_%d" % self.fresh_frames)
        if abs(ego_now.get("speed_mps", 0.0)) > 0.3:
            reasons.append("moving")
        return reasons

    def on_arm(self, message, ego_now):
        """An operator's ARM or stop for this run. Returns the reason it was
        refused, or empty."""
        if not isinstance(message, dict) or message.get("run_id") != self.run_id:
            return "wrong_run"
        if message.get("stop") is True:
            self.latch("operator_stop")
            return ""
        if message.get("enable") is not True:
            return "not_an_arm_message"
        if self.state != WAIT_DATA:
            return "state_" + self.state
        if ego_now is None:
            return "no_ego"
        reasons = self.gates(ego_now)
        if reasons:
            return "gates:" + ";".join(reasons)
        self.state = RUNNING
        self.integral = 0.0
        return ""

    def latch(self, reason):
        if self.state != STOP_LATCHED:
            self.state = STOP_LATCHED
            self.stop_reason = reason

    # ── One control tick ─────────────────────────────────────────────────────

    def command(self, ego_now, throttle, brake, steer, reason):
        self.seq += 1
        basis_ns = self.basis["sim_ns"] if self.basis else 0
        return {
            "run_id": self.run_id, "source": self.source, "command_seq": self.seq,
            "basis_sim_ns": basis_ns,
            "control_sim_ns": int(ego_now["sim_ns"]),
            "control_frame": int(ego_now["frame"]),
            "valid_until_sim_ns": basis_ns + self.age_limit_ns,
            "throttle": float(throttle), "brake": float(brake), "steer": float(steer),
            "reason": reason,
        }

    def obstacles_now(self, pose_now, lane_now, age_s, speed):
        """Every object of the basis frame in the current ego frame, with the
        decision taken on it. Returns (brake reason or "", list for the log)."""
        capture = self.basis["pose"]
        half = self.vehicle["width_m"] / 2.0 + self.c.get("corridor_margin_m", 0.3)
        bumper = self.vehicle["front_bumper_m"]
        envelope = stopping_distance(speed, self.c["brake"])
        centre = lane_now["centerline"]
        decisions = []
        brake_reason = ""

        def in_path(x, left, right):
            """True / False / None (the lane was not observed that far)."""
            if x > centre["x_max_m"]:
                return None
            if x >= centre["x_min_m"]:
                y = lane_geometry.evaluate(centre["coefficients"], x)
                low, high = y - half, y + half
            else:
                # Between the vehicle and the nearest lane point: the path
                # runs from straight ahead to that point
                y = lane_geometry.evaluate(centre["coefficients"], centre["x_min_m"])
                low, high = min(0.0, y) - half, max(0.0, y) + half
            return right >= low and left <= high

        for obs in self.basis["metric"]["observations"]:
            if obs.get("predicted_only"):
                continue
            record = {"class": obs["class"], "track_id": obs.get("track_id")}
            hazard = obs["class"] in self.hazards
            if obs["valid"]:
                x, (left, right) = obs["near_x_m"], obs["lateral_extent_m"]
                points = np.array([[x, left, 0.0], [x, right, 0.0]])
                world = capture.to_parent(points)
                moved = 0.0
                if obs.get("velocity_world_mps") is not None:
                    world[:, :2] += np.array(obs["velocity_world_mps"]) * age_s
                else:
                    # Velocity not known yet: it may be coming closer
                    moved = self.c.get("unknown_closing_mps", 2.0) * age_s
                now = pose_now.from_parent(world)
                near = float(now[:, 0].min()) - moved
                left, right = float(now[:, 1].min()), float(now[:, 1].max())
                gap = near - bumper
                path = in_path(near, left, right)
                record.update(gap_m=gap, left_m=left, right_m=right, in_path=path)
                closing = None
                if obs.get("velocity_relative_mps") is not None:
                    closing = -obs["velocity_relative_mps"][0]
                    record["closing_mps"] = closing
                decision = "clear"
                if path is None:
                    decision = "beyond_lane_support"
                    if gap <= envelope:
                        decision = "brake_uncertain_path"
                elif path:
                    if gap <= envelope:
                        decision = "brake_distance"
                    elif closing is not None and closing > 0.1 and \
                            gap / closing < self.c.get("ttc_min_s", 2.0):
                        decision = "brake_ttc"
                    else:
                        decision = "in_path_far"
                else:
                    decision = "outside_path"
            else:
                # No usable depth in the box. Where its bottom edge meets the
                # road says whether it could be in the way.
                box = obs["bbox"]
                bottom = box["y"] + box["h"]
                xy, ok = self.rig.uv_to_ground(box["x"] + box["w"] / 2.0,
                                               min(bottom, self.rig.height - 1))
                record["measurement"] = obs["reason"]
                if bottom >= self.rig.height - 2:
                    decision = "brake_unreadable_near"
                elif not ok:
                    decision = "unreadable_far"
                else:
                    now = pose_now.from_parent(
                        capture.to_parent([float(xy[0]), float(xy[1]), 0.0]))
                    gap = float(now[0]) - bumper
                    width = box["w"] * max(0.0, float(xy[0]) - self.rig.x) / self.rig.focal
                    path = in_path(float(now[0]), float(now[1]) - width / 2,
                                   float(now[1]) + width / 2)
                    record.update(gap_m=gap, in_path=path)
                    decision = ("brake_unreadable" if path is not False
                                and gap <= 2.0 * envelope else "unreadable_far")
            if decision.startswith("brake") and not hazard:
                # Logged, not braked for: the brake list is the six road users
                decision = "non_hazard_class"
            record["decision"] = decision
            decisions.append(record)
            if decision.startswith("brake") and not brake_reason:
                brake_reason = "obstacle:%s:%s" % (decision[6:], obs["class"])
        return brake_reason, decisions

    def step(self, ego_now):
        """The command for the tick that follows this ego state."""
        self.debug = {"state": self.state}
        dt = 1.0 / 30.0
        if self.last_step_ns is not None:
            dt = max(1e-3, (ego_now["sim_ns"] - self.last_step_ns) * 1e-9)
        self.last_step_ns = ego_now["sim_ns"]

        if self.state != RUNNING:
            reason = self.stop_reason if self.state == STOP_LATCHED else "wait_data"
            self.steer = 0.0
            return self.command(ego_now, 0.0, 1.0, 0.0, reason)

        def stop(reason):
            self.latch(reason)
            self.debug["state"] = self.state
            self.steer = 0.0
            return self.command(ego_now, 0.0, 1.0, 0.0, reason)

        age_ns = ego_now["sim_ns"] - self.basis["sim_ns"]
        self.debug["age_ms"] = age_ns / 1e6
        if age_ns > self.age_limit_ns:
            return stop("stale_perception")
        try:
            pose_now = Pose.from_dict(ego_now["pose"])
            speed = float(ego_now["speed_mps"])
        except (KeyError, TypeError, ValueError):
            return stop("bad_ego_state")
        if not math.isfinite(speed):
            return stop("bad_ego_state")

        # The lane as seen from where the vehicle is now
        lane_now = lane_geometry.rebase_lane(self.basis["lane"], self.basis["pose"],
                                             pose_now, self.rig.ground_z,
                                             self.lane_config)
        if not lane_now["valid"]:
            return stop("lane:" + (self.basis["lane"].get("reason") or lane_now["reason"]))
        centre = lane_now["centerline"]
        self.debug["lane"] = {"lateral_error_m": lane_now["lateral_error_m"],
                              "heading_rad": lane_now["heading_rad"],
                              "x_min_m": centre["x_min_m"], "x_max_m": centre["x_max_m"]}

        brake_reason, decisions = self.obstacles_now(pose_now, lane_now, age_ns * 1e-9,
                                                     speed)
        self.debug["obstacles"] = decisions
        self.debug["stopping_distance_m"] = stopping_distance(speed, self.c["brake"])
        if brake_reason:
            return stop(brake_reason)

        # Steering: a point on the centreline, inside what was observed
        pursuit = self.c["pure_pursuit"]
        lookahead = max(pursuit["lookahead_min_m"], pursuit["lookahead_gain_s"] * speed)
        if centre["x_min_m"] > lookahead + pursuit.get("slack_m", 6.0):
            return stop("no_lookahead_support")
        target_x = min(max(lookahead, centre["x_min_m"]), centre["x_max_m"])
        target_y = lane_geometry.evaluate(centre["coefficients"], target_x)
        wanted = self.c["steer"].get("gain", 1.0) * pure_pursuit(
            target_x, target_y, self.vehicle["wheelbase_m"],
            -self.vehicle["wheelbase_m"] / 2.0,
            math.radians(self.vehicle["max_steer_deg"]))
        wanted = max(-1.0, min(1.0, wanted))
        limit = self.c["steer"].get("rate_per_s", 2.0) * dt
        self.steer += max(-limit, min(limit, wanted - self.steer))
        self.debug["target"] = [target_x, target_y]

        # Speed: PI on the vehicle's own speed
        target_speed = min(self.c["target_speed_kmh"], 10.0) / 3.6
        error = target_speed - speed
        gains = self.c["speed"]
        # Only near the target: while the vehicle is still getting up to
        # speed the integral would wind up and carry it past 10 km/h
        if abs(error) < gains.get("integral_band_mps", 0.5):
            self.integral = max(-2.0, min(2.0, self.integral + error * dt))
        throttle = gains["kp"] * error + gains["ki"] * self.integral
        brake = 0.0
        if error < -0.5:
            # Too fast: no throttle, a light brake; never both
            throttle, brake = 0.0, min(1.0, 0.3 * -error)
        throttle = max(0.0, min(gains.get("max_throttle", 0.6), throttle))
        return self.command(ego_now, throttle, brake, self.steer, "")

    def status(self, ego_now):
        reasons = self.gates(ego_now) if self.state == WAIT_DATA else []
        return {"run_id": self.run_id, "state": self.state,
                "ready": self.state == WAIT_DATA and not reasons,
                "not_ready": reasons, "stop_reason": self.stop_reason,
                "perception_frames": self.accepted,
                "rejected": self.rejected}


# ── ROS 2 node ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="deploy/config/carla.yaml")
    parser.add_argument("--run-dir", required=True, help="directory of this run")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    import rclpy
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    with open(args.config) as f:
        config = yaml.safe_load(f)
    # The vehicle's dimensions are what the simulator measured for this run
    manifest_path = os.path.join(args.run_dir, "manifest.json")
    deadline = time.monotonic() + 120
    while not os.path.exists(manifest_path):
        if time.monotonic() > deadline:
            raise SystemExit("no manifest.json in %s" % args.run_dir)
        time.sleep(0.2)
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest["run_id"] != args.run_id:
        raise SystemExit("the run directory belongs to run %s" % manifest["run_id"])

    core = ControlCore(config, manifest["vehicle"], args.run_id, manifest["source"])
    names = protocol.topics(args.run_id, manifest["source"])
    logs = {name: open(os.path.join(args.run_dir, name + ".jsonl"), "x")
            for name in ("perception", "metric", "commands")}

    def log(name, record):
        logs[name].write(json.dumps(record, separators=(",", ":")) + "\n")

    rclpy.init()
    node = rclpy.create_node("video_analytics_controller")
    pub_control = node.create_publisher(String, names["control"], 10)
    pub_status = node.create_publisher(String, names["status"], 10)
    pub_metric = node.create_publisher(String, names["metric"], 10)
    latest = {"ego": None}

    def on_depth(message):
        if message.encoding != "32FC1":
            return
        sim_ns = message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec
        core.on_depth(sim_ns, np.frombuffer(message.data, dtype=np.float32).reshape(
            message.height, message.width))

    def on_perception(message):
        received = time.monotonic_ns()
        try:
            envelope = json.loads(message.data)
        except ValueError:
            core.reject("not_json")
            return
        now = latest["ego"]["sim_ns"] if latest["ego"] else None
        metric, reason = core.on_perception(envelope, now)
        log("perception", {"received_mono_ns": received, "rejected": reason,
                           "now_sim_ns": now, "envelope": envelope})
        if metric is not None:
            log("metric", metric)
            pub_metric.publish(String(data=json.dumps(metric)))

    def on_ego(message):
        try:
            ego = json.loads(message.data)
        except ValueError:
            return
        if core.on_ego(ego):
            return
        latest["ego"] = ego
        command = core.step(ego)
        pub_control.publish(String(data=json.dumps(command)))
        pub_status.publish(String(data=json.dumps(core.status(ego))))
        log("commands", {"issued_mono_ns": time.monotonic_ns(), "command": command,
                         "debug": core.debug})

    def on_arm(message):
        try:
            data = json.loads(message.data)
        except ValueError:
            return
        refused = core.on_arm(data, latest["ego"])
        log("commands", {"issued_mono_ns": time.monotonic_ns(), "arm": data,
                         "refused": refused, "state": core.state})
        node.get_logger().info("ARM %s: %s" % (
            "refused (%s)" % refused if refused else "accepted", core.state))

    depth_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE,
                           history=HistoryPolicy.KEEP_LAST)
    node.create_subscription(Image, names["depth"], on_depth, depth_qos)
    node.create_subscription(String, names["perception"], on_perception, 20)
    node.create_subscription(String, names["ego"], on_ego, 20)
    node.create_subscription(String, names["arm"], on_arm, 10)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for f in logs.values():
            f.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

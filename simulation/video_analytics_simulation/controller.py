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
    speed      PI on the vehicle's own speed, at most speed_cap_kmh; with
               follow_gap_m, not faster than a vehicle ahead in the lane
    brake      an object of a hazard class in the vehicle's path, closer than
               v * tau + v^2 / (2 * a_min) + margin

States:
  WAIT_DATA     brake. Leaves only on an explicit ARM for this run, and only
                when the gates pass (fresh frames with a valid lane, standing
                still, all hazard classes delivered). Nothing arms by itself.
  RUNNING       drives. Any doubt stops it: no fresh perception, no lane,
                no pose, an object in the path, an unreadable hazard.
  HOLDING       brake, for a reason that can pass: an object in the path, or
                a lane lost for a frame or two. Back to RUNNING once the lane
                is valid again and, after an object, clear_frames results in a
                row have shown the path clear as far as the vehicle would need
                to stop from its cruising speed. A prediction of a track that
                was not measured is not such a result.
  STOP_LATCHED  brake, until the episode is restarted with a new run id:
                perception too old, no usable pose, no lane ahead to steer on,
                or the lane lost for five results in a row.

What a scenario asks for comes from the run's manifest, never from the
device: its own target speed, follow_gap_m, one lane change
(lane_change: {at_tick, side}) and one park (park: {stall, handover}).
Without these keys the vehicle keeps the shared target speed and its lane.

A park (see parking.py for the geometry), backwards into a stall:
  approach  forwards along the aisle at 2 m/s at most, to the point one
            turning radius past the stall. Until the handover point a lost
            lane stops the vehicle as anywhere else; from it on the aisle is
            followed by the vehicle's own pose, since a car park has no lane
            markings. The stall must have been in the image with all four
            corners, and no road user measured inside it: otherwise the park
            is given up before the vehicle goes backwards. Braking for an
            obstacle here gives it up too.
  align     backwards, the wheel held at k of its lock, until the vehicle is
            within 12 degrees of the stall's direction
  enter     backwards and nearly straight, steering on the heading and the
            distance from the stall's axis, to the stall's centre
  parked    foot brake and parking brake, reverse off
The camera looks forward: once the vehicle goes backwards nothing sees the
stall. Reverse gear is changed only below 0.15 m/s.

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
from .parking import Park

WAIT_DATA, RUNNING, HOLDING, STOP_LATCHED = (
    "WAIT_DATA", "RUNNING", "HOLDING", "STOP_LATCHED")
# Results in a row without a valid lane before the stop is final
LANE_LATCH_FRAMES = 5


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

    def __init__(self, config, vehicle, run_id, source="carla-front", scenario=None):
        self.c = config["control"]
        scenario = scenario or {}
        self.target_speed_kmh = float(scenario.get("target_speed_kmh",
                                                   self.c["target_speed_kmh"]))
        self.follow_gap_m = scenario.get("follow_gap_m")
        change = scenario.get("lane_change")
        self.lane_change = dict(change, state="pending") if change else None
        self.park_config = self.c.get("park", {})
        self.park = Park(scenario["park"], vehicle, self.park_config) \
            if scenario.get("park") else None
        self.park_phase = "approach" if self.park else ""
        self.park_seen = False     # the stall was in the image, all of it
        self.park_travelled = 0.0  # backwards, in this phase
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
        self.clear_frames = int(self.c.get("clear_frames", 30))
        self.clear_needed = 0   # clear results still asked for before driving on
        self.clear_count = 0
        self.clear_ns = None    # the last result counted as clear
        self.lane_bad = 0       # results in a row without a valid lane
        self.lane_bad_ns = None

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
        lane_ok = lane.get("valid") or self.on_aisle(capture["pose"])
        self.fresh_frames = self.fresh_frames + 1 if (lane_ok and age_ok) else 0
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
            if not self.basis["lane"].get("valid") and not self.on_aisle(self.basis["pose"]):
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

    def on_aisle(self, pose):
        """Past the handover point of a park: no lane is asked for."""
        return self.park is not None and self.park.handed_over(pose)

    def latch(self, reason):
        if self.state != STOP_LATCHED:
            self.state = STOP_LATCHED
            self.stop_reason = reason

    # ── One control tick ─────────────────────────────────────────────────────

    def command(self, ego_now, throttle, brake, steer, reason, reverse=False,
                hand_brake=False):
        self.seq += 1
        basis_ns = self.basis["sim_ns"] if self.basis else 0
        return {
            "run_id": self.run_id, "source": self.source, "command_seq": self.seq,
            "basis_sim_ns": basis_ns,
            "control_sim_ns": int(ego_now["sim_ns"]),
            "control_frame": int(ego_now["frame"]),
            "valid_until_sim_ns": basis_ns + self.age_limit_ns,
            "throttle": float(throttle), "brake": float(brake), "steer": float(steer),
            "reverse": bool(reverse), "hand_brake": bool(hand_brake),
            "reason": reason,
        }

    def obstacles_now(self, pose_now, centre, age_s, envelope):
        """Every object of the basis frame in the current ego frame, with the
        decision taken on it. centre is the path the vehicle follows, envelope
        the gap an object in it must keep. Returns (brake reason or "",
        whether a track that was only predicted is in the path, list for the
        log)."""
        capture = self.basis["pose"]
        half = self.vehicle["width_m"] / 2.0 + self.c.get("corridor_margin_m", 0.3)
        bumper = self.vehicle["front_bumper_m"]
        extrapolate = float(self.c.get("extrapolate_m", 20.0))
        decisions = []
        brake_reason = ""
        predicted_in_path = False

        def in_path(x, left, right):
            """True / False / None (the lane was not observed that far)."""
            if x > centre["x_max_m"]:
                # A lane that is nearly straight is continued a little, for
                # this decision only: the lane itself is reported as seen
                if x > centre["x_max_m"] + extrapolate \
                        or abs(centre["coefficients"][0]) >= 1e-3:
                    return None
            if x >= centre["x_min_m"]:
                y = lane_geometry.evaluate(centre["coefficients"], x)
                low, high = y - half, y + half
            else:
                # Between the vehicle and the nearest lane point: the path
                # runs from straight ahead to that point
                y = lane_geometry.evaluate(centre["coefficients"], centre["x_min_m"])
                low, high = min(0.0, y) - half, max(0.0, y) + half
            return bool(right >= low and left <= high)

        for obs in self.basis["metric"]["observations"]:
            predicted = bool(obs.get("predicted_only"))
            record = {"class": obs["class"], "track_id": obs.get("track_id")}
            if predicted:
                record["predicted"] = True
            hazard = obs["class"] in self.hazards
            if obs["valid"]:
                x, (left, right) = obs["near_x_m"], obs["lateral_extent_m"]
                points = np.array([[x, left, 0.0], [x, right, 0.0]])
                world = capture.to_parent(points)
                moved = 0.0
                if obs.get("velocity_world_mps") is not None:
                    world[:, :2] += np.array(obs["velocity_world_mps"]) * age_s
                else:
                    # Velocity not known (yet, or the track was not measured
                    # in this frame): it may be coming closer since it was
                    # last measured
                    since = age_s + (self.basis["sim_ns"]
                                     - obs["measurement_sim_ns"]) * 1e-9
                    moved = self.c.get("unknown_closing_mps", 2.0) * since
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
                if predicted and hazard and path is not False:
                    predicted_in_path = True
                    if decision.startswith("brake"):
                        decision = "brake_predicted"
            else:
                # No usable depth in the box. Where its bottom edge meets the
                # road says whether it could be in the way.
                box = obs["bbox"]
                bottom = box["y"] + box["h"]
                xy, ok = self.rig.uv_to_ground(box["x"] + box["w"] / 2.0,
                                               min(bottom, self.rig.height - 1))
                record["measurement"] = obs["reason"]
                if obs["reason"] == "only_road":
                    # The depth was read and nothing in the box stands above
                    # the road: a shadow or a marking taken for a road user
                    decision = "flat_on_road"
                elif bottom >= self.rig.height - 2:
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
        return brake_reason, predicted_in_path, decisions

    def step(self, ego_now):
        """The command for the tick that follows this ego state."""
        self.debug = {"state": self.state}
        dt = 1.0 / 30.0
        if self.last_step_ns is not None:
            dt = max(1e-3, (ego_now["sim_ns"] - self.last_step_ns) * 1e-9)
        self.last_step_ns = ego_now["sim_ns"]

        if self.state in (WAIT_DATA, STOP_LATCHED):
            reason = self.stop_reason if self.state == STOP_LATCHED else "wait_data"
            self.steer = 0.0
            return self.command(ego_now, 0.0, 1.0, 0.0, reason)

        def stop(reason):
            self.latch(reason)
            self.debug["state"] = self.state
            self.steer = 0.0
            return self.command(ego_now, 0.0, 1.0, 0.0, reason)

        def hold(reason):
            """Brake for a reason that can pass; the reason stays the one
            that is still true."""
            if aisle:
                # On the way to the stall there is no waiting and trying
                # again: the park is given up
                return stop("park_abort:" + reason)
            self.state = HOLDING
            self.stop_reason = reason
            self.debug.update(state=HOLDING, hold=True)
            self.steer = 0.0
            self.integral = 0.0
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

        park = self.park
        aisle = False
        if park:
            self.debug["park"] = self.park_phase
            if self.park_phase != "approach":
                return self.park_backwards(ego_now, pose_now, speed, dt, stop)
            aisle = park.handed_over(pose_now)
            given_up = self.watch_stall()
            if given_up:
                return stop(given_up)

        # The lane as seen from where the vehicle is now
        lane_now = lane_geometry.rebase_lane(self.basis["lane"], self.basis["pose"],
                                             pose_now, self.rig.ground_z,
                                             self.lane_config)
        basis_ns = self.basis["sim_ns"]
        # The path: the lane's centre, or, while the vehicle is between two
        # lanes, a line beside the marking it is crossing
        centre = None
        reason = ""
        change = self.lane_change
        if change and change["state"] == "active":
            centre = self.crossing_path(lane_now, pose_now, ego_now["sim_ns"], dt)
            if centre is None and change["state"] == "active":
                reason = "lane:marking_lost"
        if aisle:
            centre = park.track(pose_now)
        if centre is None and not reason:
            if lane_now["valid"]:
                centre = lane_now["centerline"]
            else:
                reason = "lane:" + (self.basis["lane"].get("reason") or lane_now["reason"])
        if centre is None:
            # Counted per result, not per tick: one result is used for
            # several ticks when the next one is late
            if basis_ns != self.lane_bad_ns:
                self.lane_bad += 1
                self.lane_bad_ns = basis_ns
            self.clear_count = 0
            if self.lane_bad >= LANE_LATCH_FRAMES:
                return stop(reason)
            return hold(reason)
        self.lane_bad = 0
        self.debug["lane"] = {"lateral_error_m": lane_now["lateral_error_m"],
                              "heading_rad": lane_now["heading_rad"],
                              "x_min_m": centre["x_min_m"], "x_max_m": centre["x_max_m"]}

        target_speed = min(self.target_speed_kmh,
                           self.c.get("speed_cap_kmh", 30.0)) / 3.6
        # Standing after a brake, the path must be clear as far as the vehicle
        # would need from the speed it goes back to: otherwise it would creep
        # up to an obstacle that has not moved
        envelope = stopping_distance(
            max(speed, target_speed) if self.state == HOLDING else speed,
            self.c["brake"])
        brake_reason, predicted_in_path, decisions = self.obstacles_now(
            pose_now, centre, age_ns * 1e-9, envelope)
        self.debug["obstacles"] = decisions
        self.debug["stopping_distance_m"] = envelope
        if brake_reason:
            self.clear_count = 0
            self.clear_needed = self.clear_frames
            return hold(brake_reason)
        if self.state == HOLDING:
            if predicted_in_path:
                self.clear_count = 0
            elif basis_ns != self.clear_ns:
                self.clear_count += 1
            self.clear_ns = basis_ns
            if self.clear_count < self.clear_needed:
                return hold(self.stop_reason)
            self.state = RUNNING
            self.stop_reason = ""
            self.clear_needed = self.clear_count = 0
            self.debug["state"] = RUNNING

        if aisle:
            remaining = park.remaining(pose_now)
            if remaining <= 0.05:
                # At the turning point: stand, then go backwards
                self.steer = 0.0
                if speed < 0.15:
                    if not self.park_seen:
                        return stop("park_abort:stall_unseen")
                    self.park_phase = "align"
                    self.park_travelled = 0.0
                    self.integral = 0.0  # of the speed loop: a new phase
                return self.command(ego_now, 0.0, 1.0, 0.0, "park:turning_point")
            target_speed = min(target_speed, self.park_config.get("approach_speed_mps", 2.0),
                               max(0.25, 0.5 * remaining))

        # Steering: a point on the centreline, inside what was observed
        pursuit = self.c["pure_pursuit"]
        lookahead = max(pursuit["lookahead_min_m"], pursuit["lookahead_gain_s"] * speed)
        if centre["x_min_m"] > lookahead + pursuit.get("slack_m", 6.0):
            return stop("no_lookahead_support")
        target_x = min(max(lookahead, centre["x_min_m"]), centre["x_max_m"])
        target_y = lane_geometry.evaluate(centre["coefficients"], target_x)
        if change:
            self.debug["lane_change"] = change["state"]
            if change["state"] == "pending":
                self.start_lane_change(ego_now.get("tick"), lane_now, pose_now,
                                       decisions, speed)
        wanted = self.c["steer"].get("gain", 1.0) * pure_pursuit(
            target_x, target_y, self.vehicle["wheelbase_m"],
            -self.vehicle["wheelbase_m"] / 2.0,
            math.radians(self.vehicle["max_steer_deg"]))
        wanted = max(-1.0, min(1.0, wanted))
        limit = self.c["steer"].get("rate_per_s", 2.0) * dt
        self.steer += max(-limit, min(limit, wanted - self.steer))
        self.debug["target"] = [target_x, target_y]

        # Speed: PI on the vehicle's own speed
        if self.follow_gap_m:
            target_speed = self.follow(target_speed, speed, envelope, decisions)
        if change and change["state"] == "active":
            # Between two lanes: not the moment to speed up
            target_speed = min(target_speed, max(change["speed_mps"], 2.0))
        error = target_speed - speed
        gains = self.c["speed"]
        # Only near the target: while the vehicle is still getting up to
        # speed the integral would wind up and carry it past the target.
        # "Near" and the integral's limit grow with the target: at 30 km/h
        # the proportional term alone settled 1.3 m/s short.
        band = max(gains.get("integral_band_mps", 0.5), 0.2 * target_speed)
        limit = max(2.0, 0.4 * target_speed)
        if abs(error) < band:
            self.integral = max(-limit, min(limit, self.integral + error * dt))
        throttle = gains["kp"] * error + gains["ki"] * self.integral
        brake = 0.0
        if error < -0.5:
            # Too fast: no throttle, a light brake; never both
            throttle, brake = 0.0, min(1.0, 0.3 * -error)
        throttle = max(0.0, min(gains.get("max_throttle", 0.6), throttle))
        return self.command(ego_now, throttle, brake, self.steer, "")

    def watch_stall(self):
        """On the approach: whether the stall has been seen whole, and
        whether something stands in it. Returns why the park is given up, or
        empty. No detection in the stall does not make it empty: it must
        have been in the image."""
        park = self.park
        if park.in_view(self.basis["pose"], self.rig):
            self.park_seen = True
        for obs in self.basis["metric"]["observations"]:
            if obs.get("predicted_only") or not obs["valid"] \
                    or obs["class"] not in self.hazards:
                continue
            if park.contains(obs["position_world_m"],
                             -self.park_config.get("occupied_margin_m", 0.3)):
                return "park_abort:stall_occupied"
        return ""

    def park_backwards(self, ego_now, pose_now, speed, dt, stop):
        """Align, enter, parked: by the vehicle's own pose alone."""
        park = self.park
        config = self.park_config
        heading = park.heading_error(pose_now.yaw)
        u, v = park.frame([pose_now.x, pose_now.y])
        self.debug.update(park=self.park_phase, park_u_m=u, park_v_m=v,
                          park_heading_rad=heading)
        if self.park_phase == "parked":
            # The state says so, not a reason: brake held, gear forward
            return self.command(ego_now, 0.0, 1.0, 0.0, "parked", hand_brake=True)

        self.park_travelled += speed * dt
        if self.park_phase == "align":
            if abs(heading) <= math.radians(config.get("align_done_deg", 12.0)):
                self.park_phase = "enter"
                self.park_travelled = 0.0
                self.integral = 0.0
            elif self.park_travelled > 1.3 * math.pi / 2.0 * park.radius:
                return stop("park_abort:turn_overrun")
            steer = park.turn_steer()
        if self.park_phase == "enter":
            inside = u < park.length / 2.0
            if (inside and abs(v) > park.width / 2.0 - park.inset) \
                    or abs(heading) > math.radians(15.0) \
                    or u < -(park.length / 2.0 - park.inset):
                return stop("park_abort:off_the_stall")
            if u <= config.get("stop_before_m", 0.05):
                if speed < 0.15:
                    self.park_phase = "parked"
                    self.debug["park"] = "parked"
                    return self.command(ego_now, 0.0, 1.0, 0.0, "parked",
                                        hand_brake=True)
                return self.command(ego_now, 0.0, 1.0, 0.0, "park:stopping",
                                    reverse=True)
            # Backwards the rear axle moves against the heading: to come
            # back to the axis from the right the vehicle must point right
            _, rear_v = park.frame(park.rear_axle(pose_now))
            wanted = max(-0.3, min(0.3, config.get("axis_gain_rad_per_m", 0.6) * rear_v))
            steer = max(-0.3, min(0.3, config.get("heading_gain", 2.5) * (heading - wanted)))
        limit = self.c["steer"].get("rate_per_s", 2.0) * dt
        self.steer += max(-limit, min(limit, steer - self.steer))

        # Speed: reverse_speed_mps is the most it may go; it aims at three
        # quarters of that, around the throttle at which the vehicle runs at
        # about that speed, and brakes lightly above the limit.
        limit = config.get("reverse_speed_mps", 0.5)
        error = 0.75 * limit - speed
        if speed > limit:
            throttle, brake = 0.0, min(1.0, 2.0 * (speed - limit) + 0.05)
        else:
            throttle = max(0.0, min(0.4, config.get("reverse_throttle", 0.22)
                                    + config.get("reverse_kp", 0.4) * error))
            brake = 0.0
        return self.command(ego_now, throttle, brake, self.steer, "", reverse=True)

    def follow(self, target_speed, speed, envelope, decisions):
        """The target speed behind a vehicle in the lane: its speed once the
        gap is down to 80 % of follow_gap_m, blended in from follow_gap_m.
        Inside the stopping distance it is not followed, it is braked for,
        before this is reached."""
        lead = None
        for d in decisions:
            if d.get("in_path") is True and d["class"] in self.hazards \
                    and "closing_mps" in d and not d.get("predicted") \
                    and envelope < d["gap_m"] < self.follow_gap_m \
                    and (lead is None or d["gap_m"] < lead["gap_m"]):
                lead = d
        if lead is None:
            return target_speed
        lead_speed = max(0.0, speed - lead["closing_mps"])
        near = 0.8 * self.follow_gap_m
        blend = min(1.0, max(0.0, (lead["gap_m"] - near) / (self.follow_gap_m - near)))
        followed = lead_speed + blend * max(0.0, target_speed - lead_speed)
        self.debug["follow"] = {"gap_m": lead["gap_m"], "lead_speed_mps": lead_speed,
                                "target_speed_mps": min(target_speed, followed)}
        return min(target_speed, followed)

    def markings_now(self, pose_now):
        """Every marking the device's lane pixels support, fitted on the road:
        [(as seen at capture, in the current ego frame)]."""
        out = []
        for fit in lane_geometry.boundaries(self.basis["lane"].get("pixels") or [],
                                            self.rig, self.lane_config):
            moved = self.marking_now(fit, self.basis["pose"], pose_now)
            if moved is not None:
                out.append((fit, moved))
        return out

    def marking_now(self, fit, capture, pose_now):
        return lane_geometry.rebase_boundary(fit, capture, pose_now, self.rig.ground_z,
                                             self.lane_config)[0]

    def marking_at(self, marking, x=None):
        x = self.lane_config["reference_x_m"] if x is None else x
        return lane_geometry.evaluate(
            marking["coefficients"], min(max(x, marking["x_min_m"]), marking["x_max_m"]))

    def give_up_lane_change(self, reason):
        self.lane_change.update(state="skipped", reason=reason)
        self.debug.update(lane_change="skipped", lane_change_skipped=reason)

    def start_lane_change(self, tick, lane_now, pose_now, decisions, speed):
        """The scenario's lane change begins at its tick, and only when the
        image shows the marking beyond the ego lane's own on that side: the
        width of the next lane is measured, never assumed. Without it, or
        with a road user in that lane, the vehicle stays where it is."""
        change = self.lane_change
        if not isinstance(tick, int) or tick < change["at_tick"]:
            return
        name = "left" if change["side"] == "left" else "right"
        side = -1.0 if name == "left" else 1.0
        y_inner = self.marking_at(lane_now[name])
        widths = [side * (self.marking_at(m) - y_inner)
                  for _, m in self.markings_now(pose_now)]
        widths = [w for w in widths if self.lane_config["width_min_m"] <= w
                  <= self.lane_config["width_max_m"]]
        reason = "" if widths else "no_marking"
        if widths:
            low, high = sorted((y_inner, y_inner + side * min(widths)))
            if any(d["class"] in self.hazards and "left_m" in d
                   and d["right_m"] >= low and d["left_m"] <= high
                   and d["gap_m"] < change.get("clear_m", 40.0) for d in decisions):
                reason = "lane_occupied"
        if not reason:
            # The marking to cross, as the device fitted it in this result
            change.update(state="active", width_m=min(widths), shift_m=0.0, ticks=0,
                          speed_mps=speed, marking=self.basis["lane"][name],
                          marking_pose=self.basis["pose"],
                          marking_ns=self.basis["sim_ns"])
            self.debug["lane_change"] = "active"
        elif tick >= change["at_tick"] + change.get("window_ticks", 30):
            self.give_up_lane_change(reason)

    def crossing_path(self, lane_now, pose_now, now_ns, dt):
        """The path during a lane change: beside the marking being crossed,
        from the centre of the lane the vehicle leaves to the centre of the
        one it enters. None when that marking is lost, or when the change is
        over.

        While the vehicle is over the marking the lane model is of no use:
        the device reports no lane, since the marking is on neither side,
        and what the model gives for it lags the marking by half a metre and
        then mixes it with its neighbour while it moves it to another slot.
        So the marking is carried: where the last result in which it was the
        ego lane's own boundary puts it now, by the vehicle's own motion, for
        coast_s at most. The change is done when the device reports a lane
        again whose boundary on the side the vehicle came from is where that
        puts the marking.
        """
        change = self.lane_change
        left = change["side"] == "left"
        side = -1.0 if left else 1.0
        marking = self.marking_now(change["marking"], change["marking_pose"], pose_now)
        if marking is None or now_ns - change["marking_ns"] > \
                change.get("coast_s", 2.0) * 1e9:
            return None
        y = self.marking_at(marking)
        if lane_now["valid"]:
            ahead, behind = ("left", "right") if left else ("right", "left")
            if abs(self.marking_at(lane_now[behind]) - y) < 0.4:
                change["state"] = "done"
                return None
            if abs(self.marking_at(lane_now[ahead]) - y) < 0.4 \
                    and self.basis["sim_ns"] > change["marking_ns"]:
                # Not over it yet: this result's own fit of it
                change.update(marking=self.basis["lane"][ahead],
                              marking_pose=self.basis["pose"],
                              marking_ns=self.basis["sim_ns"])
                marking = lane_now[ahead]
        change["ticks"] += 1
        if change["ticks"] > change.get("timeout_ticks", 450):
            self.give_up_lane_change("timeout")
            return None
        change["shift_m"] = min(change["width_m"], change["shift_m"]
                                + change.get("rate_mps", 1.0) * dt)
        a, b, c = marking["coefficients"]
        offset = side * (change["shift_m"] - change["width_m"] / 2.0)
        return {"coefficients": [a, b, c + offset], "x_min_m": marking["x_min_m"],
                "x_max_m": marking["x_max_m"]}

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
    parser.add_argument("--config", required=True, help="simulation/config/carla.yaml")
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

    core = ControlCore(config, manifest["vehicle"], args.run_id, manifest["source"],
                       manifest.get("scenario_config"))
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

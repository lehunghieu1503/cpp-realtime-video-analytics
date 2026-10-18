"""Detections in pixels to objects in metres: range from the depth image,
and position, velocity and acceleration from a Kalman filter per track.

The device tracks boxes in the image (ByteTrack) and gives each a track id.
Here, on the host, each box is joined with the depth image of the same
simulator frame and with the ego pose at that frame:

    box + depth   ->  the nearest surface inside the box, in the ego frame
    ego pose      ->  that point in the world frame
    Kalman filter ->  world position and velocity per track id

The filter runs in the world frame because the ego vehicle moves: in the ego
frame a parked car has a velocity whenever the ego vehicle drives or turns.

Nothing here reads another actor's true position or velocity. The ego pose
and velocity are the vehicle's own odometry.
"""

import math

import numpy as np

from .calibration import DEPTH_FAR_M

DEFAULTS = {
    "box_shrink": 0.15,          # ignored at each edge of the box: background
    "max_range_m": 150.0,
    "ground_clearance_m": 0.25,  # points lower than this are the road
    "cluster_gap_m": 0.8,        # a gap in range this wide starts a new surface
    "min_points": 12,            # depth pixels a surface needs
    "min_fraction": 0.08,        # ... and this share of the usable pixels
    "max_samples": 4000,         # depth pixels read per box
    # Kalman filter, constant velocity with white acceleration noise
    "q_accel": 4.0,              # (m/s^2)^2 s: how fast velocity may change
    "r_floor_m": 0.15,           # smallest standard deviation of a measurement
    "r_range_fraction": 0.01,    # ... growing with range
    "gate": 9.21,                # chi-square, 2 degrees of freedom, 99 %
    "max_rejects": 3,            # rejected updates in a row before a restart
    "max_gap_s": 0.5,            # a track not measured for longer is retired
    "velocity_min_updates": 3,
    "velocity_min_span_s": 0.1,
    "accel_smoothing": 0.3,
    "max_tracks": 256,
}


def measure_box(bbox, depth_m, rig, config=DEFAULTS):
    """The surface seen inside a detection box.

    bbox is {"x", "y", "w", "h"} in pixels of the source image, depth_m the
    depth image of the same frame. Returns a dict: valid, reason, and when
    valid position_ego_m (median of the surface), near_x_m (its near edge),
    lateral_extent_m [left, right] and sigma_m.
    """
    height, width = depth_m.shape
    if (width, height) != (rig.width, rig.height):
        return {"valid": False, "reason": "depth_size"}
    sx, sy = bbox["w"] * config["box_shrink"], bbox["h"] * config["box_shrink"]
    x0 = max(0, int(math.floor(bbox["x"] + sx)))
    x1 = min(width, int(math.ceil(bbox["x"] + bbox["w"] - sx)))
    y0 = max(0, int(math.floor(bbox["y"] + sy)))
    y1 = min(height, int(math.ceil(bbox["y"] + bbox["h"] - sy)))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return {"valid": False, "reason": "box_outside_image"}
    step = max(1, int(math.sqrt((x1 - x0) * (y1 - y0) / config["max_samples"])))
    v, u = np.mgrid[y0:y1:step, x0:x1:step]
    depth = depth_m[v, u].astype(np.float64)
    ok = np.isfinite(depth) & (depth > 0.1) & (depth < min(config["max_range_m"],
                                                             DEPTH_FAR_M * 0.99))
    if not ok.any():
        return {"valid": False, "reason": "no_depth"}
    # Pixel centres
    points = rig.depth_to_ego(u[ok] + 0.5, v[ok] + 0.5, depth[ok])
    points = points[points[:, 2] > rig.ground_z + config["ground_clearance_m"]]
    needed = max(config["min_points"], int(config["min_fraction"] * ok.sum()))
    # A handful of pixels (a distant pedestrian) is still a surface
    needed = min(needed, max(3, int(0.5 * ok.sum())))
    if len(points) < needed:
        return {"valid": False, "reason": "only_road"}

    # Surfaces along the range axis; the nearest one with enough pixels is
    # the object, anything behind it is seen through or around it
    points = points[np.argsort(points[:, 0])]
    breaks = np.flatnonzero(np.diff(points[:, 0]) > config["cluster_gap_m"]) + 1
    for cluster in np.split(points, breaks):
        if len(cluster) >= needed:
            break
    else:
        return {"valid": False, "reason": "insufficient_support"}
    median = np.median(cluster, axis=0)
    spread = np.std(cluster[:, :2], axis=0)
    return {
        "valid": True, "reason": "",
        "position_ego_m": [float(median[0]), float(median[1])],
        "near_x_m": float(np.percentile(cluster[:, 0], 10)),
        "lateral_extent_m": [float(np.percentile(cluster[:, 1], 2)),
                             float(np.percentile(cluster[:, 1], 98))],
        "sigma_m": [float(spread[0]), float(spread[1])],
        "points": int(len(cluster)),
    }


def _ground_hit(rig, u, v):
    """Ego (x, y) where the pixel meets the road, or None."""
    xy, ok = rig.uv_to_ground(float(u), float(v))
    if not bool(np.all(ok)):
        return None
    return float(xy[0]), float(xy[1])


def measure_ground(bbox, rig, config=None):
    """The bottom edge of a box on the flat road. No depth image.

    A forward camera's near edge is the smaller ego x. A camera looking
    back (yaw past 90 degrees) reports the same corners; the caller that
    brakes behind the vehicle uses the larger x, which is closer to the
    bumper. sigma_m is metres per pixel, from the neighbouring pixels.
    """
    del config  # the depth noise model does not apply to this measurement
    u0 = float(bbox["x"])
    width = float(bbox["w"])
    v = float(bbox["y"]) + float(bbox["h"])
    u1 = u0 + width
    looking_back = abs((rig.yaw_deg + 180.0) % 360.0 - 180.0) > 90.0
    bottom_cut = v >= rig.height - 2
    if bottom_cut and not looking_back:
        # The front camera's last rows meet the road on the bonnet.
        return {"valid": False, "reason": "truncated_bottom"}
    v_use = min(v, rig.height - 1.0)
    left = _ground_hit(rig, u0, v_use)
    right = _ground_hit(rig, u1, v_use)
    side_cut = u0 < 1.0 or u1 > rig.width - 1.0
    hits = [p for p in (left, right) if p is not None]
    if side_cut or len(hits) == 1:
        out = {"valid": False, "reason": "truncated_side"}
        if hits:
            out["position_ego_m"] = [hits[0][0], hits[0][1]]
            out["near_x_m"] = hits[0][0]
        return out
    if len(hits) < 2:
        return {"valid": False, "reason": "above_horizon"}
    centre_u = u0 + width / 2.0
    centre = _ground_hit(rig, centre_u, v_use)
    if centre is None:
        centre = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
    above = _ground_hit(rig, centre_u, v_use - 1.0)
    below = _ground_hit(rig, centre_u, v_use + 1.0)
    if above is not None and below is not None:
        sigma_x = abs(above[0] - below[0]) / 2.0
    elif above is not None:
        sigma_x = abs(above[0] - centre[0])
    else:
        sigma_x = 0.5
    west = _ground_hit(rig, centre_u - 1.0, v_use)
    east = _ground_hit(rig, centre_u + 1.0, v_use)
    sigma_y = abs(west[1] - east[1]) / 2.0 if west and east else sigma_x
    xs = (left[0], right[0])
    ys = (left[1], right[1])
    return {
        "valid": True, "reason": "",
        "position_ego_m": [centre[0], centre[1]],
        "near_x_m": float(min(xs)),
        "far_x_m": float(max(xs)),
        "lateral_extent_m": [float(min(ys)), float(max(ys))],
        "sigma_m": [float(max(sigma_x, 1e-3)), float(max(sigma_y, 1e-3))],
        "points": 2,
    }


class _Track:
    def __init__(self, z, r, sim_ns):
        self.state = np.array([z[0], z[1], 0.0, 0.0])
        # Velocity unknown: a large variance, not a measured zero
        self.P = np.diag([r, r, 100.0, 100.0])
        self.first_ns = self.last_ns = sim_ns
        self.updates = 1
        self.rejects = 0
        self.accel = None
        self.accel_updates = 0
        self.velocity_prev = None
        # Of the last measurement: what a prediction is still an obstacle as
        self.cls = None
        self.edge_world = None  # near edge, left and right end, world frame


def _predict(state, P, dt, q):
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    d3, d2 = dt ** 3 / 3, dt ** 2 / 2
    Q = q * np.array([[d3, 0, d2, 0], [0, d3, 0, d2],
                      [d2, 0, dt, 0], [0, d2, 0, dt]])
    return F @ state, F @ P @ F.T + Q


class MetricTracker:
    """Per (run, source, track id): a Kalman filter on world x, y."""

    def __init__(self, rig, config=None):
        self.rig = rig
        self.config = dict(DEFAULTS, **(config or {}))
        self.tracks = {}
        self.last_ns = {}  # (run, source) -> sim_ns of the last frame taken

    def reset(self):
        self.tracks.clear()
        self.last_ns.clear()

    def update(self, perception, ego):
        """One frame: the device's result and the ego state of the same
        simulator frame. Range is the bottom edge of each box on the road.
        Returns the metric_perception dict, or None for a frame that must
        not be used (a repeat, or older than one already taken)."""
        c = self.config
        sim = perception["simulation"]
        stream = perception.get("stream", "")
        key = (sim["run_id"], stream)
        sim_ns = int(sim["sim_ns"])
        if ego["run_id"] != sim["run_id"] or int(ego["sim_ns"]) != sim_ns:
            raise ValueError("ego state is not of the perception's frame")
        if key in self.last_ns and sim_ns <= self.last_ns[key]:
            return None
        self.last_ns[key] = sim_ns

        pose = ego["pose"]
        ego_velocity = np.array(ego["velocity_world_mps"][:2], dtype=float)
        yaw_rate = float(ego.get("yaw_rate_radps", 0.0))
        R = pose.rotation[:2, :2]  # ego to world, in the road plane

        observations = []
        seen = set()
        for detection in perception.get("detections", []):
            m = measure_ground(detection["bbox"], self.rig, c)
            track_id = detection.get("track_id")
            obs = {"class": detection.get("class"), "track_id": track_id,
                   "bbox": detection["bbox"], "valid": m["valid"],
                   "reason": m["reason"], "measurement_sim_ns": sim_ns,
                   "predicted_only": False, "kinematics_valid": False,
                   "acceleration_valid": False,
                   "velocity_world_mps": None, "velocity_relative_mps": None,
                   "acceleration_world_mps2": None, "covariance": None}
            if m["valid"]:
                obs.update({k: m[k] for k in ("position_ego_m", "near_x_m",
                                              "lateral_extent_m", "sigma_m",
                                              "points")})
                z = pose.to_parent(m["position_ego_m"] + [self.rig.ground_z])[:2]
                obs["position_world_m"] = [float(z[0]), float(z[1])]
                if track_id is not None:
                    self._filter((key[0], key[1], track_id), z, m, sim_ns, obs)
                    track = self.tracks[(key[0], key[1], track_id)]
                    track.cls = obs["class"]
                    track.edge_world = pose.to_parent(
                        [[m["near_x_m"], y, self.rig.ground_z]
                         for y in m["lateral_extent_m"]])
                    seen.add(track_id)
                    self._relative(obs, R, ego_velocity, yaw_rate)
            observations.append(obs)

        # Tracks not measured in this frame: predicted for a while, flagged,
        # then retired. A prediction never shows that the road is clear: it
        # keeps the class and the near edge of the last measurement, where
        # that was, so that one missed detection does not open the road.
        for track_key in [k for k in self.tracks if k[:2] == key]:
            track = self.tracks[track_key]
            if track_key[2] in seen:
                continue
            gap = (sim_ns - track.last_ns) * 1e-9
            if gap > c["max_gap_s"]:
                del self.tracks[track_key]
                continue
            state, P = _predict(track.state, track.P, gap, c["q_accel"])
            ego_xy = pose.from_parent([state[0], state[1], pose.z])[:2]
            edge = pose.from_parent(track.edge_world)
            obs = {"class": track.cls, "track_id": track_key[2], "valid": True,
                   "reason": "", "predicted_only": True,
                   "measurement_sim_ns": track.last_ns,
                   "position_ego_m": [float(ego_xy[0]), float(ego_xy[1])],
                   "near_x_m": float(edge[:, 0].min()),
                   "lateral_extent_m": [float(edge[:, 1].min()),
                                        float(edge[:, 1].max())],
                   "position_world_m": [float(state[0]), float(state[1])],
                   "kinematics_valid": False, "acceleration_valid": False,
                   "velocity_world_mps": None, "velocity_relative_mps": None,
                   "acceleration_world_mps2": None,
                   "covariance": [float(P[0, 0]), float(P[1, 1])]}
            observations.append(obs)
        while len(self.tracks) > c["max_tracks"]:
            del self.tracks[min(self.tracks, key=lambda k: self.tracks[k].last_ns)]

        return {"simulation": sim, "stream": stream,
                "image": perception.get("image"),
                "calibration_id": self.rig.calibration_id,
                "coordinate_frame": "ego at capture: x forward, y right, metres; "
                                    "world: dead reckoning after ARM, else the pose given",
                "pose_used": {"x": float(pose.x), "y": float(pose.y),
                              "z": float(pose.z), "roll": float(pose.roll),
                              "pitch": float(pose.pitch), "yaw": float(pose.yaw)},
                "observations": observations}

    def _filter(self, track_key, z, m, sim_ns, obs):
        c = self.config
        # Pixel size on the road. The depth floor (r_floor_m, 1 % of range)
        # is not applied: it described a depth cluster, not this measurement.
        sigma = max(m["sigma_m"])
        r = sigma * sigma
        track = self.tracks.get(track_key)
        dt = (sim_ns - track.last_ns) * 1e-9 if track else 0.0
        if track is None or dt <= 0 or dt > c["max_gap_s"]:
            # New id, or the same number used again after a gap: a new object
            self.tracks[track_key] = track = _Track(z, r, sim_ns)
        else:
            state, P = _predict(track.state, track.P, dt, c["q_accel"])
            innovation = z - state[:2]
            S = P[:2, :2] + np.eye(2) * r
            distance = float(innovation @ np.linalg.solve(S, innovation))
            if distance > c["gate"]:
                track.rejects += 1
                obs["reason"] = "gated"
                if track.rejects >= c["max_rejects"]:
                    self.tracks[track_key] = track = _Track(z, r, sim_ns)
                else:
                    track.state, track.P, track.last_ns = state, P, sim_ns
            else:
                K = P[:, :2] @ np.linalg.inv(S)
                track.state = state + K @ innovation
                # Joseph form: stays symmetric and positive
                A = np.eye(4) - K @ np.eye(2, 4)
                track.P = A @ P @ A.T + K @ (np.eye(2) * r) @ K.T
                track.rejects = 0
                track.updates += 1
                previous = track.velocity_prev
                velocity = track.state[2:].copy()
                span = (sim_ns - track.first_ns) * 1e-9
                if (track.updates >= c["velocity_min_updates"]
                        and span >= c["velocity_min_span_s"]):
                    if previous is not None:
                        raw = (velocity - previous) / dt
                        k = c["accel_smoothing"]
                        track.accel = raw if track.accel is None else \
                            (1 - k) * track.accel + k * raw
                        track.accel_updates += 1
                    track.velocity_prev = velocity
                track.last_ns = sim_ns
            if not (np.isfinite(track.state).all() and np.isfinite(track.P).all()):
                self.tracks[track_key] = track = _Track(z, r, sim_ns)

        span = (track.last_ns - track.first_ns) * 1e-9
        confirmed = (track.updates >= c["velocity_min_updates"]
                     and span >= c["velocity_min_span_s"])
        obs["covariance"] = [float(v) for v in np.diag(track.P)]
        obs["filtered_world_m"] = [float(track.state[0]), float(track.state[1])]
        obs["kinematics_valid"] = bool(confirmed)
        if confirmed:
            obs["velocity_world_mps"] = [float(track.state[2]), float(track.state[3])]
            # A difference of filtered velocities: an estimate with a warm-up,
            # not a measured acceleration
            if track.accel is not None and track.accel_updates >= 3:
                obs["acceleration_world_mps2"] = [float(track.accel[0]),
                                                  float(track.accel[1])]
                obs["acceleration_valid"] = True

    @staticmethod
    def _relative(obs, R, ego_velocity, yaw_rate):
        """Velocity of the object as seen from the moving, turning ego frame."""
        if obs["velocity_world_mps"] is None:
            return
        relative = R.T @ (np.array(obs["velocity_world_mps"]) - ego_velocity)
        x, y = obs["position_ego_m"]
        # The frame turns: omega x p, with y to the right and yaw to the right
        relative -= yaw_rate * np.array([-y, x])
        obs["velocity_relative_mps"] = [float(relative[0]), float(relative[1])]

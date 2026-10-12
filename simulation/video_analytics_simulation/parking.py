"""Geometry of one perpendicular park, backwards into a stall.

The stall is a rectangle in the world, written once into the scenario
(simulation/probe-stall.py measures it): centre x, y, yaw_deg pointing out of
the stall towards the aisle, length_m along that direction, width_m across.
Nothing here asks the simulator where stalls are.

Stall frame: u out of the stall, v to the right of u, origin at the centre.

    aisle  ── the vehicle drives along v ──▶      u = track_u
                                      ╭───● turning point: rear axle one
               ┆       ┆             ╱      radius past the stall's axis
       ────────┆  ▲ u  ┆────────────╯
               ┆  │    ┆   a quarter circle backwards, wheel at k of the lock
               ┆  ●    ┆   then nearly straight back to the centre
               ┆ stall ┆

The vehicle ends nose out. Its turning radius is that of the rear axle:
wheelbase / tan(turn_gain * k * full lock), with turn_gain measured (scenario
vehicle_response, the step in reverse).
"""

import math

import numpy as np


class Park:
    def __init__(self, spec, vehicle, config):
        stall = spec["stall"]
        self.centre = np.array([float(stall["x"]), float(stall["y"])])
        self.yaw_out = float(stall["yaw_deg"])
        yaw = math.radians(self.yaw_out)
        self.u = np.array([math.cos(yaw), math.sin(yaw)])
        self.v = np.array([-math.sin(yaw), math.cos(yaw)])
        self.length = float(stall["length_m"])
        self.width = float(stall["width_m"])
        if not (self.length > 0 and self.width > 0):
            raise ValueError("the stall needs a positive length and width")
        self.k = float(config.get("k", 0.45))
        self.wheelbase = float(vehicle["wheelbase_m"])
        self.radius = self.wheelbase / math.tan(math.radians(
            float(config.get("turn_gain", 1.0)) * self.k * float(vehicle["max_steer_deg"])))
        # Where the lane model is no longer asked for: from this point on
        # the vehicle is on the aisle, which has no lane markings
        handover = spec["handover"]
        self.handover = np.array([float(handover["x"]), float(handover["y"])])
        self.track_u, v = self.frame(self.handover)
        # Along v towards the stall and past it
        self.direction = 1.0 if v < 0 else -1.0
        self.handover_v = v
        # The vehicle's origin when its rear axle is one radius past the axis
        self.stop_v = self.direction * (self.radius + self.wheelbase / 2.0)
        self.inset = float(config.get("inset_m", 0.3))

    def frame(self, xy):
        """World x, y to the stall frame (u, v)."""
        d = np.asarray(xy, dtype=float)[:2] - self.centre
        return float(d @ self.u), float(d @ self.v)

    def world(self, u, v):
        return self.centre + u * self.u + v * self.v

    def heading_error(self, yaw_deg):
        """Radians the vehicle is turned to the right of nose-out."""
        return math.radians((yaw_deg - self.yaw_out + 180.0) % 360.0 - 180.0)

    def rear_axle(self, pose):
        yaw = math.radians(pose.yaw)
        return np.array([pose.x - self.wheelbase / 2.0 * math.cos(yaw),
                         pose.y - self.wheelbase / 2.0 * math.sin(yaw)])

    def handed_over(self, pose):
        """The vehicle's origin is at or past the handover point."""
        return self.direction * (self.frame([pose.x, pose.y])[1] - self.handover_v) >= -0.5

    def remaining(self, pose):
        """Metres along the aisle to the turning point; negative past it."""
        return self.direction * (self.stop_v - self.frame([pose.x, pose.y])[1])

    def track(self, pose, ahead_m=40.0):
        """The track on the aisle as a path in the ego frame: y = b x + c."""
        _, v = self.frame([pose.x, pose.y])
        points = [list(self.world(self.track_u, v + self.direction * s)) + [pose.z]
                  for s in (0.0, ahead_m)]
        (xa, ya, _), (xb, yb, _) = pose.from_parent(points)
        b = float((yb - ya) / (xb - xa)) if abs(xb - xa) > 1e-6 else 0.0
        return {"coefficients": [0.0, b, float(ya - b * xa)], "x_min_m": 0.0,
                "x_max_m": ahead_m}

    def corners(self, z=0.0, grow=0.0):
        """The four corners in the world, (4, 3)."""
        half_l, half_w = self.length / 2.0 + grow, self.width / 2.0 + grow
        return np.array([list(self.world(su * half_l, sv * half_w)) + [z]
                         for su in (1, -1) for sv in (1, -1)])

    def contains(self, xy, inset=0.0):
        u, v = self.frame(xy)
        return abs(u) <= self.length / 2.0 - inset and abs(v) <= self.width / 2.0 - inset

    def in_view(self, capture, rig):
        """All four corners are inside the image taken from this pose."""
        uv, depth = rig.project(capture.from_parent(self.corners(capture.z + rig.ground_z)))
        return bool((depth > 0.1).all() and (uv[:, 0] >= 0).all()
                    and (uv[:, 0] < rig.width).all() and (uv[:, 1] >= 0).all()
                    and (uv[:, 1] < rig.height).all())

    def turn_steer(self):
        """The constant steer of the quarter circle backwards. Its centre is
        on the stall's side of the track: to the vehicle's right when it
        drove along +v. Backwards on that circle the wheel points the same
        way as forwards on it, and the vehicle turns the other way."""
        return self.direction * self.k

    def parked_at(self, pose):
        """The origin is inside the stall, inset, and the vehicle within 15
        degrees of nose-out."""
        return self.contains([pose.x, pose.y], self.inset) \
            and abs(self.heading_error(pose.yaw)) < math.radians(15.0)

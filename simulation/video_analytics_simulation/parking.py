"""Geometry of one perpendicular park, backwards into a stall.

A stall is a rectangle on the ground. Where it is comes from the stall model
and the rear camera (stalls.py), never from the map or the scenario.

Stall frame: u out of the stall, v to the right of u, origin at the centre.

    aisle  ── the vehicle drives along it ──▶
                                      ╭───● turning point: rear axle one
               ┆       ┆             ╱      radius past the stall's axis
       ────────┆  ▲ u  ┆────────────╯
               ┆  │    ┆   a quarter circle backwards, then nearly
               ┆  ●    ┆   straight back to the centre
               ┆ stall ┆

The vehicle passes the stall, backs along the aisle to the turning point,
turns in and ends nose out. Its turning radius is that of the rear axle:
wheelbase / tan(turn_gain * share * full lock), with turn_gain measured
(scenario vehicle_response, the step in reverse).
"""

import math

import numpy as np


class Stall:
    """A rectangle on the ground: centre x, y, yaw_deg pointing out of the
    stall towards the aisle, length_m along that direction, width_m across."""

    def __init__(self, x, y, yaw_deg, length_m, width_m):
        self.centre = np.array([float(x), float(y)])
        self.yaw_out = float(yaw_deg)
        yaw = math.radians(self.yaw_out)
        self.u = np.array([math.cos(yaw), math.sin(yaw)])
        self.v = np.array([-math.sin(yaw), math.cos(yaw)])
        self.length = float(length_m)
        self.width = float(width_m)
        if not (self.length > 0 and self.width > 0):
            raise ValueError("the stall needs a positive length and width")

    @staticmethod
    def from_dict(d):
        return Stall(d["x"], d["y"], d["yaw_deg"], d["length_m"], d["width_m"])

    def as_dict(self):
        return {"x": float(self.centre[0]), "y": float(self.centre[1]),
                "yaw_deg": self.yaw_out, "length_m": self.length, "width_m": self.width}

    def frame(self, xy):
        """World x, y to the stall frame (u, v)."""
        d = np.asarray(xy, dtype=float)[:2] - self.centre
        return float(d @ self.u), float(d @ self.v)

    def world(self, u, v):
        return self.centre + u * self.u + v * self.v

    def heading_error(self, yaw_deg):
        """Radians the vehicle is turned to the right of nose-out."""
        return math.radians((yaw_deg - self.yaw_out + 180.0) % 360.0 - 180.0)

    def corners(self, z=0.0, grow=0.0):
        """The four corners in the world, (4, 3), around the stall: the two
        of the mouth, at +v then -v, then the two of the back, at -v then +v.
        The stall model's keypoints are these, in this order."""
        half_l, half_w = self.length / 2.0 + grow, self.width / 2.0 + grow
        return np.array([list(self.world(su * half_l, sv * half_w)) + [z]
                         for su, sv in ((1, 1), (1, -1), (-1, -1), (-1, 1))])

    def contains(self, xy, inset=0.0):
        u, v = self.frame(xy)
        return abs(u) <= self.length / 2.0 - inset and abs(v) <= self.width / 2.0 - inset


def reverse_steer(heading, offset, config):
    """Steer that brings the rear axle back onto a line while going
    backwards. heading: radians the vehicle is turned to the right of the
    line's direction; offset: metres the rear axle is to the right of it.
    Backwards the rear axle moves against the heading: to come back to the
    line from the right the vehicle must point right."""
    wanted = max(-0.3, min(0.3, config.get("axis_gain_rad_per_m", 0.6) * offset))
    return max(-0.3, min(0.3, config.get("heading_gain", 2.5) * (heading - wanted)))


class Aisle:
    """The line the vehicle drives on while it looks for a stall: through
    the place it was armed at, the way it pointed. A car park has no lane
    markings to follow; this line is the vehicle's own."""

    def __init__(self, pose, vehicle, config):
        self.origin = np.array([pose.x, pose.y])
        self.yaw = float(pose.yaw)
        yaw = math.radians(self.yaw)
        self.t = np.array([math.cos(yaw), math.sin(yaw)])
        self.n = np.array([-math.sin(yaw), math.cos(yaw)])  # to its right
        self.wheelbase = float(vehicle["wheelbase_m"])
        self.max_steer = float(vehicle["max_steer_deg"])
        self.turn_gain = float(config.get("turn_gain", 1.0))
        self.inside = float(config.get("turn_ends_inside_m", 1.0))

    def rear_axle(self, pose):
        yaw = math.radians(pose.yaw)
        return np.array([pose.x - self.wheelbase / 2.0 * math.cos(yaw),
                         pose.y - self.wheelbase / 2.0 * math.sin(yaw)])

    def travelled(self, pose):
        """Metres along the aisle since it was armed."""
        return float((np.array([pose.x, pose.y]) - self.origin) @ self.t)

    def offset(self, xy):
        """Metres to the right of the aisle's line."""
        return float((np.asarray(xy, dtype=float)[:2] - self.origin) @ self.n)

    def heading_error(self, yaw_deg):
        """Radians the vehicle is turned to the right of the aisle."""
        return math.radians((yaw_deg - self.yaw + 180.0) % 360.0 - 180.0)

    def track(self, pose, ahead_m=40.0):
        """The line as a path in the ego frame: y = b x + c."""
        along = self.travelled(pose)
        points = [list(self.origin + (along + s) * self.t) + [pose.z] for s in (0.0, ahead_m)]
        (xa, ya, _), (xb, yb, _) = pose.from_parent(points)
        b = float((yb - ya) / (xb - xa)) if abs(xb - xa) > 1e-6 else 0.0
        return {"coefficients": [0.0, b, float(ya - b * xa)], "x_min_m": 0.0,
                "x_max_m": ahead_m}

    def direction(self, stall, square_deg=20.0):
        """1 when the vehicle passes this stall along its +v, -1 along its
        -v; 0 when the stall is not square to the aisle or does not open
        onto it."""
        along = float(stall.v @ self.t)
        if abs(along) < math.cos(math.radians(square_deg)):
            return 0
        out = stall.frame(self.origin)[0] - stall.length / 2.0
        if not 1.5 <= out <= 9.0:
            return 0
        return 1 if along > 0 else -1

    def past(self, stall, pose):
        """Metres the rear axle is past the stall's axis, the way the
        vehicle drove; negative before it."""
        return self.direction(stall) * stall.frame(self.rear_axle(pose))[1]

    def radius(self, stall, pose):
        """Of the quarter circle that takes the rear axle from where it is,
        out on the aisle, onto the stall's axis `turn_ends_inside_m` inside
        the mouth."""
        return stall.frame(self.rear_axle(pose))[0] - (stall.length / 2.0 - self.inside)

    def steer_share(self, radius):
        """Share of the full lock that puts the rear axle on this radius."""
        if radius <= 0.0:
            return math.inf
        return math.degrees(math.atan(self.wheelbase / radius)) \
            / (self.turn_gain * self.max_steer)

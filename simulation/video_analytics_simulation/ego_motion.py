"""Dead reckoning from speed and yaw rate.

The integrator keeps yaw in radians. Pose.yaw is degrees, which is what
every consumer of a CARLA transform expects. Noise is a fixed config, not
a filter: a wheel-radius scale and a gyro bias, plus white noise.

The point integrated is the rear axle: it moves along the heading, the
vehicle's origin does not. In a turn the origin, ahead of the axle, also
moves sideways; taken to move along the heading, a pose that backed into a
stall was 1.3 m out after 3 m of the turn.
"""

import math

import numpy as np

from .calibration import Pose


class EgoMotion:
    """Pose after ARM. Before seed(), the caller still reads the message."""

    def __init__(self, config, speed_cap_mps, rear_axle_m=0.0):
        """rear_axle_m: where the rear axle is on the vehicle's x axis, from
        its origin (negative: behind it)."""
        config = config or {}
        self.axle = -float(rear_axle_m)  # the origin is this far ahead of the axle
        if not 0.0 <= self.axle <= 3.0:
            raise ValueError("a rear axle %.2f m from the vehicle's origin" % self.axle)
        self.speed_scale = float(config.get("speed_scale", 1.0))
        self.speed_noise = float(config.get("speed_noise_mps", 0.0))
        self.yaw_bias = float(config.get("yaw_rate_bias_radps", 0.0))
        self.yaw_noise = float(config.get("yaw_rate_noise_radps", 0.0))
        self.speed_cap = float(speed_cap_mps)
        self.rng = np.random.default_rng(int(config.get("seed", 1)))
        self.seeded = False
        self.last_ns = None
        self.x = self.y = self.z = 0.0
        self.roll = self.pitch = 0.0
        self.yaw_rad = 0.0
        self.pose = None
        self.velocity = (0.0, 0.0, 0.0)
        self.yaw_rate = 0.0
        self.poses = {}
        self.samples = {}

    def seed(self, pose, sim_ns):
        self.seeded = True
        self.z = pose.z
        self.roll, self.pitch = pose.roll, pose.pitch
        self.yaw_rad = math.radians(pose.yaw)
        # The rear axle's place
        self.x = pose.x - self.axle * math.cos(self.yaw_rad)
        self.y = pose.y - self.axle * math.sin(self.yaw_rad)
        self.pose = pose
        self.last_ns = int(sim_ns)
        self.poses[self.last_ns] = pose
        self.velocity = (0.0, 0.0, 0.0)
        self.yaw_rate = 0.0
        self.samples[self.last_ns] = (pose, self.velocity, self.yaw_rate)

    def at(self, sim_ns):
        return self.poses.get(int(sim_ns))

    def accept(self, ego, reverse):
        """Integrate one sample. Empty string when it was taken.

        A rejected sample does not move the pose. speed_mps is unsigned.
        reverse comes from the gear the plant already applied.
        """
        sim_ns = int(ego["sim_ns"])
        if self.last_ns is not None and sim_ns <= self.last_ns:
            return "ego_not_newer"
        dt = (sim_ns - self.last_ns) * 1e-9
        if dt > 2.0 / 30.0 + 1e-9:
            return "ego_dt"
        try:
            speed = float(ego["speed_mps"])
            yaw_rate = float(ego["yaw_rate_radps"])
        except (KeyError, TypeError, ValueError):
            return "ego_bad"
        if not math.isfinite(speed) or speed < 0.0 or speed > self.speed_cap + 1e-9:
            return "ego_speed"
        if not math.isfinite(yaw_rate) or abs(yaw_rate) > 2.0:
            return "ego_yaw_rate"
        # The speed is the origin's; the axle's is what is left of it once
        # the origin's turning about the axle is taken out
        v = self.speed_scale * math.sqrt(max(0.0, speed ** 2 - (yaw_rate * self.axle) ** 2))
        if self.speed_noise:
            v += float(self.rng.normal(0.0, self.speed_noise))
        if reverse:
            v = -v
        w = yaw_rate + self.yaw_bias
        if self.yaw_noise:
            w += float(self.rng.normal(0.0, self.yaw_noise))
        self.yaw_rad += w * dt
        cos, sin = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        self.x += v * cos * dt
        self.y += v * sin * dt
        yaw_deg = math.degrees(self.yaw_rad)
        self.pose = Pose(self.x + self.axle * cos, self.y + self.axle * sin, self.z,
                         self.roll, self.pitch, yaw_deg)
        # The origin's: the axle's, and its turning about the axle
        self.velocity = (v * cos - w * self.axle * sin, v * sin + w * self.axle * cos, 0.0)
        self.yaw_rate = w
        self.last_ns = sim_ns
        self.poses[sim_ns] = self.pose
        self.samples[sim_ns] = (self.pose, self.velocity, self.yaw_rate)
        return ""

    def velocity_at(self, sim_ns):
        return self.samples[int(sim_ns)][1]

    def yaw_at(self, sim_ns):
        return self.samples[int(sim_ns)][2]

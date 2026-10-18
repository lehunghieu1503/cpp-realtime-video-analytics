"""Camera calibration: pixels to metres on the road, and CARLA depth decoding.

Frames used here, all in metres:

    ego      x forward, y right, z up; origin at the vehicle's origin
             (CARLA's body frame; ROS body frames have y to the left)
    camera   the same axes, at the camera
    optical  x right, y down, z forward: (x, y, z)_optical = (y, -z, x)_camera
    world    CARLA's world frame

Angles are in degrees in the configuration, as CARLA takes them, and in
radians in every result. A positive angle turns to the right.

The rig has no lens distortion: CARLA renders a pinhole image unless told
otherwise, and this module rejects a rig that asks for distortion instead of
ignoring it.
"""

import math
from dataclasses import dataclass

import numpy as np

# CARLA's depth image holds metres * 16777215 / 1000 in 24 bits
DEPTH_FAR_M = 1000.0


def rotation(roll_deg, pitch_deg, yaw_deg):
    """Rotation of a CARLA transform: columns are its forward, right, up axes."""
    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    cr, sr = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp, -cp * sr, cp * cr],
    ])


@dataclass(frozen=True)
class Pose:
    """Where a body is in its parent frame, as CARLA reports a transform."""
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    @property
    def translation(self):
        return np.array([self.x, self.y, self.z], dtype=float)

    @property
    def rotation(self):
        return rotation(self.roll, self.pitch, self.yaw)

    def to_parent(self, points):
        """Points in this body's frame, (..., 3), to the parent frame."""
        return np.asarray(points, dtype=float) @ self.rotation.T + self.translation

    def from_parent(self, points):
        return (np.asarray(points, dtype=float) - self.translation) @ self.rotation

    @staticmethod
    def from_dict(d):
        pose = Pose(*(float(d[k]) for k in ("x", "y", "z", "roll", "pitch", "yaw")))
        if not all(math.isfinite(v) for v in pose.__dict__.values()):
            raise ValueError("pose has a value that is not finite")
        return pose


def fnv1a64(text):
    value = 0xCBF29CE484222325
    for byte in text.encode():
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


@dataclass(frozen=True)
class Rig:
    """A forward camera rigidly attached to the ego vehicle, over a flat road."""
    width: int
    height: int
    fov_deg: float  # horizontal
    x: float  # camera position in the ego frame
    y: float
    z: float
    pitch_deg: float = 0.0  # positive looks up
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    ground_z: float = 0.0  # height of the road in the ego frame

    def __post_init__(self):
        values = [self.fov_deg, self.x, self.y, self.z, self.pitch_deg,
                  self.yaw_deg, self.roll_deg, self.ground_z]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("rig has a value that is not finite")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("rig needs a positive image size")
        if not 1.0 <= self.fov_deg <= 170.0:
            raise ValueError("rig needs a horizontal field of view of 1..170 degrees")
        if self.z <= self.ground_z:
            raise ValueError("the camera must be above the road")

    @staticmethod
    def from_config(c):
        """From the `calibration` section of the configuration file."""
        if any(float(v) != 0.0 for v in c.get("distortion", [])):
            raise ValueError("lens distortion is not supported by this rig")
        camera = c["camera"]
        return Rig(int(c["width"]), int(c["height"]), float(c["fov_deg"]),
                   float(camera["x"]), float(camera.get("y", 0.0)),
                   float(camera["z"]), float(camera.get("pitch_deg", 0.0)),
                   float(camera.get("yaw_deg", 0.0)),
                   float(camera.get("roll_deg", 0.0)),
                   float(c.get("ground_z", 0.0)))

    @property
    def calibration_id(self):
        """Names these exact numbers; the C++ side computes the same string."""
        text = "w=%d;h=%d;fov=%.6f;x=%.6f;y=%.6f;z=%.6f;pitch=%.6f;yaw=%.6f;" \
               "roll=%.6f;ground=%.6f" % (
                   self.width, self.height, self.fov_deg, self.x, self.y, self.z,
                   self.pitch_deg, self.yaw_deg, self.roll_deg, self.ground_z)
        return "rig_%016x" % fnv1a64(text)

    @property
    def focal(self):
        # Square pixels: the vertical focal length is the horizontal one
        return self.width / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))

    @property
    def K(self):
        f = self.focal
        return np.array([[f, 0.0, self.width / 2.0],
                         [0.0, f, self.height / 2.0],
                         [0.0, 0.0, 1.0]])

    @property
    def pose(self):
        return Pose(self.x, self.y, self.z, self.roll_deg, self.pitch_deg,
                    self.yaw_deg)

    def rays_ego(self, u, v):
        """Direction in the ego frame of the ray through each pixel, with
        length 1 along the optical axis."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        f = self.focal
        camera = np.stack([np.ones_like(u), (u - self.width / 2.0) / f,
                           -(v - self.height / 2.0) / f], axis=-1)
        return camera @ self.pose.rotation.T

    def project(self, points_ego):
        """Ego points (..., 3) to pixels (..., 2) and their depth along the
        optical axis; a depth that is not positive is behind the camera."""
        camera = self.pose.from_parent(points_ego)
        depth = camera[..., 0]
        safe = np.where(depth > 1e-9, depth, np.nan)
        f = self.focal
        u = f * camera[..., 1] / safe + self.width / 2.0
        v = -f * camera[..., 2] / safe + self.height / 2.0
        return np.stack([u, v], axis=-1), depth

    def visible_box(self, points_ego, near_m=0.15):
        """The box in the image, (x0, y0, x1, y1), of the part of a convex
        polygon (ego points, (n, 3), in order around it) that the camera
        sees; None when it sees none of it. The polygon is cut at the plane
        near_m in front of the camera, projected, and cut at the image's
        four edges."""
        ring = self.pose.from_parent(points_ego)

        def cut(polygon, inside, cross):
            kept = []
            for i in range(len(polygon)):
                a, b = polygon[i], polygon[(i + 1) % len(polygon)]
                if inside(a):
                    kept.append(a)
                if inside(a) != inside(b):
                    kept.append(cross(a, b))
            return kept

        ring = cut(list(ring), lambda p: p[0] >= near_m,
                   lambda a, b: a + (b - a) * (near_m - a[0]) / (b[0] - a[0]))
        if len(ring) < 3:
            return None
        f = self.focal
        polygon = [np.array([f * p[1] / p[0] + self.width / 2.0,
                             -f * p[2] / p[0] + self.height / 2.0]) for p in ring]
        for axis, limit, keep_below in ((0, 0.0, False), (0, float(self.width), True),
                                        (1, 0.0, False), (1, float(self.height), True)):
            polygon = cut(
                polygon,
                lambda p: p[axis] <= limit if keep_below else p[axis] >= limit,
                lambda a, b: a + (b - a) * (limit - a[axis]) / (b[axis] - a[axis]))
            if len(polygon) < 3:
                return None
        polygon = np.array(polygon)
        low, high = polygon.min(axis=0), polygon.max(axis=0)
        if high[0] - low[0] <= 0.0 or high[1] - low[1] <= 0.0:
            return None
        return float(low[0]), float(low[1]), float(high[0]), float(high[1])

    def uv_to_ground(self, u, v, max_range_m=150.0):
        """Where the ray through each pixel meets the road.

        Returns (xy, valid): ego x, y in metres, shape (..., 2), and a mask.
        Invalid: a pixel outside the image, a ray at or above the horizon, or
        a point farther than max_range_m (close to the horizon one pixel is
        many metres).
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        rays = self.rays_ego(u, v)
        down = rays[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (self.ground_z - self.z) / down
        points = self.pose.translation + rays * t[..., None]
        xy = points[..., :2]
        valid = (np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < self.width)
                 & (v >= 0) & (v < self.height) & (down < -1e-6) & (t > 0))
        with np.errstate(invalid="ignore"):
            valid &= np.hypot(xy[..., 0] - self.x, xy[..., 1] - self.y) <= max_range_m
        return np.where(valid[..., None], xy, np.nan), valid

    def depth_to_ego(self, u, v, depth_m, radial=False):
        """Pixels with their depth to ego points (..., 3).

        CARLA's depth is the distance along the optical axis. radial=True
        treats it as the distance along the ray instead; it exists so the
        plane fixture can show which one the renderer gives.
        """
        rays = self.rays_ego(u, v)
        depth_m = np.asarray(depth_m, dtype=float)
        if radial:
            rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
        return self.pose.translation + rays * depth_m[..., None]


def decode_depth(bgra):
    """CARLA's raw depth image, uint8 (H, W, 4) in B, G, R, A order, to float32
    metres. The channels are widened first: R + 256 G + 65536 B overflows 8
    and 16 bits."""
    bgra = np.asarray(bgra)
    if bgra.dtype != np.uint8 or bgra.ndim != 3 or bgra.shape[2] != 4:
        raise ValueError("depth needs the raw uint8 BGRA image")
    packed = (bgra[..., 2].astype(np.uint32)
              + (bgra[..., 1].astype(np.uint32) << 8)
              + (bgra[..., 0].astype(np.uint32) << 16))
    return (packed.astype(np.float64) * (DEPTH_FAR_M / 16777215.0)).astype(np.float32)


def right_to_left(vector):
    """A y-right vector (this module, CARLA) to ROS's y-left body convention."""
    vector = np.array(vector, dtype=float)
    vector[..., 1] *= -1
    return vector

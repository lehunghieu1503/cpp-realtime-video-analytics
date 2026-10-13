"""The bird's-eye view: the images of the cameras around the vehicle, put on
the ground as one image.

    forward is up, the vehicle's origin is in the middle, and a pixel is
    `metres_per_pixel` of ground: column u and row v show the ground at

        x = (height / 2 - v) * metres_per_pixel     (forward)
        y = (u - width / 2) * metres_per_pixel      (right)

A pixel of it is taken from the camera that is nearest to that place on the
ground, among those that have it in their image; where none has, it stays
black. The ground is taken as flat: what stands on it (a car, a person) is
drawn out along the line of sight, away from the camera, and only its foot
is where it shows.

The device makes this image (src/analytics/bev.cpp) and runs the stall model
on it. Stitcher is the reference it is tested against, through
tests/fixtures/bev.json, and what makes the images that model is trained on.
BevRig is the view as a camera: it has what the stall tracker and the viewer
ask of calibration.Rig, so a stall found in it is followed as one found in a
camera's image is.
"""

from collections import OrderedDict

import numpy as np

from .calibration import Rig

NEAR_M = 0.15  # nearer to a camera's plane than this, the ground is not its


class BevRig:
    """The view from above as a camera."""

    def __init__(self, width, height, metres_per_pixel, ground_z=0.0):
        if width <= 0 or height <= 0 or not metres_per_pixel > 0.0:
            raise ValueError("the bird's-eye view needs a size and a scale")
        self.width, self.height = int(width), int(height)
        self.metres_per_pixel = float(metres_per_pixel)
        self.ground_z = float(ground_z)
        # What a sighting is weighed by (stalls.py): how far it is from the
        # vehicle's origin, as if seen from a camera's height above it. The
        # cameras are 1 m up, 1 to 2.5 m from the origin.
        self.x = self.y = 0.0
        self.z = self.ground_z + 1.0

    @staticmethod
    def from_config(c):
        """From the `bev` section of the configuration file."""
        return BevRig(c["width"], c["height"], c["metres_per_pixel"])

    def uv_to_ground(self, u, v, max_range_m=150.0):
        """The ground under each pixel, as Rig.uv_to_ground gives it: ego
        x, y in metres, (..., 2), and a mask."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        xy = np.stack([(self.height / 2.0 - v) * self.metres_per_pixel,
                       (u - self.width / 2.0) * self.metres_per_pixel], axis=-1)
        valid = (np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < self.width)
                 & (v >= 0) & (v < self.height))
        with np.errstate(invalid="ignore"):
            valid &= np.hypot(xy[..., 0], xy[..., 1]) <= max_range_m
        return np.where(valid[..., None], xy, np.nan), valid

    def project(self, points_ego):
        """Ego points (..., 3) to pixels (..., 2), and a depth that is
        always positive: nothing on the ground is behind this camera."""
        points = np.asarray(points_ego, dtype=float)
        uv = np.stack([points[..., 1] / self.metres_per_pixel + self.width / 2.0,
                       self.height / 2.0 - points[..., 0] / self.metres_per_pixel], axis=-1)
        return uv, np.ones(points.shape[:-1])

    def visible_box(self, points_ego, near_m=0.0):
        """The box in the image, (x0, y0, x1, y1), of the part of a convex
        polygon on the ground that is inside it; None when none of it is."""
        polygon = list(self.project(points_ego)[0])
        for axis, limit, below in ((0, 0.0, False), (0, float(self.width), True),
                                   (1, 0.0, False), (1, float(self.height), True)):
            kept = []
            for i, a in enumerate(polygon):
                b = polygon[(i + 1) % len(polygon)]
                a_in = a[axis] <= limit if below else a[axis] >= limit
                b_in = b[axis] <= limit if below else b[axis] >= limit
                if a_in:
                    kept.append(a)
                if a_in != b_in:
                    kept.append(a + (b - a) * (limit - a[axis]) / (b[axis] - a[axis]))
            polygon = kept
            if len(polygon) < 3:
                return None
        polygon = np.array(polygon)
        low, high = polygon.min(axis=0), polygon.max(axis=0)
        if high[0] - low[0] <= 0.0 or high[1] - low[1] <= 0.0:
            return None
        return float(low[0]), float(low[1]), float(high[0]), float(high[1])


def cameras_of(config):
    """The cameras the bird's-eye view is made of, in the order of the `bev`
    section: (source id, Rig, port the simulator sends it on)."""
    return [(camera["source"], Rig.from_config(config[camera["calibration"]]),
             int(camera.get("port", 0))) for camera in config["bev"]["cameras"]]


class Stitcher:
    """Makes the view from the cameras' images."""

    def __init__(self, view, rigs):
        self.view, self.rigs = view, list(rigs)
        rows, cols = np.mgrid[0:view.height, 0:view.width]
        ground, _ = view.uv_to_ground(cols + 0.5, rows + 0.5)  # of pixel centres
        points = np.concatenate([ground, np.full(ground.shape[:-1] + (1,), view.ground_z)],
                                axis=-1)
        nearest = np.full(ground.shape[:-1], np.inf)
        # Which camera each pixel is taken from; -1: none sees that ground
        self.source = np.full(ground.shape[:-1], -1, dtype=np.int8)
        places = []
        for index, rig in enumerate(self.rigs):
            uv, depth = rig.project(points)
            u, v = uv[..., 0], uv[..., 1]
            with np.errstate(invalid="ignore"):
                seen = (depth > NEAR_M) & (u >= 0.5) & (u <= rig.width - 0.5) \
                    & (v >= 0.5) & (v <= rig.height - 0.5)
            distance = np.linalg.norm(points - np.array([rig.x, rig.y, rig.z]), axis=-1)
            take = seen & (distance < nearest)
            nearest[take] = distance[take]
            self.source[take] = index
            places.append((u, v))
        # Where in a camera's image each pixel is, counted from the centre
        # of its first pixel as remap does; outside it wherever the pixel is
        # another camera's or nobody's
        self.maps = [(np.where(self.source == i, u - 0.5, -10.0).astype(np.float32),
                      np.where(self.source == i, v - 0.5, -10.0).astype(np.float32))
                     for i, (u, v) in enumerate(places)]

    def blank(self):
        return np.zeros((self.view.height, self.view.width, 3), np.uint8)

    def paste(self, canvas, index, image):
        """One camera's image (BGR, as that camera gives it) onto its part
        of the view."""
        import cv2
        rig = self.rigs[index]
        if image.shape[:2] != (rig.height, rig.width):
            raise ValueError("camera %d gives %dx%d, its calibration says %dx%d" % (
                index, image.shape[1], image.shape[0], rig.width, rig.height))
        cv2.remap(image, self.maps[index][0], self.maps[index][1], cv2.INTER_LINEAR,
                  dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)
        return canvas

    def compose(self, images):
        canvas = self.blank()
        for index, image in enumerate(images):
            self.paste(canvas, index, image)
        return canvas


class Joiner:
    """Two results of one frame as one: the rear camera's own, with the
    stalls and divider ends the device found in the bird's-eye view of that
    frame in place of the rear image's. A result whose other half never
    comes is forgotten after `keep` newer ones."""

    def __init__(self, keep=30):
        self.keep = keep
        self.waiting = {"rear": OrderedDict(), "bev": OrderedDict()}

    def offer(self, kind, envelope):
        """kind: "rear" or "bev". The joined result once both are here."""
        sim_ns = (envelope.get("simulation") or {}).get("sim_ns")
        if not isinstance(sim_ns, int) or isinstance(sim_ns, bool):
            return None
        other = self.waiting["bev" if kind == "rear" else "rear"]
        if sim_ns not in other:
            mine = self.waiting[kind]
            mine[sim_ns] = envelope
            while len(mine) > self.keep:
                mine.popitem(last=False)
            return None
        mate = other.pop(sim_ns)
        rear, view = (envelope, mate) if kind == "rear" else (mate, envelope)
        return dict(rear, stalls=view.get("stalls"), stall_points=view.get("stall_points"))

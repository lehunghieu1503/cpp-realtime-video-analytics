"""One window with the whole run, for a person watching: the simulator's
own images, with what the device found in them drawn on top.

    front camera                      boxes, track ids, lane lines
    rear camera (scenarios that park) stalls: green empty, red occupied
                                      ends of dividers: crosses
                                      the stalls the controller follows, and
                                      in yellow the one it parks in
    two lines: tick, state, phase, speed; what is applied to the vehicle

    +--------------------------------+----------------+
    | the vehicle from behind (chase)|  bird's-eye    |
    | or, without it, front camera   |  view          |
    |                                +----------------+
    |                                |  map           |
    +--------+--------+--------+-----+                |
    | front  | rear   | left   | right|               |
    +--------+--------+--------+-----+----------------+
    | two lines of state                              |

Only the panels the run has are there: rear, left, right and the map come
with the scenarios that park, the view from above with those that make one,
the chase image with --follow. The window fits a 1920x1080 screen and can be
resized.

The map is drawn, not seen: the stalls the controller follows, where it
believes they are around the vehicle, forward up. Occupied ones hold a car,
the one it parks in is blue.

The bird's-eye view (the `bev` section of the configuration) is the view from
above of the cameras around the vehicle, forward up; the stalls and the ends
are then found in it and drawn on it.

In a scenario that parks there are no lane lines: the device does not run
the lane model there (simulation/run.sh leaves it out of its config).

An image is shown with the result of its own frame, joined by sim_ns, so it
is as late as that result. Until the first result an image is shown as it
is. Nothing in the control path reads this module; without --view it is
never imported.
"""

import math
from collections import OrderedDict

import numpy as np

from .bev import BevRig
from .calibration import Pose

WINDOW = "CARLA cameras"
LABELS = {"front": "FRONT CAMERA", "rear": "REAR CAMERA",
          "left": "LEFT CAMERA", "right": "RIGHT CAMERA"}
KEPT = 45  # images a late result can still be joined with
SIDE = 600  # pixels a side of the view from above, and of the map
MAP_M = 20.0  # metres a side of the map
# Stalls the controller follows but has seen whole fewer times than this are
# not drawn: guesses from the ends of dividers, often askew beside parked
# cars. shortcut: the value of park.min_sightings in carla.yaml, read it from
# there if that changes
MIN_HITS = 10
# Of two stalls that cover each other more than this share of the smaller,
# only the one seen last is drawn. The controller may follow one stall twice:
# its pose drifts, and a stall seen again far from where it was first put
# starts a new track. The one seen last is where the stall is from the pose
# it has now.
MAX_OVERLAP = 0.1
SAME_TIME_NS = 500_000_000  # stalls seen this close in time count as seen together


class Viewer:
    def __init__(self, rigs, pose_at, scale=0.5, vehicle=None):
        """rigs: name -> Rig, in the order shown. pose_at(sim_ns): the ego
        pose of that frame as calibration.Pose, or None. vehicle: the
        manifest's, to draw it into a view from above."""
        import cv2
        self.cv2 = cv2
        self.rigs = rigs
        self.pose_at = pose_at
        self.vehicle = vehicle
        self.scale = scale
        self.images = {name: OrderedDict() for name in rigs}
        self.results = {name: None for name in rigs}
        self.panels = {name: np.zeros((int(r.height * scale), int(r.width * scale), 3),
                                      np.uint8) for name, r in rigs.items()}
        self.park = None
        self.drawn = set()  # ids of the stalls followed that are drawn
        self.sprite = None  # the vehicle from above, BGRA, nose up (sprite_from)
        self.parked = None  # the same, grey: the cars in occupied stalls
        self.history = None  # the last view from above, filled, and its pose
        self.chase = None
        self.size = None  # of the picture the window was last fitted to
        self.state = ""
        self.closed = False
        self.map_rig = BevRig(SIDE, SIDE, MAP_M / SIDE) if "rear" in rigs else None
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    def frame(self, name, sim_ns, bgra):
        """A camera image of this tick, as the simulator got it."""
        size = (self.panels[name].shape[1], self.panels[name].shape[0])
        small = self.cv2.resize(bgra[:, :, :3], size, interpolation=self.cv2.INTER_AREA)
        images = self.images[name]
        images[sim_ns] = small
        while len(images) > KEPT:
            images.popitem(last=False)
        if self.results[name] is None:
            self.panels[name] = small

    def chase_frame(self, bgra):
        """The image of the camera behind the vehicle; nothing is drawn on it."""
        self.chase = bgra[:, :, :3]

    def result(self, name, envelope):
        """The device's result for one frame of this camera."""
        sim_ns = (envelope.get("simulation") or {}).get("sim_ns")
        image = self.images[name].get(sim_ns)
        if image is None:
            return
        self.results[name] = sim_ns
        panel = image.copy()
        self.draw(name, panel, envelope, sim_ns)
        self.panels[name] = panel

    def status(self, status):
        self.state = status.get("state", "")
        if status.get("stop_reason"):
            self.state += " " + status["stop_reason"]
        self.park = status.get("park")
        self.drawn = self.stalls_drawn(self.park) if self.park else set()

    def draw(self, name, panel, envelope, sim_ns):
        cv2, k = self.cv2, self.scale

        def point(x, y):
            return int(round(x * k)), int(round(y * k))

        for item in envelope.get("detections", []):
            box = item["bbox"]
            cv2.rectangle(panel, point(box["x"], box["y"]),
                          point(box["x"] + box["w"], box["y"] + box["h"]), (255, 160, 0), 1)
            label = item["class"] + (" #%d" % item["track_id"] if "track_id" in item else "")
            cv2.putText(panel, label, point(box["x"], box["y"] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 160, 0), 1, cv2.LINE_AA)
        for points in (envelope.get("lane") or {}).get("pixels", []):
            if len(points) > 1:
                cv2.polylines(panel, [np.array([point(*p) for p in points])], False,
                              (0, 255, 255), 2, cv2.LINE_AA)
        for stall in envelope.get("stalls", []):
            colour = (0, 0, 255) if stall["class"] == "occupied" else (0, 255, 0)
            corners = stall["corners"]
            for i, (x, y, confidence) in enumerate(corners):
                if confidence < 0.5:
                    continue
                cv2.circle(panel, point(x, y), 3, colour, -1, cv2.LINE_AA)
                nx, ny, other = corners[(i + 1) % 4]
                if other >= 0.5:
                    cv2.line(panel, point(x, y), point(nx, ny), colour, 1, cv2.LINE_AA)
        for end in envelope.get("stall_points", []):
            cv2.drawMarker(panel, point(end["x"], end["y"]),
                           (255, 0, 255) if end["class"] == "back" else (255, 255, 0),
                           cv2.MARKER_CROSS, 9, 2, cv2.LINE_8)
        if "stalls" in envelope and self.park:
            self.draw_tracks(name, panel, sim_ns)

    def draw_tracks(self, name, panel, sim_ns):
        """The stalls the controller follows, from the world back into this
        image: where it believes they are."""
        cv2, rig = self.cv2, self.rigs[name]
        # The controller's own pose when it gives one: it reckons its place
        # itself, and its stalls are where it takes them to be from there
        pose = Pose.from_dict(self.park["pose"]) if self.park.get("pose") \
            else self.pose_at(sim_ns)
        if pose is None:
            return
        for row in self.park.get("stalls", []):
            if not self.shown(row):
                continue
            world = np.array([[x, y, pose.z + rig.ground_z] for x, y in row["corners"]])
            uv, depth = rig.project(pose.from_parent(world))
            if (depth < 0.3).any():
                continue  # partly behind the camera: its outline would be wrong
            chosen = row["id"] == self.park.get("target")
            outline = np.array([[int(round(u * self.scale)), int(round(v * self.scale))]
                                for u, v in uv])
            cv2.polylines(panel, [outline], True, (0, 255, 255) if chosen else (255, 255, 255),
                          3 if chosen else 1, cv2.LINE_AA)

    def show(self, tick, speed_mps, applied):
        """Draws the window. False once the person has closed it or pressed q."""
        if self.closed:
            return False
        return self.display(self.picture(tick, speed_mps, applied))

    def picture(self, tick, speed_mps, applied):
        """The window's picture. Touches no window: it may be made on another
        thread than the one that shows it."""
        cv2 = self.cv2
        panels = {name: self.with_vehicle(name, panel) for name, panel in self.panels.items()}
        right = [panels.pop("bev")] if "bev" in panels else []
        if self.map_rig:
            right.append(self.map())
        big = self.chase if self.chase is not None else panels.pop("front")
        width = big.shape[1]
        column = [big]
        if panels:
            # Four to a row whatever their number: a camera keeps one size
            small = [cv2.resize(p, (width // 4, p.shape[0] * (width // 4) // p.shape[1]),
                                interpolation=cv2.INTER_AREA) for p in panels.values()]
            for name, image in zip(panels, small):
                self.label(image, LABELS.get(name, name))
            row = np.hstack(small)
            column.append(np.pad(row, ((0, 0), (0, width - row.shape[1]), (0, 0))))
        view = np.vstack(column)
        if right:
            side = min(SIDE, view.shape[0] // len(right))
            column = np.vstack([cv2.resize(p, (side, side), interpolation=cv2.INTER_AREA)
                                for p in right])
            view = np.hstack([view, np.pad(column, ((0, view.shape[0] - column.shape[0]),
                                                    (0, 0), (0, 0)))])
        widest = view.shape[1]
        bar = np.zeros((46, widest, 3), np.uint8)
        gear = "R" if applied.get("reverse") else "D"
        lines = ("tick %d  %s  %s%4.1f km/h  %s" % (
            tick, self.state, "park: %s  " % self.park["phase"] if self.park else "",
            speed_mps * 3.6, gear),
            "throttle %.2f  brake %.2f  steer %+.2f  %s" % (
            applied.get("throttle", 0.0), applied.get("brake", 0.0),
            applied.get("steer", 0.0), applied.get("reason", "")))
        for i, text in enumerate(lines):
            cv2.putText(bar, text, (8, 18 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([view, bar])

    def display(self, picture):
        """Shows a picture made by picture(). False once the person has
        closed the window or pressed q."""
        if self.closed:
            return False
        cv2 = self.cv2
        if self.size != picture.shape[:2]:
            # A window that may be resized opens small, whatever it shows
            self.size = picture.shape[:2]
            cv2.resizeWindow(WINDOW, picture.shape[1], picture.shape[0])
        cv2.imshow(WINDOW, picture)
        # Closed by its button: Qt says not visible; GTK, which answers -1 for
        # visible whatever the window, no longer has it (-1 for any property)
        try:
            gone = cv2.waitKey(1) & 0xFF == ord("q") \
                or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) == 0 \
                or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_AUTOSIZE) < 0
        except cv2.error:
            gone = True  # Qt, once its window is closed, raises for any property
        if gone:
            self.close()
        return not self.closed

    def shown(self, row):
        """Whether a stall the controller follows is drawn."""
        return row["id"] in self.drawn

    def stalls_drawn(self, park):
        """Ids of the stalls followed worth drawing: seen often enough, and
        none covering one seen later, or as late and more often. The one it
        parks in always."""
        cv2, target = self.cv2, park.get("target")
        rows = sorted((row for row in park.get("stalls", [])
                       if row.get("hits", MIN_HITS) >= MIN_HITS or row["id"] == target),
                      key=lambda row: (row["id"] != target,
                                       -(row.get("seen_ns", 0) // SAME_TIME_NS),
                                       -row.get("hits", 0)))
        kept = []
        for row in rows:
            corners = np.array(row["corners"], np.float32)
            area = cv2.contourArea(corners)
            if all(cv2.intersectConvexConvex(corners, other)[0]
                   <= MAX_OVERLAP * min(area, cv2.contourArea(other))
                   for _, other in kept):
                kept.append((row["id"], corners))
        return {key for key, _ in kept}

    def map(self):
        """The stalls the controller follows, around the vehicle, forward up."""
        cv2, rig = self.cv2, self.map_rig
        panel = np.full((rig.height, rig.width, 3), (60, 52, 46), np.uint8)
        park = self.park or {}
        if park.get("pose"):
            # Its own pose, of the frame the corners are in
            pose = Pose.from_dict(park["pose"])
            for row in park.get("stalls", []):
                if not self.shown(row):
                    continue
                world = np.array([[x, y, pose.z] for x, y in row["corners"]])
                corners = rig.project(pose.from_parent(world))[0]
                outline = np.round(corners).astype(np.int32)
                if row["id"] == park.get("target"):
                    tint = panel.copy()
                    cv2.fillPoly(tint, [outline], (200, 120, 30), cv2.LINE_AA)
                    cv2.addWeighted(tint, 0.6, panel, 0.4, 0.0, panel)
                elif not row.get("free", True) and self.parked is not None:
                    self.paste(panel, self.parked, self.parked_in(corners, rig))
                elif not row.get("free", True):
                    # A car, a little smaller than the stall
                    middle = corners.mean(axis=0)
                    car = np.round(middle + 0.7 * (corners - middle)).astype(np.int32)
                    cv2.fillPoly(panel, [car], (170, 170, 170), cv2.LINE_AA)
                    cv2.polylines(panel, [car], True, (90, 90, 90), 2, cv2.LINE_AA)
                cv2.polylines(panel, [outline], True, (235, 235, 235), 2, cv2.LINE_AA)
        if self.vehicle:
            self.draw_vehicle(panel, rig, 1.0)
        return panel

    def label(self, image, text):
        """Which camera an image is, white on a dark box in its top left."""
        cv2 = self.cv2
        (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        image[:h + base + 8, :w + 12] //= 3
        cv2.putText(image, text, (6, h + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    def with_vehicle(self, name, panel):
        """A view from above with the vehicle drawn where it is: no camera
        sees it, and the ground under it is black."""
        rig = self.rigs[name]
        if not self.vehicle or not hasattr(rig, "metres_per_pixel"):
            return panel
        panel = panel.copy()
        self.draw_vehicle(panel, rig, self.scale)
        return panel

    def draw_vehicle(self, panel, rig, scale):
        """The vehicle seen from above, nose up, where rig puts its origin."""
        cv2, v = self.cv2, self.vehicle
        front, half = v["front_bumper_m"], v["width_m"] / 2.0
        rear = v.get("rear_bumper_m", -front)
        outline = np.array([[front, -half, 0.0], [front, half, 0.0],
                            [rear, half, 0.0], [rear, -half, 0.0]])
        nose = np.array([[front - 0.3, 0.0, 0.0], [front - 1.4, -half + 0.25, 0.0],
                         [front - 1.4, half - 0.25, 0.0]])
        body, tip = (np.round(rig.project(p)[0] * scale).astype(np.int32)
                     for p in (outline, nose))
        if self.sprite is not None:
            # Front left, front right, rear left
            self.paste(panel, self.sprite, rig.project(outline[[0, 1, 3]])[0] * scale)
            return
        cv2.fillPoly(panel, [body], (120, 70, 40), cv2.LINE_AA)
        cv2.polylines(panel, [body], True, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.fillPoly(panel, [tip], (255, 255, 255), cv2.LINE_AA)

    def set_sprite(self, sprite):
        """The vehicle from above (sprite_from): drawn in place of its
        outline, and grey in the occupied stalls of the map."""
        cv2 = self.cv2
        grey = cv2.cvtColor(cv2.cvtColor(sprite[:, :, :3], cv2.COLOR_BGR2GRAY),
                            cv2.COLOR_GRAY2BGR)
        self.parked = np.dstack([cv2.addWeighted(grey, 0.6, grey, 0.0, 70), sprite[:, :, 3]])
        self.sprite = sprite

    def parked_in(self, corners, rig):
        """Where the car in a stall goes: its front left, front right and
        rear left, in pixels, nose to the stall's mouth (its first two
        corners)."""
        v, k = self.vehicle, 1.0 / rig.metres_per_pixel
        middle = corners.mean(axis=0)
        along = (corners[0] + corners[1]) / 2.0 - middle
        along /= np.linalg.norm(along)
        across = corners[1] - corners[0]
        across /= np.linalg.norm(across)
        length = (v["front_bumper_m"] - v.get("rear_bumper_m", -v["front_bumper_m"])) * k
        half = v["width_m"] / 2.0 * k
        front = middle + along * length / 2.0
        return np.array([front - across * half, front + across * half,
                         front - along * length - across * half])

    def paste(self, panel, sprite, corners):
        """sprite onto panel, its corners top left, top right and bottom left
        at those three points; its transparent pixels leave the panel."""
        cv2 = self.cv2
        corners = np.float32(corners)
        # Down to about its size first: an affine warp alone aliases
        size = (max(2, int(round(np.linalg.norm(corners[1] - corners[0])))),
                max(2, int(round(np.linalg.norm(corners[2] - corners[0])))))
        small = cv2.resize(sprite, size, interpolation=cv2.INTER_AREA)
        h, w = small.shape[:2]
        matrix = cv2.getAffineTransform(np.float32([[0, 0], [w, 0], [0, h]]), corners)
        x, y, bw, bh = cv2.boundingRect(np.round(np.vstack([
            corners, [corners[1] + corners[2] - corners[0]]])).astype(np.int32))
        x0, y0 = max(x - 1, 0), max(y - 1, 0)
        x1, y1 = min(x + bw + 1, panel.shape[1]), min(y + bh + 1, panel.shape[0])
        if x1 <= x0 or y1 <= y0:
            return
        matrix[:, 2] -= (x0, y0)
        warped = cv2.warpAffine(small, matrix, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                borderValue=(0, 0, 0, 0))
        alpha = warped[:, :, 3:].astype(np.float32) / 255.0
        roi = panel[y0:y1, x0:x1]
        roi[:] = (roi * (1.0 - alpha) + warped[:, :, :3] * alpha).astype(np.uint8)

    def fill(self, view, unseen, sim_ns):
        """The view from above with the ground no camera sees now taken from
        the views before it, carried by how the vehicle moved since. For the
        person watching: the device's own view keeps those pixels black."""
        cv2, rig = self.cv2, self.rigs["bev"]
        pose = self.pose_at(sim_ns)
        if pose is None:
            return view
        if self.history is not None:
            before, at = self.history
            view = view.copy()
            view[unseen] = cv2.warpAffine(before, carried(rig, at, pose),
                                          (rig.width, rig.height),
                                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)[unseen]
        self.history = (view, pose)
        return view

    def pump(self):
        """Lets the window answer while nothing new is drawn (the run is
        waiting for the device)."""
        if not self.closed:
            self.cv2.waitKey(1)

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.cv2.destroyWindow(WINDOW)
            except self.cv2.error:
                pass  # closed by its button already


def sprite_from(bgra, tags, car_tags, fov_deg, distance_m, vehicle):
    """The vehicle from above, BGRA, nose up, cut to its box: a camera
    straight above its origin looking down, image top forward, gave bgra;
    tags is the semantic camera's tag of each pixel, car_tags the vehicles'.
    distance_m: from the camera to about half the vehicle's height."""
    height, width = bgra.shape[:2]
    metres = distance_m * 2.0 * math.tan(math.radians(fov_deg) / 2.0) / width
    front, half = vehicle["front_bumper_m"], vehicle["width_m"] / 2.0
    rear = vehicle.get("rear_bumper_m", -front)
    top, bottom = (int(round(height / 2.0 - x / metres)) for x in (front, rear))
    left, right = (int(round(width / 2.0 + y / metres)) for y in (-half, half))
    sprite = np.ascontiguousarray(bgra[top:bottom, left:right]).copy()
    sprite[:, :, 3] = np.where(np.isin(tags[top:bottom, left:right], car_tags), 255, 0)
    return sprite


def carried(rig, before, now):
    """The 2x3 matrix from a pixel of the view from above at pose `now` to
    the pixel of the same ground in the view at pose `before`."""
    import cv2
    pixels = np.float32([[0, 0], [rig.width - 1, 0], [0, rig.height - 1]])
    ground, _ = rig.uv_to_ground(pixels[:, 0], pixels[:, 1])
    points = np.concatenate([ground, np.zeros((3, 1))], axis=-1)
    then = before.from_parent(now.to_parent(points))
    return cv2.getAffineTransform(pixels, np.float32(rig.project(then)[0]))

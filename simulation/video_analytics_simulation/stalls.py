"""Parking stalls: what the device finds in the rear camera's image, put on
the ground and followed from frame to frame.

The device's stall model (src/inference/stall-engine.hpp) gives two things:

  ends    where a divider between two stalls ends, at the mouth of the
          stalls or at the line across their back: one pixel each. A small
          thing with a sharp place.
  stalls  a class, empty or occupied, and four corners. A stall is large and
          its corners are far from its middle: the model puts them roughly,
          and now and then in another order.

So a stall is measured by the ends and judged by the stall detections:

  An end the model sees is put on the ground with the rear camera's
  calibration, the ground taken as flat, and moved to the world with the
  vehicle's own pose at that frame. Under a camera h metres up, a point d
  metres away moves by (d^2 + h^2) / (f h) metres along the line of sight
  for one pixel: 3 cm at 3 m, 23 cm at 9 m. A track keeps an end as the mean
  of its sightings, each weighted by the inverse square of that number; an
  end that has left the image keeps what was learned of it while it was near.

  Two ends at the mouth, one stall's width apart, are a stall. It opens
  away from the ends at the back of its dividers; while none of those is
  seen, towards where the camera was when its mouth was first seen: the
  vehicle drives on the aisle. Its direction is square to the line through
  all the mouth ends of its row, which is longer than its own mouth. Its
  length comes from the ends at the back of its dividers once they are seen;
  until then it is taken as 5 m, which only moves where its back is.

  A stall detection votes, empty or occupied, for the stall whose box in the
  image is its own: the two are compared in pixels. On the ground the
  detection's far corners would be metres off.

Nothing here reads the map or where the stalls really are.
"""

import math
from collections import deque

import numpy as np

from .parking import Stall

DEFAULTS = {
    "confidence": 0.5,        # of an end, of a stall detection, of one of its corners
    "end_range_m": 10.0,      # an end farther from the camera is not used
    "end_gate_m": 0.7,        # a sighting is of the end this near to it, or of a new one
    # Far away a sighting can land beyond that gate and start an end of its
    # own beside the real one: an end this near to one seen four times as
    # often is dropped
    "ghost_m": 0.9,
    "forget": 0.98,           # what an end knows counts this much less per new sighting
    "end_hits": 4,            # sightings before an end marks anything
    "width_m": [2.2, 3.6],    # two mouth ends this far apart are a stall
    "row_reach_m": 14.0,      # mouth ends of the same row, for its direction
    "row_gate_m": 0.5,        # ... and this near to the line through the stall's own two
    "length_m": [4.0, 6.0],   # what a stall's length may be
    "length_default_m": 5.0,  # until the ends at its back say how long it is
    # A stall detection votes for the stall whose box in the image overlaps
    # its own by this much (intersection over union), and is 12 pixels high
    "vote_overlap": 0.5,
    "vote_height_px": 12.0,
    "window": 15,             # votes the class is read from
    "free_share": 0.2,        # occupied in more than this share of them: not free
    "min_hits": 8,            # votes before a stall says it is free
    "stale_s": 2.0,           # an end seen fewer than end_hits times is dropped after this
}


class End:
    """Where a divider ends, on the ground, in the world."""

    def __init__(self, end_id):
        self.id = end_id
        self.xy = np.zeros(2)
        self.weight = 0.0
        self.hits = 0
        self.back = 0  # sightings that said "at the back"
        self.last_ns = 0
        self.nearest_m = math.inf

    def add(self, xy, weight, back, distance, sim_ns, forget):
        known = self.weight * forget
        self.xy = (known * self.xy + weight * xy) / (known + weight)
        self.weight = known + weight
        self.hits += 1
        self.back += 1 if back else 0
        self.last_ns = sim_ns
        self.nearest_m = min(self.nearest_m, distance)

    @property
    def at_the_back(self):
        return 2 * self.back > self.hits


class StallTrack:
    """The stall between two neighbouring ends at the mouth."""

    def __init__(self, track_id, plus, minus, out, window):
        self.id = track_id
        self.plus, self.minus = plus, minus  # its mouth ends, at +v and at -v
        self.out = out                       # unit vector out of the stall, at first sight
        self.classes = deque(maxlen=window)  # 1: occupied
        self.hits = 0                        # votes
        self.backs = ()  # the ends at the back of its two dividers, once seen
        self.fitted = None

    @property
    def last_ns(self):
        """When one of its ends was last seen."""
        return max(end.last_ns for end in (self.plus, self.minus) + self.backs)

    def stall(self):
        return self.fitted

    def occupied_share(self):
        return sum(self.classes) / len(self.classes) if self.classes else 1.0

    def free(self, config):
        """Voted on often enough, and empty nearly every time."""
        return self.hits >= config["min_hits"] \
            and self.occupied_share() <= config["free_share"]


class StallTracker:
    def __init__(self, rig, config=None):
        self.rig = rig
        self.capture = None  # the pose of the latest frame: what the tracks are held in
        self.config = dict(DEFAULTS, **(config or {}))
        self.ends = []
        self.tracks = []
        self.next_id = 1
        self.height = rig.z - rig.ground_z

    def ground(self, u, v, capture, max_range_m):
        """Pixels to the world, with the distance from the camera on the
        ground and whether the pixel is on the ground within range."""
        xy, ok = self.rig.uv_to_ground(u, v, max_range_m)
        xy = np.where(ok[..., None], xy, 0.0)
        world = capture.to_parent(np.concatenate(
            [xy, np.full(xy.shape[:-1] + (1,), self.rig.ground_z)], axis=-1))[..., :2]
        return world, np.hypot(xy[..., 0] - self.rig.x, xy[..., 1] - self.rig.y), ok

    def update(self, stalls, points, capture, sim_ns):
        """The stall detections and divider ends of one frame, taken from the
        pose `capture`. Returns what was made of them, for the log."""
        config = self.config
        self.capture = capture
        record = {"ends": self.see_ends(points, capture, sim_ns)}
        stale = sim_ns - int(config["stale_s"] * 1e9)
        self.ends = [e for e in self.ends
                     if (e.hits >= config["end_hits"] or e.last_ns >= stale)
                     and not any(o.hits >= 4 * e.hits
                                 and np.hypot(*(o.xy - e.xy)) < config["ghost_m"]
                                 for o in self.ends)]
        self.pair_ends(capture)
        for track in self.tracks:
            self.fit(track)
        record["votes"] = self.vote(stalls, capture)
        return record

    def see_ends(self, points, capture, sim_ns):
        config = self.config
        made = []
        for point in sorted(points, key=lambda p: -float(p["confidence"])):
            if float(point["confidence"]) < config["confidence"]:
                continue
            xy, distance, ok = self.ground(np.array(float(point["x"])),
                                           np.array(float(point["y"])), capture,
                                           config["end_range_m"])
            if not ok:
                continue
            distance = float(distance)
            end = min(self.ends, key=lambda e: np.hypot(*(e.xy - xy)), default=None)
            if end is None or np.hypot(*(end.xy - xy)) > config["end_gate_m"]:
                end = End(self.next_id)
                self.next_id += 1
                self.ends.append(end)
            end.add(xy, 1.0 / (distance ** 2 + self.height ** 2) ** 2,
                    point["class"] == "back", distance, sim_ns, config["forget"])
            made.append(end.id)
        return made

    def mouths(self):
        return [e for e in self.ends
                if e.hits >= self.config["end_hits"] and not e.at_the_back]

    def pair_ends(self, capture):
        """A stall for every two neighbouring mouth ends that have none yet;
        and none for a pair that is no longer one."""
        low, high = self.config["width_m"]
        mouths = self.mouths()
        alive = {id(e) for e in mouths}
        self.tracks = [t for t in self.tracks
                       if id(t.plus) in alive and id(t.minus) in alive
                       and low <= np.hypot(*(t.plus.xy - t.minus.xy)) <= high]
        paired = {frozenset((t.plus.id, t.minus.id)) for t in self.tracks}
        camera = capture.to_parent([self.rig.x, self.rig.y, self.rig.z])[:2]
        for i, a in enumerate(mouths):
            for b in mouths[i + 1:]:
                if frozenset((a.id, b.id)) in paired \
                        or not low <= np.hypot(*(a.xy - b.xy)) <= high:
                    continue
                # Out of the stall: square to its mouth, to the camera's side
                along = (a.xy - b.xy) / np.hypot(*(a.xy - b.xy))
                out = np.array([along[1], -along[0]])
                if out @ (camera - (a.xy + b.xy) / 2.0) < 0:
                    out = -out
                # +v is to the right of out: (-out_y, out_x)
                plus, minus = (a, b) if along @ np.array([-out[1], out[0]]) > 0 else (b, a)
                self.tracks.append(StallTrack(self.next_id, plus, minus, out,
                                              self.config["window"]))
                self.next_id += 1

    def fit(self, track):
        """The stall of a track, from its ends as they are now."""
        config = self.config
        a, b = track.plus.xy, track.minus.xy
        middle = (a + b) / 2.0
        # The row's direction: the line through every mouth end on it
        row = np.array([e.xy for e in self.mouths()
                        if np.hypot(*(e.xy - middle)) <= config["row_reach_m"]])
        along = (a - b) / np.hypot(*(a - b))
        if len(row) > 2:
            beside = np.abs((row - middle) @ np.array([along[1], -along[0]]))
            row = row[beside <= config["row_gate_m"]]
        if len(row) > 2:
            _, _, axes = np.linalg.svd(row - row.mean(axis=0))
            along = axes[0] if axes[0] @ along > 0 else -axes[0]
        # +v, from the end at -v to the one at +v, is to the right of out
        out = np.array([along[1], -along[0]])

        def backs_behind(out):
            """The ends at the back of its two dividers, with how deep the
            stall is at each, were it to open towards `out`."""
            found = []
            for end in self.ends:
                if end.hits < config["end_hits"] or not end.at_the_back:
                    continue
                for corner in (a, b):
                    depth = -float((end.xy - corner) @ out)
                    if config["length_m"][0] <= depth <= config["length_m"][1] \
                            and abs(float((end.xy - corner) @ along)) <= config["row_gate_m"]:
                        found.append((end, depth))
            return found

        backs = backs_behind(out)
        if not backs and backs_behind(-out):
            # Its back is on the side it was taken to open to: a row seen
            # from behind. It opens the other way.
            track.plus, track.minus = track.minus, track.plus
            along, out = -along, -out
            backs = backs_behind(out)
        track.out = out
        track.backs = tuple(end for end, _ in backs)
        length = float(np.mean([depth for _, depth in backs])) if backs \
            else config["length_default_m"]
        centre = middle - length / 2.0 * out
        track.fitted = Stall(centre[0], centre[1], math.degrees(math.atan2(out[1], out[0])),
                             length, float(np.hypot(*(a - b))))

    def vote(self, stalls, capture):
        config = self.config
        ground_z = capture.z + self.rig.ground_z
        boxes = []
        for track in self.tracks:
            box = self.rig.visible_box(capture.from_parent(track.fitted.corners(ground_z)))
            if box is not None and box[3] - box[1] >= config["vote_height_px"]:
                boxes.append((track, box))
        best = {}  # track id -> (overlap, occupied): one vote a frame
        for detection in stalls:
            if float(detection["confidence"]) < config["confidence"]:
                continue
            b = detection["bbox"]
            mine = (float(b["x"]), float(b["y"]), float(b["x"]) + float(b["w"]),
                    float(b["y"]) + float(b["h"]))
            overlaps = []
            for track, box in boxes:
                x = min(mine[2], box[2]) - max(mine[0], box[0])
                y = min(mine[3], box[3]) - max(mine[1], box[1])
                if x > 0 and y > 0:
                    union = (mine[2] - mine[0]) * (mine[3] - mine[1]) \
                        + (box[2] - box[0]) * (box[3] - box[1]) - x * y
                    overlaps.append((x * y / union, track))
            if not overlaps:
                continue
            overlap, track = max(overlaps, key=lambda o: o[0])
            if overlap >= config["vote_overlap"] and overlap > best.get(track.id, (0.0,))[0]:
                best[track.id] = (overlap, detection["class"] == "occupied")
        for track in self.tracks:
            if track.id in best:
                track.classes.append(1 if best[track.id][1] else 0)
                track.hits += 1
        return sorted(best)

    def get(self, track_id):
        return next((t for t in self.tracks if t.id == track_id), None)

    def beside(self, track, side):
        """The track of the stall next to this one, on the side of +v
        (side 1) or -v (side -1): the one that shares that mouth end."""
        shared = track.plus if side > 0 else track.minus
        return next((t for t in self.tracks if t is not track
                     and (t.plus is shared or t.minus is shared)), None)

    def summary(self):
        """Every stall followed, for the log and for a viewer."""
        return [dict(track.fitted.as_dict(), id=track.id, hits=track.hits,
                     occupied_share=round(track.occupied_share(), 2),
                     free=track.free(self.config), backs=len(track.backs),
                     nearest_m=round(min(track.plus.nearest_m, track.minus.nearest_m), 2))
                for track in self.tracks if track.fitted is not None]

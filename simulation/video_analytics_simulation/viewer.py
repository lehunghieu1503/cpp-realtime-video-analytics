"""A window with the run's cameras, for a person watching: the simulator's
own images, with what the device found in them drawn on top.

    front camera                      rear camera (scenarios that park)
    boxes, track ids, lane lines      stalls: green empty, red occupied
                                      ends of dividers: crosses
                                      the stalls the controller follows, and
                                      in yellow the one it parks in
    one line: tick, state, phase, speed, what is applied to the vehicle

An image is shown with the result of its own frame, joined by sim_ns, so it
is as late as that result. Until the first result an image is shown as it
is. Nothing in the control path reads this module; without --view it is
never imported.
"""

from collections import OrderedDict

import numpy as np

WINDOW = "CARLA cameras"
KEPT = 45  # images a late result can still be joined with


class Viewer:
    def __init__(self, rigs, pose_at, scale=0.5):
        """rigs: name -> Rig, in the order shown. pose_at(sim_ns): the ego
        pose of that frame as calibration.Pose, or None."""
        import cv2
        self.cv2 = cv2
        self.rigs = rigs
        self.pose_at = pose_at
        self.scale = scale
        self.images = {name: OrderedDict() for name in rigs}
        self.results = {name: None for name in rigs}
        self.panels = {name: np.zeros((int(r.height * scale), int(r.width * scale), 3),
                                      np.uint8) for name, r in rigs.items()}
        self.park = None
        self.state = ""
        self.closed = False
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

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
        cv2, rig, pose = self.cv2, self.rigs[name], self.pose_at(sim_ns)
        if pose is None:
            return
        for row in self.park.get("stalls", []):
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
        cv2 = self.cv2
        row = np.hstack(list(self.panels.values()))
        bar = np.zeros((28, row.shape[1], 3), np.uint8)
        gear = "R" if applied.get("reverse") else "D"
        text = "tick %d  %s  %s%4.1f km/h  %s  throttle %.2f  brake %.2f  steer %+.2f  %s" % (
            tick, self.state, "park: %s  " % self.park["phase"] if self.park else "",
            speed_mps * 3.6, gear, applied.get("throttle", 0.0), applied.get("brake", 0.0),
            applied.get("steer", 0.0), applied.get("reason", ""))
        cv2.putText(bar, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.imshow(WINDOW, np.vstack([row, bar]))
        if cv2.waitKey(1) & 0xFF == ord("q") \
                or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            self.close()
        return not self.closed

    def pump(self):
        """Lets the window answer while nothing new is drawn (the run is
        waiting for the device)."""
        if not self.closed:
            self.cv2.waitKey(1)

    def close(self):
        if not self.closed:
            self.closed = True
            self.cv2.destroyWindow(WINDOW)

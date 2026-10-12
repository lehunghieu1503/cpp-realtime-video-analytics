"""What the simulator and the controller agree on, without CARLA or ROS:
frame identity, the frame ledger, the caches, and the command envelope with
its lease.

A frame is identified by (run_id, source, sim_ns). sim_ns is the simulator's
time of the frame in nanoseconds, computed once, and is also the timestamp
the frame travels with. Nothing is ever joined by arrival order, by a local
counter or by the nearest time.
"""

import json
import math
import os
import re
import secrets
import threading
import time
from collections import OrderedDict

RUN_ID = re.compile(r"^[A-Za-z0-9_]{1,64}$")
HAZARD_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")


def new_run_id():
    return "r_%s_%s" % (time.strftime("%Y%m%d_%H%M%S"), secrets.token_hex(2))


def sim_ns_of(seconds):
    """CARLA's elapsed seconds as integer nanoseconds."""
    return int(round(seconds * 1e9))


def topics(run_id, source):
    source = re.sub(r"\W", "_", source)
    base = "simulation/%s" % run_id
    return {
        "perception": "%s/%s/perception" % (base, source),
        "depth": "%s/%s/depth" % (base, source),
        "camera_info": "%s/%s/camera_info" % (base, source),
        "metric": "%s/%s/metric_perception" % (base, source),
        "ego": base + "/ego",
        "control": base + "/control",
        "arm": base + "/arm",
        "status": base + "/status",
    }


class Ledger:
    """Append-only record of every frame the camera produced.

    A frame is registered when its image arrives, before any queue can drop
    it; what happens to it afterwards is appended under the same identity.
    A frame that was produced and then lost stays in the file, so it stays
    in the denominator of every rate computed from it. A write that fails
    raises: a run with an incomplete ledger must not look complete.
    """

    def __init__(self, path):
        self.path = path
        self.file = open(path, "x")
        self.lock = threading.Lock()
        self.registered = set()

    def write(self, record):
        with self.lock:
            self.file.write(json.dumps(record, separators=(",", ":")) + "\n")
            self.file.flush()

    def register(self, frame, sim_ns, mono_ns, **more):
        self.registered.add(frame)
        self.write(dict(event="rgb", frame=frame, sim_ns=sim_ns, mono_ns=mono_ns,
                        **more))

    def update(self, frame, event, **more):
        self.write(dict(event=event, frame=frame, **more))

    def close(self):
        with self.lock:
            self.file.close()


def read_ledger(path):
    """Ledger events merged per frame, and the ticks.

    frames[frame] has the fields of every event of that frame, `registered`
    (its RGB image arrived) and `status`: the last thing that happened to it,
    "rgb" (registered only), "paired", "sent", or "dropped" with a `reason`.
    ticks[frame] is what the simulator expected to produce.
    """
    frames = {}
    ticks = {}
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            event = record.pop("event")
            frame = record.pop("frame")
            if event == "tick":
                ticks[frame] = record
                continue
            entry = frames.setdefault(frame, {"status": None, "registered": False})
            entry.update(record)
            if event == "rgb":
                entry["registered"] = True
                entry["status"] = "rgb"
            elif event not in ("depth", "artifact"):
                entry["status"] = event
    return frames, ticks


class LiveCache:
    """The frames a late result can still be joined with: bounded, keyed by
    sim_ns. An entry that was pushed out makes its result unusable; it does
    not disappear from the ledger."""

    def __init__(self, max_entries):
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.evicted = 0

    def put(self, sim_ns, entry):
        self.entries[sim_ns] = entry
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)
            self.evicted += 1

    def get(self, sim_ns):
        return self.entries.get(sim_ns)

    def __contains__(self, sim_ns):
        return sim_ns in self.entries

    def __len__(self):
        return len(self.entries)


class Pairer:
    """Pairs the RGB and the depth image of the same simulator frame.

    By the frame number each image carries, with equal timestamps, never by
    the order the callbacks ran in. Holds a bounded number of frames waiting
    for their other half; one that waited too long is returned as lost.
    """

    def __init__(self, max_waiting=8):
        self.max_waiting = max_waiting
        self.waiting = OrderedDict()

    def add(self, kind, frame, sim_ns, data):
        """Returns (pair, lost): pair is (frame, sim_ns, rgb, depth) or None;
        lost is a list of (frame, reason)."""
        lost = []
        other = "depth" if kind == "rgb" else "rgb"
        entry = self.waiting.setdefault(frame, {})
        if kind in entry:
            lost.append((frame, "duplicate_" + kind))
            return None, lost
        entry[kind] = (sim_ns, data)
        pair = None
        if other in entry:
            del self.waiting[frame]
            if entry[other][0] != sim_ns:
                lost.append((frame, "timestamp_mismatch"))
            else:
                pair = (frame, sim_ns, entry["rgb"][1], entry["depth"][1])
        while len(self.waiting) > self.max_waiting:
            old, halves = self.waiting.popitem(last=False)
            lost.append((old, "missing_depth" if "rgb" in halves else "missing_rgb"))
        return pair, lost


# ── Command envelope ─────────────────────────────────────────────────────────

COMMAND_FIELDS = ("run_id", "source", "command_seq", "basis_sim_ns",
                  "control_sim_ns", "control_frame", "valid_until_sim_ns",
                  "throttle", "brake", "steer", "reason")
# May be left out, and are then false
OPTIONAL_FLAGS = ("reverse", "hand_brake")


def is_driving(command):
    """Whether a command does more than brake. Reverse gear and the parking
    brake count whatever the pedals say: a command with either of them goes
    through every check a command that moves the vehicle goes through."""
    return (command["throttle"] > 0.0 or command["brake"] < 1.0
            or command.get("reverse") is True or command.get("hand_brake") is True)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def check_command(command, run_id, source, last_seq, now_sim_ns, limits,
                  known_basis, known_control):
    """Empty when the vehicle may act on this command; the reason otherwise.

    known_basis(sim_ns) and known_control(sim_ns, frame) say whether the
    simulator produced a camera frame, and an ego state, with that identity.
    limits: age_limit_ns (perception the command rests on) and
    pose_budget_ns (ego state it was computed for).
    """
    if not isinstance(command, dict):
        return "not_an_object"
    for field in COMMAND_FIELDS:
        if field not in command:
            return "missing_" + field
    if command["run_id"] != run_id:
        return "wrong_run"
    if command["source"] != source:
        return "wrong_source"
    for field in ("command_seq", "basis_sim_ns", "control_sim_ns", "control_frame",
                  "valid_until_sim_ns"):
        if not _is_int(command[field]) or command[field] < 0:
            return "bad_" + field
    for field in ("throttle", "brake", "steer"):
        if not _is_number(command[field]):
            return "bad_" + field
    if not 0.0 <= command["throttle"] <= 1.0 or not 0.0 <= command["brake"] <= 1.0:
        return "out_of_range"
    if not -1.0 <= command["steer"] <= 1.0:
        return "out_of_range"
    if command["throttle"] > 0.0 and command["brake"] > 0.0:
        return "throttle_and_brake"
    for field in OPTIONAL_FLAGS:
        if field in command and not isinstance(command[field], bool):
            return "bad_" + field
    if not isinstance(command["reason"], str):
        return "bad_reason"
    if command["command_seq"] <= last_seq:
        return "old_sequence"
    if command["control_sim_ns"] > now_sim_ns or command["basis_sim_ns"] > now_sim_ns:
        return "from_the_future"
    if not known_control(command["control_sim_ns"], command["control_frame"]):
        return "unknown_control_pose"
    if now_sim_ns - command["control_sim_ns"] > limits["pose_budget_ns"]:
        return "stale_control_pose"
    if is_driving(command):
        # A command that lets the vehicle move must rest on a camera frame
        # this simulator produced, and recent enough
        if not known_basis(command["basis_sim_ns"]):
            return "unknown_basis"
        if command["valid_until_sim_ns"] > command["basis_sim_ns"] + limits["age_limit_ns"]:
            return "validity_too_long"
        if now_sim_ns > command["valid_until_sim_ns"]:
            return "expired"
    return ""


BRAKE = {"throttle": 0.0, "brake": 1.0, "steer": 0.0, "reverse": False,
         "hand_brake": False}


class Lease:
    """The simulator's own decision on what to apply at each tick.

    It does not trust the controller to be alive or right: without an armed
    run, a valid command that is still in force in simulation time, and a
    command received recently in wall time, it brakes. A rejected command
    does not extend anything.

    Every brake it decides on itself is the foot brake alone: reverse gear
    and the parking brake are applied only from a command that passed as
    one that drives. What a command gives as its reason decides nothing.
    """

    def __init__(self, run_id, source, limits):
        self.run_id = run_id
        self.source = source
        self.limits = limits  # age_limit_ns, pose_budget_ns, wall_lease_ns
        self.armed = False
        self.stopped = False  # latched until the episode ends
        self.last_seq = -1
        self.command = None
        self.received_mono_ns = None
        self.rejected = {}

    def arm(self, message):
        """An operator's message for this run: enable, or stop."""
        if not isinstance(message, dict) or message.get("run_id") != self.run_id:
            return "wrong_run"
        if message.get("stop") is True:
            self.stopped = True
            self.armed = False
            return ""
        if message.get("enable") is not True:
            return "not_an_arm_message"
        if self.stopped:
            return "stopped_until_reset"
        self.armed = True
        return ""

    def offer(self, command, now_sim_ns, mono_ns, known_basis, known_control):
        reason = check_command(command, self.run_id, self.source, self.last_seq,
                               now_sim_ns, self.limits, known_basis, known_control)
        if reason:
            self.rejected[reason] = self.rejected.get(reason, 0) + 1
            return reason
        self.last_seq = command["command_seq"]
        self.command = command
        self.received_mono_ns = mono_ns
        return ""

    def control(self, now_sim_ns, mono_ns):
        """(throttle, brake, steer, reason, command_seq) to apply now."""
        def brake(reason, seq=None):
            return dict(BRAKE, reason=reason, command_seq=seq)

        if self.stopped:
            return brake("stopped")
        command = self.command
        if command is None:
            return brake("no_command")
        seq = command["command_seq"]
        if mono_ns - self.received_mono_ns > self.limits["wall_lease_ns"]:
            return brake("controller_silent", seq)
        if not is_driving(command):
            return dict(BRAKE, steer=command["steer"],
                        reason=command["reason"] or "commanded_brake",
                        command_seq=seq)
        if not self.armed:
            return brake("not_armed", seq)
        if now_sim_ns > command["valid_until_sim_ns"]:
            return brake("command_expired", seq)
        reverse = command.get("reverse") is True
        # Standing on the brake, in a forward gear: held, not driven. The
        # log keeps the controller's word for why (it is "parked").
        held = command["throttle"] == 0.0 and command["brake"] >= 1.0 and not reverse
        return dict(throttle=command["throttle"], brake=command["brake"],
                    steer=command["steer"], reverse=reverse,
                    hand_brake=command.get("hand_brake") is True,
                    reason=(command["reason"] or "held") if held else "",
                    command_seq=seq)


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)

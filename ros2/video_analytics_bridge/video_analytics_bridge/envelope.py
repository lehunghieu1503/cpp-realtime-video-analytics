"""What the bridge checks before it republishes a simulation frame.

A result of a simulation carries the identity of the simulator's frame:

    {"stream": "carla-front", "frame_id": 42,
     "simulation": {"run_id": "r_20261011_ab12", "sim_ns": 1533333333},
     "image": {"width": 1280, "height": 720},
     "detections": [...], "lane": {...}}

No ROS import here, so the checks can be tested without ROS.
"""
import math
import re

RUN_ID = re.compile(r"^[A-Za-z0-9_]{1,64}$")
NS_PER_S = 1000000000


def topic_name(source):
    """A source id as part of a topic name: letters, digits and underscores."""
    return re.sub(r"\W", "_", source)


def is_int(value):
    # bool is an int in Python; true is not a number of nanoseconds
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def check(frame):
    """Empty when the frame can be published as a simulation frame; the
    reason otherwise."""
    if not isinstance(frame, dict):
        return "not an object"
    simulation = frame.get("simulation")
    if not isinstance(simulation, dict):
        return "no simulation block"
    run_id = simulation.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID.match(run_id):
        return "bad run_id"
    sim_ns = simulation.get("sim_ns")
    # A float would have lost nanoseconds on the way
    if not is_int(sim_ns) or sim_ns < 0:
        return "bad sim_ns"
    if not isinstance(frame.get("stream"), str) or not frame["stream"]:
        return "no stream"
    image = frame.get("image")
    if (not isinstance(image, dict) or not is_int(image.get("width"))
            or not is_int(image.get("height")) or image["width"] <= 0
            or image["height"] <= 0):
        return "bad image size"
    detections = frame.get("detections")
    if not isinstance(detections, list):
        return "no detections list"
    for item in detections:
        if not isinstance(item, dict) or not isinstance(item.get("class"), str):
            return "bad detection"
        box = item.get("bbox")
        if not isinstance(box, dict) or not all(is_number(box.get(k)) for k in "xywh"):
            return "bad bbox"
        if not is_number(item.get("confidence")):
            return "bad confidence"
        if "track_id" in item and not is_int(item["track_id"]):
            return "bad track_id"
    lane = frame.get("lane")
    if lane is not None:
        if not isinstance(lane, dict) or not isinstance(lane.get("valid"), bool):
            return "bad lane"
        if lane["valid"]:
            for side in ("left", "right", "centerline"):
                curve = lane.get(side)
                if (not isinstance(curve, dict)
                        or not isinstance(curve.get("coefficients"), list)
                        or len(curve["coefficients"]) != 3
                        or not all(is_number(v) for v in curve["coefficients"])
                        or not is_number(curve.get("x_min_m"))
                        or not is_number(curve.get("x_max_m"))):
                    return "bad lane curve"
            if not all(is_number(lane.get(k)) for k in (
                    "lateral_error_m", "heading_rad", "curvature_inv_m",
                    "reference_x_m")):
                return "bad lane numbers"
    return ""


def stamp(sim_ns):
    """Simulation nanoseconds as a ROS time: (sec, nanosec), nothing rounded."""
    return sim_ns // NS_PER_S, sim_ns % NS_PER_S


def topics(frame):
    """Topic names of a checked simulation frame."""
    base = "simulation/%s/%s" % (frame["simulation"]["run_id"],
                                 topic_name(frame["stream"]))
    return base + "/perception", base + "/lane"


def optical_frame(frame):
    """header.frame_id: the camera's optical frame of this run."""
    return "%s/%s_optical" % (frame["simulation"]["run_id"],
                              topic_name(frame["stream"]))

"""CARLA adapter: owns the simulation clock, the ego vehicle and its sensors.

Each tick, in this order:

  1. apply to the vehicle what the lease allows (see protocol.Lease): the
     controller's command if it is valid, armed and fresh, full brake
     otherwise
  2. world.tick(): the simulation advances 1/30 s. It never waits for the
     device's result; a late result means a brake, not a paused world
  3. take the RGB and depth images of that tick, pair them by frame number
  4. send the RGB image to the device as one JPEG with timestamp sim_ns
     (appsrc ! gdppay ! tcpserversink), publish depth, camera info, the ego
     state and /clock on ROS 2, and write the run's records

What the controller may use: RGB (through the device), depth, the camera
calibration and the ego vehicle's own pose and velocity. The positions of
the other actors and the map's lane centre go to truth.jsonl only, for the
evaluation; nothing in the control path reads them.

Records in the run directory:
  manifest.json   versions, map, rig, vehicle, settings of this run
  frames.jsonl    the ledger: every tick and every image, and what became of it
  ego.jsonl       ego state per tick (what a replay needs with the depth)
  depth/          lossless depth of paired frames (bounded, see the config)
  truth.jsonl     other actors, lane centre, collisions: evaluation only
  controls.jsonl  what was applied at each tick and why
"""

import argparse
import array
import json
import math
import os
import queue
import signal
import sys
import threading
import time
import zlib

import numpy as np
import yaml

from . import protocol
from .calibration import Rig, decode_depth

SOURCE = "carla-front"


def mono_ns():
    return time.monotonic_ns()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def scenario_config(path, name):
    with open(path) as f:
        data = yaml.safe_load(f)
    if name not in data["scenarios"]:
        raise SystemExit("unknown scenario %s; have: %s" % (
            name, ", ".join(sorted(data["scenarios"]))))
    return dict(data.get("defaults", {}), **data["scenarios"][name])


def pose_dict(transform):
    l, r = transform.location, transform.rotation
    return {"x": l.x, "y": l.y, "z": l.z, "roll": r.roll, "pitch": r.pitch,
            "yaw": r.yaw}


# ── Sending the image ────────────────────────────────────────────────────────

class JpegSender:
    """One JPEG per buffer, timestamped with sim_ns, in GDP framing over TCP.

    The device connects as a client. A client that cannot keep up is
    disconnected by the sink (buffers-max); bytes of a frame are never
    dropped from the middle of the stream. A frame is dropped whole, here,
    before it is encoded, when the previous ones have not left yet.
    """

    def __init__(self, bind, port, max_queued, client_max_buffers):
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        self.Gst = Gst
        Gst.init(None)
        for element in ("appsrc", "gdppay", "tcpserversink"):
            if Gst.ElementFactory.find(element) is None:
                raise SystemExit("GStreamer element %s is not installed" % element)
        self.pipeline = Gst.parse_launch(
            "appsrc name=carla_src is-live=true format=time do-timestamp=false "
            "caps=image/jpeg,framerate=30/1 ! gdppay ! "
            "tcpserversink name=sink host=%s port=%d sync=false async=false "
            "buffers-max=%d recover-policy=none" % (bind, port, client_max_buffers))
        self.source = self.pipeline.get_by_name("carla_src")
        self.sink = self.pipeline.get_by_name("sink")
        self.max_queued = max_queued
        self.pipeline.set_state(Gst.State.PLAYING)

    def clients(self):
        return self.sink.get_property("num-handles")

    def send(self, jpeg, sim_ns):
        """False when the frame was not handed over (queue full)."""
        if self.source.get_property("current-level-buffers") >= self.max_queued:
            return False
        buffer = self.Gst.Buffer.new_wrapped(jpeg)
        buffer.pts = sim_ns
        buffer.dts = self.Gst.CLOCK_TIME_NONE
        return self.source.emit("push-buffer", buffer) == self.Gst.FlowReturn.OK

    def close(self):
        self.pipeline.set_state(self.Gst.State.NULL)


class DepthWriter:
    """Writes the raw depth of paired frames, losslessly, off the tick thread.

    Encoding "carla_bgr24_zlib": the B, G, R bytes of CARLA's depth image as
    they came, zlib level 1. calibration.decode_depth() of them is the depth
    the live run used.
    """

    def __init__(self, directory, max_frames, ledger):
        self.directory = directory
        self.remaining = max_frames
        self.ledger = ledger
        self.queue = queue.Queue(maxsize=8)
        self.bytes = 0
        self.seconds = 0.0
        if max_frames > 0:
            os.makedirs(directory)
        # zlib takes about 30 ms a frame, one tick: two threads keep up
        self.threads = [threading.Thread(target=self.run, daemon=True)
                        for _ in range(2)]
        for thread in self.threads:
            thread.start()

    def offer(self, frame, sim_ns, bgra):
        if self.remaining <= 0:
            return
        self.remaining -= 1
        try:
            self.queue.put_nowait((frame, sim_ns, bgra))
        except queue.Full:
            self.ledger.update(frame, "artifact", depth_artifact=None,
                               depth_artifact_error="writer_queue_full")

    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            frame, sim_ns, bgra = item
            start = time.monotonic()
            raw = np.ascontiguousarray(bgra[:, :, :3]).tobytes()
            name = "%d.bgr24.z" % sim_ns
            try:
                with open(os.path.join(self.directory, name), "xb") as f:
                    data = zlib.compress(raw, 1)
                    f.write(data)
                self.bytes += len(data)
                self.ledger.update(frame, "artifact", depth_artifact="depth/" + name,
                                   depth_crc32=zlib.crc32(raw),
                                   depth_shape=[bgra.shape[0], bgra.shape[1]])
            except OSError as error:
                self.ledger.update(frame, "artifact", depth_artifact=None,
                                   depth_artifact_error=str(error))
            self.seconds += time.monotonic() - start

    def close(self):
        for thread in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join(timeout=30)


# ── Scripted actors (the scene; their state is truth, for evaluation only) ───

class ScriptedActor:
    """Another road user, moved along a lane by a script, without physics:
    the same run gives the same positions."""

    def __init__(self, carla, world, base_waypoint, spec, dt, index=0):
        self.carla = carla
        self.spec = spec
        self.dt = dt
        self.name = spec.get("name", spec["class"])
        self.cls = spec["class"]
        self.base = base_waypoint
        library = world.get_blueprint_library()
        blueprint = library.find(spec["blueprint"])
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "true")
        self.placements = spec.get("sweep", {}).get("distances_m")
        self.hold = spec.get("sweep", {}).get("hold_ticks", 30)
        self.s = float(spec.get("ahead_m", self.placements[0] if self.placements else 20.0))
        self.speed = float(spec.get("speed_mps", 0.0))
        self.lane_offset = float(spec.get("lane", 0))
        self.visible = False
        # physics: true lets the actor hit things (the collision negative
        # control); it is then driven by a velocity, not placed
        self.physics = bool(spec.get("physics", False))
        start = self.transform_at(0)[0]
        if not self.physics:
            # Scripted actors may start at the same place (one after another
            # in a sweep): each is created at a height of its own and put in
            # place by its first step
            start.location.z += 50.0 + 10.0 * index
        self.actor = world.spawn_actor(blueprint, start)
        self.actor.set_simulate_physics(self.physics)
        self.extent = self.actor.bounding_box.extent
        self.box_centre = self.actor.bounding_box.location
        self.previous_velocity = (0.0, 0.0)

    def lane_position(self, tick):
        """Lane index relative to the ego lane (fractional during a cut-in)."""
        cut = self.spec.get("cut_in")
        if not cut or tick < cut["start_tick"]:
            return float(self.spec.get("lane", 0))
        progress = min(1.0, (tick - cut["start_tick"]) / float(cut["duration_ticks"]))
        start = float(self.spec.get("lane", 0))
        # Smooth step: no lateral jump at either end
        progress = progress * progress * (3 - 2 * progress)
        return start + (float(cut["to_lane"]) - start) * progress

    def transform_at(self, tick):
        spec = self.spec
        if "at" in spec:
            # A place in the world, not along the ego vehicle's lane
            at = spec["at"]
            return self.carla.Transform(
                self.carla.Location(float(at["x"]), float(at["y"]),
                                    float(at.get("z", 0.0)) + float(spec.get("z_m", 0.3))),
                self.carla.Rotation(yaw=float(at.get("yaw_deg", 0.0)))), 0.0
        if self.placements:
            # Placements are counted from the tick the actor appears
            since = max(0, tick - self.spec.get("from_tick", 0))
            index = min(since // self.hold, len(self.placements) - 1)
            s = float(self.placements[index])
            speed = 0.0
        else:
            start = spec.get("move_from_tick", 0)
            t = max(0, tick - start) * self.dt
            accel = float(spec.get("accel_mps2", 0.0))
            v0 = float(spec.get("speed_mps", 0.0))
            v_max = float(spec.get("max_speed_mps", 1e9))
            v_min = float(spec.get("min_speed_mps", 0.0))
            speed = min(v_max, max(v_min, v0 + accel * t))
            # Time at which the speed limit is reached; constant after it
            if accel != 0.0:
                t_limit = max(0.0, ((v_max if accel > 0 else v_min) - v0) / accel)
            else:
                t_limit = float("inf")
            t_a = min(t, t_limit)
            s = float(spec.get("ahead_m", 20.0)) + v0 * t_a + 0.5 * accel * t_a * t_a \
                + speed * max(0.0, t - t_limit)
        waypoints = self.base.next(max(0.1, s))
        waypoint = waypoints[0] if waypoints else self.base
        transform = waypoint.transform
        yaw = math.radians(transform.rotation.yaw)
        lateral = self.lane_position(tick) * waypoint.lane_width + \
            float(spec.get("lateral_m", 0.0))
        location = self.carla.Location(
            transform.location.x - math.sin(yaw) * lateral,
            transform.location.y + math.cos(yaw) * lateral,
            transform.location.z + float(spec.get("z_m", 0.3 if self.cls != "person" else 1.0)))
        rotation = self.carla.Rotation(yaw=transform.rotation.yaw + float(spec.get("yaw_deg", 0.0)))
        return self.carla.Transform(location, rotation), speed

    def step(self, tick):
        """The command that puts the actor where it is at this tick."""
        tick = max(0, tick)  # before the run starts: the first placement
        spec = self.spec
        present = spec.get("from_tick", 0) <= tick < spec.get("until_tick", 10 ** 9)
        transform, speed = self.transform_at(tick)
        self.visible = present
        self.speed = speed
        command = self.carla.command
        if self.physics:
            yaw = math.radians(self.actor.get_transform().rotation.yaw)
            v = float(spec.get("speed_mps", 0.0)) if tick >= spec.get("move_from_tick", 0) else 0.0
            return command.ApplyTargetVelocity(self.actor, self.carla.Vector3D(
                v * math.cos(yaw), v * math.sin(yaw), 0.0))
        if not present:
            # Parked far below the road instead of destroyed: ids stay stable
            transform.location.z -= 500.0
        return command.ApplyTransform(self.actor, transform)

    def truth(self, snapshot_actor):
        """State at the tick just simulated, from the simulator's snapshot."""
        t = snapshot_actor.get_transform()
        return {"id": self.actor.id, "name": self.name, "class": self.cls,
                "present": self.visible, "pose": pose_dict(t),
                "speed_mps": self.speed,
                "extent": [self.extent.x, self.extent.y, self.extent.z],
                "box_centre": [self.box_centre.x, self.box_centre.y, self.box_centre.z]}


# ── The adapter ──────────────────────────────────────────────────────────────

class Simulator:
    def __init__(self, args):
        import carla
        self.carla = carla
        self.args = args
        self.config = load_config(args.config)
        self.scenario = scenario_config(args.scenarios, args.scenario)
        self.c = self.config["carla"]
        self.run_id = args.run_id
        if not protocol.RUN_ID.match(self.run_id):
            raise SystemExit("bad run id")
        self.run_dir = args.run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self.rig = Rig.from_config(self.config["calibration"])
        self.dt = 1.0 / float(self.c.get("fps", 30))
        # A scenario may lower it: a scene with more detail makes larger frames
        self.jpeg_quality = int(self.scenario.get("jpeg_quality",
                                                  self.c.get("jpeg_quality", 75)))
        self.stop_requested = False
        self.actors = []
        self.sensors = []
        self.scripted = []
        self.original_settings = None
        self.world = None

        # Refuses a directory that already holds a run
        self.ledger = protocol.Ledger(os.path.join(self.run_dir, "frames.jsonl"))
        self.ego_log = open(os.path.join(self.run_dir, "ego.jsonl"), "x")
        self.truth_log = open(os.path.join(self.run_dir, "truth.jsonl"), "x")
        self.control_log = open(os.path.join(self.run_dir, "controls.jsonl"), "x")
        self.depth_writer = DepthWriter(
            os.path.join(self.run_dir, "depth"),
            int(self.scenario.get("depth_artifact_frames",
                                  self.c.get("depth_artifact_frames", 0))),
            self.ledger)

        control = self.config["control"]
        self.limits = {
            "age_limit_ns": int(control["age_limit_ms"] * 1e6),
            "pose_budget_ns": int(control["pose_budget_ms"] * 1e6),
            "wall_lease_ns": int(control["wall_lease_ms"] * 1e6),
        }
        self.lease = protocol.Lease(self.run_id, SOURCE, self.limits)
        self.sent = protocol.LiveCache(int(self.c.get("live_cache_entries", 90)))
        self.ego_states = protocol.LiveCache(int(self.c.get("live_cache_entries", 90)))
        self.pairer = protocol.Pairer()
        self.image_queue = queue.Queue(maxsize=int(self.c.get("callback_queue", 8)))
        self.output_queue = queue.Queue(maxsize=2)
        if int(self.scenario.get("rgb_artifact_every",
                                 self.c.get("rgb_artifact_every", 0))):
            os.makedirs(os.path.join(self.run_dir, "rgb"))
        self.events = []  # collisions and lane invasions, from sensor threads
        self.events_lock = threading.Lock()
        self.pending_commands = []
        self.pending_arm = []
        self.controller_ready = False
        self.controller_frames = 0
        self.arm_sent = 0.0  # when ARM was last sent
        self.stats = {"ticks": 0, "sent": 0, "dropped": 0, "jpeg_bytes": 0,
                      "encode_ms": 0.0, "tick_ms": 0.0, "wait_images_ms": 0.0}

    # ── ROS ──────────────────────────────────────────────────────────────────

    def start_ros(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import String
        self.rclpy = rclpy
        self.String, self.Image, self.CameraInfo, self.Clock = String, Image, CameraInfo, Clock
        rclpy.init()
        self.node = rclpy.create_node("video_analytics_simulator")
        names = protocol.topics(self.run_id, SOURCE)
        self.topic_names = names
        # Depth is large: keep one, the consumer joins by timestamp anyway
        sensor = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE,
                            history=HistoryPolicy.KEEP_LAST)
        self.pub_depth = self.node.create_publisher(Image, names["depth"], sensor)
        self.pub_info = self.node.create_publisher(CameraInfo, names["camera_info"], 5)
        self.pub_ego = self.node.create_publisher(String, names["ego"], 10)
        self.pub_arm = self.node.create_publisher(String, names["arm"], 10)
        self.pub_control = self.node.create_publisher(String, names["control"], 10)
        self.pub_clock = self.node.create_publisher(Clock, "/clock", 10)
        self.node.create_subscription(
            String, names["control"],
            lambda m: self.pending_commands.append((m.data, mono_ns())), 20)
        self.node.create_subscription(
            String, names["arm"], lambda m: self.pending_arm.append(m.data), 10)
        self.node.create_subscription(String, names["status"], self.on_status, 10)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def on_status(self, message):
        try:
            status = json.loads(message.data)
        except ValueError:
            return
        if status.get("run_id") != self.run_id:
            return
        self.controller_ready = status.get("ready") is True
        self.controller_frames = int(status.get("perception_frames", 0))

    def spin(self):
        # Everything that is waiting, without blocking the tick
        for _ in range(12):
            self.executor.spin_once(timeout_sec=0)

    # ── CARLA ────────────────────────────────────────────────────────────────

    def connect(self):
        carla = self.carla
        self.client = carla.Client(self.c.get("host", "127.0.0.1"),
                                   int(self.c.get("port", 2000)))
        self.client.set_timeout(60.0)
        self.server_version = self.client.get_server_version()
        client_version = self.client.get_client_version()
        if self.server_version != client_version:
            raise SystemExit("CARLA server %s and client %s differ" % (
                self.server_version, client_version))
        town = self.scenario.get("map", self.c.get("map", "Town04"))
        self.world = self.client.get_world()
        if self.world.get_map().name.split("/")[-1] != town:
            self.client.load_world(town)
            # The map before leaves its textures in video memory until the
            # next load, and the server starts with its largest map: with
            # them, the device's models did not fit beside it on 8 GB
            self.world = self.client.reload_world()
        else:
            # The map is kept: after some twenty loads in a row the server
            # rendered wrong colours. What an earlier run left behind (it
            # was killed) is removed instead.
            for actor in self.world.get_actors():
                if actor.type_id.startswith(("vehicle.", "walker.", "sensor.")):
                    actor.destroy()
        self.original_settings = self.world.get_settings()
        # A run that was killed left the world waiting for ticks
        self.original_settings.synchronous_mode = False
        self.original_settings.fixed_delta_seconds = None
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.dt
        # fixed_delta_seconds <= max_substep_delta_time * max_substeps
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        self.world.apply_settings(settings)
        self.world.set_weather(getattr(carla.WeatherParameters,
                                       self.scenario.get("weather", "ClearNoon")))
        self.map = self.world.get_map()

    def spawn(self):
        carla = self.carla
        library = self.world.get_blueprint_library()
        route = self.config["routes"][self.scenario.get("route", "straight")]
        waypoint = self.map.get_waypoint(
            carla.Location(route["x"], route["y"], 0.0), project_to_road=True)
        self.base_waypoint = waypoint
        transform = waypoint.transform
        # A route with a heading is a pose of its own (a car park has no
        # lane to snap to); its z is the ground there
        self.route_ground_z = None
        if "yaw_deg" in route:
            self.route_ground_z = float(route.get("z", 0.0))
            transform = carla.Transform(
                carla.Location(float(route["x"]), float(route["y"]), self.route_ground_z),
                carla.Rotation(yaw=float(route["yaw_deg"])))
        transform.location.z += 0.3
        transform.rotation.yaw += float(self.scenario.get("ego_yaw_offset_deg", 0.0))
        yaw = math.radians(transform.rotation.yaw)
        lateral = float(self.scenario.get("ego_lateral_m", 0.0))
        transform.location.x -= math.sin(yaw) * lateral
        transform.location.y += math.cos(yaw) * lateral
        blueprint = library.find(self.c.get("ego_blueprint", "vehicle.tesla.model3"))
        blueprint.set_attribute("role_name", "hero")
        self.ego = self.world.spawn_actor(blueprint, transform)
        self.actors.append(self.ego)

        rig = self.rig
        mount = carla.Transform(carla.Location(rig.x, rig.y, rig.z),
                                carla.Rotation(roll=rig.roll_deg, pitch=rig.pitch_deg,
                                               yaw=rig.yaw_deg))

        def camera(kind, extra):
            bp = library.find(kind)
            bp.set_attribute("image_size_x", str(rig.width))
            bp.set_attribute("image_size_y", str(rig.height))
            bp.set_attribute("fov", str(rig.fov_deg))
            bp.set_attribute("sensor_tick", "0.0")
            for name, value in extra.items():
                if bp.has_attribute(name):
                    bp.set_attribute(name, value)
            # Rigid: a spring arm would move the camera against the calibration
            sensor = self.world.spawn_actor(
                bp, mount, attach_to=self.ego,
                attachment_type=carla.AttachmentType.Rigid)
            self.sensors.append(sensor)
            return sensor

        # A pinhole image: the calibration has no distortion terms
        rgb = camera("sensor.camera.rgb", {
            "lens_k": "0.0", "lens_kcube": "0.0", "lens_circle_multiplier": "0.0",
            "lens_circle_falloff": "0.0", "chromatic_aberration_intensity": "0.0",
            "motion_blur_intensity": "0.0"})
        depth = camera("sensor.camera.depth", {})
        rgb.listen(lambda image: self.on_image("rgb", image))
        depth.listen(lambda image: self.on_image("depth", image))

        for kind in ("sensor.other.collision", "sensor.other.lane_invasion"):
            sensor = self.world.spawn_actor(library.find(kind), carla.Transform(),
                                            attach_to=self.ego)
            sensor.listen(lambda event, kind=kind: self.on_event(kind, event))
            self.sensors.append(sensor)

        for index, spec in enumerate(self.scenario.get("actors", [])):
            self.scripted.append(ScriptedActor(carla, self.world, waypoint, spec,
                                               self.dt, index))

        physics = self.ego.get_physics_control()
        wheels = physics.wheels
        # Wheels: front left, front right, rear left, rear right; positions
        # in centimetres, in the world frame
        wheelbase = math.hypot(wheels[0].position.x - wheels[2].position.x,
                               wheels[0].position.y - wheels[2].position.y) / 100.0
        box = self.ego.bounding_box
        self.vehicle = {
            "blueprint": self.ego.type_id,
            "wheelbase_m": wheelbase,
            "max_steer_deg": wheels[0].max_steer_angle,
            "width_m": 2.0 * box.extent.y,
            "front_bumper_m": box.location.x + box.extent.x,
            "rear_bumper_m": box.location.x - box.extent.x,
            "mass_kg": physics.mass,
        }

    def on_image(self, kind, image):
        """Sensor thread. The ledger entry comes first: whatever happens to
        the image next, the frame was produced and is counted."""
        sim_ns = protocol.sim_ns_of(image.timestamp)
        if kind == "rgb":
            self.ledger.register(image.frame, sim_ns, mono_ns())
        else:
            self.ledger.update(image.frame, "depth", depth_mono_ns=mono_ns())
        data = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            image.height, image.width, 4).copy()
        try:
            self.image_queue.put_nowait((kind, image.frame, sim_ns, data))
        except queue.Full:
            self.ledger.update(image.frame, "dropped", reason="callback_queue_full",
                               kind=kind)

    def on_event(self, kind, event):
        record = {"event": kind.split(".")[-1], "frame": event.frame}
        if kind.endswith("collision"):
            record["other"] = event.other_actor.type_id
            record["other_id"] = event.other_actor.id
        else:
            record["markings"] = [str(m.type) for m in event.crossed_lane_markings]
        with self.events_lock:
            self.events.append(record)

    # ── One tick ─────────────────────────────────────────────────────────────

    def ego_state(self, snapshot, frame, sim_ns, tick):
        actor = snapshot.find(self.ego.id)
        velocity = actor.get_velocity()
        angular = actor.get_angular_velocity()  # degrees per second
        return {"run_id": self.run_id, "frame": frame, "sim_ns": sim_ns,
                # The scenario's own count: what its from_tick and at_tick mean
                "tick": tick,
                "pose": pose_dict(actor.get_transform()),
                "velocity_world_mps": [velocity.x, velocity.y, velocity.z],
                "speed_mps": math.hypot(velocity.x, velocity.y),
                "yaw_rate_radps": math.radians(angular.z),
                "mono_ns": mono_ns()}

    def lane_truth(self, ego_transform):
        """The map's centre of the lane the ego vehicle is in, ahead of it."""
        waypoint = self.map.get_waypoint(ego_transform.location, project_to_road=True)
        points = []
        current = waypoint
        for _ in range(22):
            l = current.transform.location
            points.append([l.x, l.y])
            following = current.next(2.0)
            if not following:
                break
            current = following[0]
        return {"lane_id": waypoint.lane_id, "road_id": waypoint.road_id,
                "width_m": waypoint.lane_width,
                "centre_yaw_deg": waypoint.transform.rotation.yaw,
                "centre": [waypoint.transform.location.x, waypoint.transform.location.y],
                "centre_ahead": points}

    def draw_markers(self):
        """Red points at known places in the ego frame of the start pose, for
        the reprojection check: where they appear in the image must be where
        the calibration projects them. Drawn twice a second and kept for a
        second: drawn for one tick at a time, a point was now and then missing
        from a frame."""
        points = self.scenario.get("markers_ego_m")
        if not points:
            return
        if not hasattr(self, "marker_world"):
            start = self.ego.get_transform()
            self.marker_world = [start.transform(self.carla.Location(*p)) for p in points]
            protocol.write_json(os.path.join(self.run_dir, "markers.json"), {
                "ego_pose": pose_dict(start), "points_ego_m": points,
                "points_world_m": [[l.x, l.y, l.z] for l in self.marker_world]})
        self.marker_ticks = getattr(self, "marker_ticks", -1) + 1
        if self.marker_ticks % 15:
            return
        for location in self.marker_world:
            self.world.debug.draw_point(location, size=0.06,
                                        color=self.carla.Color(255, 0, 0),
                                        life_time=1.0)

    def open_loop(self, tick):
        """A scripted command, for the scenarios that measure the vehicle
        itself (steering sign and gain, braking). They do not use the
        controller: no perception is involved."""
        steps = self.scenario.get("open_loop")
        if not steps:
            return None
        current = {"throttle": 0.0, "brake": 1.0, "steer": 0.0}
        for step in steps:
            if tick >= step["from_tick"]:
                current = {k: float(step.get(k, 0.0)) for k in ("throttle", "brake", "steer")}
                current["reverse"] = bool(step.get("reverse", False))
        return dict(current, reason="open_loop", command_seq=None)

    def fault(self, kind, tick):
        for f in self.scenario.get("faults", []):
            if f["kind"] == kind and f.get("from_tick", 0) <= tick < f.get("until_tick", 10 ** 9):
                return f
        return None

    def take_commands(self, now_sim_ns):
        self.spin()
        for data in self.pending_arm:
            try:
                reason = self.lease.arm(json.loads(data))
            except ValueError:
                reason = "not_json"
            self.log_control({"event": "arm", "message": data[:200], "rejected": reason})
        self.pending_arm.clear()
        for data, received in self.pending_commands:
            try:
                command = json.loads(data)
            except ValueError:
                command = None
            reason = self.lease.offer(
                command, now_sim_ns, received,
                known_basis=lambda ns: ns in self.sent,
                known_control=lambda ns, frame: (
                    ns in self.ego_states and self.ego_states.get(ns)["frame"] == frame))
            if reason:
                self.log_control({"event": "rejected", "reason": reason,
                                  "command": command if isinstance(command, dict) else None})
        self.pending_commands.clear()

    def log_control(self, record):
        self.control_log.write(json.dumps(record, separators=(",", ":")) + "\n")

    def collect_images(self, frame):
        """The RGB and depth image of this tick's frame; (rgb, depth, sim_ns)
        or None. Waiting here is for the simulator's own sensors, not for
        the device."""
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                kind, image_frame, sim_ns, data = self.image_queue.get(
                    timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                break
            pair, lost = self.pairer.add(kind, image_frame, sim_ns, data)
            for lost_frame, reason in lost:
                self.ledger.update(lost_frame, "dropped", reason=reason)
            if pair and pair[0] == frame:
                return pair[2], pair[3], pair[1]
        return None

    def publish_state(self, sim_ns, ego):
        sec, nanosec = divmod(sim_ns, 1000000000)
        clock = self.Clock()
        clock.clock.sec, clock.clock.nanosec = sec, nanosec
        self.pub_clock.publish(clock)
        self.pub_ego.publish(self.String(data=json.dumps(ego)))

    def publish_depth(self, sim_ns, depth_m):
        sec, nanosec = divmod(sim_ns, 1000000000)
        image = self.Image()
        image.header.stamp.sec, image.header.stamp.nanosec = sec, nanosec
        image.header.frame_id = "%s/%s_optical" % (self.run_id, SOURCE.replace("-", "_"))
        image.height, image.width = depth_m.shape
        image.encoding = "32FC1"
        image.is_bigendian = 0
        image.step = depth_m.shape[1] * 4
        # Not `image.data = ...`: that setter checks the 3.7 MB byte by byte
        # in Python, 400 ms a frame
        data = array.array("B")
        data.frombytes(depth_m.tobytes())
        image._data = data
        self.pub_depth.publish(image)
        info = self.CameraInfo()
        info.header = image.header
        info.height, info.width = self.rig.height, self.rig.width
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        K = self.rig.K
        info.k = [float(v) for v in K.reshape(-1)]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [K[0, 0], 0.0, K[0, 2], 0.0, 0.0, K[1, 1], K[1, 2], 0.0,
                  0.0, 0.0, 1.0, 0.0]
        self.pub_info.publish(info)

    def output_loop(self, sender):
        """Output thread: one paired frame at a time, in the order produced.
        Depth first, so that it is on the host before the device's result
        for the same frame can come back."""
        import cv2
        # Optimised Huffman tables: the same pixels in 14 % fewer bytes, for
        # 2 ms more per frame
        quality = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality,
                   cv2.IMWRITE_JPEG_OPTIMIZE, 1]
        keep_every = int(self.scenario.get("rgb_artifact_every",
                                           self.c.get("rgb_artifact_every", 0)))
        while True:
            item = self.output_queue.get()
            if item is None:
                return
            frame, sim_ns, rgb, depth_raw, drop_rgb, drop_depth = item
            if not drop_depth:
                self.publish_depth(sim_ns, decode_depth(depth_raw))
                self.depth_writer.offer(frame, sim_ns, depth_raw)
            if drop_rgb:
                self.ledger.update(frame, "dropped", reason="fault_drop_rgb")
                self.stats["dropped"] += 1
                continue
            encode = time.monotonic()
            ok, jpeg = cv2.imencode(".jpg", rgb[:, :, :3], quality)
            jpeg = jpeg.tobytes()
            encode_ms = (time.monotonic() - encode) * 1e3
            if ok and sender.send(jpeg, sim_ns):
                self.sent.put(sim_ns, frame)
                self.ledger.update(frame, "sent", jpeg_bytes=len(jpeg),
                                   jpeg_crc32=zlib.crc32(jpeg),
                                   encode_ms=round(encode_ms, 3),
                                   sent_mono_ns=mono_ns(), clients=sender.clients())
                self.stats["sent"] += 1
                self.stats["jpeg_bytes"] += len(jpeg)
                self.stats["encode_ms"] += encode_ms
                if keep_every and self.stats["sent"] % keep_every == 1 % keep_every:
                    with open(os.path.join(self.run_dir, "rgb", "%d.jpg" % sim_ns),
                              "wb") as f:
                        f.write(jpeg)
            else:
                self.ledger.update(frame, "dropped", reason="sender_queue_full")
                self.stats["dropped"] += 1

    def tick(self, tick):
        started = time.monotonic()
        last = self.last_ego

        # 1. What the vehicle does during this tick
        self.take_commands(last["sim_ns"] if last else 0)
        if self.args.arm and tick >= 0 and self.controller_ready \
                and time.monotonic() - self.arm_sent > 1.0:
            # The operator's explicit action, taken once the controller says
            # its gates pass; without --arm nobody arms. The controller may
            # still refuse (a gate closed on the way): asked again, at most
            # once a second, while it reports ready.
            self.pub_arm.publish(self.String(data=json.dumps(
                {"run_id": self.run_id, "enable": True})))
            self.arm_sent = time.monotonic()
        forged = self.fault("forge_commands", tick)
        if forged:
            for bad in forged["commands"]:
                self.pub_control.publish(self.String(data=json.dumps(bad)))
        applied = self.lease.control(last["sim_ns"] if last else 0, mono_ns())
        open_loop = self.open_loop(tick)
        if open_loop is not None:
            applied = open_loop
        # One synchronous batch: every command is in the simulator before the
        # tick. Sent one by one they are not waited for, and now and then an
        # actor was still at its old place in the tick that followed.
        commands = [self.carla.command.ApplyVehicleControl(
            self.ego, self.carla.VehicleControl(
                throttle=float(applied["throttle"]), brake=float(applied["brake"]),
                steer=float(applied["steer"]),
                # Changing `reverse` puts the gearbox in -1 or 1 by itself;
                # manual_gear_shift stays off
                reverse=bool(applied.get("reverse", False)),
                hand_brake=bool(applied.get("hand_brake", False))))]
        commands += [actor.step(tick) for actor in self.scripted]
        self.client.apply_batch_sync(commands, False)
        self.draw_markers()
        frame_before = last["frame"] if last else None
        apply_mono = mono_ns()

        # 2. Advance; never waits for the device
        tick_mono = mono_ns()
        frame = self.world.tick()
        snapshot = self.world.get_snapshot()
        sim_ns = protocol.sim_ns_of(snapshot.timestamp.elapsed_seconds)
        self.ledger.update(frame, "tick", sim_ns=sim_ns, tick=tick, mono_ns=tick_mono)
        self.stats["tick_ms"] += (mono_ns() - tick_mono) / 1e6
        ego = self.ego_state(snapshot, frame, sim_ns, tick)
        self.ego_states.put(sim_ns, ego)
        self.last_ego = ego
        self.ego_log.write(json.dumps(ego, separators=(",", ":")) + "\n")
        self.log_control({"event": "applied", "tick": tick, "frame_before": frame_before,
                          "frame_after": frame, "sim_ns_after": sim_ns,
                          "apply_mono_ns": apply_mono, "armed": self.lease.armed,
                          **applied})

        ego_transform = snapshot.find(self.ego.id).get_transform()
        if self.args.follow:
            # Chase view for a person watching the server's window
            yaw = math.radians(ego_transform.rotation.yaw)
            at = ego_transform.location
            self.world.get_spectator().set_transform(self.carla.Transform(
                self.carla.Location(at.x - 8.0 * math.cos(yaw),
                                    at.y - 8.0 * math.sin(yaw), at.z + 4.0),
                self.carla.Rotation(pitch=-15.0, yaw=ego_transform.rotation.yaw)))
        with self.events_lock:
            events, self.events = self.events, []
        self.truth_log.write(json.dumps({
            "frame": frame, "sim_ns": sim_ns, "tick": tick,
            "ego": {"pose": ego["pose"], "speed_mps": ego["speed_mps"]},
            "lane": self.lane_truth(ego_transform),
            "actors": [a.truth(snapshot.find(a.actor.id)) for a in self.scripted],
            "events": events}, separators=(",", ":")) + "\n")

        # 3. This tick's images
        wait = time.monotonic()
        images = self.collect_images(frame)
        self.stats["wait_images_ms"] += (time.monotonic() - wait) * 1e3
        self.publish_state(sim_ns, ego)
        if images is None:
            if frame in self.ledger.registered:
                self.ledger.update(frame, "dropped", reason="not_paired")
        elif images[2] != sim_ns:
            self.ledger.update(frame, "dropped", reason="image_time_differs")
        else:
            self.ledger.update(frame, "paired")
            # 4. Encoding and publishing take longer than what is left of the
            # tick, so they run on their own thread, in order. Its queue is
            # short: a frame that cannot get in is dropped whole, and counted.
            item = (frame, sim_ns, images[0], images[1],
                    bool(self.fault("drop_rgb", tick)),
                    bool(self.fault("drop_depth", tick)))
            try:
                self.output_queue.put_nowait(item)
            except queue.Full:
                self.ledger.update(frame, "dropped", reason="sender_queue_full")
                self.stats["dropped"] += 1
        if tick >= 0:
            self.stats["ticks"] += 1

        kill = self.fault("kill_controller", tick)
        if kill and self.args.controller_pid and not getattr(self, "killed", False):
            os.kill(self.args.controller_pid, signal.SIGKILL)
            self.killed = True
            self.log_control({"event": "fault", "kind": "kill_controller", "tick": tick})

        # Real time when the machine allows: the device sees 30 frames a second
        remaining = self.dt - (time.monotonic() - started)
        if remaining > 0 and not self.args.no_pacing:
            time.sleep(remaining)

    # ── Run ──────────────────────────────────────────────────────────────────

    def manifest(self, sender_description, ground_measured):
        def version(command):
            import subprocess
            try:
                return subprocess.run(command, capture_output=True, text=True,
                                      timeout=10).stdout.strip().split("\n")[0]
            except Exception:  # noqa: BLE001  (a missing tool is recorded as such)
                return "unknown"
        return {
            "run_id": self.run_id, "source": SOURCE, "scenario": self.args.scenario,
            "scenario_config": self.scenario,
            "commit": version(["git", "rev-parse", "HEAD"]),
            "branch": version(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "carla_server": self.server_version,
            "carla_client": self.client.get_client_version(),
            "map": self.map.name, "weather": self.scenario.get("weather", "ClearNoon"),
            "fixed_delta_seconds": self.dt, "max_substeps": 10,
            "max_substep_delta_time": 0.01,
            "calibration": self.config["calibration"],
            "calibration_id": self.rig.calibration_id,
            "focal_px": self.rig.focal,
            "ground_z_measured": ground_measured,
            "vehicle": self.vehicle,
            "detector": {k: self.config.get(k) for k in (
                "model", "backend", "precision", "confidence", "classes")},
            "lane": self.config.get("lane"),
            "metric": self.config.get("metric", {}),
            "control": self.config["control"],
            "jpeg_quality": self.jpeg_quality,
            "sender": sender_description,
            "gstreamer": version(["gst-inspect-1.0", "--version"]),
            "depth_encoding": "carla_bgr24_zlib",
            "armed_by_operator": bool(self.args.arm),
            "started_unix": time.time(),
        }

    def run(self):
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop_requested", True))
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop_requested", True))
        sender = None
        try:
            self.connect()
            self.spawn()
            self.start_ros()
            s = self.c.get("sender", {})
            sender = JpegSender(s.get("bind", "0.0.0.0"), int(s.get("port", 5000)),
                                int(s.get("max_queued", 2)),
                                int(s.get("client_max_buffers", 30)))
            self.last_ego = None
            self.ego.apply_control(self.carla.VehicleControl(throttle=0.0, brake=1.0))
            # Let the vehicle settle on its wheels before anything is measured
            for _ in range(int(self.c.get("settle_ticks", 30))):
                self.world.tick()
            while True:  # images of the settling ticks are not part of the run
                try:
                    self.image_queue.get_nowait()
                except queue.Empty:
                    break
            location = self.ego.get_transform().location
            road = self.map.get_waypoint(location, project_to_road=True)
            ground = (road.transform.location.z if self.route_ground_z is None
                      else self.route_ground_z) - location.z
            protocol.write_json(os.path.join(self.run_dir, "manifest.json"),
                                self.manifest(s, ground))
            if abs(ground - self.rig.ground_z) > 0.05:
                print("WARNING: the road is %.3f m from the vehicle origin, the "
                      "calibration says %.3f" % (ground, self.rig.ground_z), flush=True)
            # Images of the settling ticks were registered; mark them
            for frame in sorted(self.ledger.registered):
                self.ledger.update(frame, "dropped", reason="settling")
            self.ledger.registered.clear()

            ticks = int(self.args.ticks or self.scenario.get("ticks", 600))
            print("run %s scenario %s: %d ticks, waiting for a client on port %d"
                  % (self.run_id, self.args.scenario, ticks, int(s.get("port", 5000))),
                  flush=True)
            started = time.monotonic()
            # The run starts once the device is connected, or after the wait.
            # The world keeps ticking meanwhile: a server that has just loaded
            # this map still holds the textures of the one before, and lets
            # go of them only while it renders. Standing still it kept 6 GB
            # of video memory and the device's models did not fit beside it.
            wait_until = time.monotonic() + float(self.args.wait_client)
            while sender.clients() == 0 and time.monotonic() < wait_until \
                    and not self.stop_requested:
                self.world.tick()
                while not self.image_queue.empty():
                    self.image_queue.get_nowait()
                time.sleep(0.1)
            for frame in sorted(self.ledger.registered):
                self.ledger.update(frame, "dropped", reason="waiting_for_the_device")
            self.ledger.registered.clear()
            output = threading.Thread(target=self.output_loop, args=(sender,),
                                      daemon=True)
            output.start()
            # Before the run: the same loop, vehicle braked, until the first
            # result has come all the way back (the device may still be
            # loading its models). These ticks carry tick -1 and are not part
            # of the run's frames.
            preroll = time.monotonic() + float(self.args.wait_result)
            while sender.clients() > 0 and self.controller_frames == 0 \
                    and time.monotonic() < preroll and not self.stop_requested:
                self.tick(-1)
            for key in self.stats:  # the rates are those of the run itself
                self.stats[key] = 0
            started = time.monotonic()
            for tick in range(ticks):
                if self.stop_requested:
                    break
                self.tick(tick)
            elapsed = time.monotonic() - started
            self.output_queue.put(None)
            output.join(timeout=10)
            n = max(1, self.stats["ticks"])
            summary = {
                "ticks": self.stats["ticks"], "wall_seconds": elapsed,
                "wall_fps": self.stats["ticks"] / elapsed if elapsed > 0 else 0.0,
                "simulated_fps": 1.0 / self.dt,
                "frames_sent": self.stats["sent"], "frames_dropped": self.stats["dropped"],
                "jpeg_mean_bytes": self.stats["jpeg_bytes"] / max(1, self.stats["sent"]),
                "jpeg_mbit_per_s": self.stats["jpeg_bytes"] * 8e-6 / elapsed if elapsed else 0,
                "encode_mean_ms": self.stats["encode_ms"] / max(1, self.stats["sent"]),
                "tick_mean_ms": self.stats["tick_ms"] / n,
                "wait_images_mean_ms": self.stats["wait_images_ms"] / n,
                "commands_rejected": self.lease.rejected,
                "live_cache_evicted": self.sent.evicted,
                "depth_artifact_bytes": self.depth_writer.bytes,
                "depth_artifact_write_seconds": self.depth_writer.seconds,
            }
            protocol.write_json(os.path.join(self.run_dir, "simulator-summary.json"),
                                summary)
            print(json.dumps(summary), flush=True)
        finally:
            self.close(sender)

    def close(self, sender):
        """Only what this run created; the world gets its settings back."""
        if self.world is not None:
            try:
                self.ego.apply_control(self.carla.VehicleControl(throttle=0.0, brake=1.0))
            except Exception:  # noqa: BLE001
                pass
            for sensor in self.sensors:
                try:
                    sensor.stop()
                    sensor.destroy()
                except Exception:  # noqa: BLE001
                    pass
            for actor in [a.actor for a in self.scripted] + self.actors:
                try:
                    actor.destroy()
                except Exception:  # noqa: BLE001
                    pass
            if self.original_settings is not None:
                self.world.apply_settings(self.original_settings)
        if sender:
            sender.close()
        self.depth_writer.close()
        for f in (self.ego_log, self.truth_log, self.control_log):
            f.close()
        self.ledger.close()
        if getattr(self, "rclpy", None) and self.rclpy.ok():
            self.node.destroy_node()
            self.rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="simulation/config/carla.yaml")
    parser.add_argument("--scenarios", required=True,
                        help="simulation/config/carla-scenarios.yaml")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--run-dir", required=True, help="new directory of this run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ticks", type=int, default=0, help="override the scenario")
    parser.add_argument("--arm", action="store_true",
                        help="send ARM once the controller reports its gates pass")
    parser.add_argument("--controller-pid", type=int, default=0,
                        help="for the kill_controller fault")
    parser.add_argument("--wait-client", type=float, default=30.0,
                        help="seconds to wait for the device to connect")
    parser.add_argument("--wait-result", type=float, default=120.0,
                        help="seconds to wait, ticking, for the first result")
    parser.add_argument("--follow", action="store_true",
                        help="keep the server's spectator camera behind the ego vehicle")
    parser.add_argument("--no-pacing", action="store_true",
                        help="tick as fast as possible instead of in real time")
    args = parser.parse_args()
    Simulator(args).run()


if __name__ == "__main__":
    main()

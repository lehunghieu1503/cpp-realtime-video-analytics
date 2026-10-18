#!/usr/bin/env python3
"""Measure a car park's stalls, and make the images the stall model is trained on.
Run by hand, with a CARLA server on port 2000; nothing calls this.

Usage:
  simulation/stall-dataset.py map  [--map Town05] [--centre X Y] [--span M]
  simulation/stall-dataset.py make --out data/stalls [--count 4400] [--seed 1]
                                   [--preview 12] [--bev]

map   CARLA has no list of parking stalls, so the paint is measured: a camera
      looking straight down gives the pixels the server labels RoadLines, and
      the dividers and the lines at the back of the rows are read from them.
      Writes simulation/config/carla-stalls.yaml: every stall as centre x, y,
      yaw_deg out of the stall, length_m, width_m. Rows along the world's x
      or y axis only.

make  The rear camera of simulation/config/carla.yaml (rear_calibration), on
      the ego vehicle, at poses a vehicle has while it passes a row, turns in
      backwards and stands in a stall, and at random ones. Cars are put in
      some of the stalls, the map's own parked cars are shown or hidden, the
      sun and the clouds change. Each image gets its labels in Ultralytics
      pose format:

        class cx cy w h  x0 y0 v0  x1 y1 v1  x2 y2 v2  x3 y3 v3     (normalised)

      Stalls, one label per stall that is large enough in the image: class
      0 is an empty stall, 1 one with a vehicle in it. The box is that of
      the part of the stall inside the image. The four points are the
      stall's corners in the order of parking.Stall.corners(): the two of
      the mouth, then the two of the back. v is 2 for a corner that is in
      the image and not hidden (the depth image of the same frame says so),
      0 otherwise.

      Ends of dividers, one label per end that is in the image and not
      hidden: class 2 where a divider ends at the mouth of the stalls, 3
      where it meets the line across at their back. The box is that of
      0.8 m of ground around the end; the four points are all the end
      itself. A stall says whether it is empty and roughly where it is; an
      end is a small thing with a sharp place, which is what a stall is
      measured by.

      Writes OUT/images, OUT/labels and OUT/data.yaml.

      --bev: the images are the bird's-eye view the device makes of the
      cameras of the `bev` section (video_analytics_simulation/bev.py), each
      camera's image first through JPEG as the device gets it. The labels
      are the same, in that view: a corner or an end is seen where the view
      has a camera's pixel and that camera has nothing in front of it.

The stalls file is truth: it labels these images and the evaluation reads it.
Nothing in the control path does.
"""

import argparse
import math
import os
import queue
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from video_analytics_simulation import bev  # noqa: E402
from video_analytics_simulation.calibration import Pose, Rig, decode_depth  # noqa: E402
from video_analytics_simulation.parking import Stall  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STALLS = os.path.join(HERE, "config", "carla-stalls.yaml")
# Smaller than this in the image, a stall gets no label: the model is not
# asked for what it cannot see
MIN_BOX = (24, 10)
END_GROUND_M = 0.8      # the ground around a divider's end that is its box
MIN_END_BOX = (10, 5)
CAR_LABELS = ("Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Vehicles")


def connect(args, town):
    import carla
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != town:
        client.load_world(town)
        world = client.reload_world()
    for actor in world.get_actors():
        if actor.type_id.startswith(("vehicle.", "walker.", "sensor.")):
            actor.destroy()
    original = world.get_settings()
    original.synchronous_mode = False
    original.fixed_delta_seconds = None
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / 30.0
    world.apply_settings(settings)
    return carla, client, world, original


class Camera:
    """A sensor whose images are taken one tick at a time."""

    def __init__(self, carla, world, kind, transform, width, height, fov, parent=None):
        bp = world.get_blueprint_library().find(kind)
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))
        # A pinhole image, as the simulator's cameras give
        for name in ("lens_k", "lens_kcube", "lens_circle_multiplier",
                     "lens_circle_falloff", "chromatic_aberration_intensity",
                     "motion_blur_intensity"):
            if bp.has_attribute(name):
                bp.set_attribute(name, "0.0")
        self.sensor = world.spawn_actor(bp, transform, attach_to=parent,
                                        attachment_type=carla.AttachmentType.Rigid)
        self.shape = (height, width, 4)
        self.queue = queue.Queue()
        self.sensor.listen(self.queue.put)

    def image(self, frame):
        """The image of this frame; older ones are thrown away."""
        while True:
            image = self.queue.get(timeout=20)
            if image.frame >= frame:
                return np.frombuffer(image.raw_data, dtype=np.uint8).reshape(self.shape).copy()

    def close(self):
        self.sensor.stop()
        self.sensor.destroy()


# ── map ──────────────────────────────────────────────────────────────────────

def measure(args):
    import cv2
    carla, client, world, original = connect(args, args.map)
    parked = set()
    try:
        # The map's own parked cars and the heads of its lamp posts hide the
        # paint under them
        for name in CAR_LABELS + ("Poles",):
            if hasattr(carla.CityObjectLabel, name):
                parked |= {o.id for o in world.get_environment_objects(
                    getattr(carla.CityObjectLabel, name))}
        world.enable_environment_objects(parked, False)
        cx, cy = args.centre
        height, size, fov = 150.0, 2400, 2.0 * math.degrees(math.atan(args.span / 300.0))
        scale = args.span / size  # metres a pixel
        # Looking straight down with yaw 0: the image's right is the world's
        # +y, its top the world's +x
        camera = Camera(carla, world, "sensor.camera.semantic_segmentation",
                        carla.Transform(carla.Location(cx, cy, height),
                                        carla.Rotation(pitch=-90.0)), size, size, fov)
        for _ in range(3):
            frame = world.tick()
        tags = camera.image(frame)[:, :, 2]
        camera.close()

        def world_xy(u, v):
            return cx - (v - size / 2.0) * scale, cy + (u - size / 2.0) * scale

        label = carla.CityObjectLabel
        lines = (tags == int(label.RoadLines)).astype(np.uint8)
        cars = np.isin(tags, [int(getattr(label, n)) for n in CAR_LABELS
                              if hasattr(label, n)]).astype(np.uint8)
        ground = (tags == int(label.Ground)).astype(np.uint8)
        # The car park: the largest piece of ground, with what stands on it
        count, pieces, stats, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(
            ground | lines | cars, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)))
        if count < 2:
            raise SystemExit("no ground in the image: wrong --centre?")
        park = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        left, top, wide, high, _ = stats[park]
        area = {"x": [round(float(world_xy(0, top + high)[0]), 2),
                      round(float(world_xy(0, top)[0]), 2)],
                "y": [round(float(world_xy(left, 0)[1]), 2),
                      round(float(world_xy(left + wide, 0)[1]), 2)]}
        lines &= (pieces == park).astype(np.uint8)

        stalls = []
        long = int(1.5 / scale)
        # A row along y has dividers along x: long in the image's v. The
        # transposed image gives the rows along x.
        for along_y in (True, False):
            mask = lines if along_y else lines.T.copy()
            backs = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, long), np.uint8))
            dividers = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((long, 1), np.uint8))
            n, _, s, c = cv2.connectedComponentsWithStats(dividers & (1 - backs))
            # The lines across, as the image rows of their centres
            rows = np.flatnonzero(backs.any(axis=1))
            across = [run.mean() for run in np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
                      if len(run)]
            found = []
            for i in range(1, n):
                if s[i, cv2.CC_STAT_HEIGHT] * scale < 3.0:
                    continue
                first, last = s[i, cv2.CC_STAT_TOP], s[i, cv2.CC_STAT_TOP] + s[i, cv2.CC_STAT_HEIGHT]
                # The back: the end that has a line across within a metre
                # of it; the mouth is open. Whatever stands over the paint
                # there (a lamp) leaves the divider short of that line.
                ends = sorted((min(abs(row - first), abs(row - last)), row) for row in across)
                if not ends or ends[0][0] * scale > 1.0:
                    continue
                back = ends[0][1]
                mouth = last if abs(back - first) < abs(back - last) else first
                found.append((round(back), c[i][0], back, mouth))
            found.sort()
            for a, b in zip(found, found[1:]):
                width = (b[1] - a[1]) * scale
                if a[0] != b[0] or not 2.2 <= width <= 3.6:
                    continue
                back, mouth = (a[2] + b[2]) / 2.0, (a[3] + b[3]) / 2.0
                u, v = (a[1] + b[1]) / 2.0, (back + mouth) / 2.0
                if not along_y:
                    u, v = v, u
                x, y = world_xy(u, v)
                # Out of the stall: from the back to the mouth
                out = (180.0 if mouth > back else 0.0) if along_y else \
                    (90.0 if mouth > back else -90.0)
                stalls.append({"x": round(x, 2), "y": round(y, 2), "yaw_deg": out,
                               "length_m": round(abs(mouth - back) * scale, 2),
                               "width_m": round(width, 2)})
        if not stalls:
            raise SystemExit("no stalls found")
        stalls.sort(key=lambda s: (s["yaw_deg"], -s["x"], -s["y"]))
        hit = world.project_point(carla.Location(stalls[0]["x"], stalls[0]["y"], 20.0),
                                  carla.Vector3D(0, 0, -1), 40.0)
        data = {}
        if os.path.exists(STALLS):
            with open(STALLS) as f:
                data = yaml.safe_load(f) or {}
        data[args.map] = {"ground_z": round(hit.location.z, 3), "area": area,
                          "stalls": stalls}
        with open(STALLS, "w") as f:
            f.write("# The stalls of the car parks, measured on the paint by\n"
                    "# simulation/stall-dataset.py map: centre x, y, yaw_deg out of the\n"
                    "# stall towards its aisle, length_m along that direction, width_m\n"
                    "# between the centres of its dividers. area is the car park's ground.\n"
                    "# Truth: for the stall model's training labels and for the evaluation.\n"
                    "# Nothing in the control path reads this file.\n")
            for town, park in data.items():
                f.write("%s:\n  ground_z: %s\n  area: {x: %s, y: %s}\n  stalls:\n" % (
                    town, park["ground_z"], park["area"]["x"], park["area"]["y"]))
                for s in park["stalls"]:
                    f.write("    - {x: %s, y: %s, yaw_deg: %s, length_m: %s, width_m: %s}\n" % (
                        s["x"], s["y"], s["yaw_deg"], s["length_m"], s["width_m"]))
        widths = [s["width_m"] for s in stalls]
        print("%s: %d stalls, %.2f to %.2f m wide, ground at z = %.3f; written to %s" % (
            args.map, len(stalls), min(widths), max(widths), hit.location.z, STALLS))
    finally:
        world.enable_environment_objects(parked, True)
        world.apply_settings(original)


# ── make ─────────────────────────────────────────────────────────────────────

def box_corners(x, y, yaw_deg, half_length, half_width):
    yaw = math.radians(yaw_deg)
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    right = np.array([-math.sin(yaw), math.cos(yaw)])
    return np.array([[x, y] + a * half_length * forward + b * half_width * right
                     for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))])


def boxes_overlap(a, b):
    """Two rectangles, as corners (4, 2): separating axes."""
    for corners in (a, b):
        for i in range(2):
            axis = corners[i + 1] - corners[i]
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def divider_ends(stalls):
    """Where the dividers end: every corner of every stall, once (two stalls
    share a divider), as (x, y, class)."""
    ends = []
    for stall in stalls:
        for k, corner in enumerate(stall.corners()[:, :2]):
            cls = 2 if k < 2 else 3
            if not any(c == cls and np.hypot(x - corner[0], y - corner[1]) < 0.5
                       for x, y, c in ends):
                ends.append((float(corner[0]), float(corner[1]), cls))
    return ends


class CameraView:
    """What one camera shows of the world, from where the vehicle is."""

    def __init__(self, rig, capture, depth_m):
        self.rig, self.capture, self.depth_m = rig, capture, depth_m
        self.width, self.height = rig.width, rig.height

    def box(self, outline):
        return self.rig.visible_box(self.capture.from_parent(outline), 0.15)

    def sees(self, point):
        """The pixel of a world point, or None when it is outside the image
        or hidden."""
        rig, f = self.rig, self.rig.focal
        # x forward, y right, z up
        x, y, z = rig.pose.from_parent(self.capture.from_parent(point))
        if x < 0.15:
            return None
        u, v = f * y / x + rig.width / 2.0, -f * z / x + rig.height / 2.0
        if not (0 <= u < rig.width and 0 <= v < rig.height):
            return None
        # Not hidden: the depth image has the ground there, not something
        # nearer. One pixel is many metres of ground far away, so the
        # tolerance grows with the distance.
        tolerance = 0.3 + 1.5 * x ** 2 / (f * (rig.z - rig.ground_z))
        return (u, v) if self.depth_m[int(v), int(u)] > x - tolerance else None


class BevView:
    """What the bird's-eye view shows: a place on the ground is seen where
    the view has a camera's pixel and that camera sees the place."""

    def __init__(self, stitcher, capture, depths):
        self.stitcher, self.capture = stitcher, capture
        self.cameras = [CameraView(rig, capture, depth_m)
                        for rig, depth_m in zip(stitcher.rigs, depths)]
        self.width, self.height = stitcher.view.width, stitcher.view.height

    def box(self, outline):
        return self.stitcher.view.visible_box(self.capture.from_parent(outline))

    def sees(self, point):
        (u, v), _ = self.stitcher.view.project(self.capture.from_parent(point))
        if not (0 <= u < self.width and 0 <= v < self.height):
            return None
        source = self.stitcher.source[int(v), int(u)]
        if source < 0 or self.cameras[source].sees(point) is None:
            return None
        return float(u), float(v)


def label(cls, outline, points, needed, min_box, view):
    """One label line, or None: the box of the ground polygon `outline`
    (world, (n, 3)) as far as it is in the image, and the four `points`
    (world, (4, 3)), of which `needed` must be seen."""
    w, h = view.width, view.height
    box = view.box(outline)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    if x1 - x0 < min_box[0] or y1 - y0 < min_box[1]:
        return None

    numbers = [(x0 + x1) / 2.0 / w, (y0 + y1) / 2.0 / h, (x1 - x0) / w, (y1 - y0) / h]
    visible = 0
    for point in points:
        seen = view.sees(point)
        if seen:
            visible += 1
            numbers += [seen[0] / w, seen[1] / h, 2]
        else:
            numbers += [0.0, 0.0, 0]
    if visible < needed:
        return None
    return "%d " % cls + " ".join(
        "%d" % n if isinstance(n, int) else "%.6f" % n for n in numbers)


def labels_of(stalls, occupied, ends, view, ground_z):
    lines = [label(1 if taken else 0, stall.corners(ground_z), stall.corners(ground_z),
                   2, MIN_BOX, view)
             for stall, taken in zip(stalls, occupied)]
    half = END_GROUND_M / 2.0
    for x, y, cls in ends:
        around = np.array([[x + a * half, y + b * half, ground_z]
                           for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
        lines.append(label(cls, around, np.tile([x, y, ground_z], (4, 1)), 4, MIN_END_BOX,
                           view))
    return [line for line in lines if line]


def ego_pose(rng, stalls, free, wheelbase):
    """(x, y, yaw_deg) of the vehicle's origin: one of the poses a vehicle
    has around a stall, or anywhere."""
    kind = rng.random()
    stall = stalls[int(rng.choice(free)) if len(free) and rng.random() < 0.8
                   else int(rng.integers(len(stalls)))]
    half = stall.length / 2.0
    if kind < 0.35:
        # Passing the row on its aisle, either way
        side = rng.choice((-1.0, 1.0))
        u, v = half + rng.uniform(3.0, 6.8), rng.uniform(-13.0, 13.0)
        heading = side * 90.0 + rng.normal(0.0, 5.0)
    elif kind < 0.65:
        # Turning in backwards: the rear axle on a circle that ends on the axis
        side = rng.choice((-1.0, 1.0))
        radius, end = rng.uniform(4.5, 7.5), half + rng.uniform(-1.8, 1.5)
        phi = math.radians(rng.uniform(0.0, 100.0))
        rear = np.array([end + radius * math.sin(phi), side * radius * (1.0 - math.cos(phi))])
        rear += rng.normal(0.0, 0.3, 2)
        heading = math.degrees(side * phi) + rng.normal(0.0, 6.0)
        u = rear[0] + wheelbase / 2.0 * math.cos(math.radians(heading))
        v = rear[1] + wheelbase / 2.0 * math.sin(math.radians(heading))
    elif kind < 0.8:
        # Backing straight in, to where it stands
        heading = rng.normal(0.0, 6.0)
        rear = np.array([rng.uniform(-half + 0.9, half + 4.0), rng.normal(0.0, 0.3)])
        u = rear[0] + wheelbase / 2.0 * math.cos(math.radians(heading))
        v = rear[1] + wheelbase / 2.0 * math.sin(math.radians(heading))
    else:
        u, v, heading = rng.uniform(-12.0, 12.0), rng.uniform(-15.0, 15.0), \
            rng.uniform(-180.0, 180.0)
    x, y = stall.world(u, v)
    return float(x), float(y), stall.yaw_out + heading


def make(args):
    import cv2
    with open(args.config) as f:
        config = yaml.safe_load(f)
    town = args.map
    # The rear camera's image, or the view from above of several cameras
    rigs = [rig for _, rig, _ in bev.cameras_of(config)] if args.bev \
        else [Rig.from_config(config["rear_calibration"])]
    stitcher = bev.Stitcher(bev.BevRig.from_config(config["bev"]), rigs) if args.bev else None
    with open(STALLS) as f:
        park = yaml.safe_load(f)[town]
    stalls = [Stall.from_dict(s) for s in park["stalls"]]
    ends = divider_ends(stalls)
    ground_z, area = park["ground_z"], park["area"]
    rng = np.random.default_rng(args.seed)
    for part in ("images/train", "images/val", "labels/train", "labels/val", "preview"):
        os.makedirs(os.path.join(args.out, part), exist_ok=True)

    carla, client, world, original = connect(args, town)
    library = world.get_blueprint_library()
    spawned, cameras, statics, ego = [], [], [], None
    try:
        # The map's own parked cars: boxes in the world
        for name in CAR_LABELS:
            if hasattr(carla.CityObjectLabel, name):
                statics += world.get_environment_objects(getattr(carla.CityObjectLabel, name))
        statics = [o for o in statics
                   if area["x"][0] < o.bounding_box.location.x < area["x"][1]
                   and area["y"][0] < o.bounding_box.location.y < area["y"][1]]
        static_boxes = [box_corners(o.bounding_box.location.x, o.bounding_box.location.y,
                                    o.bounding_box.rotation.yaw, o.bounding_box.extent.x,
                                    o.bounding_box.extent.y) for o in statics]
        print("%d stalls, %d ends of dividers, %d cars of the map in the car park" % (
            len(stalls), len(ends), len(statics)))
        cars = [b for b in library.filter("vehicle.*")
                if b.get_attribute("number_of_wheels").as_int() == 4
                and not any(n in b.id for n in ("firetruck", "ambulance", "carlacola",
                                                "european_hgv", "fusorosa", "sprinter"))]

        blueprint = library.find(config["carla"].get("ego_blueprint", "vehicle.tesla.model3"))
        # On the aisle in front of the first stall, along it
        start = stalls[0].world(stalls[0].length / 2.0 + 4.5, 0.0)
        ego = world.spawn_actor(blueprint, carla.Transform(
            carla.Location(float(start[0]), float(start[1]), ground_z + 0.3),
            carla.Rotation(yaw=stalls[0].yaw_out + 90.0)))
        for _ in range(40):
            world.tick()
        # Where its origin is over the ground once it stands on its wheels
        settled = ego.get_transform()
        height = settled.location.z
        wheels = ego.get_physics_control().wheels
        wheelbase = math.hypot(wheels[0].position.x - wheels[2].position.x,
                               wheels[0].position.y - wheels[2].position.y) / 100.0
        extent, offset = ego.bounding_box.extent, ego.bounding_box.location.x
        ego.set_simulate_physics(False)
        for rig in rigs:
            mount = carla.Transform(carla.Location(rig.x, rig.y, rig.z),
                                    carla.Rotation(roll=rig.roll_deg, pitch=rig.pitch_deg,
                                                   yaw=rig.yaw_deg))
            cameras += [Camera(carla, world, kind, mount, rig.width, rig.height,
                               rig.fov_deg, ego)
                        for kind in ("sensor.camera.rgb", "sensor.camera.depth")]

        made = previews = 0
        boxes, occupied = [], []
        while made < args.count:
            if made % args.scene == 0:
                # A new scene: which cars stand where, and the light
                for actor in spawned:
                    actor.destroy()
                spawned = []
                own = rng.random() < 0.4
                world.enable_environment_objects({o.id for o in statics}, own)
                boxes = list(static_boxes) if own else []
                share = rng.uniform(0.05, 0.3) if own else rng.uniform(0.1, 0.7)
                for stall in stalls:
                    if rng.random() > share or any(
                            stall.contains(b.mean(axis=0)) for b in boxes):
                        continue
                    x, y = stall.world(rng.normal(0.0, 0.25), rng.normal(0.0, 0.12))
                    yaw = stall.yaw_out + rng.choice((0.0, 180.0)) + rng.normal(0.0, 3.0)
                    actor = world.try_spawn_actor(
                        cars[int(rng.integers(len(cars)))],
                        carla.Transform(carla.Location(float(x), float(y), ground_z + 0.3),
                                        carla.Rotation(yaw=float(yaw))))
                    if actor is not None:
                        spawned.append(actor)
                for _ in range(25):  # onto their wheels, then they stay
                    world.tick()
                for actor in spawned:
                    actor.set_simulate_physics(False)
                    t, e = actor.get_transform(), actor.bounding_box.extent
                    boxes.append(box_corners(t.location.x, t.location.y, t.rotation.yaw,
                                             e.x, e.y))
                occupied = [any(stall.contains(b.mean(axis=0)) for b in boxes)
                            for stall in stalls]
                # The light of the runs (ClearNoon) four times in ten; the sun
                # elsewhere, clouds and a wet ground otherwise
                weather = carla.WeatherParameters()  # the preset itself is shared
                for name in dir(carla.WeatherParameters.ClearNoon):
                    value = getattr(carla.WeatherParameters.ClearNoon, name)
                    if isinstance(value, float):
                        setattr(weather, name, value)
                if rng.random() > 0.4:
                    weather.cloudiness = float(rng.uniform(0, 90))
                    weather.sun_altitude_angle = float(rng.uniform(12, 85))
                    weather.sun_azimuth_angle = float(rng.uniform(0, 360))
                    weather.wetness = float(rng.choice((0.0, 0.0, 40.0)))
                world.set_weather(weather)
            free = [i for i, taken in enumerate(occupied) if not taken]
            x, y, yaw = ego_pose(rng, stalls, free, wheelbase)
            yaw_rad = math.radians(yaw)
            body = box_corners(x + offset * math.cos(yaw_rad), y + offset * math.sin(yaw_rad),
                               yaw, extent.x, extent.y)
            if not (area["x"][0] + 0.3 < body[:, 0].min() and body[:, 0].max() < area["x"][1] - 0.3
                    and area["y"][0] + 0.3 < body[:, 1].min()
                    and body[:, 1].max() < area["y"][1] - 0.3):
                continue
            if any(boxes_overlap(body, b) for b in boxes):
                continue
            ego.set_transform(carla.Transform(carla.Location(x, y, height),
                                              carla.Rotation(yaw=yaw)))
            world.tick()
            frame = world.tick()  # the image of the tick after the move
            images = [camera.image(frame)[:, :, :3] for camera in cameras[0::2]]
            depths = [decode_depth(camera.image(frame)) for camera in cameras[1::2]]
            capture = Pose(x, y, height, settled.rotation.roll, settled.rotation.pitch, yaw)
            # As the device gets them: JPEG, at the qualities the runs use
            quality = [cv2.IMWRITE_JPEG_QUALITY, int(rng.choice((30, 50, 75, 75, 90)))]
            if stitcher:
                image = stitcher.compose([
                    cv2.imdecode(cv2.imencode(".jpg", sent, quality)[1], cv2.IMREAD_COLOR)
                    for sent in images])
                view, quality = BevView(stitcher, capture, depths), \
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
            else:
                image, view = images[0], CameraView(rigs[0], capture, depths[0])
            lines = labels_of(stalls, occupied, ends, view, ground_z)
            part = "val" if (made // args.scene) % 10 == 9 else "train"
            name = "%s_%05d" % (town.lower(), made)
            cv2.imwrite(os.path.join(args.out, "images", part, name + ".jpg"), image, quality)
            with open(os.path.join(args.out, "labels", part, name + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            if previews < args.preview and made % max(1, args.count // max(1, args.preview)) == 0:
                drawn = image.copy()
                for line in lines:
                    n = [float(v) for v in line.split()]
                    colour = ((0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255))[int(n[0])]
                    cv2.rectangle(drawn, (int((n[1] - n[3] / 2) * view.width),
                                          int((n[2] - n[4] / 2) * view.height)),
                                  (int((n[1] + n[3] / 2) * view.width),
                                   int((n[2] + n[4] / 2) * view.height)), colour, 1)
                    for k in range(4):
                        if n[7 + 3 * k]:
                            at = (int(n[5 + 3 * k] * view.width),
                                  int(n[6 + 3 * k] * view.height))
                            cv2.circle(drawn, at, 6, colour, 2)
                            cv2.putText(drawn, str(k), at, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (255, 255, 255), 2)
                cv2.imwrite(os.path.join(args.out, "preview", name + ".jpg"), drawn)
                previews += 1
            made += 1
            if made % 200 == 0:
                print("%d of %d" % (made, args.count), flush=True)
        with open(os.path.join(args.out, "data.yaml"), "w") as f:
            yaml.safe_dump({"path": os.path.abspath(args.out), "train": "images/train",
                            "val": "images/val", "kpt_shape": [4, 3],
                            # Mirrored, the corners at +v and -v change places
                            "flip_idx": [1, 0, 3, 2],
                            "names": {0: "empty", 1: "occupied", 2: "mouth", 3: "back"}},
                           f, sort_keys=False)
        print("%d images in %s" % (made, args.out))
    finally:
        for camera in cameras:
            camera.close()
        for actor in spawned + ([ego] if ego else []):
            actor.destroy()
        world.enable_environment_objects({o.id for o in statics}, True)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        world.apply_settings(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    commands = parser.add_subparsers(dest="command", required=True)
    m = commands.add_parser("map")
    m.add_argument("--map", default="Town05")
    m.add_argument("--centre", type=float, nargs=2, default=[-7.0, -30.0], metavar=("X", "Y"))
    m.add_argument("--span", type=float, default=80.0, help="metres across, looked at from above")
    k = commands.add_parser("make")
    k.add_argument("--map", default="Town05")
    k.add_argument("--config", default=os.path.join(HERE, "config", "carla.yaml"))
    k.add_argument("--out", required=True)
    k.add_argument("--count", type=int, default=4400)
    k.add_argument("--scene", type=int, default=20, help="images per arrangement of cars")
    k.add_argument("--seed", type=int, default=1)
    k.add_argument("--preview", type=int, default=12,
                   help="images written again with their labels drawn, to look at")
    k.add_argument("--bev", action="store_true",
                   help="images of the bird's-eye view instead of the rear camera's")
    args = parser.parse_args()
    return measure(args) if args.command == "map" else make(args)


if __name__ == "__main__":
    sys.exit(main())

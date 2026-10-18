#!/usr/bin/env python3
"""Make the images the lane model is trained on, with the lane boundaries
the map gives for each. Run by hand, with a CARLA server on port 2000;
nothing calls this.

Usage:
  simulation/lane-dataset.py --out data/lanes [--map Town04] [--count 6000]
                             [--seed 1] [--preview 16]

The front camera of simulation/config/carla.yaml (calibration), on the ego
vehicle, on every driving lane of the map: mostly near the middle of a lane
and along it, as a vehicle that keeps its lane, and one time in three
anywhere across it, as one that changes lane. Other cars stand ahead in
half of the images. The light is that of the runs four times in ten; a sun
elsewhere, clouds, a wet road, night and rain otherwise.

The labels are the four lines the lane model has a place for, in its order:
the far side of the lane to the left, the two sides of the lane the vehicle
is in, the far side of the lane to the right. The map says where a lane's
sides are; they are followed 100 m ahead, straight on where a lane
branches, and drawn into the image with the pose the
vehicle has. A line is labelled from the first place the map has paint on
it to the last: through the gaps of a broken line, through an opening
without paint and behind a car, as a person labelling it would, but not
before the paint begins or after it ends.

Writes OUT/images/{train,val}/NAME.jpg and OUT/{train,val}.jsonl, a line
per image:

    {"image": "images/train/NAME.jpg", "lanes": [[x, ...] * 4], ...}

`lanes` holds, for each line, its column in the image at each of the rows
of lane.anchor_rows(height), or -1. Stretches of 50 m of a lane are in one
part or the other, one in ten in val: a val image is not of a place a
train image shows from 2 m further on.

The map is truth: it labels these images. Nothing in the control path
reads it.
"""

import argparse
import json
import math
import os
import sys
import zlib

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from video_analytics_simulation import lane  # noqa: E402
from video_analytics_simulation.calibration import Pose, Rig  # noqa: E402

REACH_M = 100.0
STEP_M = 1.0
COLOURS = ((255, 160, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255))
RAIN = ("HardRainNoon", "MidRainyNoon", "WetCloudyNoon", "SoftRainSunset")


def lines_ahead(carla, start):
    """The four lines from this waypoint on, each as world points (n, 3) of
    the stretch that is labelled."""
    points = [[], [], [], []]
    painted = [[], [], [], []]
    ended = [False] * 4
    none = carla.LaneMarkingType.NONE
    w = start
    for _ in range(int(REACH_M / STEP_M) + 1):
        at = w.transform.location
        centre = np.array([at.x, at.y, at.z])
        r = w.transform.get_right_vector()
        right = np.array([r.x, r.y, r.z])
        half = w.lane_width / 2.0
        rows = [None,
                (centre - right * half, w.left_lane_marking.type != none),
                (centre + right * half, w.right_lane_marking.type != none),
                None]
        for slot, sign, other in ((0, -1.0, w.get_left_lane()), (3, 1.0, w.get_right_lane())):
            if other is None or other.lane_type != carla.LaneType.Driving:
                continue
            # The far side of a lane that runs the other way is its right
            same = (other.lane_id > 0) == (w.lane_id > 0)
            marking = (other.right_lane_marking if (sign > 0) == same
                       else other.left_lane_marking)
            rows[slot] = (centre + sign * right * (half + other.lane_width),
                          marking.type != none)
        for slot, row in enumerate(rows):
            # A line ends where the lane beside ends: what comes after a gap
            # is another line
            if row is None:
                ended[slot] = len(points[slot]) > 0
            if row is None or ended[slot]:
                continue
            points[slot].append(row[0])
            painted[slot].append(row[1])
        following = w.next(STEP_M)
        if not following:
            break
        if len(following) > 1:
            # Straight on: branches leave each other slowly, so each is
            # judged by where it points 15 m further
            yaw = w.transform.rotation.yaw
            following.sort(key=lambda n: abs(
                ((n.next(15.0) or [n])[0].transform.rotation.yaw - yaw + 180.0)
                % 360.0 - 180.0))
        w = following[0]
    lines = []
    for slot in range(4):
        paint = np.nonzero(painted[slot])[0]
        lines.append(np.array(points[slot][paint[0]:paint[-1] + 1]) if len(paint)
                     else np.zeros((0, 3)))
    return lines


def label(lines, capture, rig, rows):
    """Each line's column in the image at each of the rows, or -1."""
    labels = []
    for points in lines:
        if len(points) < 2:
            labels.append([-1.0] * len(rows))
            continue
        uv, depth = rig.project(capture.from_parent(points))
        columns = lane.boundary_at_rows(uv[depth > 0.5], rows, rig.width)
        labels.append([round(float(c), 1) for c in columns])
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town04")
    parser.add_argument("--config", default=os.path.join(HERE, "config", "carla.yaml"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=6000)
    parser.add_argument("--scene", type=int, default=10, help="images per light")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--preview", type=int, default=16,
                        help="images written again with their labels drawn, to look at")
    args = parser.parse_args()

    import carla
    import cv2
    with open(args.config) as f:
        config = yaml.safe_load(f)
    rig = Rig.from_config(config["calibration"])
    rows = lane.anchor_rows(rig.height)
    rng = np.random.default_rng(args.seed)
    for part in ("images/train", "images/val", "preview"):
        os.makedirs(os.path.join(args.out, part), exist_ok=True)

    sys.path.insert(0, HERE)
    dataset = __import__("stall-dataset")  # its server connection and camera
    _, client, world, original = dataset.connect(args, args.map)
    town = world.get_map()
    waypoints = [w for w in town.generate_waypoints(2.0)
                 if w.lane_type == carla.LaneType.Driving]
    print("%d places on the driving lanes of %s" % (len(waypoints), args.map))
    library = world.get_blueprint_library()
    cars = [b for b in library.filter("vehicle.*")
            if b.get_attribute("number_of_wheels").as_int() == 4]
    under = carla.Transform(carla.Location(0.0, 0.0, -100.0))  # where unused cars wait
    spawned, camera, ego = [], None, None
    files = {part: open(os.path.join(args.out, part + ".jsonl"), "w")
             for part in ("train", "val")}
    try:
        first = waypoints[0].transform
        ego = world.spawn_actor(
            library.find(config["carla"].get("ego_blueprint", "vehicle.tesla.model3")),
            carla.Transform(first.location + carla.Location(z=0.3), first.rotation))
        for _ in range(40):
            world.tick()
        # Where its origin is over the road, and how it leans, on its wheels
        settled = ego.get_transform()
        height = settled.location.z - first.location.z
        lean = settled.rotation.pitch - first.rotation.pitch
        ego.set_simulate_physics(False)
        for i in rng.choice(len(cars), 6, replace=False):
            actor = world.try_spawn_actor(cars[int(i)], carla.Transform(
                carla.Location(first.location.x, first.location.y,
                               first.location.z + 20.0 + 5.0 * len(spawned))))
            if actor is not None:
                actor.set_simulate_physics(False)
                spawned.append(actor)
        camera = dataset.Camera(
            carla, world, "sensor.camera.rgb",
            carla.Transform(carla.Location(rig.x, rig.y, rig.z),
                            carla.Rotation(roll=rig.roll_deg, pitch=rig.pitch_deg,
                                           yaw=rig.yaw_deg)),
            rig.width, rig.height, rig.fov_deg, ego)

        made = previews = 0
        counts = {"train": 0, "val": 0}
        weather_name = ""
        while made < args.count:
            if made % args.scene == 0:
                kind = rng.random()
                night = False
                weather = carla.WeatherParameters()  # the presets are shared
                preset = (carla.WeatherParameters.ClearNoon if kind < 0.7 else
                          carla.WeatherParameters.ClearNight if kind < 0.85 else
                          getattr(carla.WeatherParameters, str(rng.choice(RAIN))))
                for name in dir(preset):
                    if isinstance(getattr(preset, name), float):
                        setattr(weather, name, getattr(preset, name))
                if kind < 0.4:
                    weather_name = "noon"
                elif kind < 0.7:
                    weather_name = "day"
                    weather.cloudiness = float(rng.uniform(0, 90))
                    weather.sun_altitude_angle = float(rng.uniform(8, 85))
                    weather.sun_azimuth_angle = float(rng.uniform(0, 360))
                    weather.wetness = float(rng.choice((0.0, 0.0, 40.0)))
                elif kind < 0.85:
                    weather_name, night = "night", True
                    weather.cloudiness = float(rng.uniform(0, 60))
                else:
                    weather_name = "rain"
                world.set_weather(weather)
                lights = carla.VehicleLightState.NONE
                if night and rng.random() < 0.5:
                    lights = carla.VehicleLightState(carla.VehicleLightState.LowBeam
                                                     | carla.VehicleLightState.Position)
                    weather_name = "night_lights"
                ego.set_light_state(lights)

            w = waypoints[int(rng.integers(len(waypoints)))]
            half = w.lane_width / 2.0
            keeping = rng.random() < 0.67
            across = float(np.clip(rng.normal(0.0, 0.25), -0.8, 0.8) if keeping
                           else rng.uniform(-half, half))
            turned = float(np.clip(rng.normal(0.0, 1.5 if keeping else 3.0), -8.0, 8.0))
            base = w.transform
            r = base.get_right_vector()
            up = base.get_up_vector()
            ego.set_transform(carla.Transform(
                carla.Location(base.location.x + r.x * across + up.x * height,
                               base.location.y + r.y * across + up.y * height,
                               base.location.z + r.z * across + up.z * height),
                carla.Rotation(roll=base.rotation.roll,
                               pitch=base.rotation.pitch + lean + float(rng.normal(0.0, 0.3)),
                               yaw=base.rotation.yaw + turned)))
            # Cars ahead, in this lane and the ones beside it
            ahead = int(rng.integers(1, 5)) if rng.random() < 0.5 else 0
            for i, actor in enumerate(spawned):
                place = None
                if i < ahead:
                    side = (w, w, w.get_left_lane(), w.get_right_lane())[int(rng.integers(4))]
                    if side is not None and side.lane_type == carla.LaneType.Driving \
                            and (side.lane_id > 0) == (w.lane_id > 0):
                        place = side.next(float(rng.uniform(9.0, 70.0)))
                if place:
                    t = place[0].transform
                    actor.set_transform(carla.Transform(
                        t.location + carla.Location(z=0.05),
                        carla.Rotation(roll=t.rotation.roll, pitch=t.rotation.pitch,
                                       yaw=t.rotation.yaw + float(rng.normal(0.0, 2.0)))))
                else:
                    actor.set_transform(under)
            # The exposure follows the light: longer after a new one
            for _ in range(12 if made % args.scene == 0 else 1):
                world.tick()
            frame = world.tick()
            image = camera.image(frame)[:, :, :3]
            t = ego.get_transform()
            capture = Pose(t.location.x, t.location.y, t.location.z, t.rotation.roll,
                           t.rotation.pitch, t.rotation.yaw)
            at = town.get_waypoint(t.location, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
            labels = label(lines_ahead(carla, at), capture, rig, rows)
            if not any(x >= 0 for line in labels[1:3] for x in line):
                continue  # no side of its own lane in the image: nothing to learn
            stretch = "%s/%d/%d/%d" % (args.map, w.road_id, w.lane_id, int(w.s // 50.0))
            part = "val" if zlib.crc32(stretch.encode()) % 10 == 0 else "train"
            name = "%s_%05d" % (args.map.lower(), made)
            path = os.path.join("images", part, name + ".jpg")
            cv2.imwrite(os.path.join(args.out, path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            files[part].write(json.dumps({
                "image": path, "width": rig.width, "height": rig.height, "lanes": labels,
                "map": args.map, "road": w.road_id, "lane": w.lane_id, "s": round(w.s, 1),
                "junction": w.is_junction, "across_m": round(across, 2),
                "turned_deg": round(turned, 2), "cars": ahead, "light": weather_name,
            }) + "\n")
            counts[part] += 1
            if previews < args.preview and made % max(1, args.count // max(1, args.preview)) == 0:
                drawn = image.copy()
                for line, colour in zip(labels, COLOURS):
                    for x, y in zip(line, rows):
                        if x >= 0:
                            cv2.circle(drawn, (int(x), y), 4, colour, -1)
                cv2.imwrite(os.path.join(args.out, "preview", name + ".jpg"), drawn)
                previews += 1
            made += 1
            if made % 200 == 0:
                print("%d of %d" % (made, args.count), flush=True)
        print("%d train and %d val images in %s" % (counts["train"], counts["val"], args.out))
    finally:
        for f in files.values():
            f.close()
        if camera:
            camera.close()
        for actor in spawned + ([ego] if ego else []):
            actor.destroy()
        world.set_weather(carla.WeatherParameters.ClearNoon)
        world.apply_settings(original)


if __name__ == "__main__":
    sys.exit(main())

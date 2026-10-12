#!/usr/bin/env python3
"""Find one parking stall for the park show and print it. Run once, by hand;
nothing calls this.

Usage: simulation/probe-stall.py [--map Town05] [--row-x X] [--y-from Y] [--y-to Y]
                                 [--yaw-out DEG] [--margin M] [--k K] [--turn-gain G]
                                 [--fov DEG] [--camera-x M]

CARLA has no list of parking stalls. Its map has lanes of type Parking, which
in Town05 are strips along the kerbs of streets, not the stalls of the car
park. So this measures the paint: rays cast down along a row of stalls find
the dividers (the server labels them RoadLines), the cars standing in the row
and how far the open ground reaches in front of it.

  --row-x, --y-from, --y-to   a line through the middle of a row of stalls,
                              along world y (default: the row of the Town05
                              car park that opens to its widest aisle)
  --yaw-out                   the direction out of the stalls, degrees
  --margin                    a stall must be wider than the vehicle by this
                              much, metres (default 1.0)
  --k                         share of the full steering lock used to turn in
  --turn-gain                 measured wheel angle over the nominal one
                              (scenario vehicle_response, the reverse step)

  --fov, --camera-x           the front camera, as in carla.yaml

A stall is a candidate when it is empty, its neighbours on both sides are
empty, it is wide enough, and the aisle in front of it is wider than the
turning radius. The vehicle drives along the aisle, passes the stall and
turns into it backwards; before that its front camera must have had all four
corners of the stall in the image, from where it starts. Prints the widest
candidate for which that holds, as the lines to put into
simulation/config/carla.yaml and carla-scenarios.yaml. Exit code 1 when there
is none: no numbers are made up.
"""

import argparse
import math
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="Town05")
    parser.add_argument("--row-x", type=float, default=9.85)
    parser.add_argument("--y-from", type=float, default=-49.0)
    parser.add_argument("--y-to", type=float, default=-12.0)
    parser.add_argument("--yaw-out", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--k", type=float, default=0.45)
    parser.add_argument("--turn-gain", type=float, default=0.88)
    parser.add_argument("--fov", type=float, default=52.0)
    parser.add_argument("--camera-x", type=float, default=0.6)
    parser.add_argument("--blueprint", default="vehicle.tesla.model3")
    args = parser.parse_args()

    import carla
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != args.map:
        world = client.load_world(args.map)
    town = world.get_map()
    lines = carla.CityObjectLabel.RoadLines
    flat = (lines, carla.CityObjectLabel.Ground, carla.CityObjectLabel.Roads)

    def down(x, y):
        """(height, label) of what is at x, y, seen from above."""
        hit = world.project_point(carla.Location(x, y, 20.0), carla.Vector3D(0, 0, -1), 40.0)
        return (hit.location.z, hit.label) if hit else (None, None)

    # The vehicle that is to fit: its box and its wheels, as the simulator
    # reads them
    vehicle = world.spawn_actor(world.get_blueprint_library().find(args.blueprint),
                                carla.Transform(carla.Location(0, 0, 300.0)))
    vehicle.set_simulate_physics(False)
    try:
        wheels = vehicle.get_physics_control().wheels
        wheelbase = math.hypot(wheels[0].position.x - wheels[2].position.x,
                               wheels[0].position.y - wheels[2].position.y) / 100.0
        max_steer = wheels[0].max_steer_angle
        width = 2.0 * vehicle.bounding_box.extent.y
        length = 2.0 * vehicle.bounding_box.extent.x
    finally:
        vehicle.destroy()
    radius = wheelbase / math.tan(math.radians(args.turn_gain * args.k * max_steer))
    print("vehicle %s: %.2f x %.2f m, wheelbase %.2f m, full lock %.0f deg; rear axle "
          "radius at %.2f of the lock: %.2f m" % (args.blueprint, length, width, wheelbase,
                                                  max_steer, args.k, radius))

    # Lanes of type Parking, to say where they are and that they are not stalls
    near = town.get_waypoint(carla.Location(args.row_x, (args.y_from + args.y_to) / 2, 0),
                             project_to_road=True, lane_type=carla.LaneType.Parking)
    if near is not None:
        l = near.transform.location
        print("nearest lane of type Parking: road %d, %.0f m from the row, %.1f m wide, "
              "along a street: not a stall" % (
                  near.road_id, math.hypot(l.x - args.row_x,
                                           l.y - (args.y_from + args.y_to) / 2),
                  near.lane_width))

    out = math.radians(args.yaw_out)
    if abs(math.sin(out)) > 1e-6:
        raise SystemExit("this probe measures a row along world y: --yaw-out 0 or 180")
    sign = 1.0 if math.cos(out) > 0 else -1.0
    step = 0.02
    count = int(round((args.y_to - args.y_from) / step))
    row = [(args.y_from + i * step,) + down(args.row_x, args.y_from + i * step)
           for i in range(count + 1)]
    ground = sorted(z for _, z, label in row if label in flat and z is not None)
    if not ground:
        raise SystemExit("nothing flat under the row: wrong --row-x?")
    ground = ground[len(ground) // 2]
    # Dividers: runs of paint across the line
    dividers, run = [], []
    for y, _, label in row:
        if label == lines:
            run.append(y)
        elif run:
            dividers.append((run[0] + run[-1]) / 2.0)
            run = []
    print("ground at z = %.3f; %d dividers at y = %s" % (
        ground, len(dividers), ", ".join("%.2f" % y for y in dividers)))

    def occupied(low, high):
        return any(low < y < high and (label not in flat or z > ground + 0.2)
                   for y, z, label in row)

    def extent_along_x(y):
        """How far the divider at y reaches towards the aisle and away from it."""
        ends = []
        for direction in (sign, -sign):
            x = args.row_x
            while abs(x - args.row_x) < 8.0 and down(x + direction * step, y)[1] == lines:
                x += direction * step
            ends.append(x)
        return ends  # [mouth, back]

    def aisle(mouth_x, y):
        """Open, level ground in front of the stall, up to the first thing
        that stands on it. A fence of thin bars is missed from above: rays
        along the ground, at several heights and offsets, find it."""
        x = mouth_x
        while abs(x - mouth_x) < 30.0:
            z, label = down(x + sign * 0.05, y)
            if z is None or label not in flat or abs(z - ground) > 0.08:
                break
            x += sign * 0.05
        free = abs(x - mouth_x)
        for i in range(-15, 16):
            for height in (0.35, 0.55, 0.75, 0.95):
                hits = world.cast_ray(
                    carla.Location(mouth_x + sign * 0.2, y + 0.1 * i, ground + height),
                    carla.Location(mouth_x + sign * 30.0, y + 0.1 * i, ground + height))
                if hits:
                    free = min(free, abs(hits[0].location.x - mouth_x))
        return free

    def along_aisle(x, y, direction):
        """How far the vehicle's width of open, level ground goes along y."""
        while abs(y) < 500.0:
            for offset in (-width / 2 - 0.3, 0.0, width / 2 + 0.3):
                z, label = down(x + offset, y + direction * 0.1)
                if z is None or label not in flat or abs(z - ground) > 0.08:
                    return y
            y += direction * 0.1

    stalls = []
    for low, high in zip(dividers, dividers[1:]):
        stalls.append({"low": low, "high": high, "width": high - low,
                       "occupied": occupied(low + 0.2, high - 0.2)})
    candidates = []
    for i, stall in enumerate(stalls):
        why = []
        if stall["occupied"]:
            why.append("occupied")
        if i == 0 or i == len(stalls) - 1:
            why.append("end of the row")
        elif stalls[i - 1]["occupied"] or stalls[i + 1]["occupied"]:
            why.append("a neighbour is occupied")
        if stall["width"] <= width + args.margin:
            why.append("%.2f m wide, vehicle %.2f m + %.2f m" % (
                stall["width"], width, args.margin))
        centre_y = (stall["low"] + stall["high"]) / 2.0
        if not why:
            mouth, back = extent_along_x(stall["low"])
            stall.update(mouth=mouth, back=back, aisle=aisle(mouth, centre_y))
            if stall["aisle"] < radius:
                why.append("aisle %.1f m, turning radius %.1f m" % (stall["aisle"], radius))
        print("stall %2d  y %.2f..%.2f  %s" % (i, stall["low"], stall["high"],
                                                "; ".join(why) or "candidate"))
        if not why:
            candidates.append(stall)
    if not candidates:
        print("no stall in this row can be used", file=sys.stderr)
        return 1

    swing = math.hypot(radius + width / 2.0, wheelbase / 2.0 + length / 2.0) - radius
    chosen = None
    for stall in sorted(candidates, key=lambda s: -s["width"]):
        depth = abs(stall["mouth"] - stall["back"])
        centre_x = (stall["mouth"] + stall["back"]) / 2.0
        centre_y = (stall["low"] + stall["high"]) / 2.0
        mouth_m = abs(stall["mouth"] - centre_x)
        # The track in the aisle. Not farther out than where the front of
        # the vehicle, swinging out while it turns in backwards, stays 0.3 m
        # inside the aisle. Not nearer than where the turn ends 1 m inside
        # the mouth: by then the vehicle is within 11 degrees of straight,
        # and the stalls either side are empty. The nearest track is taken:
        # from it the camera sees the back of the stall at the smallest angle.
        track = mouth_m - 1.0 + radius
        if track > mouth_m + stall["aisle"] - swing - 0.3:
            print("stall at y %.2f: aisle %.1f m, too narrow to turn in on a radius of "
                  "%.1f m with the front swinging out %.1f m" % (
                      centre_y, stall["aisle"], radius, swing))
            continue
        track_x = centre_x + sign * track
        seen = []
        for heading in (-1.0, 1.0):
            # Starts where the aisle begins, half a metre inside it
            first, last = along_aisle(track_x, centre_y, -heading), \
                along_aisle(track_x, centre_y, heading)
            start_y = first + heading * (length / 2 + 0.5)
            camera_y = start_y + heading * args.camera_x
            angles = [math.degrees(math.atan2(abs(track_x - x), heading * (y - camera_y)))
                      for x in (stall["mouth"], stall["back"])
                      for y in (stall["low"], stall["high"])]
            # Room beyond the stall to stop one radius past its axis
            beyond = heading * (last - (centre_y + heading * (
                radius + wheelbase / 2 + length / 2)))
            seen.append("%.1f" % max(angles))
            if max(angles) < args.fov / 2.0 - 0.5 and beyond > 0.5:
                chosen = (stall, depth, centre_x, centre_y, track_x, heading, start_y,
                          max(angles))
                break
        if chosen:
            break
        print("stall at y %.2f: from the two ends of the aisle its corners are up to %s "
              "deg off the camera's axis, the image ends at %.1f" % (
                  centre_y, " and ".join(seen), args.fov / 2.0))
    if not chosen:
        print("no stall in this row can be used", file=sys.stderr)
        return 1
    stall, depth, centre_x, centre_y, track_x, heading, start_y, widest = chosen
    print()
    print("# simulation/config/carla.yaml, under routes:")
    print("  carpark: {x: %.2f, y: %.2f, z: %.3f, yaw_deg: %.1f}" % (
        track_x, start_y, ground, 90.0 * heading))
    print("# simulation/config/carla-scenarios.yaml, under the scenario's park:")
    print("    stall: {x: %.2f, y: %.2f, yaw_deg: %.1f, length_m: %.2f, width_m: %.2f}" % (
        centre_x, centre_y, args.yaw_out, depth, stall["width"]))
    print("# measured on the paint by ray casts, %s; margin to the vehicle %.2f m; "
          "aisle %.1f m; both neighbours empty; from the start its corners are within "
          "%.1f deg of the camera's axis" % (
              town.name.split("/")[-1], stall["width"] - width, stall["aisle"], widest))
    return 0


if __name__ == "__main__":
    sys.exit(main())

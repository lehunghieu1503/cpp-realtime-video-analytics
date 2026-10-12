import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from video_analytics_simulation import protocol  # noqa: E402

MS = 1000000
LIMITS = {"age_limit_ns": 200 * MS, "pose_budget_ns": 100 * MS,
          "wall_lease_ns": 300 * MS}


def command(**changes):
    c = {"run_id": "r1", "source": "carla-front", "command_seq": 5,
         "basis_sim_ns": 900 * MS, "control_sim_ns": 1000 * MS, "control_frame": 30,
         "valid_until_sim_ns": 1100 * MS, "throttle": 0.3, "brake": 0.0,
         "steer": 0.1, "reason": ""}
    c.update(changes)
    return c


def check(c, last_seq=4, now=1000 * MS, basis=True, control=True):
    return protocol.check_command(c, "r1", "carla-front", last_seq, now, LIMITS,
                                  lambda ns: basis, lambda ns, frame: control)


class Identity(unittest.TestCase):
    def test_sim_ns(self):
        self.assertEqual(protocol.sim_ns_of(0.0), 0)
        self.assertEqual(protocol.sim_ns_of(1 / 30), 33333333)
        # 46 ticks of 1/30 s, as CARLA adds them up
        self.assertEqual(protocol.sim_ns_of(1.5333333333), 1533333333)
        self.assertTrue(protocol.RUN_ID.match(protocol.new_run_id()))
        self.assertNotEqual(protocol.new_run_id(), protocol.new_run_id())

    def test_topics_are_per_run_and_source(self):
        a = protocol.topics("r1", "carla-front")
        self.assertEqual(a["perception"], "simulation/r1/carla_front/perception")
        self.assertEqual(a["control"], "simulation/r1/control")
        self.assertNotEqual(a["depth"], protocol.topics("r2", "carla-front")["depth"])


class LedgerTest(unittest.TestCase):
    def test_a_lost_frame_stays_in_the_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "frames.jsonl")
            ledger = protocol.Ledger(path)
            for frame in range(10, 16):
                ledger.update(frame, "tick", sim_ns=frame * 33, tick=frame - 10)
            ledger.register(10, 330, 1)           # sent
            ledger.update(10, "depth")
            ledger.update(10, "paired")
            ledger.update(10, "sent", jpeg_bytes=1000)
            ledger.register(11, 363, 2)           # depth never came
            ledger.update(11, "dropped", reason="missing_depth")
            ledger.register(12, 396, 3)           # callback queue was full
            ledger.update(12, "dropped", reason="callback_queue_full", kind="rgb")
            ledger.register(13, 429, 4)           # paired, sender queue full
            ledger.update(13, "depth")
            ledger.update(13, "paired")
            ledger.update(13, "dropped", reason="sender_queue_full")
            ledger.update(14, "depth")            # RGB was never produced
            ledger.update(14, "dropped", reason="missing_rgb")
            ledger.close()

            frames, ticks = protocol.read_ledger(path)
            self.assertEqual(len(ticks), 6)
            generated = [f for f, e in frames.items() if e["registered"]]
            self.assertEqual(sorted(generated), [10, 11, 12, 13])
            self.assertEqual(frames[10]["status"], "sent")
            self.assertEqual(frames[11]["reason"], "missing_depth")
            self.assertEqual(frames[12]["reason"], "callback_queue_full")
            self.assertEqual(frames[13]["status"], "dropped")
            # Produced by the camera or not: two different losses
            self.assertFalse(frames[14]["registered"])
            self.assertNotIn(15, frames)  # ticked, no image at all

            # A run directory is never reused
            with self.assertRaises(FileExistsError):
                protocol.Ledger(path)

    def test_a_failed_write_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = protocol.Ledger(os.path.join(directory, "frames.jsonl"))
            ledger.file.close()
            with self.assertRaises(ValueError):
                ledger.register(1, 33, 1)


class PairerTest(unittest.TestCase):
    def test_pairs_by_frame_not_by_arrival(self):
        pairer = protocol.Pairer()
        self.assertEqual(pairer.add("depth", 7, 700, "d7"), (None, []))
        self.assertEqual(pairer.add("rgb", 8, 800, "c8"), (None, []))
        self.assertEqual(pairer.add("rgb", 7, 700, "c7"), ((7, 700, "c7", "d7"), []))
        self.assertEqual(pairer.add("depth", 8, 800, "d8"), ((8, 800, "c8", "d8"), []))

    def test_losses_are_reported(self):
        pairer = protocol.Pairer(max_waiting=2)
        pairer.add("rgb", 1, 100, "c1")
        pairer.add("depth", 2, 200, "d2")
        pair, lost = pairer.add("rgb", 3, 300, "c3")
        self.assertIsNone(pair)
        self.assertEqual(lost, [(1, "missing_depth")])
        pair, lost = pairer.add("rgb", 4, 400, "c4")
        self.assertEqual(lost, [(2, "missing_rgb")])
        pair, lost = pairer.add("depth", 3, 301, "d3")
        self.assertIsNone(pair)
        self.assertEqual(lost, [(3, "timestamp_mismatch")])
        pair, lost = pairer.add("rgb", 4, 400, "again")
        self.assertEqual(lost, [(4, "duplicate_rgb")])
        self.assertLessEqual(len(pairer.waiting), 2)


class CacheTest(unittest.TestCase):
    def test_bounded(self):
        cache = protocol.LiveCache(3)
        for ns in range(5):
            cache.put(ns, "e%d" % ns)
        self.assertEqual(len(cache), 3)
        self.assertEqual(cache.evicted, 2)
        self.assertIsNone(cache.get(1))
        self.assertEqual(cache.get(4), "e4")
        self.assertIn(2, cache)


class CommandTest(unittest.TestCase):
    def test_a_good_command(self):
        self.assertEqual(check(command()), "")
        self.assertEqual(check(command(throttle=0.0, brake=1.0)), "")

    def test_rejections(self):
        cases = [
            (dict(run_id="r0"), {}, "wrong_run"),
            (dict(source="rear"), {}, "wrong_source"),
            (dict(command_seq=4), {}, "old_sequence"),
            (dict(command_seq=3), {}, "old_sequence"),
            (dict(command_seq=5.0), {}, "bad_command_seq"),
            (dict(throttle=float("nan")), {}, "bad_throttle"),
            (dict(steer=float("inf")), {}, "bad_steer"),
            (dict(brake=True), {}, "bad_brake"),
            (dict(throttle=1.2), {}, "out_of_range"),
            (dict(steer=-1.01), {}, "out_of_range"),
            (dict(brake=0.2), {}, "throttle_and_brake"),
            (dict(reason=None), {}, "bad_reason"),
            (dict(control_sim_ns=1033 * MS), {}, "from_the_future"),
            (dict(basis_sim_ns=1001 * MS), {}, "from_the_future"),
            (dict(control_sim_ns=880 * MS), {}, "stale_control_pose"),
            ({}, dict(control=False), "unknown_control_pose"),
            ({}, dict(basis=False), "unknown_basis"),
            (dict(valid_until_sim_ns=1101 * MS), {}, "validity_too_long"),
            ({}, dict(now=1101 * MS, control=True), "expired"),
        ]
        for changes, context, reason in cases:
            c = command(**changes)
            if reason == "expired":
                c["control_sim_ns"] = 1101 * MS
            self.assertEqual(check(c, **context), reason, changes)
        for field in protocol.COMMAND_FIELDS:
            c = command()
            del c[field]
            self.assertEqual(check(c), "missing_" + field)
        self.assertEqual(check("go"), "not_an_object")

    def test_a_brake_command_needs_no_basis(self):
        # Braking is always allowed: before the first perception there is no
        # camera frame to rest on
        c = command(throttle=0.0, brake=1.0, basis_sim_ns=0, valid_until_sim_ns=0)
        self.assertEqual(check(c, basis=False), "")


class LeaseTest(unittest.TestCase):
    def setUp(self):
        self.lease = protocol.Lease("r1", "carla-front", LIMITS)
        self.known = dict(known_basis=lambda ns: True,
                          known_control=lambda ns, frame: True)

    def offer(self, c, now=1000 * MS, mono=0):
        return self.lease.offer(c, now, mono, **self.known)

    def test_brakes_until_armed_with_a_valid_command(self):
        self.assertEqual(self.lease.control(0, 0)["reason"], "no_command")
        self.assertEqual(self.offer(command()), "")
        applied = self.lease.control(1000 * MS, 10 * MS)
        self.assertEqual((applied["throttle"], applied["brake"], applied["reason"]),
                         (0.0, 1.0, "not_armed"))
        self.assertEqual(self.lease.arm({"run_id": "r0", "enable": True}), "wrong_run")
        self.assertEqual(self.lease.arm({"run_id": "r1"}), "not_an_arm_message")
        self.assertEqual(self.lease.arm({"run_id": "r1", "enable": 1}),
                         "not_an_arm_message")
        self.assertFalse(self.lease.armed)
        self.assertEqual(self.lease.arm({"run_id": "r1", "enable": True}), "")
        applied = self.lease.control(1000 * MS, 10 * MS)
        self.assertEqual((applied["throttle"], applied["brake"], applied["steer"]),
                         (0.3, 0.0, 0.1))

    def test_expires_in_simulation_time(self):
        self.lease.arm({"run_id": "r1", "enable": True})
        self.offer(command())
        self.assertEqual(self.lease.control(1100 * MS, 0)["reason"], "")
        applied = self.lease.control(1100 * MS + 1, 0)
        self.assertEqual((applied["throttle"], applied["brake"], applied["reason"]),
                         (0.0, 1.0, "command_expired"))

    def test_controller_silent_in_wall_time(self):
        self.lease.arm({"run_id": "r1", "enable": True})
        self.offer(command(), mono=0)
        self.assertEqual(self.lease.control(1000 * MS, 300 * MS)["reason"], "")
        # The simulation clock did not move, the wall clock did
        self.assertEqual(self.lease.control(1000 * MS, 301 * MS)["reason"],
                         "controller_silent")

    def test_a_rejected_command_extends_nothing(self):
        self.lease.arm({"run_id": "r1", "enable": True})
        self.offer(command(), mono=0)
        for bad in (command(command_seq=6, run_id="r0"),
                    command(command_seq=6, throttle=float("nan")),
                    command(command_seq=5)):
            self.assertNotEqual(self.offer(bad, mono=250 * MS), "")
        self.assertEqual(self.lease.control(1000 * MS, 301 * MS)["reason"],
                         "controller_silent")
        self.assertEqual(sum(self.lease.rejected.values()), 3)

    def test_stop_latches(self):
        self.lease.arm({"run_id": "r1", "enable": True})
        self.offer(command())
        self.assertEqual(self.lease.arm({"run_id": "r1", "stop": True}), "")
        self.assertEqual(self.lease.control(1000 * MS, 0)["reason"], "stopped")
        self.assertEqual(self.lease.arm({"run_id": "r1", "enable": True}),
                         "stopped_until_reset")
        self.offer(command(command_seq=6))
        self.assertEqual(self.lease.control(1000 * MS, 0)["brake"], 1.0)

    def test_a_commanded_brake_is_applied_unarmed(self):
        self.offer(command(throttle=0.0, brake=1.0, reason="obstacle:distance:car"))
        applied = self.lease.control(1000 * MS, 0)
        self.assertEqual((applied["brake"], applied["reason"]),
                         (1.0, "obstacle:distance:car"))


class ReverseAndParkingBrake(unittest.TestCase):
    """reverse and hand_brake: optional, false when left out, and counted as
    driving whatever the pedals say."""
    STANDING = dict(throttle=0.0, brake=1.0)

    def setUp(self):
        self.lease = protocol.Lease("r1", "carla-front", LIMITS)
        self.known = dict(known_basis=lambda ns: True,
                          known_control=lambda ns, frame: True)

    def offer(self, c, now=1000 * MS, mono=0, **known):
        return self.lease.offer(c, now, mono, **dict(self.known, **known))

    def arm(self):
        self.assertEqual(self.lease.arm({"run_id": "r1", "enable": True}), "")

    def test_a_command_without_the_two_fields_is_valid_and_they_are_false(self):
        self.assertEqual(check(command()), "")
        self.arm()
        self.offer(command())
        applied = self.lease.control(1000 * MS, 0)
        self.assertIs(applied["reverse"], False)
        self.assertIs(applied["hand_brake"], False)
        self.assertEqual(applied["throttle"], 0.3)

    def test_they_must_be_booleans(self):
        for field in ("reverse", "hand_brake"):
            for bad in (1, 0, "true", None, 1.0):
                self.assertEqual(check(command(**{field: bad})), "bad_" + field, bad)
            self.assertEqual(check(command(**{field: True})), "")
            self.assertEqual(check(command(**{field: False})), "")
        # Reverse does not allow both pedals either
        self.assertEqual(check(command(reverse=True, brake=0.5)), "throttle_and_brake")

    def test_standing_with_either_flag_is_checked_like_a_command_that_drives(self):
        for flag in ("reverse", "hand_brake"):
            forged = command(**self.STANDING, **{flag: True}, reason="parked")
            self.assertTrue(protocol.is_driving(forged))
            # A brake needs no basis; this is not just a brake
            self.assertEqual(check(dict(forged, basis_sim_ns=0, valid_until_sim_ns=0),
                                   basis=False), "unknown_basis")
            self.assertEqual(check(dict(forged, valid_until_sim_ns=1101 * MS)),
                             "validity_too_long")
            self.assertEqual(check(dict(forged, run_id="r0")), "wrong_run")
            self.assertEqual(check(dict(forged, control_sim_ns=1101 * MS), now=1101 * MS),
                             "expired")
        self.assertFalse(protocol.is_driving(command(**self.STANDING)))
        self.assertFalse(protocol.is_driving(command(**self.STANDING, reverse=False,
                                                     hand_brake=False)))

    def test_not_armed_neither_flag_reaches_the_vehicle(self):
        for flag in ("reverse", "hand_brake"):
            self.setUp()
            self.assertEqual(self.offer(command(**self.STANDING, **{flag: True},
                                                reason="parked")), "")
            applied = self.lease.control(1000 * MS, 0)
            self.assertEqual(applied["reason"], "not_armed")
            self.assertEqual((applied["throttle"], applied["brake"]), (0.0, 1.0))
            self.assertIs(applied["reverse"], False)
            self.assertIs(applied["hand_brake"], False)

    def test_a_parking_brake_from_another_run_is_rejected(self):
        self.arm()
        self.assertEqual(self.offer(command(**self.STANDING, hand_brake=True,
                                            run_id="r0", reason="parked")), "wrong_run")
        applied = self.lease.control(1000 * MS, 0)
        self.assertEqual(applied["reason"], "no_command")
        self.assertIs(applied["hand_brake"], False)

    def test_armed_and_in_date_reverse_reaches_the_vehicle(self):
        self.arm()
        self.assertEqual(self.offer(command(throttle=0.2, reverse=True, steer=-0.45)), "")
        applied = self.lease.control(1000 * MS, 0)
        self.assertEqual((applied["throttle"], applied["brake"], applied["steer"]),
                         (0.2, 0.0, -0.45))
        self.assertIs(applied["reverse"], True)
        self.assertIs(applied["hand_brake"], False)
        self.assertEqual(applied["reason"], "")
        # Standing in reverse gear, between two moves: still reverse, not "held"
        self.offer(command(**self.STANDING, reverse=True, command_seq=6))
        applied = self.lease.control(1000 * MS, 0)
        self.assertIs(applied["reverse"], True)
        self.assertEqual(applied["reason"], "")

    def test_parked_is_held_with_the_parking_brake(self):
        self.arm()
        self.offer(command(**self.STANDING, hand_brake=True, reason="parked"))
        applied = self.lease.control(1000 * MS, 0)
        self.assertEqual((applied["throttle"], applied["brake"]), (0.0, 1.0))
        self.assertIs(applied["hand_brake"], True)
        self.assertIs(applied["reverse"], False)
        self.assertEqual(applied["reason"], "parked")  # not a driving tick in the log

    def test_every_brake_the_lease_decides_on_has_both_flags_false(self):
        reverse = command(throttle=0.2, reverse=True, hand_brake=True)
        cases = {}
        cases["no_command"] = self.lease.control(0, 0)
        self.offer(reverse)
        cases["not_armed"] = self.lease.control(1000 * MS, 0)
        self.arm()
        cases["command_expired"] = self.lease.control(1100 * MS + 1, 0)
        cases["controller_silent"] = self.lease.control(1000 * MS, 301 * MS)
        self.offer(command(**self.STANDING, command_seq=6, reason="obstacle:distance:car"))
        cases["obstacle:distance:car"] = self.lease.control(1000 * MS, 0)
        self.offer(dict(reverse, command_seq=7))
        self.lease.arm({"run_id": "r1", "stop": True})
        cases["stopped"] = self.lease.control(1000 * MS, 0)
        for reason, applied in cases.items():
            self.assertEqual(applied["reason"], reason)
            self.assertEqual((applied["throttle"], applied["brake"]), (0.0, 1.0), reason)
            self.assertIs(applied["reverse"], False, reason)
            self.assertIs(applied["hand_brake"], False, reason)

    def test_the_reason_is_not_authority(self):
        # "parked" as a reason, without the flag: an ordinary brake
        self.offer(command(**self.STANDING, reason="parked"))
        applied = self.lease.control(1000 * MS, 0)
        self.assertIs(applied["hand_brake"], False)
        self.assertEqual(applied["reason"], "parked")


if __name__ == "__main__":
    unittest.main()

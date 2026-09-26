# 합성 랜드마크로 판정 로직을 검증한다.  실행: python -m unittest discover tests
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import posture_logic  # noqa: E402
import posture_tracker as pt  # noqa: E402
import replay  # noqa: E402
from db_manager import DBManager  # noqa: E402
from recorder import Recorder, read_recording  # noqa: E402

FPS = 30


def pose(ear_y=250, ear_half=30, tilt_dy=0, sh_half=100):
    """정면 상반신. 기본값이 바른 자세: 목 길이비 150/60=2.5."""
    return {
        "l_sh": [320 - sh_half, 400 + tilt_dy, 0.99], "r_sh": [320 + sh_half, 400 - tilt_dy, 0.99],
        "l_ear": [320 - ear_half, ear_y, 0.99], "r_ear": [320 + ear_half, ear_y, 0.99],
    }


UPRIGHT = pose()
TURTLE = pose(ear_y=265, ear_half=33)  # 귀가 내려오고 머리가 커짐 → 목 길이비 -18%
NOD = pose(ear_y=275)                   # 고개 숙임: 귀가 내려옴 → 목 길이비 -17% (거북목과 구분하지 않음)
TILT = pose(tilt_dy=14)                 # 어깨 기울기 0.14 (임계 0.07의 2배)


def calibrated_tracker():
    samples = [pt.calibration_sample(pt.measurement_from_points(UPRIGHT)) for _ in range(30)]
    calib = posture_logic.baseline_from_samples(samples)
    tracker = pt.PostureTracker()
    tracker.set_baseline(calib["baseline"], calib["thresholds"], 0.0)
    return tracker


def run(tracker, points, t0, seconds, brightness=100):
    """t0부터 seconds 동안 같은 자세를 FPS로 넣는다. (다음 시각, 이벤트 목록, 알림 목록)을 돌려준다."""
    events, notify = [], []
    n = int(round(seconds * FPS))
    for i in range(n):
        m = pt.measurement_from_points(points) if points else None
        res = tracker.process(t0 + i / FPS, m, brightness)
        events += res["events"]
        notify += res["notify"]
    return t0 + n / FPS, events, notify


class MetricTests(unittest.TestCase):
    def test_looking_down_counts_as_turtle(self):
        # 정면 카메라에서는 고개 숙임과 거북목이 구분되지 않아(실측) 숙임도 거북목 점수를 받는다
        b = {"width": 200, "tilt": 0, "neck": 2.5}
        for points in (TURTLE, NOD):
            m = pt.measurement_from_points(points)
            ev = posture_logic.evaluate_posture(m["l_sh"], m["r_sh"], m["l_ear"], m["r_ear"], baseline=b)
            self.assertGreater(ev["scores"]["turtle"], 1)

    def test_ears_hidden_means_no_turtle_score(self):
        b = {"width": 200, "tilt": 0, "neck": 2.5}
        m = pt.measurement_from_points(TURTLE)
        ev = posture_logic.evaluate_posture(m["l_sh"], m["r_sh"], m["l_ear"], m["r_ear"], ear_visible=False, baseline=b)
        self.assertIsNone(ev["scores"]["turtle"])
        self.assertIsNotNone(ev["scores"]["tilt"])

    def test_baseline_from_samples(self):
        self.assertIsNone(posture_logic.baseline_from_samples([]))
        samples = [{"width": 200, "tilt": 0.0, "neck": 2.5} for _ in range(30)]
        calib = posture_logic.baseline_from_samples(samples)
        self.assertEqual(calib["missing"], [])
        self.assertAlmostEqual(calib["baseline"]["neck"], 2.5)
        self.assertEqual(calib["thresholds"], posture_logic.default_thresholds())
        # 귀가 절반 이상 안 보였으면 거북목 기준을 잡지 않는다
        samples = [{"width": 200, "tilt": 0.0, "neck": 2.5 if i < 10 else None} for i in range(30)]
        self.assertEqual(posture_logic.baseline_from_samples(samples)["missing"], ["turtle"])

    def test_merge_thresholds_widens_to_current_defaults(self):
        merged = posture_logic.merge_thresholds({"tilt": 0.05, "neck_drop": 0.2, "nod": 0.15})
        self.assertEqual(merged["tilt"], posture_logic.SHOULDER_TILT_THRESHOLD)  # 예전 기본값 0.05 → 현재 기본값
        self.assertEqual(merged["neck_drop"], 0.2)                               # 자동으로 넓어진 값은 유지
        self.assertEqual(merged["shoulder_grow"], posture_logic.SHOULDER_GROW_RATIO)  # 없던 항목은 기본값
        self.assertNotIn("nod", merged)                                          # 없어진 항목은 무시


class IssueStateTests(unittest.TestCase):
    def feed(self, state, scores, t0=0.0):
        changes = []
        for i, s in enumerate(scores):
            c = state.update(t0 + i / FPS, s)
            if c:
                changes.append((round(t0 + i / FPS, 2), c))
        return t0 + len(scores) / FPS, changes

    def test_flicker_near_threshold_still_confirms(self):
        # 6프레임 중 1프레임이 임계 아래로 튀어도(83%가 임계 이상) 확정돼야 한다.
        # 예전 방식은 한 프레임만 내려가도 유지 시간이 0이 돼 영영 확정되지 않았다.
        scores = [0.95 if i % 6 == 0 else 1.05 for i in range(FPS * 3)]
        _, changes = self.feed(pt.IssueState("turtle"), scores)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0][1], "start")
        self.assertAlmostEqual(changes[0][0], 2.0, delta=0.1)

    def test_mostly_below_does_not_confirm(self):
        scores = [1.05 if i % 2 else 0.9 for i in range(FPS * 5)]
        _, changes = self.feed(pt.IssueState("turtle"), scores)
        self.assertEqual(changes, [])

    def test_hysteresis(self):
        state = pt.IssueState("turtle")
        t, _ = self.feed(state, [1.2] * (FPS * 3))
        self.assertTrue(state.confirmed)
        t, changes = self.feed(state, [0.8] * (FPS * 3), t)  # 임계 아래지만 복귀 기준(0.7) 위 → 유지
        self.assertEqual(changes, [])
        _, changes = self.feed(state, [0.5] * (FPS * 2), t)
        self.assertEqual([c for _, c in changes], ["end"])

    def test_gap_resets_history(self):
        state = pt.IssueState("turtle")
        self.feed(state, [1.2] * FPS)                      # 1초
        _, changes = self.feed(state, [1.2] * FPS, 2.5)    # 1.5초 끊긴 뒤 1초 → 합쳐서 2초여도 확정 안 됨
        self.assertEqual(changes, [])


class TrackerTests(unittest.TestCase):
    def test_turtle_confirm_and_time_accounting(self):
        tr = calibrated_tracker()
        t, ev, _ = run(tr, UPRIGHT, 1000.0, 10)
        self.assertEqual(ev, [])
        t, ev, notify = run(tr, TURTLE, t, 10)
        self.assertEqual([(e["type"], e["issue"]) for e in ev], [("start", "turtle")])
        self.assertEqual(notify, ["turtle"])
        buckets = tr.pop_time()
        monitored = sum(b["monitored"] for b in buckets.values())
        bad = sum(b["bad"] for b in buckets.values())
        self.assertAlmostEqual(monitored, 20, delta=0.1)
        self.assertTrue(7 < bad < 8.5, bad)  # 평활 지연 + 2초 유지 조건만큼 늦게 확정
        self.assertEqual(tr.pop_time(), {})

    def test_short_look_down_is_ignored_but_long_is_turtle(self):
        tr = calibrated_tracker()
        t, _, _ = run(tr, UPRIGHT, 0.0, 5)
        t, ev, _ = run(tr, NOD, t, 1.0)    # 키보드를 잠깐 봄
        t, ev2, _ = run(tr, UPRIGHT, t, 3)
        self.assertEqual(ev + ev2, [])
        t, ev, _ = run(tr, NOD, t, 5)      # 계속 숙이고 있음
        self.assertEqual([(e["type"], e["issue"]) for e in ev], [("start", "turtle")])

    def test_issues_have_independent_timers(self):
        tr = calibrated_tracker()
        t, _, _ = run(tr, UPRIGHT, 0.0, 5)
        # 거북목 → 거북목+기울어짐 → 기울어짐. 어느 한 항목이라도 임계를 넘은 구간은 2초가 넘게 이어지지만
        # (예전의 정상/이상 단일 상태였다면 확정), 항목별로는 각각 1.6초 미만이라 확정되면 안 된다.
        both = pose(ear_y=265, ear_half=33, tilt_dy=14)
        ev, any_run, longest_any, above = [], 0.0, 0.0, {i: 0.0 for i in posture_logic.ISSUES}
        for points, secs in ((TURTLE, 1.1), (both, 1.0), (TILT, 1.0)):
            for _ in range(int(round(secs * FPS))):
                res = tr.process(t, pt.measurement_from_points(points), 100)
                t += 1 / FPS
                ev += res["events"]
                over = [i for i, s in res["eval"]["scores"].items() if (s or 0) >= 1]
                any_run = any_run + 1 / FPS if over else 0.0
                longest_any = max(longest_any, any_run)
                for i in over:
                    above[i] += 1 / FPS
        # 임계값이나 지표를 바꾸면 자세별 점수가 달라져 이 전제가 깨질 수 있으므로 함께 확인한다
        self.assertGreater(longest_any, 2.1)
        self.assertLess(max(above.values()), 1.6)
        self.assertEqual(ev, [])
        self.assertEqual(tr.confirmed_issues(), [])

    def test_absence_ends_confirmed_issue(self):
        tr = calibrated_tracker()
        t, _, _ = run(tr, TURTLE, 0.0, 5)
        self.assertEqual(tr.confirmed_issues(), ["turtle"])
        t, ev, _ = run(tr, None, t, pt.ABSENCE_END_SECONDS + 0.5)
        self.assertEqual([(e["type"], e["issue"]) for e in ev], [("end", "turtle")])

    def test_notify_cooldown_and_renotify(self):
        tr = calibrated_tracker()
        t, _, n1 = run(tr, TURTLE, 0.0, 5)
        t, _, _ = run(tr, UPRIGHT, t, 5)
        t, ev, n2 = run(tr, TURTLE, t, 5)  # 60초 안에 다시 확정 → 기록은 하되 알림은 생략
        self.assertIn(("start", "turtle"), [(e["type"], e["issue"]) for e in ev])
        self.assertEqual((n1, n2), (["turtle"], []))
        _, _, n3 = run(tr, TURTLE, t, pt.RENOTIFY_SECONDS)
        self.assertEqual(n3, ["turtle"])

    def test_no_baseline_no_judgement(self):
        tr = pt.PostureTracker()
        _, ev, _ = run(tr, TURTLE, 0.0, 5)
        self.assertEqual(ev, [])
        self.assertEqual(tr.pop_time(), {})


# 앉은 거리 변화 자동 보정용 자세. 원근 때문에 가까이 앉으면 바른 자세여도 목 길이비가 줄어드는 상황을 흉내 낸다.
CLOSE = pose(ear_y=248, ear_half=37, sh_half=120)          # 바른 자세로 당겨 앉음: 어깨너비 +20%, 목 길이비 -18%
CLOSE_TURTLE = pose(ear_y=268, ear_half=40, sh_half=120)   # 당겨 앉은 자리에서 거북목
SLOUCH = pose(ear_y=268, ear_half=33, sh_half=107)         # 빠르게 상체를 숙임: 어깨너비 +7%, 목 길이비 크게 감소


def run_steps(tracker, steps, t0=0.0):
    """[(시작 자세, 끝 자세, 초)] 순서로, 두 자세 사이를 선형으로 옮겨 가며 넣는다. (시각, 이벤트, 기준 이동 목록)."""
    events, shifts, t = [], [], t0
    for a, b, secs in steps:
        n = int(round(secs * FPS))
        for i in range(n):
            f = (i + 1) / n
            points = {k: [a[k][j] + (b[k][j] - a[k][j]) * f for j in range(3)] for k in a}
            res = tracker.process(t, pt.measurement_from_points(points), 100)
            events += [(e["type"], e["issue"]) for e in res["events"]]
            if res["seat_shift"]:
                shifts.append(res["seat_shift"])
            t += 1 / FPS
    return t, events, shifts


class SeatMoveTests(unittest.TestCase):
    def test_pulling_chair_in_shifts_baseline_instead_of_turtle(self):
        tr = calibrated_tracker()
        _, events, shifts = run_steps(tr, [(UPRIGHT, UPRIGHT, 5), (UPRIGHT, CLOSE, 1.0), (CLOSE, CLOSE, 12)])
        self.assertNotIn(("start", "turtle"), events)
        self.assertIn(("start", "lean"), events)  # 가까이 앉은 것 자체는 모니터 근접으로 잡혀야 한다
        self.assertEqual(len(shifts), 1)
        self.assertAlmostEqual(shifts[0]["shift"], -0.18, delta=0.02)
        self.assertAlmostEqual(tr.baseline["neck_calibrated"], 2.5)

    def test_same_move_without_compensation_is_false_turtle(self):
        # 보정이 없으면(예전 동작) 같은 동작이 거북목으로 오판된다 — 위 테스트가 의미 있는지 확인
        tr = calibrated_tracker()
        original = pt.SEAT_MOVE_WIDTH_CHANGE
        pt.SEAT_MOVE_WIDTH_CHANGE = 99
        try:
            _, events, _ = run_steps(tr, [(UPRIGHT, UPRIGHT, 5), (UPRIGHT, CLOSE, 1.0), (CLOSE, CLOSE, 12)])
        finally:
            pt.SEAT_MOVE_WIDTH_CHANGE = original
        self.assertIn(("start", "turtle"), events)

    def test_turtle_after_moving_is_still_detected(self):
        tr = calibrated_tracker()
        _, events, _ = run_steps(tr, [(UPRIGHT, UPRIGHT, 5), (UPRIGHT, CLOSE, 1.0), (CLOSE, CLOSE, 12),
                                      (CLOSE, CLOSE_TURTLE, 0.5), (CLOSE_TURTLE, CLOSE_TURTLE, 5)])
        self.assertIn(("start", "turtle"), events)

    def test_quick_slouch_is_not_a_seat_move(self):
        tr = calibrated_tracker()
        _, events, shifts = run_steps(tr, [(UPRIGHT, UPRIGHT, 5), (UPRIGHT, SLOUCH, 1.0), (SLOUCH, SLOUCH, 6)])
        self.assertEqual(shifts, [])
        self.assertIn(("start", "turtle"), events)

    def test_no_shift_when_already_turtle_before_moving(self):
        tr = calibrated_tracker()
        _, events, shifts = run_steps(tr, [(TURTLE, TURTLE, 5), (TURTLE, CLOSE_TURTLE, 1.0),
                                           (CLOSE_TURTLE, CLOSE_TURTLE, 12)])
        self.assertEqual(shifts, [])
        self.assertIn(("start", "turtle"), events)
        self.assertNotIn(("end", "turtle"), events)

    def test_shift_beyond_limit_is_rejected(self):
        # 당기면서 자세도 크게 무너지면(목 길이비 변화가 한도 초과) 기준을 옮기지 않고 거북목으로 본다
        collapsed = pose(ear_y=285, ear_half=37, sh_half=120)
        tr = calibrated_tracker()
        _, events, shifts = run_steps(tr, [(UPRIGHT, UPRIGHT, 5), (UPRIGHT, collapsed, 1.0), (collapsed, collapsed, 12)])
        self.assertEqual(shifts, [])
        self.assertIn(("start", "turtle"), events)

    def test_recalibration_clears_seat_shift(self):
        tr = calibrated_tracker()
        run_steps(tr, [(UPRIGHT, UPRIGHT, 5), (UPRIGHT, CLOSE, 1.0), (CLOSE, CLOSE, 12)])
        self.assertIn("neck_calibrated", tr.baseline)
        fresh = calibrated_tracker().baseline
        tr.set_baseline(fresh, posture_logic.default_thresholds(), 100.0)
        self.assertNotIn("neck_calibrated", tr.baseline)
        self.assertFalse(tr.seat_moving)


class DBTests(unittest.TestCase):
    def test_time_based_score(self):
        db = DBManager(":memory:")
        tr = calibrated_tracker()
        t = 1_700_000_000.0
        t, ev, _ = run(tr, UPRIGHT, t, 60)
        t, ev2, _ = run(tr, TURTLE, t, 60)
        ids = {}
        for e in ev + ev2 + tr.end_all(t):
            if e["type"] == "start":
                ids[e["issue"]] = db.start_event(e["issue"], e["t"], e["metric"])
            else:
                db.end_event(ids.pop(e["issue"]), e["t"])
        db.add_time(tr.pop_time())
        date = next(iter(db._conn.execute("SELECT date FROM posture_time")))[0]
        s = db.get_daily_summary(date)
        self.assertEqual(s["event_counts"]["turtle"], 1)
        self.assertAlmostEqual(s["monitored_minutes"], 2, delta=0.01)
        self.assertTrue(50 <= s["score"] <= 53, s["score"])
        end_ts = db._conn.execute("SELECT end_ts FROM posture_events").fetchone()[0]
        self.assertIsNotNone(end_ts)
        db.close()

    def test_short_session_has_no_score(self):
        db = DBManager(":memory:")
        db.add_time({("2026-09-24", 10): {"monitored": 30, "bad": 30, **{i: 0 for i in posture_logic.ISSUES}}})
        self.assertIsNone(db.get_daily_summary("2026-09-24")["score"])
        db.close()


class ReplayTests(unittest.TestCase):
    def test_roundtrip_with_labels(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            rec = Recorder(path)
            rec.write_baseline(0.0, None, None)
            t = 0.0
            for i in range(FPS * 3):  # 기준 자세 측정 구간 (판정에서 빠져야 함)
                rec.write_frame(t, 100, TURTLE, None, calib=True)
                t += 1 / FPS
            tr = calibrated_tracker()
            rec.write_baseline(t, tr.baseline, tr.thresholds)
            for points, label, secs in ((UPRIGHT, "normal", 5), (TURTLE, "turtle", 5), (NOD, "nod", 10)):
                for _ in range(FPS * secs):
                    rec.write_frame(t, 100, points, label)
                    t += 1 / FPS
            rec.close()

            frames, events = replay.replay(read_recording(path))
            self.assertEqual(len(frames), FPS * 20)
            starts = [e["issue"] for e in events if e["type"] == "start"]
            self.assertEqual(starts, ["turtle"])
            out = io.StringIO()
            replay.summarize(frames, events, out)
            self.assertIn("바른 자세인데 이상 판정 (오탐): 0%", out.getvalue())

            frames, events = replay.replay(read_recording(path), threshold_overrides={"neck_drop": 0.3})
            self.assertEqual([e for e in events if e["type"] == "start"], [])
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()

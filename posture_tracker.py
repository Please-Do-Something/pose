# posture_tracker.py
# [Layer 3] 프레임 단위 판정을 시간축으로 묶는 순수 로직 (OpenCV/PyQt5 비의존).
# 카메라 워커와 녹화 재생(replay.py)이 같은 코드를 쓰도록, 시각(now)은 항상 인자로 받는다.
import math
import time
from collections import deque

import numpy as np

import posture_logic
from posture_logic import ISSUES

# ---- 인식 품질 ----
EAR_VISIBILITY_MIN = 0.5       # 귀 랜드마크 신뢰도가 이보다 낮으면(고개를 심하게 돌리거나 가려진 경우) 거북목 판정을 건너뜀
SHOULDER_VISIBILITY_MIN = 0.5  # 어깨가 화면 밖이거나 가려지면 판정/기준 자세 샘플에서 제외
MIN_BRIGHTNESS = 3             # 화면 평균 밝기(0~255)가 이보다 낮으면(카메라 가림/완전 암흑) 판정을 보류

# ---- 평활 ----
# 랜드마크 지수평활 시간상수(초). 원값으로 판정하면 경계에서 깜빡인다.
# 어깨는 옷/배경과 경계가 흐려 귀보다 훨씬 많이 흔들리므로 더 강하게 평활한다.
SMOOTH_TAU = {"l_sh": 1.0, "r_sh": 1.0, "l_ear": 0.5, "r_ear": 0.5}
MEDIAN_WINDOW = 5              # 평활 전에 최근 N프레임 중앙값을 취해 한 프레임짜리 튐을 제거

# ---- 확정/복귀 ----
# 이상 자세가 "확정"되려면 최근 이 시간 동안의 프레임 중 CONFIRM_FRACTION 이상이 임계를 넘어야 한다.
SUSTAIN_SECONDS = {"turtle": 2.0, "lean": 2.0, "tilt": 2.0}
RECOVER_SECONDS = 1.0          # 이상 → 정상 복귀는 더 빨리 인정해 교정한 자세가 바로 반영되게 함
# 한 프레임이라도 임계 아래로 내려가면 유지 시간이 0으로 돌아가던 방식은 임계 근처에서 영영 확정되지 않았다.
# 비율로 보면 몇 프레임 튀어도 판정이 버틴다. 대신 실제로는 유지 시간의 80%(2초면 1.6초)만 넘어도 확정된다.
CONFIRM_FRACTION = 0.8
# 히스테리시스: 점수 1.0 이상이면 이상으로 들어가고, 확정된 뒤에는 이 값 아래로 내려와야 정상으로 본다.
EXIT_SCORE = 0.7
MAX_GAP = 1.0                  # 판정 프레임 사이 공백이 이보다 길면 평활/유지 이력을 처음부터 다시 쌓음
HOLD_DISPLAY_DELAY = 1.0       # 판정 보류가 이 시간 넘게 이어질 때만 "판정 보류" 문구를 띄움
ABSENCE_END_SECONDS = 5.0      # 판정 보류가 이만큼 이어지면(자리를 비움 등) 확정된 이상 자세를 끝난 것으로 처리

# ---- 앉은 거리 변화 자동 보정 ----
# 의자를 당기거나 빼면 원근 때문에 바른 자세여도 목 길이비가 바뀐다(귀와 어깨의 깊이 차, 카메라 높이 탓).
# 원근 계수는 카메라/체형마다 달라 미리 알 수 없으므로, 자리를 옮기는 "순간"을 잡아 그 직전·직후의
# 목 길이비 비율만큼 거북목 기준을 옮긴다. 실제로 생긴 차이를 그대로 쓰므로 카메라별 조정이 필요 없다.
# 어깨너비가 짧은 시간에 크게 변하면 자리를 옮긴 것으로 본다. 상체만 숙이는 동작은 어깨너비 변화가 작고
# (카메라 모델: 어깨 5cm + 머리 10cm 앞으로 → +7%), 천천히 무너지는 자세는 창 안에서 변화가 작아 걸리지 않는다.
SEAT_MOVE_WIDTH_CHANGE = 0.10  # 어깨너비가 SEAT_MOVE_WINDOW 안에 이 비율 이상 변하면 이동 시작 (약 7cm 이상 이동)
SEAT_MOVE_WINDOW = 3.0         # 이동 감지 비교 구간 [초]
SEAT_SETTLE_TOLERANCE = 0.02   # 어깨너비가 SEAT_SETTLE_SECONDS 동안 이 비율 안에서 머물면 자리를 잡은 것으로 봄
SEAT_SETTLE_SECONDS = 1.5
SEAT_MOVE_TIMEOUT = 10.0       # 이동 시작 후 이 시간 안에 자리를 잡지 않으면 보정 취소
MAX_SEAT_SHIFT = 0.20          # 한 번 이동으로 옮길 수 있는 거북목 기준 폭 (넘으면 자세도 바뀐 것으로 보고 보정 안 함)
MAX_TOTAL_SEAT_SHIFT = 0.30    # 캘리브레이션 때 잡은 기준에서 누적으로 옮길 수 있는 폭

# ---- 알림 ----
NOTIFY_COOLDOWN = 60.0         # 같은 항목은 이 시간 안에 다시 확정돼도 알리지 않음 (자세를 고쳤다 다시 무너질 때 알림 폭주 방지)
RENOTIFY_SECONDS = 300.0       # 이상 자세가 이어지면 이 간격으로 다시 알림


def measurement_from_points(points):
    """
    랜드마크 좌표/신뢰도로 판정 입력을 만든다.
    points: {"l_sh", "r_sh", "l_ear", "r_ear"} → (x, y, visibility), 좌표는 픽셀 (다른 키는 무시).
    녹화 파일에도 이 points를 그대로 저장하므로, 재생 시에도 같은 기준으로 신뢰도를 판정한다.
    """
    m = {key: np.array(points[key][:2], dtype=float) for key in SMOOTH_TAU}
    vis = {key: float(points[key][2]) for key in SMOOTH_TAU}
    m["vis"] = vis
    m["ear_visible"] = vis["l_ear"] >= EAR_VISIBILITY_MIN and vis["r_ear"] >= EAR_VISIBILITY_MIN
    m["shoulder_visible"] = vis["l_sh"] >= SHOULDER_VISIBILITY_MIN and vis["r_sh"] >= SHOULDER_VISIBILITY_MIN
    return m


class Smoother:
    """최근 MEDIAN_WINDOW 프레임 중앙값으로 튀는 값을 걸러낸 뒤 시간 기반 지수평활한다."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._smoothed = {}
        self._history = {}
        self._last = None

    def update(self, m, now):
        dt = None if self._last is None else now - self._last
        self._last = now
        if dt is None or dt > MAX_GAP:
            # 오래 끊겼다가 다시 잡히면 예전 값에 끌려가지 않도록 새로 시작
            self._smoothed.clear()
            self._history.clear()

        keys = ["l_sh", "r_sh"]
        if m["ear_visible"]:
            keys += ["l_ear", "r_ear"]  # 안 보일 때의 좌표는 추정치라 평활에 섞지 않음
        for key in keys:
            hist = self._history.setdefault(key, deque(maxlen=MEDIAN_WINDOW))
            hist.append(m[key])
            value = np.median(np.array(hist), axis=0)
            prev = self._smoothed.get(key)
            if prev is None:
                self._smoothed[key] = value
            else:
                alpha = 1 - math.exp(-dt / SMOOTH_TAU[key])
                self._smoothed[key] = prev + alpha * (value - prev)

        out = dict(m)
        out.update({k: v for k, v in self._smoothed.items() if k in keys})
        return out


def calibration_sample(sm):
    """평활된 측정값에서 기준 자세 샘플 하나를 뽑는다 (posture_logic.baseline_from_samples 입력)."""
    result = posture_logic.evaluate_posture(
        sm["l_sh"], sm["r_sh"], left_ear=sm["l_ear"], right_ear=sm["r_ear"], ear_visible=sm["ear_visible"],
    )
    return {"width": result["shoulder_width"], "tilt": result["tilt_ratio"],
            "neck": result["neck_ratio"]}


class IssueState:
    """한 판정 항목의 확정 상태. 최근 일정 시간 동안 조건을 만족한 프레임 비율로 확정/복귀를 정한다."""

    def __init__(self, issue):
        self.issue = issue
        self.confirmed = False
        self.confirmed_at = None
        self._window = deque()       # (시각, 전환 조건 만족 여부)
        self._tracking_since = None  # 이력을 끊김 없이 쌓기 시작한 시각

    def clear_window(self):
        self._window.clear()
        self._tracking_since = None

    def required_seconds(self):
        return RECOVER_SECONDS if self.confirmed else SUSTAIN_SECONDS[self.issue]

    def progress(self):
        """(전환 조건을 만족한 프레임 비율, 이력이 쌓인 시간) — 디버그 표시용."""
        if not self._window:
            return 0.0, 0.0
        frac = sum(flag for _, flag in self._window) / len(self._window)
        return frac, self._window[-1][0] - self._tracking_since

    def update(self, now, score):
        """점수를 반영하고, 상태가 바뀌면 "start"/"end"를 돌려준다."""
        if self._window and now - self._window[-1][0] > MAX_GAP:
            self.clear_window()
        if self.confirmed:
            # 판정할 수 없는 프레임(None)은 정상 쪽으로 친다 (예: 귀가 안 보이면 거북목은 풀림)
            flag = score is None or score < EXIT_SCORE
        else:
            flag = score is not None and score >= 1.0
        if self._tracking_since is None:
            self._tracking_since = now
        self._window.append((now, flag))

        need = self.required_seconds()
        while self._window[0][0] < now - need:
            self._window.popleft()
        frac = sum(f for _, f in self._window) / len(self._window)
        if now - self._tracking_since < need or frac < CONFIRM_FRACTION:
            return None

        self.confirmed = not self.confirmed
        self.confirmed_at = now if self.confirmed else None
        self.clear_window()
        return "start" if self.confirmed else "end"


class PostureTracker:
    """
    프레임마다 process()를 호출하면 평활 → 지표 계산 → 항목별 확정 → 시간 집계 → 알림 판단까지 한다.
    DB 기록/알림 전송은 하지 않고, 돌려준 events/notify를 호출측(카메라 워커)이 처리한다.
    """

    def __init__(self, baseline=None, thresholds=None):
        self.smoother = Smoother()
        self.baseline = baseline
        self.thresholds = thresholds or posture_logic.default_thresholds()
        self.states = {issue: IssueState(issue) for issue in ISSUES}
        self._last_notified = {}
        self._last_judged = None
        self._hold_since = None
        self._time = {}  # (날짜, 시) → {"monitored", "bad", 항목별 초}
        self._seat_hist = deque()  # (시각, 어깨너비, 목 길이비, 거북목 점수) 최근 판정 프레임
        self._seat_move = None     # 자리 이동 중이면 {"start": 시각, "pre": 이동 직전 샘플 또는 None}

    @property
    def seat_moving(self):
        """자리를 옮기는 중이라 거북목 판정을 멈춘 상태인지 (디버그 표시용)."""
        return self._seat_move is not None

    # ---- 상태 조회 ----
    def confirmed_issues(self):
        return [issue for issue in ISSUES if self.states[issue].confirmed]

    def set_baseline(self, baseline, thresholds, now):
        """기준이 바뀌면 이전 기준으로 확정된 이상 자세는 끝낸다. 끝난 이벤트 목록을 돌려준다."""
        events = self.end_all(now)
        self.baseline, self.thresholds = baseline, thresholds
        self.smoother.reset()
        self._seat_hist.clear()
        self._seat_move = None
        return events

    def end_all(self, now):
        events = []
        for state in self.states.values():
            if state.confirmed:
                state.confirmed, state.confirmed_at = False, None
                events.append({"type": "end", "issue": state.issue, "t": now})
            state.clear_window()
        self._last_judged = None
        return events

    def pop_time(self):
        """마지막 호출 이후 쌓인 시간 집계를 꺼낸다. {(YYYY-MM-DD, 시): {"monitored", "bad", turtle, ...}}"""
        out, self._time = self._time, {}
        return out

    # ---- 프레임 처리 ----
    def process(self, now, m, brightness):
        """
        m: measurement_from_points 결과(사람이 없으면 None).
        반환: {"hold": 보류 사유|None, "hold_shown": 사용자에게 보일 보류 사유|None,
               "sm": 평활값|None, "eval": evaluate_posture 결과|None,
               "events": [{"type": "start"/"end", "issue", "t", "metric"?}], "notify": [항목],
               "seat_shift": 이번 프레임에 자리 이동을 반영해 거북목 기준을 옮겼으면 {"shift", "width_change"}}
        """
        hold = None
        if m is None:
            hold = "사람 인식 안 됨"
        elif brightness < MIN_BRIGHTNESS:
            hold = "조명이 너무 어두움"
        elif not m["shoulder_visible"]:
            hold = "어깨 인식 불안정"

        result = {"hold": hold, "hold_shown": None, "sm": None, "eval": None, "events": [], "notify": [],
                  "seat_shift": None}
        if self.baseline is None:
            return result  # 기준 자세 설정 전에는 판정/기록/알림을 하지 않는다

        if hold is not None:
            if self._hold_since is None:
                self._hold_since = now
            held = now - self._hold_since
            if held >= HOLD_DISPLAY_DELAY:
                result["hold_shown"] = hold
            if held >= ABSENCE_END_SECONDS:
                result["events"] = self.end_all(now)
            return result
        self._hold_since = None

        sm = self.smoother.update(m, now)
        ev = posture_logic.evaluate_posture(
            sm["l_sh"], sm["r_sh"], left_ear=sm["l_ear"], right_ear=sm["r_ear"],
            ear_visible=sm["ear_visible"], baseline=self.baseline, thresholds=self.thresholds,
        )
        result["sm"], result["eval"] = sm, ev

        result["seat_shift"] = self._track_seat(now, ev)

        metric = {"turtle": ev["neck_ratio"], "lean": ev["shoulder_width"],
                  "tilt": ev["tilt_ratio"]}
        for issue in ISSUES:
            if issue == "turtle" and self._seat_move is not None:
                # 자리를 옮기는 동안의 목 길이비 변화는 원근 탓이라 거북목 판정을 멈춘다.
                # (모니터 근접은 가까이 앉은 것 자체가 판정 대상이라 계속 본다)
                continue
            state = self.states[issue]
            change = state.update(now, ev["scores"][issue])
            if change == "start":
                result["events"].append({"type": "start", "issue": issue, "t": now, "metric": metric[issue]})
                last = self._last_notified.get(issue)
                if last is None or now - last >= NOTIFY_COOLDOWN:
                    result["notify"].append(issue)
            elif change == "end":
                result["events"].append({"type": "end", "issue": issue, "t": now})
            elif state.confirmed and now - self._last_notified.get(issue, state.confirmed_at) >= RENOTIFY_SECONDS:
                result["notify"].append(issue)
        for issue in result["notify"]:
            self._last_notified[issue] = now

        self._account_time(now)
        return result

    def _track_seat(self, now, ev):
        """
        앉은 거리가 바뀌는 순간을 감지하고, 자리를 잡으면 거북목 기준을 옮긴다.
        기준을 옮겼으면 {"shift": 목 길이비 변화율, "width_change": 어깨너비 변화율}, 아니면 None.
        """
        hist = self._seat_hist
        if hist and now - hist[-1][0] > MAX_GAP:
            hist.clear()  # 판정이 끊겼으면 전후 비교가 의미 없다
        hist.append((now, ev["shoulder_width"], ev["neck_ratio"], ev["scores"]["turtle"]))
        while hist[0][0] < now - SEAT_MOVE_WINDOW:
            hist.popleft()
        width = ev["shoulder_width"]

        if self._seat_move is None:
            oldest = hist[0]
            if abs(width / oldest[1] - 1) < SEAT_MOVE_WIDTH_CHANGE:
                return None
            # 이동 직전이 정상 자세였을 때만 기준을 옮긴다 (거북목 상태로 옮기면 그 자세가 기준에 흡수됨)
            _, _, pre_neck, pre_score = oldest
            usable = (pre_neck is not None and pre_score is not None and pre_score < EXIT_SCORE
                      and not self.states["turtle"].confirmed)
            self._seat_move = {"start": now, "pre": oldest if usable else None}
            self.states["turtle"].clear_window()
            return None

        move = self._seat_move
        if now - move["start"] > SEAT_MOVE_TIMEOUT:
            self._seat_move = None  # 계속 움직이면 보정하지 않고 판정을 재개
            return None
        settled = [h[1] for h in hist if h[0] >= now - SEAT_SETTLE_SECONDS]
        if hist[0][0] > now - SEAT_SETTLE_SECONDS or max(settled) / min(settled) - 1 > SEAT_SETTLE_TOLERANCE:
            return None

        # 자리를 잡았다: 비교 기록을 새로 시작해 이동 구간과 다시 비교되지 않게 한다
        self._seat_move = None
        hist.clear()
        pre, neck = move["pre"], ev["neck_ratio"]
        if pre is None or neck is None or not self.baseline.get("neck"):
            return None
        shift = neck / pre[2]
        calibrated = self.baseline.get("neck_calibrated", self.baseline["neck"])
        new_neck = self.baseline["neck"] * shift
        if abs(shift - 1) > MAX_SEAT_SHIFT or abs(new_neck / calibrated - 1) > MAX_TOTAL_SEAT_SHIFT:
            return None
        self.baseline = {**self.baseline, "neck": new_neck, "neck_calibrated": calibrated}
        return {"shift": shift - 1, "width_change": width / pre[1] - 1}

    def _account_time(self, now):
        """판정한 시간과 그중 이상 자세로 확정돼 있던 시간을 시간대별로 쌓는다 (보류 구간은 빼고)."""
        last, self._last_judged = self._last_judged, now
        if last is None or now - last > MAX_GAP:
            return
        dt = now - last
        lt = time.localtime(now)
        bucket = self._time.setdefault((time.strftime("%Y-%m-%d", lt), lt.tm_hour),
                                       {"monitored": 0.0, "bad": 0.0, **{i: 0.0 for i in ISSUES}})
        bucket["monitored"] += dt
        confirmed = self.confirmed_issues()
        if confirmed:
            bucket["bad"] += dt
        for issue in confirmed:
            bucket[issue] += dt

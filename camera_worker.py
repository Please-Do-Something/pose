# camera_worker.py
# [Layer 2] 웹캠 캡처 + MediaPipe 랜드마크 추출을 담당하는 QThread.
# 메인 GUI 스레드가 프레임 처리 중 멈추지 않도록 별도 스레드에서 동작한다.
# 판정 자체(평활/확정/시간 집계)는 posture_tracker에 있고, 여기서는 그 결과를 DB/알림/화면에 연결한다.
import json
import math
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

import posture_logic
import posture_tracker
from posture_logic import ISSUE_LABELS, ISSUES
from recorder import LABELS, Recorder

try:
    from winotify import Notification
    NOTIFY_AVAILABLE = True
except ImportError:
    NOTIFY_AVAILABLE = False

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# mediapipe pose 랜드마크 번호: 7/8=왼쪽/오른쪽 귀, 11/12=왼쪽/오른쪽 어깨
# (화면을 좌우반전한 뒤 추론하므로 mediapipe의 "왼쪽"은 실제로는 사용자의 오른쪽이다)
LANDMARK_INDEX = {"l_ear": 7, "r_ear": 8, "l_sh": 11, "r_sh": 12}

# 기준 자세를 저장해 두는 파일. 앱을 다시 켜도 기준 자세를 새로 잡을 필요가 없게 한다.
BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline.json")
BASELINE_VERSION = 2  # 저장하는 기준값의 계산식이 바뀌면 올린다 (예전 파일은 무시하고 재설정 요청)

# 어두운 화면에서 랜드마크 인식이 흔들리지 않도록 모델 입력만 감마 보정으로 밝힌다. 화면 표시는 원본 그대로.
TARGET_BRIGHTNESS = 110        # 보정 목표 평균 밝기
MIN_GAMMA = 0.25               # 너무 어두운 화면을 과하게 펴면 노이즈만 커지므로 보정 한도

CALIB_COUNTDOWN_SECONDS = 3.0  # 버튼을 누른 뒤 바른 자세를 잡을 준비 시간
# 이 시간 동안 판정과 같은 평활을 거친 값을 모아, 중앙값으로 기준을 잡고 흔들림 폭으로 자동 임계를 정한다.
# (카운트다운 동안 평활을 미리 돌려 두므로 측정 시작 시점엔 평활값이 안정돼 있다.)
# 흔들림은 1초 단위로 천천히 출렁이므로 너무 짧게 재면 과소평가된다.
CALIB_SAMPLE_SECONDS = 3.0
NOTICE_SECONDS = 3.0           # 캘리브레이션 결과 문구를 영상 위에 띄워두는 시간
TIME_FLUSH_SECONDS = 10.0      # 판정 시간 집계를 DB에 쓰는 주기 (비정상 종료 시 이만큼만 잃음)

COLOR_NORMAL = (50, 205, 154)
COLOR_ABNORMAL = (0, 0, 255)
COLOR_INFO = (0, 215, 255)

FONT_PATH = "C:/Windows/Fonts/malgunbd.ttf"
_FONT_CACHE = {}


def _points(landmarks, w, h):
    """판정에 쓰는 랜드마크만 픽셀 좌표 + 신뢰도로 뽑는다 (녹화 파일에도 이 형태로 저장)."""
    out = {}
    for key, idx in LANDMARK_INDEX.items():
        lm = landmarks[idx]
        out[key] = [round(lm.x * w, 2), round(lm.y * h, 2), round(lm.visibility, 3)]
    return out


def _put_text_kr(image, text, pos, font_size, color_bgr, from_bottom=False):
    """
    cv2.putText는 한글 미지원이라 PIL로 그린 뒤 다시 OpenCV 이미지로 변환한다.
    text에 줄바꿈이 있으면 여러 줄로 그린다. from_bottom이면 pos[1]을 아래쪽 여백으로 본다.
    """
    if font_size not in _FONT_CACHE:
        _FONT_CACHE[font_size] = ImageFont.truetype(FONT_PATH, font_size)
    font = _FONT_CACHE[font_size]

    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    pad = 6
    if from_bottom:
        text_bottom = draw.textbbox((0, 0), text, font=font)[3]
        pos = (pos[0], img_pil.height - pos[1] - text_bottom)
    bbox = draw.textbbox(pos, text, font=font)
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=(0, 0, 0, 150),
    )
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(pos, text, font=font, fill=color_rgb + (255,))

    combined = Image.alpha_composite(img_pil, overlay).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)


def _normalize_brightness(frame, brightness):
    """어두운 프레임을 감마 보정으로 TARGET_BRIGHTNESS 근처까지 밝힌다 (밝은 프레임은 그대로)."""
    if brightness >= TARGET_BRIGHTNESS or brightness <= 0:
        return frame
    gamma = max(math.log(TARGET_BRIGHTNESS / 255) / math.log(brightness / 255), MIN_GAMMA)
    lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(frame, lut)


def _notify_windows(title, message):
    if not NOTIFY_AVAILABLE:
        return
    try:
        Notification(app_id="자세 교정 알림", title=title, msg=message, duration="short").show()
    except Exception as e:
        print(f"[알림 실패] {e}")


class CameraWorker(QThread):
    frame_ready = pyqtSignal(QImage)
    status_updated = pyqtSignal(dict)
    calibration_finished = pyqtSignal(bool, str)  # (성공 여부, 사용자 안내 문구)

    def __init__(self, db_manager, parent=None, baseline_path=BASELINE_PATH):
        super().__init__(parent)
        self._db = db_manager
        self._running = False

        self._baseline_path = baseline_path
        self.tracker = posture_tracker.PostureTracker()
        self._open_events = {}  # 항목 → posture_events 행 id (복귀 시 end_ts를 채우기 위함)
        self._last_flush = time.time()

        # 캘리브레이션은 GUI 스레드에서 요청만 하고, 실제 측정/기준값 갱신은 워커 스레드에서 한다.
        self._calib_requested = False
        self._calib_started_at = None
        self._calib_samples = []
        self._notice = None  # (문구, 색, 만료 시각)
        self._hold_reason = None

        self.debug_enabled = False  # GUI 체크박스로 켜고 끔. 켜면 판정에 쓰는 수치를 영상 좌하단에 표시
        self._fps = None
        self._last_frame_at = None

        # 튜닝용 녹화. 파일 열기/닫기도 워커 스레드에서 한다 (GUI는 원하는 상태만 바꿈).
        self.recording_wanted = False
        self.label = None  # 녹화 중 정답 라벨 (recorder.LABELS의 키)
        self._recorder = None

        self.baseline_loaded = self._load_baseline()
        if self.baseline_loaded:
            msg = "저장된 기준 자세를 불러왔습니다 (카메라 위치를 바꿨다면 다시 설정하세요)"
            self._notice = (msg, COLOR_NORMAL, time.time() + NOTICE_SECONDS + 2)

    @property
    def has_baseline(self):
        return self.tracker.baseline is not None

    @property
    def recording_path(self):
        return self._recorder.path if self._recorder else None

    def calibrate(self):
        """기준 자세 측정을 요청한다. GUI 버튼에서 호출 (카운트다운 → 샘플 수집 → 기준 확정)."""
        self._calib_requested = True

    def _load_baseline(self):
        try:
            with open(self._baseline_path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") != BASELINE_VERSION:
                # 판정 지표 계산식이 바뀌어 예전에 저장한 기준값과는 비교할 수 없다
                self._notice = ("판정 방식이 바뀌어 기준 자세를 다시 설정해야 합니다",
                                COLOR_INFO, time.time() + NOTICE_SECONDS + 2)
                return False
            baseline = {"width": float(data["width"]), "tilt": float(data["tilt"]), "neck": data.get("neck")}
            # 예전 파일에 없거나 현재 기본값보다 좁은 임계는 기본값을 쓴다 (없어진 항목의 값은 무시)
            thresholds = posture_logic.merge_thresholds(data.get("thresholds"))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return False
        self.tracker.set_baseline(baseline, thresholds, time.time())
        return True

    def _save_baseline(self):
        data = {"version": BASELINE_VERSION, **self.tracker.baseline,
                "thresholds": self.tracker.thresholds,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        try:
            with open(self._baseline_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"[기준 자세 저장 실패] {e}")

    def _calib_phase(self, now):
        if self._calib_started_at is None:
            return None
        elapsed = now - self._calib_started_at
        if elapsed < CALIB_COUNTDOWN_SECONDS:
            return "countdown"
        if elapsed < CALIB_COUNTDOWN_SECONDS + CALIB_SAMPLE_SECONDS:
            return "sampling"
        return "done"

    def _finish_calibration(self, now):
        samples = self._calib_samples
        self._calib_started_at = None
        self._calib_samples = []

        calib = posture_logic.baseline_from_samples(samples)
        if calib is None:
            msg = "자세를 인식하지 못했습니다. 어깨가 화면에 보이게 앉은 뒤 다시 시도하세요"
            if self.has_baseline:
                msg += " (이전 기준 자세를 유지합니다)"
            self._notice = (msg, COLOR_ABNORMAL, now + NOTICE_SECONDS)
            self.calibration_finished.emit(False, msg)
            return

        self._handle_events(self.tracker.set_baseline(calib["baseline"], calib["thresholds"], now))
        if self._recorder:
            self._recorder.write_baseline(now, self.tracker.baseline, self.tracker.thresholds)

        msg = "기준 자세가 설정되었습니다"
        if "turtle" in calib["missing"]:
            msg += " (귀가 잘 보이지 않아 거북목은 판정하지 않습니다)"
        color = COLOR_NORMAL
        if calib["capped"]:
            msg += (f"\n카메라 인식이 많이 흔들려 {', '.join(calib['capped'])} 판정이 둔할 수 있습니다. "
                    "조명을 밝게 하거나 상반신이 잘 보이게 해 보세요")
            color = COLOR_INFO

        self._save_baseline()
        self._notice = (msg, color, now + NOTICE_SECONDS + (3 if calib["capped"] else 0))
        self.calibration_finished.emit(True, msg)

    def _handle_events(self, events, notify=()):
        """트래커가 알려준 확정/복귀를 DB에 남기고, 알릴 항목이 있으면 토스트를 띄운다."""
        for e in events:
            if e["type"] == "start":
                self._open_events[e["issue"]] = self._db.start_event(e["issue"], e["t"], e["metric"])
            else:
                event_id = self._open_events.pop(e["issue"], None)
                if event_id is not None:
                    self._db.end_event(event_id, e["t"])
        if notify:
            _notify_windows("자세 교정 알림", " / ".join(ISSUE_LABELS[i] for i in notify))

    def _flush_time(self, now, force=False):
        if force or now - self._last_flush >= TIME_FLUSH_SECONDS:
            self._last_flush = now
            self._db.add_time(self.tracker.pop_time())

    def _sync_recorder(self, now):
        if self.recording_wanted and self._recorder is None:
            try:
                self._recorder = Recorder()
            except OSError as e:
                self.recording_wanted = False
                self._notice = (f"녹화 파일을 만들 수 없습니다: {e}", COLOR_ABNORMAL, now + NOTICE_SECONDS)
                return
            self._recorder.write_baseline(now, self.tracker.baseline, self.tracker.thresholds)
        elif not self.recording_wanted and self._recorder is not None:
            path = self._recorder.path
            self._recorder.close()
            self._recorder = None
            self._notice = (f"녹화 저장: {os.path.basename(path)}", COLOR_NORMAL, now + NOTICE_SECONDS)

    def _debug_text(self, m, frame):
        """판정에 실제로 쓰이는 수치를 기준값/임계값/확정 진행 상황과 나란히 보여준다."""
        t = self.tracker
        b, th = t.baseline, t.thresholds
        v = (frame and frame["sm"]) or m
        ev = (frame and frame["eval"]) or posture_logic.evaluate_posture(
            v["l_sh"], v["r_sh"], left_ear=v["l_ear"], right_ear=v["r_ear"],
            ear_visible=v["ear_visible"], baseline=b, thresholds=th)
        scores = ev["scores"]
        defaults = posture_logic.default_thresholds()

        def auto_tag(key):
            # 카메라 흔들림 때문에 기본값보다 넓어진 임계에 표시
            return " 자동" if th[key] > defaults[key] + 1e-9 else ""

        def state_tag(issue):
            s = t.states[issue]
            frac, span = s.progress()
            if s.confirmed:
                return "  [확정" + (f", 복귀 {frac * 100:.0f}%" if frac > 0 else "") + "]"
            if frac > 0:
                return f"  [{frac * 100:.0f}% {span:.1f}/{s.required_seconds():.0f}s]"
            return "  [초과]" if (scores[issue] or 0) >= 1 else ""

        lines = []
        if ev["neck_ratio"] is None:
            lines.append("거북목: 귀 안 보임 → 판정 생략")
        elif not b or not b.get("neck"):
            lines.append(f"목 길이비: {ev['neck_ratio']:.3f} | 기준 없음")
        else:
            drop = 1 - ev["neck_ratio"] / b["neck"]
            lines.append(f"목 길이비: {ev['neck_ratio']:.3f} | 기준 {b['neck']:.3f} → 감소 {drop * 100:+.0f}% "
                         f"(임계 {th['neck_drop'] * 100:.0f}%{auto_tag('neck_drop')}){state_tag('turtle')}")

        width = ev["shoulder_width"]
        if b:
            lines.append(f"어깨너비: {width:.0f}px | 기준 {b['width']:.0f} → x{width / b['width']:.2f} "
                         f"(임계 x{th['shoulder_grow']:.2f}{auto_tag('shoulder_grow')}){state_tag('lean')}")
            lines.append(f"어깨기울기: {ev['tilt_ratio']:+.3f} | 기준 {b['tilt']:+.3f} → "
                         f"차이 {abs(ev['tilt_ratio'] - b['tilt']):.3f} "
                         f"(임계 {th['tilt']:.3f}{auto_tag('tilt')}){state_tag('tilt')}")
        else:
            lines.append(f"어깨너비: {width:.0f}px | 어깨기울기: {ev['tilt_ratio']:+.3f} | 기준 없음")

        vis = m["vis"]
        lines.append(f"신뢰도: 귀 L{vis['l_ear']:.2f} R{vis['r_ear']:.2f} | 어깨 L{vis['l_sh']:.2f} "
                     f"R{vis['r_sh']:.2f}")
        return "\n".join(lines)

    def stop(self):
        self._running = False
        self.wait()

    def run(self):
        cap = cv2.VideoCapture(0)
        self._running = True

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self._running and cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)  # 거울처럼 보이도록 좌우 반전
                h, w = frame.shape[:2]

                now = time.time()
                if self._last_frame_at is not None:
                    inst_fps = 1.0 / max(now - self._last_frame_at, 1e-6)
                    self._fps = inst_fps if self._fps is None else self._fps * 0.9 + inst_fps * 0.1
                self._last_frame_at = now

                self._sync_recorder(now)
                if self._calib_requested:
                    self._calib_requested = False
                    self._calib_started_at = now
                    self._calib_samples = []
                    # 측정 중에는 판정하지 않으므로 진행 중인 이상 자세는 여기서 끝낸다.
                    # 이전 자세의 평활값이 새 기준에 섞이지 않도록 평활도 처음부터 다시 시작.
                    self._handle_events(self.tracker.end_all(now))
                    self.tracker.smoother.reset()
                calib_phase = self._calib_phase(now)

                brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
                rgb = cv2.cvtColor(_normalize_brightness(frame, brightness), cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                image = frame

                points = _points(results.pose_landmarks.landmark, w, h) if results.pose_landmarks else None
                m = posture_tracker.measurement_from_points(points) if points else None
                if self._recorder:
                    self._recorder.write_frame(now, brightness, points, self.label, calib=calib_phase is not None)

                tracked = None
                if calib_phase in ("countdown", "sampling"):
                    self._hold_reason = None
                    if m is not None and m["shoulder_visible"]:
                        # 판정과 같은 평활을 카운트다운부터 돌려 두고, 측정 구간에서는 판정이 실제로 보게 될
                        # 값을 모은다. 원값으로 흔들림을 재면 평활 후보다 훨씬 커서 자동 임계가 과하게 넓어진다.
                        sm = self.tracker.smoother.update(m, now)
                        if calib_phase == "sampling":
                            self._calib_samples.append(posture_tracker.calibration_sample(sm))
                else:
                    tracked = self.tracker.process(now, m, brightness)
                    self._handle_events(tracked["events"], tracked["notify"])
                    self._hold_reason = tracked["hold_shown"]
                self._flush_time(now)

                confirmed = self.tracker.confirmed_issues()
                status = "이상" if confirmed else "정상"
                issue_labels = [ISSUE_LABELS[i] for i in confirmed]

                debug_text = None
                if m is not None:
                    if self.debug_enabled:
                        debug_text = self._debug_text(m, tracked)
                    style = mp_drawing.DrawingSpec(
                        color=COLOR_ABNORMAL if confirmed else COLOR_NORMAL,
                        thickness=2, circle_radius=2,
                    )
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=style, connection_drawing_spec=style,
                    )

                if calib_phase == "done":
                    self._finish_calibration(now)
                    calib_phase = None

                if calib_phase == "countdown":
                    remaining = math.ceil(CALIB_COUNTDOWN_SECONDS - (now - self._calib_started_at))
                    image = _put_text_kr(image, f"바른 자세로 앉아주세요... {remaining}", (10, 8), 26, COLOR_INFO)
                elif calib_phase == "sampling":
                    image = _put_text_kr(image, "기준 자세 측정 중... 움직이지 마세요", (10, 8), 26, COLOR_INFO)
                elif not self.has_baseline:
                    guide_text = "바른 자세로 앉은 뒤 '기준 자세 설정' 버튼을 누르세요"
                    image = _put_text_kr(image, guide_text, (10, 8), 26, COLOR_NORMAL)
                elif self._hold_reason is not None:
                    image = _put_text_kr(image, f"판정 보류: {self._hold_reason}", (10, 8), 26, COLOR_INFO)
                else:
                    status_text = f"상태: {status}" + (f" ({', '.join(issue_labels)})" if issue_labels else "")
                    image = _put_text_kr(image, status_text, (10, 8), 26,
                                         COLOR_ABNORMAL if confirmed else COLOR_NORMAL)

                bottom_lines = []
                if self._recorder:
                    bottom_lines.append("● 녹화 중 · 라벨(숫자키): " + (LABELS[self.label] if self.label else "없음"))
                if self.debug_enabled:
                    bottom_lines.append(debug_text or "사람 인식 안 됨")
                    bottom_lines.append(f"밝기 {brightness:.0f} | " + (f"FPS {self._fps:.0f}" if self._fps else "FPS -"))
                if bottom_lines:
                    image = _put_text_kr(image, "\n".join(bottom_lines), (10, 10), 15, COLOR_INFO, from_bottom=True)

                if self._notice is not None:
                    text, color, expires_at = self._notice
                    if now < expires_at:
                        image = _put_text_kr(image, text, (10, 50), 20, color)
                    else:
                        self._notice = None

                rgb_out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                fh, fw, ch = rgb_out.shape
                qimg = QImage(rgb_out.data, fw, fh, ch * fw, QImage.Format_RGB888).copy()
                self.frame_ready.emit(qimg)
                self.status_updated.emit({
                    "status": status,
                    "issues": issue_labels,
                    "baseline_set": self.has_baseline,
                    "calibrating": calib_phase is not None,
                    "hold_reason": self._hold_reason,
                    "recording_path": self.recording_path,
                })

        # 종료 시 진행 중인 이상 자세를 닫고 남은 시간 집계를 저장한다
        now = time.time()
        self._handle_events(self.tracker.end_all(now))
        self._flush_time(now, force=True)
        self.recording_wanted = False
        self._sync_recorder(now)
        cap.release()

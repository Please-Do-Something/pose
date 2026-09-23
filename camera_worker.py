# camera_worker.py
# [Layer 2] 웹캠 캡처 + MediaPipe 랜드마크 추출을 담당하는 QThread.
# 메인 GUI 스레드가 프레임 처리 중 멈추지 않도록 별도 스레드에서 동작한다.
import time

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

import posture_logic

try:
    from winotify import Notification
    NOTIFY_AVAILABLE = True
except ImportError:
    NOTIFY_AVAILABLE = False

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# mediapipe pose 랜드마크 번호: 7/8=왼쪽/오른쪽 귀, 11/12=왼쪽/오른쪽 어깨
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12

SUSTAIN_SECONDS = 2.0          # 자세 변화가 "진짜"로 인정되려면 최소 이만큼 연속 유지돼야 함
EAR_VISIBILITY_MIN = 0.5       # 귀 랜드마크 신뢰도가 이보다 낮으면(고개를 심하게 돌리거나 가려진 경우) 거북목 판정을 건너뜀
FORWARD_OFFSET_MARGIN = 0.04   # 기준보다 귀가 이만큼(정규화 z) 더 카메라 쪽으로 나오면 거북목으로 판정. 작을수록 예민해짐

COLOR_NORMAL = (50, 205, 154)
COLOR_ABNORMAL = (0, 0, 255)

FONT_PATH = "C:/Windows/Fonts/malgunbd.ttf"
_FONT_CACHE = {}


def _landmark_xy(landmarks, idx, w, h):
    """정규화된(0~1) mediapipe 좌표를 실제 픽셀 좌표로 바꾼다."""
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def _put_text_kr(image, text, pos, font_size, color_bgr):
    """cv2.putText는 한글 미지원이라 PIL로 그린 뒤 다시 OpenCV 이미지로 변환한다."""
    if font_size not in _FONT_CACHE:
        _FONT_CACHE[font_size] = ImageFont.truetype(FONT_PATH, font_size)
    font = _FONT_CACHE[font_size]

    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    pad = 6
    bbox = draw.textbbox(pos, text, font=font)
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=(0, 0, 0, 150),
    )
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(pos, text, font=font, fill=color_rgb + (255,))

    combined = Image.alpha_composite(img_pil, overlay).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)


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

    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self._db = db_manager
        self._running = False

        self._latest_landmarks = None
        self._frame_size = (0, 0)

        self.baseline_width = None
        self.baseline_forward_offset = None

        self._raw_status = "정상"
        self._status_since = time.time()
        self._confirmed_status = "정상"

    def calibrate(self):
        """현재 프레임의 랜드마크를 기준 자세로 저장한다. GUI 버튼/키에서 호출."""
        if self._latest_landmarks is None:
            return
        landmarks = self._latest_landmarks
        w, h = self._frame_size
        l_sh = _landmark_xy(landmarks, L_SHOULDER, w, h)
        r_sh = _landmark_xy(landmarks, R_SHOULDER, w, h)
        self.baseline_width = float(np.linalg.norm(l_sh - r_sh))

        l_ear_lm, r_ear_lm = landmarks[L_EAR], landmarks[R_EAR]
        if l_ear_lm.visibility >= EAR_VISIBILITY_MIN and r_ear_lm.visibility >= EAR_VISIBILITY_MIN:
            ear_depth = (l_ear_lm.z + r_ear_lm.z) / 2
            shoulder_depth = (landmarks[L_SHOULDER].z + landmarks[R_SHOULDER].z) / 2
            self.baseline_forward_offset = shoulder_depth - ear_depth
        else:
            self.baseline_forward_offset = None

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
                self._frame_size = (w, h)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                image = frame
                issues = []

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    self._latest_landmarks = landmarks

                    l_ear_lm, r_ear_lm = landmarks[L_EAR], landmarks[R_EAR]
                    ear_visible = (l_ear_lm.visibility >= EAR_VISIBILITY_MIN
                                   and r_ear_lm.visibility >= EAR_VISIBILITY_MIN)
                    l_sh = _landmark_xy(landmarks, L_SHOULDER, w, h)
                    r_sh = _landmark_xy(landmarks, R_SHOULDER, w, h)
                    ear_depth = (l_ear_lm.z + r_ear_lm.z) / 2
                    shoulder_depth = (landmarks[L_SHOULDER].z + landmarks[R_SHOULDER].z) / 2

                    result = posture_logic.evaluate_posture(
                        l_sh, r_sh,
                        ear_depth=ear_depth,
                        shoulder_depth=shoulder_depth,
                        ear_visible=ear_visible,
                        baseline_width=self.baseline_width,
                        baseline_forward_offset=self.baseline_forward_offset,
                        forward_offset_margin=FORWARD_OFFSET_MARGIN,
                    )
                    issues = result["issues"]

                    new_raw_status = "이상" if issues else "정상"
                    if new_raw_status != self._raw_status:
                        self._raw_status = new_raw_status
                        self._status_since = time.time()

                    held_seconds = time.time() - self._status_since
                    if held_seconds >= SUSTAIN_SECONDS and self._raw_status != self._confirmed_status:
                        if self._confirmed_status == "정상" and self._raw_status == "이상":
                            self._db.log_status(result["is_turtle"], result["is_imbalanced"], result["forward_offset"])
                            _notify_windows("자세 교정 알림", " / ".join(issues))
                        self._confirmed_status = self._raw_status

                    style = mp_drawing.DrawingSpec(
                        color=COLOR_ABNORMAL if self._confirmed_status == "이상" else COLOR_NORMAL,
                        thickness=2, circle_radius=2,
                    )
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=style, connection_drawing_spec=style,
                    )

                if self.baseline_width is None:
                    guide_text = "바른 자세로 앉은 뒤 '기준 자세 설정' 버튼을 누르세요"
                    image = _put_text_kr(image, guide_text, (10, 8), 26, COLOR_NORMAL)
                else:
                    status_text = f"상태: {self._confirmed_status}" + (f" ({', '.join(issues)})" if issues else "")
                    color = COLOR_ABNORMAL if self._confirmed_status == "이상" else COLOR_NORMAL
                    image = _put_text_kr(image, status_text, (10, 8), 26, color)

                rgb_out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                fh, fw, ch = rgb_out.shape
                qimg = QImage(rgb_out.data, fw, fh, ch * fw, QImage.Format_RGB888).copy()
                self.frame_ready.emit(qimg)
                self.status_updated.emit({
                    "status": self._confirmed_status,
                    "issues": issues,
                    "baseline_set": self.baseline_width is not None,
                })

        cap.release()

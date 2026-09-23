# recorder.py
# 튜닝용 랜드마크 녹화/읽기. 영상은 저장하지 않고 판정에 쓰는 5개 랜드마크 좌표/신뢰도와 밝기만 남긴다.
# 한 줄에 JSON 하나(JSONL):
#   {"type": "baseline", "t", "baseline", "thresholds"}  — 녹화 시작 시점과 기준 자세를 다시 잡을 때마다
#   {"type": "frame", "t", "brightness", "points": {...}|null, "label": 라벨|null, "calib"?: true}
import json
import os
import time

RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")

# 녹화 중 GUI 숫자키로 붙이는 정답 라벨. replay.py가 판정 결과와 비교한다.
LABELS = {
    "normal": "바른 자세",
    "turtle": "거북목",
    "nod": "고개 숙임",
    "tilt": "어깨 기울임",
    "lean": "화면에 가까이",
}


class Recorder:
    def __init__(self, path=None):
        if path is None:
            os.makedirs(RECORDINGS_DIR, exist_ok=True)
            path = os.path.join(RECORDINGS_DIR, time.strftime("%Y%m%d_%H%M%S") + ".jsonl")
        self.path = path
        self._f = open(path, "w", encoding="utf-8")

    def write_baseline(self, t, baseline, thresholds):
        self._write({"type": "baseline", "t": t, "baseline": baseline, "thresholds": thresholds})

    def write_frame(self, t, brightness, points, label, calib=False):
        """calib: 기준 자세 측정 중인 프레임 (앱은 이 프레임을 판정하지 않으므로 재생에서도 건너뜀)."""
        frame = {"type": "frame", "t": round(t, 4), "brightness": round(brightness, 1),
                 "points": points, "label": label}
        if calib:
            frame["calib"] = True
        self._write(frame)

    def _write(self, obj):
        self._f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def close(self):
        self._f.close()


def read_recording(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

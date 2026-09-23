# posture_logic.py
# [Layer 3] 순수 자세 판정 로직. OpenCV/PyQt5에 의존하지 않아 단독 테스트가 가능하다.
# 여기서는 한 프레임의 지표와 "임계 대비 점수"만 계산한다. 시간에 걸친 확정/복귀는 posture_tracker가 맡는다.
import numpy as np

# 판정 항목. 키는 DB/녹화 파일에 그대로 저장되므로 바꾸지 않는다.
ISSUES = ("turtle", "lean", "tilt")
ISSUE_LABELS = {
    "turtle": "거북목 의심",
    "lean": "화면에 너무 가까움(구부정)",
    "tilt": "어깨 기울어짐",
}

# 기본 임계값. 인식이 안정적인 카메라에서 쓰는 최소값이며, 흔들리는 카메라에서는 auto_thresholds가 넓힌다.
# 목 길이비가 기준보다 이 비율 이상 줄면 거북목. 고개를 숙여도 똑같이 줄어들어 숙임도 함께 잡힌다 (neck_ratio 참고).
NECK_DROP_RATIO = 0.15
# 어깨 기울기가 기준 대비 이만큼(약 4도) 변하면 기울어짐.
# 실측(2026-09-24 7분 녹화): 바른 자세에서도 화면이 밝아지면(모니터 빛) 기울기가 0.015~0.05 더 크게 잡혔고,
# 마우스 쪽 어깨가 올라가는 것까지 겹쳐 0.05에서는 바른 자세의 17%가 기울어짐으로 확정됐다. 0.07에서는 1회(6초).
SHOULDER_TILT_THRESHOLD = 0.07
SHOULDER_GROW_RATIO = 1.15      # 어깨너비가 기준의 이 배수를 넘으면 화면에 너무 가까움

# 자동 임계값: 기준 자세 측정 중 값이 흔들린 폭(표준편차)의 이 배수보다 임계가 좁으면 넓힌다.
# 가만히 있어도 흔들리는 만큼은 자세 변화로 보지 않기 위함 (웹캠/조명/옷마다 흔들림이 다름).
NOISE_MULTIPLIER = 3.0
# 흔들림이 너무 커도 임계를 무한정 넓히면 판정이 사실상 꺼지므로 상한을 둔다.
MAX_NECK_DROP_RATIO = 0.30
MAX_SHOULDER_GROW_RATIO = 1.30
MAX_SHOULDER_TILT_THRESHOLD = 0.12

CALIB_MIN_SAMPLES = 20  # 기준 자세 측정에서 유효 샘플이 이보다 적으면 실패로 처리


def default_thresholds():
    return {
        "neck_drop": NECK_DROP_RATIO,
        "shoulder_grow": SHOULDER_GROW_RATIO,
        "tilt": SHOULDER_TILT_THRESHOLD,
    }


def merge_thresholds(saved):
    """
    저장된 임계(baseline.json/녹화 파일)를 불러온다. 없는 항목은 기본값을 쓰고, 기본값보다 좁은 항목은
    기본값으로 넓힌다. 저장된 임계는 "기본값과 흔들림 중 큰 값"이므로, 기본값을 올린 뒤에도 재설정 없이 반영된다.
    """
    thresholds = default_thresholds()
    for key, value in (saved or {}).items():
        if key in thresholds:
            thresholds[key] = max(thresholds[key], float(value))
    return thresholds


def _mid_and_distance(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    return (left + right) / 2, float(np.linalg.norm(left - right))


def neck_ratio(left_ear, right_ear, left_shoulder, right_shoulder):
    """
    목 길이비 = 어깨 중점과 귀 중점의 세로 간격 / 양 귀 사이 거리 (화면 좌표 x,y 기반).
    고개가 앞으로 빠지면 (1) 귀가 어깨 쪽으로 내려와 세로 간격이 줄고 (2) 머리가 카메라에 가까워져
    귀 사이 거리가 늘어나므로, 두 효과가 모두 값을 줄이는 방향으로 작용한다.
    카메라와의 거리가 바뀌면 분자/분모가 같이 변해 비율은 유지된다.
    고개를 숙여도 귀가 내려와 값이 줄어들어, 오래 숙이고 있으면 거북목으로 잡힌다(짧은 숙임은 유지 조건이 거름).

    고개 숙임 구분을 시도했다가 뺀 이력 (2026-09-24 실측 녹화 2개, 약 8분):
    코-귀 높이차로 숙임을 가려 거북목에서 빼려 했으나, 이 정면 카메라에서는 고개를 내밀 때도 코가 귀보다
    0.10~0.17(귀거리 기준) 내려가 가볍게 숙인 자세(0.17)와 구분되지 않았다. 크게 숙여도 귀가 같이 내려가
    0.15밖에 차이 나지 않았다. 예외를 두면 거북목 탐지율이 7%까지 떨어져 숙임도 거북목으로 보기로 했다.

    판정 방식 변경 이력:
    - mediapipe z(깊이) 사용 → 같은 자세에서도 조명에 따라 판정 여유만큼 값이 달라짐 (밝기 7↔27에서 -0.02~+0.04)
    - 분모를 어깨너비로 사용 → 어두우면 어깨가 약 7% 넓게 잡혀 비율이 함께 흔들림
      (실측: 밝기 32→8에서 세로간격/어깨너비 0.60→0.56, 귀거리/어깨너비 0.425→0.40)
    - 분모를 귀 사이 거리로 사용 (현재) → 같은 실측값으로 1.41→1.40, 조명 영향이 거의 상쇄됨
    """
    ear_mid, ear_distance = _mid_and_distance(left_ear, right_ear)
    if ear_distance == 0:
        return None
    shoulder_mid, _ = _mid_and_distance(left_shoulder, right_shoulder)
    return float(shoulder_mid[1] - ear_mid[1]) / ear_distance


def shoulder_tilt(left_shoulder, right_shoulder):
    """
    좌우 어깨 높이차를 어깨너비로 정규화한 부호 있는 기울기. left_shoulder(mediapipe 11번)가 더 낮으면 양수.
    카메라 워커는 화면을 좌우반전한 뒤 추론하므로 11번은 실제로는 사용자의 오른쪽 어깨다
    (실측: 11번이 화면 오른쪽에 잡힘). 판정은 기준 대비 차이의 절댓값만 쓰므로 부호는 표시에만 영향이 있다.
    """
    width = abs(left_shoulder[0] - right_shoulder[0])
    if width == 0:
        return 0.0
    return (left_shoulder[1] - right_shoulder[1]) / width


def shoulder_width(left_shoulder, right_shoulder):
    return _mid_and_distance(left_shoulder, right_shoulder)[1]


def evaluate_posture(left_shoulder, right_shoulder,
                     left_ear=None, right_ear=None, ear_visible=True,
                     baseline=None, thresholds=None):
    """
    한 프레임의 지표를 계산하고, 항목마다 "임계 대비 얼마나 벗어났는지" 점수를 매긴다.
    점수 1.0 = 임계에 딱 닿음, 0 = 기준 자세와 같음. 판정할 수 없는 항목은 None.
    모든 판정은 캘리브레이션 때 잡은 사용자 본인의 기준값 대비 변화량으로 한다
    (사람마다/웹캠 위치마다 평상시 값이 달라 고정값으로는 오탐이 잦음).

    baseline: {"width", "tilt", "neck"(None 가능)} 또는 None(캘리브레이션 전)
    thresholds: default_thresholds()와 같은 키. None이면 기본값.
    """
    thresholds = thresholds or default_thresholds()
    width = shoulder_width(left_shoulder, right_shoulder)
    tilt = shoulder_tilt(left_shoulder, right_shoulder)
    ears_ok = ear_visible and left_ear is not None and right_ear is not None
    neck = neck_ratio(left_ear, right_ear, left_shoulder, right_shoulder) if ears_ok else None

    scores = {issue: None for issue in ISSUES}
    if baseline:
        scores["lean"] = (width / baseline["width"] - 1) / (thresholds["shoulder_grow"] - 1)
        scores["tilt"] = abs(tilt - baseline["tilt"]) / thresholds["tilt"]
        if neck is not None and baseline.get("neck"):
            scores["turtle"] = (1 - neck / baseline["neck"]) / thresholds["neck_drop"]

    return {
        "scores": scores,
        "neck_ratio": neck,
        "shoulder_width": width,
        "tilt_ratio": tilt,
    }


def _widen(default, noise, cap):
    """기본 임계와 흔들림*배수 중 큰 값, 단 상한 이내. (임계값, 상한에 걸렸는지)를 돌려준다."""
    wanted = max(default, NOISE_MULTIPLIER * noise)
    return min(wanted, cap), wanted > cap


def auto_thresholds(necks, widths, tilts):
    """
    기준 자세를 유지하는 동안 모은 값들(판정과 같은 평활을 거친 값)의 흔들림으로
    이 카메라/환경에 맞는 임계값을 정한다. 목 길이비와 어깨너비는 기준 대비 비율로 판정하므로
    상대 흔들림(표준편차/중앙값)을, 기울기는 차이로 판정하므로 절대 흔들림을 쓴다.
    반환: default_thresholds()의 키 + "capped": 상한에 걸린 항목 이름 목록
    """
    capped = []
    result = default_thresholds()

    if len(necks) >= 2 and np.median(necks) > 0:
        result["neck_drop"], hit = _widen(NECK_DROP_RATIO, float(np.std(necks) / np.median(necks)),
                                          MAX_NECK_DROP_RATIO)
        if hit:
            capped.append(ISSUE_LABELS["turtle"])

    grow_margin, hit = _widen(SHOULDER_GROW_RATIO - 1, float(np.std(widths) / np.median(widths)),
                              MAX_SHOULDER_GROW_RATIO - 1)
    result["shoulder_grow"] = 1 + grow_margin
    if hit:
        capped.append(ISSUE_LABELS["lean"])

    result["tilt"], hit = _widen(SHOULDER_TILT_THRESHOLD, float(np.std(tilts)), MAX_SHOULDER_TILT_THRESHOLD)
    if hit:
        capped.append(ISSUE_LABELS["tilt"])

    result["capped"] = capped
    return result


def baseline_from_samples(samples):
    """
    기준 자세 측정 샘플({"width", "tilt", "neck"|None})로 기준값과 자동 임계를 정한다.
    샘플이 부족하면 None. 귀가 절반 이상의 프레임에서 보였을 때만 거북목 기준을 잡는다.
    반환: {"baseline": {...}, "thresholds": {...}, "capped": [...], "missing": ["turtle"] 또는 []}
    """
    if len(samples) < CALIB_MIN_SAMPLES:
        return None

    widths = [s["width"] for s in samples]
    tilts = [s["tilt"] for s in samples]
    necks = [s["neck"] for s in samples if s.get("neck") is not None]
    if len(necks) < len(samples) / 2:
        necks = []
    baseline = {
        "width": float(np.median(widths)),
        "tilt": float(np.median(tilts)),
        "neck": float(np.median(necks)) if necks else None,
    }
    auto = auto_thresholds(necks, widths, tilts)
    capped = auto.pop("capped")
    missing = ["turtle"] if baseline["neck"] is None else []
    return {"baseline": baseline, "thresholds": auto, "capped": capped, "missing": missing}

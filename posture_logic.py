# posture_logic.py
# [Layer 3] 순수 자세 판정 로직. OpenCV/PyQt5에 의존하지 않아 단독 테스트가 가능하다.
import numpy as np


def check_forward_head(ear_depth, shoulder_depth, baseline_offset=None,
                        margin=0.04, fixed_threshold=0.08):
    """
    귀와 어깨의 카메라 기준 상대 깊이(mediapipe z)로 거북목(고개가 앞으로 빠짐)을 판정한다.
    mediapipe z는 카메라에 가까울수록 값이 작아지므로, offset = shoulder_depth - ear_depth가
    클수록 귀가 어깨보다 카메라 쪽(앞)으로 나와 있다는 뜻이다.

    x,y 화면 좌표만 쓰는 각도 기반 판정과 달리, 이 값은 고개를 좌우로 돌리는 동작에는
    거의 영향을 안 받는다 (한쪽 귀는 카메라에 가까워지고 반대쪽은 멀어져 평균이 상쇄됨).
    반면 고개를 앞으로 내미는 동작(진짜 거북목)에는 양쪽 귀가 함께 가까워지므로 그대로 반응한다.
    """
    offset = shoulder_depth - ear_depth
    if baseline_offset is not None:
        return offset > baseline_offset + margin, offset
    return offset > fixed_threshold, offset


def check_shoulder_imbalance(left_shoulder, right_shoulder, threshold=0.05):
    """좌우 어깨 높이차를 어깨너비로 정규화해 비교한다."""
    width = abs(left_shoulder[0] - right_shoulder[0])
    if width == 0:
        return False, 0.0
    ratio = abs(left_shoulder[1] - right_shoulder[1]) / width
    return ratio > threshold, ratio


def check_leaning_forward(shoulder_width, baseline_width, ratio_threshold=1.15):
    """어깨너비가 기준(캘리브레이션) 대비 크게 늘어나면 화면 쪽으로 상체를 기울인 것으로 본다."""
    if not baseline_width:
        return False
    return shoulder_width > baseline_width * ratio_threshold


def evaluate_posture(left_shoulder, right_shoulder,
                      ear_depth=None, shoulder_depth=None, ear_visible=True,
                      baseline_width=None, baseline_forward_offset=None,
                      forward_offset_margin=0.04, shoulder_tilt_threshold=0.05,
                      shoulder_grow_ratio=1.15):
    """
    세 가지 판정을 한 번에 수행한다.
    baseline_forward_offset이 설정돼 있으면(캘리브레이션 완료) 고정 threshold 대신
    사용자 본인의 평상시 귀-어깨 깊이차를 기준으로 판정한다. 사람마다/웹캠 위치마다
    평상시 값이 달라 고정값만으로는 오탐이 잦기 때문이다.
    """
    issues = []

    is_turtle = False
    forward_offset = None
    if ear_visible and ear_depth is not None and shoulder_depth is not None:
        is_turtle, forward_offset = check_forward_head(
            ear_depth, shoulder_depth, baseline_forward_offset, forward_offset_margin
        )
    if is_turtle:
        issues.append("거북목 의심")

    shoulder_width = float(np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder)))
    if check_leaning_forward(shoulder_width, baseline_width, shoulder_grow_ratio):
        issues.append("화면에 너무 가까움(구부정)")

    is_imbalanced, tilt_ratio = check_shoulder_imbalance(
        left_shoulder, right_shoulder, shoulder_tilt_threshold
    )
    if is_imbalanced:
        issues.append("어깨 기울어짐")

    return {
        "issues": issues,
        "is_turtle": is_turtle,
        "is_imbalanced": is_imbalanced,
        "forward_offset": forward_offset,
        "shoulder_width": shoulder_width,
        "tilt_ratio": tilt_ratio,
    }

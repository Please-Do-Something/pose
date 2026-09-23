# 프로젝트 구조 — 자세 교정 모니터

웹캠으로 실시간 자세(거북목, 어깨 기울어짐, 화면에 너무 가까이 붙음)를 감지해 알려주고, 기록을 대시보드로 보여주는 PyQt5 데스크톱 앱.

## 디렉터리 구조

```
pose/
├── main.py              # [Layer 1] PyQt5 GUI 메인 윈도우 & 대시보드
├── camera_worker.py     # [Layer 2] 웹캠 캡처 & MediaPipe 랜드마크 추출 (QThread)
├── posture_logic.py     # [Layer 3] 순수 자세 판정 로직 (OpenCV/PyQt5 비의존)
├── db_manager.py         # [Layer 4] SQLite 저장 & 집계
├── posture.db             # db_manager.py가 생성하는 SQLite DB (런타임 생성, 최초 실행 전엔 없음)
├── requirements.txt       # 의존성 목록
├── architect.txt           # 원본 설계 스펙 (영문, AI 코드 생성용 프롬프트 문서)
├── README.md               # 프로젝트 개요 & 실행법
└── venv/                    # 가상환경 (Python 3.11.9)
```

## 아키텍처: 4-Layer 구조

```
┌─────────────────────────────────────────────────────────┐
│ main.py (Layer 1 — GUI)                                   │
│  MainWindow(QMainWindow)                                  │
│   - 좌측: 웹캠 프리뷰(QLabel) + 상태 텍스트 + 캘리브레이션 버튼 │
│   - 우측: 대시보드(오늘 점수 / 거북목 횟수 / 시간대별 차트)    │
│   - QTimer로 30초마다 대시보드 갱신                          │
└───────────────┬─────────────────────────────────────────┘
                │ pyqtSignal: frame_ready(QImage), status_updated(dict)
┌───────────────▼─────────────────────────────────────────┐
│ camera_worker.py (Layer 2 — QThread)                       │
│  CameraWorker                                              │
│   - cv2.VideoCapture(0)로 프레임 캡처                        │
│   - mediapipe.solutions.pose로 랜드마크 추출                  │
│   - 귀(7,8)/어깨(11,12) 좌표 → posture_logic 호출              │
│   - 2초(SUSTAIN_SECONDS) 이상 유지된 상태만 "확정"으로 인정        │
│   - 확정 시 DB 기록 + winotify 토스트 알림                       │
│   - 프레임에 스켈레톤/한글 상태 텍스트(PIL 합성) 오버레이 후 GUI로 전달  │
└───────────────┬─────────────────────────────────────────┘
                │ 함수 호출 (직접 의존, 신호 아님)
┌───────────────▼─────────────────────────────────────────┐
│ posture_logic.py (Layer 3 — 순수 로직, 단위 테스트 가능)         │
│  - check_forward_head (z 깊이 기반 거북목 판정) /                │
│    check_shoulder_imbalance / check_leaning_forward             │
│  - evaluate_posture(...) → {issues, is_turtle, is_imbalanced, │
│    forward_offset, shoulder_width, tilt_ratio}                 │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ db_manager.py (Layer 4 — 저장/집계, main.py에서 직접 사용)     │
│  DBManager (SQLite: posture.db)                             │
│   - init_db(): posture_logs 테이블 생성                        │
│   - log_status(is_turtle, is_imbalanced, forward_offset)       │
│   - get_daily_summary(date_str) → 점수/횟수/시간대별 집계 dict      │
└─────────────────────────────────────────────────────────┘
```

`main.py`가 `DBManager` 인스턴스를 만들어 `CameraWorker`에 주입하므로, DB 기록은 워커 스레드 안에서 바로 이뤄진다(대시보드 갱신은 메인 스레드의 타이머가 별도로 조회).

## 판정 로직 (posture_logic.py)

캘리브레이션(`기준 자세 설정` 버튼) 이전/이후로 판정 기준이 다르다:

| 지표 | 캘리브레이션 전 | 캘리브레이션 후 |
|---|---|---|
| 거북목 | 귀-어깨 카메라 상대깊이차 > 0.08 (고정, `fixed_threshold`) | 깊이차가 기준보다 0.04 이상 증가 (`FORWARD_OFFSET_MARGIN`) |
| 구부정(화면에 가까움) | 판정 안 함 | 어깨너비가 기준 대비 1.15배 이상 |
| 어깨 기울어짐 | 좌우 어깨 높이차/너비 > 0.05 (항상 동일) | 동일 |

상태 변화는 `SUSTAIN_SECONDS`(2초) 이상 유지돼야 "확정"되어 DB 기록/알림이 발생한다 (짧은 흔들림에 의한 오탐 방지).

**거북목 판정은 x,y 화면 각도가 아니라 z(깊이)값을 쓴다.** mediapipe 랜드마크의 z는 카메라와의 상대 거리(가까울수록 값이 작음)를 나타내는데, `offset = 어깨 평균깊이 - 귀 평균깊이`가 클수록 귀가 어깨보다 카메라 쪽으로 나와 있다는 뜻이다. 이 값은 고개를 좌우로 돌려도 거의 안 변한다(한쪽 귀는 가까워지고 반대쪽은 멀어져 평균이 상쇄됨). 반면 고개를 앞으로 내밀면(진짜 거북목) 양쪽 귀가 함께 가까워지므로 offset이 커진다. x,y 각도 기반 판정은 좌우 회전과 전방 숙임을 구분 못 해 오탐이 잦았던 이전 방식의 결함이었다.

## 데이터베이스 스키마 (posture.db)

```sql
CREATE TABLE posture_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    is_turtle BOOLEAN NOT NULL,
    is_imbalanced BOOLEAN NOT NULL,
    neck_angle REAL   -- 컬럼명은 유지되지만 저장되는 값은 forward_offset(z 깊이차)이다
)
```

- `log_status()`는 "정상 → 이상"으로 확정 전환되는 순간에만 1건 INSERT (지속 시간이 아닌 이벤트 단위 기록)
- `get_daily_summary()`는 해당 날짜의 레코드를 모두 읽어 시간대별(0~23시) 카운트와 `100 - 이벤트수*2` 점수를 계산

## 의존성 (requirements.txt)

| 패키지 | 용도 |
|---|---|
| opencv-python | 웹캠 캡처, 이미지 처리 |
| mediapipe | 자세(Pose) 랜드마크 추출 AI 모델 |
| PyQt5 | GUI 프레임워크 |
| pyqtgraph | 대시보드 시간대별 분포 바 차트 |
| pyinstaller | .exe 빌드 |
| numpy | 벡터/각도 계산 |
| Pillow | 한글 텍스트 오버레이 (cv2.putText 한글 미지원 우회) |
| winotify | Windows 토스트 알림 (미설치 시 알림만 비활성화, 앱은 정상 동작) |

가상환경: `venv/` (Python 3.11.9)

## 실행

```bash
.\venv\Scripts\Activate.ps1
python main.py
```

최초 실행 후 카메라 앞에 바른 자세로 앉아 **'기준 자세 설정'** 버튼을 눌러 캘리브레이션해야 판정이 정확해진다.

## 빌드 (.exe)

```bash
pyinstaller -w -F main.py
```

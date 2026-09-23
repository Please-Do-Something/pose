# 자세 교정 모니터

웹캠으로 실시간 자세(거북목, 어깨 기울어짐)를 감지해 알려주고, 기록을 대시보드로 보여주는 데스크톱 앱.

## 구조

- `posture_logic.py` — [Layer 3] OpenCV/PyQt5에 의존하지 않는 순수 판정 로직 (한 프레임의 지표와 임계 대비 점수, 기준 자세/자동 임계).
- `posture_tracker.py` — [Layer 3] 평활, 항목별 확정/복귀, 판정 시간 집계, 알림 판단 (순수 로직).
- `recorder.py` / `replay.py` — 튜닝용 랜드마크 녹화와, 녹화를 같은 판정 코드로 재생·채점하는 CLI (`python replay.py recordings/<파일>.jsonl`).
- `tests/` — 합성 데이터 단위 테스트 (`python -m unittest discover tests`).
- `camera_worker.py` — [Layer 2] 웹캠 캡처 및 MediaPipe 랜드마크 추출을 처리하는 QThread. 프레임과 판정 결과를 Qt 시그널로 GUI에 전달한다.
- `db_manager.py` — [Layer 4] SQLite(`posture.db`)에 자세 이상 이벤트를 기록하고, 대시보드용 통계를 집계한다.
- `main.py` — [Layer 1] PyQt5 GUI 진입점. 좌측에 실시간 웹캠 화면, 우측에 오늘의 자세 점수/거북목 경고 횟수/시간대별 분포 차트를 표시한다.

## 실행

```bash
pip install -r requirements.txt
python main.py
```

'기준 자세 설정' 버튼을 눌러 현재 앉은 자세를 기준으로 캘리브레이션한 뒤 사용한다.

## 빌드 (.exe)

```bash
pyinstaller -w -F main.py
```

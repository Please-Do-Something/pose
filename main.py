# main.py
# [Layer 1] PyQt5 GUI 메인 윈도우 & 대시보드.
import sys
from datetime import datetime

# camera_worker(mediapipe)는 PyQt5보다 먼저 import해야 한다.
# PyQt5를 먼저 import하면 mediapipe의 네이티브 DLL(_framework_bindings) 로딩이
# 깨져 "DLL 초기화 루틴을 실행할 수 없습니다" ImportError가 발생한다 (Windows DLL 충돌).
from camera_worker import CameraWorker
from db_manager import DBManager
from posture_logic import ISSUE_LABELS
from recorder import LABELS

import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QShortcut, QVBoxLayout, QWidget,
)

# 녹화 중 숫자키로 붙이는 정답 라벨 (0 = 라벨 없음)
LABEL_KEYS = {"0": None, "1": "normal", "2": "turtle", "3": "nod", "4": "tilt", "5": "lean"}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("자세 교정 모니터")

        self.db = DBManager()
        self.worker = CameraWorker(self.db)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status_updated.connect(self.on_status)
        self.worker.calibration_finished.connect(self.on_calibration_finished)

        self._build_ui()

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_dashboard)
        self.refresh_timer.start(30_000)  # 30초마다 대시보드 갱신

        self.worker.start()
        self.refresh_dashboard()

    def _build_ui(self):
        central = QWidget()
        layout = QHBoxLayout(central)

        # 왼쪽: 웹캠 피드 + 상태 + 캘리브레이션 버튼
        left = QVBoxLayout()
        self.video_label = QLabel("카메라를 시작하는 중...")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.status_label = QLabel("상태: -")
        self.calibrate_btn = QPushButton("기준 자세 다시 설정" if self.worker.has_baseline else "기준 자세 설정")
        self.calibrate_btn.clicked.connect(self.start_calibration)
        left.addWidget(self.video_label)
        left.addWidget(self.status_label)
        left.addWidget(self.calibrate_btn)

        self.debug_check = QCheckBox("디버그 정보 표시 (판정 수치)")
        self.debug_check.toggled.connect(self.on_debug_toggled)
        left.addWidget(self.debug_check)

        # 튜닝용: 랜드마크를 recordings/에 저장하고 replay.py로 판정을 재생/채점한다
        label_help = ", ".join(f"{k}={LABELS[v] if v else '없음'}" for k, v in LABEL_KEYS.items())
        self.record_check = QCheckBox("판정 데이터 녹화 (튜닝용)")
        self.record_check.setToolTip(f"녹화 중 숫자키로 지금 자세의 정답 라벨을 붙입니다: {label_help}")
        self.record_check.toggled.connect(self.on_record_toggled)
        left.addWidget(self.record_check)
        for key, label in LABEL_KEYS.items():
            QShortcut(QKeySequence(key), self, activated=lambda label=label: self.set_label(label))

        # 오른쪽: 대시보드 (오늘의 자세 점수, 거북목 경고 횟수, 시간대별 분포)
        right = QVBoxLayout()
        dashboard_box = QGroupBox("오늘의 대시보드")
        dashboard_layout = QVBoxLayout()
        self.score_label = QLabel("자세 점수: -")
        self.turtle_label = QLabel("이상 자세 횟수: -")
        self.turtle_label.setWordWrap(True)
        self.chart = pg.PlotWidget()
        self.chart.setBackground("w")
        self.chart.setLabel("left", "이상 자세 시간(분)")
        self.chart.setLabel("bottom", "시간대(시)")
        dashboard_layout.addWidget(self.score_label)
        dashboard_layout.addWidget(self.turtle_label)
        dashboard_layout.addWidget(self.chart)
        dashboard_box.setLayout(dashboard_layout)
        right.addWidget(dashboard_box)

        layout.addLayout(left, 2)
        layout.addLayout(right, 1)
        self.setCentralWidget(central)

    def on_frame(self, qimg):
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio
        ))

    def on_debug_toggled(self, checked):
        self.worker.debug_enabled = checked

    def on_record_toggled(self, checked):
        self.worker.recording_wanted = checked
        if not checked:
            self.worker.label = None

    def set_label(self, label):
        self.worker.label = label
        if self.record_check.isChecked():
            self.statusBar().showMessage(f"라벨: {LABELS[label] if label else '없음'}", 3000)

    def start_calibration(self):
        self.calibrate_btn.setEnabled(False)
        self.calibrate_btn.setText("기준 자세 측정 중...")
        self.worker.calibrate()

    def on_calibration_finished(self, success, message):
        self.calibrate_btn.setEnabled(True)
        self.calibrate_btn.setText("기준 자세 다시 설정" if self.worker.has_baseline else "기준 자세 설정")
        self.statusBar().showMessage(message.replace("\n", " "), 8000)
        if not success:
            QMessageBox.warning(self, "기준 자세 설정 실패", message)

    def on_status(self, info):
        if info["calibrating"]:
            self.status_label.setText("기준 자세 측정 중... 바른 자세를 유지하세요")
            return
        if not info["baseline_set"]:
            self.status_label.setText("바른 자세로 앉은 뒤 '기준 자세 설정' 버튼을 누르세요")
            return
        if info["hold_reason"]:
            self.status_label.setText(f"판정 보류: {info['hold_reason']}")
            return
        text = f"상태: {info['status']}"
        if info["issues"]:
            text += " (" + ", ".join(info["issues"]) + ")"
        self.status_label.setText(text)

    def refresh_dashboard(self):
        today = datetime.now().strftime("%Y-%m-%d")
        summary = self.db.get_daily_summary(today)
        if summary["score"] is None:
            self.score_label.setText("자세 점수: - (측정 시간이 부족합니다)")
        else:
            self.score_label.setText(
                f"자세 점수: {summary['score']}  (측정 {summary['monitored_minutes']:.0f}분 중 "
                f"이상 자세 {summary['bad_minutes']:.0f}분)")
        counts = summary["event_counts"]
        self.turtle_label.setText("이상 자세 횟수: " + ", ".join(
            f"{ISSUE_LABELS[issue]} {n}" for issue, n in counts.items()))

        self.chart.clear()
        hours = list(range(24))
        bar = pg.BarGraphItem(x=hours, height=summary["hourly_bad_minutes"], width=0.6, brush="r")
        self.chart.addItem(bar)

    def closeEvent(self, event):
        self.worker.stop()
        self.db.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1000, 560)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

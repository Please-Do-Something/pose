# main.py
# [Layer 1] PyQt5 GUI 메인 윈도우 & 대시보드.
import sys
from datetime import datetime

# camera_worker(mediapipe)는 PyQt5보다 먼저 import해야 한다.
# PyQt5를 먼저 import하면 mediapipe의 네이티브 DLL(_framework_bindings) 로딩이
# 깨져 "DLL 초기화 루틴을 실행할 수 없습니다" ImportError가 발생한다 (Windows DLL 충돌).
from camera_worker import CameraWorker
from db_manager import DBManager

import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QVBoxLayout, QWidget,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("자세 교정 모니터")

        self.db = DBManager()
        self.worker = CameraWorker(self.db)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status_updated.connect(self.on_status)

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
        self.calibrate_btn = QPushButton("기준 자세 설정")
        self.calibrate_btn.clicked.connect(self.worker.calibrate)
        left.addWidget(self.video_label)
        left.addWidget(self.status_label)
        left.addWidget(self.calibrate_btn)

        # 오른쪽: 대시보드 (오늘의 자세 점수, 거북목 경고 횟수, 시간대별 분포)
        right = QVBoxLayout()
        dashboard_box = QGroupBox("오늘의 대시보드")
        dashboard_layout = QVBoxLayout()
        self.score_label = QLabel("자세 점수: -")
        self.turtle_label = QLabel("거북목 경고 횟수: -")
        self.chart = pg.PlotWidget()
        self.chart.setBackground("w")
        self.chart.setLabel("left", "발생 횟수")
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

    def on_status(self, info):
        if not info["baseline_set"]:
            self.status_label.setText("바른 자세로 앉은 뒤 '기준 자세 설정' 버튼을 누르세요")
            return
        text = f"상태: {info['status']}"
        if info["issues"]:
            text += " (" + ", ".join(info["issues"]) + ")"
        self.status_label.setText(text)

    def refresh_dashboard(self):
        today = datetime.now().strftime("%Y-%m-%d")
        summary = self.db.get_daily_summary(today)
        self.score_label.setText(f"자세 점수: {summary['score']}")
        self.turtle_label.setText(f"거북목 경고 횟수: {summary['turtle_count']}")

        self.chart.clear()
        hours = list(range(24))
        bar = pg.BarGraphItem(x=hours, height=summary["hourly_counts"], width=0.6, brush="r")
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

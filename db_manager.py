# db_manager.py
# [Layer 4] 로컬 SQLite 저장소 관리 및 대시보드용 집계 쿼리.
import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "posture.db")


class DBManager:
    def __init__(self, db_path=DB_PATH):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self.init_db()

    def init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS posture_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                is_turtle BOOLEAN NOT NULL,
                is_imbalanced BOOLEAN NOT NULL,
                neck_angle REAL
            )
        """)
        self._conn.commit()

    def log_status(self, is_turtle, is_imbalanced, neck_angle):
        """이상 자세로 '확정'되는 순간에만 호출된다 (1초 미만의 흔들림은 호출측에서 걸러짐)."""
        self._conn.execute(
            "INSERT INTO posture_logs (timestamp, is_turtle, is_imbalanced, neck_angle) "
            "VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), bool(is_turtle), bool(is_imbalanced), neck_angle),
        )
        self._conn.commit()

    def get_daily_summary(self, date_str):
        """date_str: 'YYYY-MM-DD'. 대시보드용 총계와 시간대별 발생 횟수를 반환한다."""
        cur = self._conn.execute(
            "SELECT timestamp, is_turtle, is_imbalanced FROM posture_logs WHERE timestamp LIKE ?",
            (f"{date_str}%",),
        )
        rows = cur.fetchall()

        hourly_counts = [0] * 24
        turtle_count = 0
        imbalance_count = 0
        for timestamp, is_turtle, is_imbalanced in rows:
            hour = int(timestamp[11:13])
            hourly_counts[hour] += 1
            turtle_count += int(is_turtle)
            imbalance_count += int(is_imbalanced)

        total_events = len(rows)
        score = max(0, 100 - total_events * 2)

        return {
            "date": date_str,
            "score": score,
            "turtle_count": turtle_count,
            "imbalance_count": imbalance_count,
            "hourly_counts": hourly_counts,
        }

    def close(self):
        self._conn.close()

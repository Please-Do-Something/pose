# db_manager.py
# [Layer 4] 로컬 SQLite 저장소 관리 및 대시보드용 집계 쿼리.
import os
import sqlite3
import threading
from datetime import datetime

from posture_logic import ISSUES

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "posture.db")
MIN_SCORE_SECONDS = 60  # 판정 시간이 이보다 짧으면 점수를 내지 않음 (몇 초 측정으로 0점/100점이 나오지 않도록)


def _iso(ts):
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


class DBManager:
    def __init__(self, db_path=DB_PATH):
        # 카메라 워커 스레드(기록)와 GUI 스레드(대시보드 조회)가 함께 쓰므로 연결 사용을 잠금으로 직렬화한다
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self.init_db()

    def init_db(self):
        with self._lock:
            # 예전 방식(확정 순간 1건씩)의 기록. 과거 데이터 보존용으로 테이블만 남겨 둔다.
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS posture_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    is_turtle BOOLEAN NOT NULL,
                    is_imbalanced BOOLEAN NOT NULL,
                    neck_angle REAL
                )
            """)
            # 이상 자세 한 번 = 한 행. 복귀하면 end_ts가 채워진다 (앱이 비정상 종료되면 NULL로 남음).
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS posture_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    issue TEXT NOT NULL,
                    start_ts DATETIME NOT NULL,
                    end_ts DATETIME,
                    metric REAL
                )
            """)
            # 시간대별 판정 시간과 그중 이상 자세였던 시간(초). 점수는 이 비율로 계산한다.
            issue_cols = ", ".join(f"{issue}_sec REAL NOT NULL DEFAULT 0" for issue in ISSUES)
            self._conn.execute(f"""
                CREATE TABLE IF NOT EXISTS posture_time (
                    date TEXT NOT NULL,
                    hour INTEGER NOT NULL,
                    monitored_sec REAL NOT NULL DEFAULT 0,
                    bad_sec REAL NOT NULL DEFAULT 0,
                    {issue_cols},
                    PRIMARY KEY (date, hour)
                )
            """)
            self._conn.commit()

    def start_event(self, issue, ts, metric):
        """이상 자세로 확정된 순간 호출. 복귀 때 end_event에 넘길 행 id를 돌려준다."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO posture_events (issue, start_ts, metric) VALUES (?, ?, ?)",
                (issue, _iso(ts), metric),
            )
            self._conn.commit()
            return cur.lastrowid

    def end_event(self, event_id, ts):
        with self._lock:
            self._conn.execute("UPDATE posture_events SET end_ts = ? WHERE id = ?", (_iso(ts), event_id))
            self._conn.commit()

    def add_time(self, buckets):
        """PostureTracker.pop_time() 결과를 누적한다."""
        if not buckets:
            return
        cols = ["monitored_sec", "bad_sec"] + [f"{issue}_sec" for issue in ISSUES]
        keys = ["monitored", "bad"] + list(ISSUES)
        updates = ", ".join(f"{c} = {c} + excluded.{c}" for c in cols)
        sql = (f"INSERT INTO posture_time (date, hour, {', '.join(cols)}) "
               f"VALUES (?, ?, {', '.join('?' * len(cols))}) "
               f"ON CONFLICT(date, hour) DO UPDATE SET {updates}")
        with self._lock:
            self._conn.executemany(sql, [(d, h, *(v[k] for k in keys)) for (d, h), v in buckets.items()])
            self._conn.commit()

    def get_daily_summary(self, date_str):
        """date_str: 'YYYY-MM-DD'. 대시보드용 점수/항목별 횟수/시간대별 이상 자세 시간을 반환한다."""
        with self._lock:
            time_rows = self._conn.execute(
                "SELECT hour, monitored_sec, bad_sec FROM posture_time WHERE date = ?", (date_str,)
            ).fetchall()
            event_rows = self._conn.execute(
                "SELECT issue, COUNT(*) FROM posture_events WHERE start_ts LIKE ? GROUP BY issue",
                (f"{date_str}%",),
            ).fetchall()

        hourly_bad_minutes = [0.0] * 24
        monitored = bad = 0.0
        for hour, m_sec, b_sec in time_rows:
            hourly_bad_minutes[hour] = b_sec / 60
            monitored += m_sec
            bad += b_sec

        counts = {issue: 0 for issue in ISSUES}
        # 예전에 있던 항목(고개 숙임 "nod")의 기록은 집계에서 뺀다
        counts.update({issue: n for issue, n in event_rows if issue in counts})
        # 점수 = 판정한 시간 중 바른 자세였던 비율. 짧게 여러 번 숙인 사람보다 오래 거북목이던 사람이 더 낮게 나온다.
        score = round(100 * (1 - bad / monitored)) if monitored >= MIN_SCORE_SECONDS else None

        return {
            "date": date_str,
            "score": score,
            "monitored_minutes": monitored / 60,
            "bad_minutes": bad / 60,
            "event_counts": counts,
            "turtle_count": counts["turtle"],
            "hourly_bad_minutes": hourly_bad_minutes,
        }

    def close(self):
        with self._lock:
            self._conn.close()

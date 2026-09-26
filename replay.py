# replay.py
# 녹화한 랜드마크(recordings/*.jsonl)를 앱과 같은 판정 코드(posture_tracker)로 다시 돌려 결과를 요약한다.
# 임계값/판정 로직을 바꾼 뒤 같은 녹화로 오탐·미탐이 어떻게 달라지는지 비교하는 용도.
#
#   python replay.py recordings/20260924_101500.jsonl
#   python replay.py rec.jsonl --set neck_drop=0.12 --set tilt=0.08  # 임계값을 바꿔 보기
#   python replay.py rec.jsonl --baseline baseline.json              # 녹화 속 기준 대신 다른 기준 파일로
#   python replay.py rec.jsonl --csv out.csv                         # 프레임별 수치를 CSV로
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict

import posture_logic
import posture_tracker
from posture_logic import ISSUE_LABELS, ISSUES
from recorder import LABELS, read_recording


def load_baseline_file(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    baseline = {"width": float(data["width"]), "tilt": float(data["tilt"]), "neck": data.get("neck")}
    return baseline, posture_logic.merge_thresholds(data.get("thresholds"))


def replay(records, baseline_override=None, threshold_overrides=None):
    """
    녹화 레코드를 판정한다. 반환: 프레임별 결과 목록과 확정/복귀 이벤트 목록.
    앱과 마찬가지로 기준 자세 측정 중인 프레임은 판정하지 않고, 기준이 바뀌면 판정 상태를 새로 시작한다.
    """
    tracker = posture_tracker.PostureTracker()
    overrides = threshold_overrides or {}

    def apply(baseline, thresholds, t):
        # 앱이 기준 파일을 불러올 때처럼 현재 기본값보다 좁은 임계는 넓힌다 (예전 녹화도 현재 로직으로 재생)
        thresholds = {**posture_logic.merge_thresholds(thresholds), **overrides}
        return tracker.set_baseline(baseline, thresholds, t)

    if baseline_override:
        apply(*baseline_override, 0.0)

    frames, events = [], []
    for rec in records:
        if rec["type"] == "baseline":
            if not baseline_override and rec["baseline"] is not None:
                events += apply(rec["baseline"], rec["thresholds"], rec["t"])
            continue
        if rec.get("calib"):
            continue
        m = posture_tracker.measurement_from_points(rec["points"]) if rec["points"] else None
        res = tracker.process(rec["t"], m, rec["brightness"])
        events += res["events"]
        frames.append({
            "t": rec["t"],
            "label": rec.get("label"),
            "judged": res["eval"] is not None,
            "scores": res["eval"]["scores"] if res["eval"] else {},
            "eval": res["eval"],
            "confirmed": set(tracker.confirmed_issues()),
            "notify": res["notify"],
            "seat_shift": res["seat_shift"],
        })
    if frames:
        events += tracker.end_all(frames[-1]["t"])
    return frames, events


def summarize(frames, events, out=sys.stdout):
    def p(*args):
        print(*args, file=out)

    if not frames:
        p("판정할 프레임이 없습니다.")
        return
    judged = [f for f in frames if f["judged"]]
    duration = frames[-1]["t"] - frames[0]["t"]
    p(f"길이 {duration:.0f}초, 프레임 {len(frames)}개, 판정 {len(judged)}개 "
      f"(보류/기준 없음 {len(frames) - len(judged)}개)")
    if not judged:
        p("기준 자세가 없어 판정하지 않았습니다. --baseline 으로 기준 파일을 지정하세요.")
        return

    p("\n[항목별 확정 결과]")
    starts = Counter(e["issue"] for e in events if e["type"] == "start")
    notifies = Counter(i for f in frames for i in f["notify"])
    for issue in ISSUES:
        secs = _confirmed_seconds(frames, issue)
        p(f"  {ISSUE_LABELS[issue]:<16} 확정 {starts[issue]}회, {secs:.0f}초, 알림 {notifies[issue]}회")
    shifts = [f["seat_shift"]["shift"] for f in frames if f["seat_shift"]]
    if shifts:
        p(f"  앉은 거리 변화로 거북목 기준 자동 이동 {len(shifts)}회: "
          + ", ".join(f"{v * 100:+.0f}%" for v in shifts))

    by_label = defaultdict(list)
    for f in judged:
        by_label[f["label"]].append(f)
    if set(by_label) == {None}:
        p("\n라벨이 없어 채점은 생략합니다 (녹화 중 숫자키로 라벨을 붙이세요).")
        return

    p("\n[라벨별 판정 비율] 순간=그 프레임 점수가 임계 이상, 확정=유지 조건까지 통과")
    header = "  라벨         프레임 " + " ".join(f"{ISSUE_LABELS[i][:6]:>14}" for i in ISSUES)
    p(header)
    for label in [*LABELS, None]:
        fs = by_label.get(label)
        if not fs:
            continue
        cells = []
        for issue in ISSUES:
            inst = sum((f["scores"].get(issue) or 0) >= 1 for f in fs) / len(fs)
            conf = sum(issue in f["confirmed"] for f in fs) / len(fs)
            cells.append(f"{inst * 100:5.0f}%/{conf * 100:4.0f}%")
        name = LABELS[label] if label else "(라벨 없음)"
        p(f"  {name:<10} {len(fs):>6} " + " ".join(f"{c:>14}" for c in cells))

    p("\n[핵심 지표] (확정 기준, 프레임 비율)")
    _headline(p, by_label.get("normal"), lambda f: bool(f["confirmed"]), "바른 자세인데 이상 판정 (오탐)")
    for label, issue in (("turtle", "turtle"), ("tilt", "tilt"), ("lean", "lean")):
        _headline(p, by_label.get(label), lambda f, i=issue: i in f["confirmed"],
                  f"{LABELS[label]} 라벨에서 {ISSUE_LABELS[issue]} 확정 (탐지율, 유지 시간만큼 늦게 잡힘)")
    _headline(p, by_label.get("nod"), lambda f: "turtle" in f["confirmed"],
              "고개 숙임 라벨에서 거북목 확정 (오래 숙이면 거북목으로 보는 게 의도된 동작)")


def _headline(p, fs, pred, text):
    if fs:
        p(f"  {text}: {sum(map(pred, fs)) / len(fs) * 100:.0f}% ({len(fs)}프레임)")


def _confirmed_seconds(frames, issue):
    total = 0.0
    for prev, cur in zip(frames, frames[1:]):
        if issue in prev["confirmed"] and cur["t"] - prev["t"] <= posture_tracker.MAX_GAP:
            total += cur["t"] - prev["t"]
    return total


def write_csv(frames, path):
    cols = ["t", "label", "judged", "neck_ratio", "shoulder_width", "tilt_ratio"]
    cols += [f"score_{i}" for i in ISSUES] + ["confirmed"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for fr in frames:
            ev = fr["eval"] or {}
            w.writerow([fr["t"], fr["label"] or "", int(fr["judged"]),
                        *(ev.get(k) for k in ("neck_ratio", "shoulder_width", "tilt_ratio")),
                        *(fr["scores"].get(i) for i in ISSUES),
                        "|".join(sorted(fr["confirmed"]))])


def main(argv=None):
    ap = argparse.ArgumentParser(description="녹화한 랜드마크로 자세 판정을 재생/채점한다")
    ap.add_argument("recording")
    ap.add_argument("--baseline", help="녹화 속 기준 대신 쓸 baseline.json")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help=f"임계값 덮어쓰기 ({', '.join(posture_logic.default_thresholds())})")
    ap.add_argument("--csv", help="프레임별 수치를 저장할 CSV 경로")
    args = ap.parse_args(argv)

    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        if key not in posture_logic.default_thresholds():
            ap.error(f"알 수 없는 임계값: {key}")
        overrides[key] = float(value)

    baseline = load_baseline_file(args.baseline) if args.baseline else None
    frames, events = replay(read_recording(args.recording), baseline, overrides)
    summarize(frames, events)
    if args.csv:
        write_csv(frames, args.csv)
        print(f"\nCSV 저장: {args.csv}")


if __name__ == "__main__":
    main()

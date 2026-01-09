import json
import os
import csv
import math

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT = os.path.join(ROOT, "output", "punish_report.json")
OUT_JSON = os.path.join(ROOT, "output", "segment_summary.json")
OUT_CSV = os.path.join(ROOT, "output", "segment_summary.csv")

SEGMENT_SECONDS = 60  # 1-minute segments


def main(segment_seconds=SEGMENT_SECONDS):
    if not os.path.exists(INPUT):
        print("No punish_report.json found at", INPUT)
        return

    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("rows", [])

    # ensure every row has start_time; if not, skip
    times = [r.get("start_time") for r in rows if r.get("start_time") is not None]
    if not times:
        print("No timestamped rows found")
        return

    max_time = max(times)
    n_segments = int(math.floor(max_time / segment_seconds)) + 1

    segments = []
    for s in range(n_segments):
        start_t = s * segment_seconds
        end_t = (s + 1) * segment_seconds
        seg_rows = [r for r in rows if r.get("start_time") is not None and start_t <= r.get("start_time") < end_t]
        total = len(seg_rows)
        whiffs = len([r for r in seg_rows if r.get("type") == "whiff_punish"])
        whiff_punishable = len([r for r in seg_rows if r.get("type") == "whiff_punish" and r.get("punishable")])
        pun_jumps = len([r for r in seg_rows if r.get("type") == "punishable_jump" and r.get("punishable")])
        segments.append({
            "segment_index": s,
            "start_time": start_t,
            "end_time": end_t,
            "total_opportunities": total,
            "whiffs": whiffs,
            "whiff_punishable": whiff_punishable,
            "punishable_jumps": pun_jumps,
        })

    # write JSON and CSV
    # escreve resumo por segmento em JSON
    with open(OUT_JSON, "w", encoding="utf-8") as jf:
        json.dump({"segment_seconds": segment_seconds, "segments": segments}, jf, indent=2)

    with open(OUT_CSV, "w", newline='', encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=["segment_index", "start_time", "end_time", "total_opportunities", "whiffs", "whiff_punishable", "punishable_jumps"])
        writer.writeheader()
        for seg in segments:
            writer.writerow(seg)

    print("Wrote segment summaries:", OUT_JSON, OUT_CSV)


if __name__ == '__main__':
    import sys
    seg = SEGMENT_SECONDS
    if len(sys.argv) > 1:
        try:
            seg = int(sys.argv[1])
        except Exception:
            pass
    main(seg)

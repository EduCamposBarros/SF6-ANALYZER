import json
import csv
import os
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "output", "results.json")
OUT_CSV = os.path.join(ROOT, "output", "punish_report.csv")
OUT_JSON = os.path.join(ROOT, "output", "punish_report.json")
VIDEO = os.path.join(ROOT, "Match.mp4")


def seconds_from_frame(frame_id, fps):
    return frame_id / fps if fps and fps > 0 else None


def main():
    """
    Exporta oportunidades detectadas em CSV e JSON.

    - Lê `output/results.json` (campo `frame_data.summary`) e extrai `whiff_punishes`
      e `punishable_jumps`.
    - Tenta converter frames para tempo usando o `fps` do arquivo `Match.mp4` quando disponível.
    - Gera `output/punish_report.csv` e `output/punish_report.json`.
    """

    if not os.path.exists(RESULTS):
        print("No results.json found at", RESULTS)
        return

    with open(RESULTS, "r", encoding="utf-8") as f:
        res = json.load(f)

    summary = res.get("frame_data", {}).get("summary", {})
    whiffs = summary.get("whiff_punishes", [])
    jumps = summary.get("punishable_jumps", [])

    # try to read video fps
    fps = None
    if os.path.exists(VIDEO):
        try:
            cap = cv2.VideoCapture(VIDEO)
            fps = cap.get(cv2.CAP_PROP_FPS) or None
            cap.release()
        except Exception:
            fps = None

    rows = []
    for w in whiffs:
        start = w.get("start")
        end = w.get("end")
        rows.append({
            "type": "whiff_punish",
            "player": w.get("attacker"),
            "start_frame": start,
            "end_frame": end,
            "punishable": bool(w.get("punishable")),
            "start_time": seconds_from_frame(start, fps),
            "end_time": seconds_from_frame(end, fps),
        })

    for j in jumps:
        rows.append({
            "type": "punishable_jump",
            "player": j.get("player"),
            "start_frame": j.get("start"),
            "end_frame": j.get("land"),
            "punishable": bool(j.get("punishable")),
            "start_time": seconds_from_frame(j.get("start"), fps),
            "end_time": seconds_from_frame(j.get("land"), fps),
        })

    # write CSV
    with open(OUT_CSV, "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = ["type", "player", "start_frame", "end_frame", "punishable", "start_time", "end_time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # write JSON
    with open(OUT_JSON, "w", encoding="utf-8") as jf:
        json.dump({"counts": {"whiffs": len(whiffs), "punishable_jumps": len([j for j in jumps if j.get('punishable')])}, "rows": rows}, jf, indent=2)

    print("Exported punish report:", OUT_CSV, OUT_JSON)


if __name__ == '__main__':
    main()

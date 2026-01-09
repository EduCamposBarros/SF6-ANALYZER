"""
Gera um vídeo de debug com overlays úteis para inspeção manual.

O vídeo contém:
- bounding boxes dos jogadores
- estado detectado (`p1:...`, `p2:...`)
- indicação de hitspark (círculo e label) quando detectado

Use este arquivo para validar visualmente falsos-positivos ou posicionamento
de bboxes e para ajustar parâmetros dos detectores.
"""

import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2
from vision.character_detection import detect_characters
from vision.effects_detection import detect_effects
from vision.state_detection import detect_state, StateDetectorConfig
from config import DAMAGE_PER_HIT
import json

# load results if present to overlay whiff/punish annotations
RESULTS_PATH = os.path.join("output", "results.json")
results_data = None
if os.path.exists(RESULTS_PATH):
    try:
        with open(RESULTS_PATH, "r", encoding="utf-8") as rf:
            results_data = json.load(rf)
    except Exception:
        results_data = None

VIDEO = "Match.mp4"
OUT = "output/debug_overlay.mp4"

# configurações padrão do detector de estado (podem ser sobrescritas pelo tuning)
cfg = StateDetectorConfig()

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit(f"Cannot open video {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT, fourcc, fps, (w, h))

ret, prev = cap.read()
if not ret:
    raise SystemExit("Video empty")

frame_idx = 0
print("Writing debug overlay to", OUT)
# vida inicial (0-100)
life_p1 = 100
life_p2 = 100
prev_p1_bbox = None
prev_p2_bbox = None
while True:
    ret, frame = cap.read()
    if not ret:
        break

    prev_bboxes = (prev_p1_bbox, prev_p2_bbox) if (prev_p1_bbox is not None and prev_p2_bbox is not None) else None
    bboxes = detect_characters(frame, prev, prev_bboxes)
    if not bboxes:
        writer.write(frame)
        frame_idx += 1
        prev = frame
        continue

    p1_bbox, p2_bbox = bboxes
    # detect effects between prev and current
    eff1 = detect_effects(frame, prev, p1_bbox, p2_bbox, mean_diff_thresh=30.0)
    # detect states (use previous bbox for motion comparisons)
    s1 = detect_state(frame, p1_bbox, prev, prev_p1_bbox, config=cfg)
    s2 = detect_state(frame, p2_bbox, prev, prev_p2_bbox, config=cfg)

    out = frame.copy()
    # draw bboxes
    def draw_bbox(img, box, color, label=None):
        # expect box as (x1,y1,x2,y2)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if label:
            # draw label inside the bbox at top-left corner
            cv2.rectangle(img, (x1, y1), (x1 + 120, y1 + 20), color, -1)
            cv2.putText(img, label, (x1 + 4, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    draw_bbox(out, p1_bbox, (0, 200, 0), f"p1:{s1}")
    draw_bbox(out, p2_bbox, (0, 0, 200), f"p2:{s2}")

    if eff1:
        # mark center with a small circle
        x1, y1, x2, y2 = map(int, p1_bbox)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(out, (cx, cy), 8, (0, 255, 255), -1)
        cv2.putText(out, f"hitspark", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # reduce life based on detected hits
        try:
            for e in eff1:
                t = e.get("target")
                if t == "p2":
                    life_p2 = max(0, life_p2 - DAMAGE_PER_HIT)
                elif t == "p1":
                    life_p1 = max(0, life_p1 - DAMAGE_PER_HIT)
        except Exception:
            pass

    # overlay whiff/punish markers if results.json is available
    if results_data:
        windows = results_data.get("frame_data", {}).get("windows", [])
        # total whiff count for legend
        total_whiffs = sum(1 for w in windows if w.get("whiff"))
        # find index of current whiff window (1-based)
        cur_whiff_index = None
        # find whiff windows covering this frame
        for idx, win in enumerate(windows):
            if win.get("whiff") and win.get("start") <= frame_idx <= win.get("end"):
                cur_whiff_index = sum(1 for w in windows[:idx+1] if w.get("whiff"))
                # draw a prominent red label
                cv2.putText(out, "WHIFF", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                if win.get("punishable"):
                    cv2.putText(out, "PUNISHABLE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)

                # highlight attacker bbox for visibility
                attacker = win.get("attacker")
                if attacker == "p1":
                    color = (0, 0, 255)
                    bbox = p1_bbox
                    target_bbox = p2_bbox
                else:
                    color = (0, 0, 255)
                    bbox = p2_bbox
                    target_bbox = p1_bbox

                try:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 4)
                    cv2.putText(out, f"WHIFF->{attacker}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
                except Exception:
                    pass

                # mark target center with a circle
                try:
                    tx1, ty1, tx2, ty2 = map(int, target_bbox)
                    tcx, tcy = (tx1 + tx2) // 2, (ty1 + ty2) // 2
                    cv2.circle(out, (tcx, tcy), 12, (255, 0, 0), -1)
                except Exception:
                    pass

                break

        # draw legend with total whiffs and current whiff index
        legend_text = f"Whiffs: {total_whiffs}"
        if cur_whiff_index is not None:
            legend_text += f" (current {cur_whiff_index})"
        # tracker mode: prefer CSRT if manager has active trackers
        try:
            from vision.tracker import get_manager

            mgr = get_manager()
            tracker_active = mgr.trackers.get("p1") is not None or mgr.trackers.get("p2") is not None
            mode = "CSRT" if tracker_active else "TEMPLATE"
        except Exception:
            mode = "TEMPLATE"

        legend_text += f"  |  Mode: {mode}"
        # draw background for legend
        cv2.rectangle(out, (8, 82), (8 + 380, 82 + 26), (30, 30, 30), -1)
        cv2.putText(out, legend_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # overlay frame index
    cv2.putText(out, f"frame:{frame_idx}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # draw life bars and percentages
    def draw_life_bar(img, life, left=True):
        # life: 0-100
        bar_w = int(w * 0.3)
        bar_h = 18
        margin = 10
        pct = max(0, min(100, life))
        fill_w = int(bar_w * (pct / 100.0))
        if left:
            x0 = margin
            y0 = margin
            x1 = x0 + bar_w
            y1 = y0 + bar_h
            cv2.rectangle(img, (x0, y0), (x1, y1), (50, 50, 50), -1)
            cv2.rectangle(img, (x0, y0), (x0 + fill_w, y1), (0, 200, 0), -1)
            cv2.putText(img, f"P1: {pct}%", (x0 + 4, y0 + bar_h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            x1 = w - margin
            y0 = margin
            x0 = x1 - bar_w
            y1 = y0 + bar_h
            cv2.rectangle(img, (x0, y0), (x1, y1), (50, 50, 50), -1)
            cv2.rectangle(img, (x1 - fill_w, y0), (x1, y1), (0, 0, 200), -1)
            cv2.putText(img, f"P2: {pct}%", (x0 + 4, y0 + bar_h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    draw_life_bar(out, life_p1, left=True)
    draw_life_bar(out, life_p2, left=False)

    writer.write(out)
    frame_idx += 1
    prev = frame
    prev_p1_bbox = p1_bbox
    prev_p2_bbox = p2_bbox

cap.release()
writer.release()
print("Debug video written:", OUT)

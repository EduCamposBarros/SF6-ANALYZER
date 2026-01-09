"""Valida a configuração ajustada executando uma checagem rápida sobre o vídeo.
Gera um resumo em `output/validate_report.json` contendo: total_frames, hits_count, coverage, attack_rate.
"""
import json
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vision.effects_detection import detect_effects
from vision.character_detection import detect_characters
from vision.tuned_state_config import get_default_config
from vision.state_detection import detect_state
import cv2

VIDEO = "Match.mp4"
MAX_FRAMES = 600

cfg = get_default_config()

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit(f"Cannot open video {VIDEO}")

frames = []
for i in range(MAX_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

bboxes = []
prev_b = None
prev_fr = None
for i, f in enumerate(frames):
    pb1, pb2 = detect_characters(f, prev_fr, prev_b)
    bboxes.append((pb1, pb2))
    prev_fr = f
    prev_b = (pb1, pb2)

hits = []
attack_frames = set()

for i, frame in enumerate(frames):
    p1_bbox, p2_bbox = bboxes[i]
    prev_frame = frames[i-1] if i>0 else None
    prev_p1 = bboxes[i-1][0] if i>0 else None
    prev_p2 = bboxes[i-1][1] if i>0 else None

    effs = detect_effects(frame, prev_frame, p1_bbox, p2_bbox, mean_diff_thresh=40.0, blur_ksize=5, morph_kernel=5, binary_thresh=30)
    if effs:
        for e in effs:
            hits.append(i)

    s1 = detect_state(frame, p1_bbox, prev_frame, prev_p1, config=cfg)
    s2 = detect_state(frame, p2_bbox, prev_frame, prev_p2, config=cfg)
    if s1 == "attack_active":
        attack_frames.add(i)
    if s2 == "attack_active":
        attack_frames.add(i)

if not hits:
    coverage = 0.0
else:
    covered = 0
    for h in hits:
        window = range(max(0, h-6), h+1)
        if any(f in attack_frames for f in window):
            covered += 1
    coverage = covered / len(hits)

attack_rate = len(attack_frames) / len(frames) if frames else 0

report = {
    "total_frames": len(frames),
    "hits_count": len(hits),
    "coverage": coverage,
    "attack_rate": attack_rate,
}

os.makedirs("output", exist_ok=True)
with open("output/validate_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))

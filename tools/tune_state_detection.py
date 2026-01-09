import json
import itertools
import os
import sys

# Ensure project root is on sys.path when running the script directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vision.state_detection import StateDetectorConfig, detect_state
from vision.effects_detection import detect_effects
from vision.character_detection import detect_characters
import cv2

VIDEO = "Match.mp4"
# Use a smaller sample for faster iteration; increase to 500+ for full runs
MAX_FRAMES = 300

# Parameter grid (state detector)
# Expanded grid to improve coverage — moderate size to keep runtime reasonable
area_thresh_vals = [15000, 18000, 21000]
area_fallback_vals = [2500, 3000]
mean_color_vals = [30, 35, 40]
jump_cy_delta_vals = [0.04, 0.06]
drive_cx_factor_vals = [0.08, 0.12]
motion_thresh_vals = [1.5, 3.0, 5.0]

results = []

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit(f"Cannot open video {VIDEO}")

# read all frames up to MAX_FRAMES
frames = []
for i in range(MAX_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

print(f"Loaded {len(frames)} frames for tuning")

# precompute bboxes per frame using detect_characters with simple tracking
bboxes = []
prev_b = None
prev_fr = None
for i, f in enumerate(frames):
    pb1, pb2 = detect_characters(f, prev_fr, prev_b)
    # convert (x,y,w,h) -> (x1,y1,x2,y2) for compatibility with detect_state
    def to_xyxy(b):
        x, y, w, h = map(int, b)
        return (x, y, x + w, y + h)

    bboxes.append((to_xyxy(pb1), to_xyxy(pb2)))
    prev_fr = f
    prev_b = (pb1, pb2)

# effects detector preprocessing grid — mais restritivo para reduzir falsos positivos
# aumentamos os thresholds e os tamanhos de kernel para limpeza mais forte
mean_diff_thresh_vals = [40.0, 60.0]
blur_k_sizes = [5, 7]
morph_k_sizes = [5, 7]
binary_thresholds = [50, 60]

# Sweep parameter grid
try:
    for area_t, area_fb, mean_c, jump_delta, drive_factor in itertools.product(
        area_thresh_vals, area_fallback_vals, mean_color_vals, jump_cy_delta_vals, drive_cx_factor_vals
    ):
        for motion_thresh in motion_thresh_vals:
            cfg = StateDetectorConfig(
                area_attack_threshold=area_t,
                area_attack_fallback=area_fb,
                mean_color_block_threshold=mean_c,
                jump_cy_delta=jump_delta,
                drive_cx_delta_factor=drive_factor,
                motion_thresh=motion_thresh,
            )

            # First, sweep effects detector preprocessing to find a reasonable baseline of hits
            effects_candidates = []
            for mean_diff_thresh, blur_k, morph_k, bin_th in itertools.product(
                mean_diff_thresh_vals, blur_k_sizes, morph_k_sizes, binary_thresholds
            ):
                hits_local = []
                for i in range(1, len(frames)):
                    eff = detect_effects(
                        frames[i], frames[i - 1], bboxes[i][0], bboxes[i][1],
                        mean_diff_thresh=mean_diff_thresh, blur_ksize=blur_k, morph_kernel=morph_k, binary_thresh=bin_th
                    )
                    if eff:
                        hits_local.append(i)

                effects_candidates.append(
                    {"mean_diff_thresh": mean_diff_thresh, "blur": blur_k, "morph": morph_k, "bin": bin_th, "hits_count": len(hits_local), "hits": hits_local}
                )

            # choose candidate that yields a hits_count in a reasonable range
            MIN_HITS = max(5, int(0.01 * len(frames)))
            MAX_HITS = max(10, int(0.5 * len(frames)))
            best_effect = None
            best_e_score = None
            for c in effects_candidates:
                hc = c["hits_count"]
                if hc < MIN_HITS or hc > MAX_HITS:
                    continue
                escore = abs(hc - (MIN_HITS + MAX_HITS) / 2)
                if best_e_score is None or escore < best_e_score:
                    best_e_score = escore
                    best_effect = c

            if best_effect is None:
                effects_candidates.sort(key=lambda x: abs(x["hits_count"] - (len(frames) * 0.12)))
                best_effect = effects_candidates[0]

            print(f"Chosen effects config: mean_diff_thresh={best_effect['mean_diff_thresh']}, blur={best_effect['blur']}, morph={best_effect['morph']}, bin={best_effect['bin']}, hits_count={best_effect['hits_count']}")

            # Use chosen hits as baseline
            hits = best_effect["hits"]

            attack_frames = set()
            # run lightweight state detection over sample frames with current cfg
            for i, frame in enumerate(frames):
                p1_bbox, p2_bbox = bboxes[i]
                prev_frame = frames[i - 1] if i > 0 else None
                prev_p1_bbox, prev_p2_bbox = bboxes[i - 1] if i > 0 else (None, None)
                s1 = detect_state(frame, p1_bbox, prev_frame, prev_p1_bbox, config=cfg)
                s2 = detect_state(frame, p2_bbox, prev_frame, prev_p2_bbox, config=cfg)
                if s1 == "attack_active":
                    attack_frames.add(i)
                if s2 == "attack_active":
                    attack_frames.add(i)

            if not hits:
                # still record a result with zero coverage so user can inspect
                results.append({
                    "config": {"area_t": area_t, "area_fb": area_fb, "mean_c": mean_c, "jump_delta": jump_delta, "drive_factor": drive_factor, "motion_thresh": motion_thresh},
                    "coverage": 0.0,
                    "attack_rate": len(attack_frames) / len(frames) if frames else 0.0,
                    "score": - (len(attack_frames) / len(frames) if frames else 0.0),
                })
                continue

            # measure coverage: fraction of hit frames that have attack_active within window
            covered = 0
            coverage_window = 8
            for h in hits:
                window = range(max(0, h - coverage_window), h + 1)
                if any(f in attack_frames for f in window):
                    covered += 1

            coverage = covered / len(hits) if hits else 0.0
            attack_rate = len(attack_frames) / len(frames) if frames else 0.0

            # compute a score balancing coverage vs attack rate (penalize high attack_rate)
            score = coverage - 0.6 * attack_rate

            results.append(
                {
                    "config": {
                        "area_t": area_t,
                        "area_fb": area_fb,
                        "mean_c": mean_c,
                        "jump_delta": jump_delta,
                        "drive_factor": drive_factor,
                        "motion_thresh": motion_thresh,
                    },
                    "coverage": coverage,
                    "attack_rate": attack_rate,
                    "score": score,
                }
            )
except KeyboardInterrupt:
    print("Tuning interrupted by user — writing partial report")

# sort by score desc then attack_rate asc
results_sorted = sorted(results, key=lambda r: (-r.get("score", 0.0), r.get("attack_rate", 0.0)))
report = {"total_frames": len(frames), "hits_detected": len(hits) if 'hits' in locals() else 0, "results": results_sorted[:20]}

with open("output/tuning_report.json", "w") as f:
    json.dump(report, f, indent=2)
print("Tuning complete — report saved to output/tuning_report.json")
# also print top 5 results for quick inspection
for r in results_sorted[:5]:
    cfg = r["config"]
    print(f"cfg={cfg} coverage={r.get('coverage',0.0):.3f} attack_rate={r.get('attack_rate',0.0):.3f} score={r.get('score',0.0):.4f}")

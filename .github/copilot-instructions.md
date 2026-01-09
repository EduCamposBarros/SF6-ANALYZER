# Copilot / AI Agent Instructions for SF6 analyzer

This repository analyzes gameplay video frames to extract frame-level state and events. Below are focused, actionable guidelines to help AI coding agents be immediately productive.

## Big-picture architecture
- **Video input**: code under `video/` (eg. `video/extract_frames.py`) is responsible for turning source videos into frame images or arrays.
- **Vision**: `vision/` contains detectors (`character_detection.py`, `state_detection.py`, `effects_detection.py`) that should accept a frame and return detections (bounding boxes, labels, confidence). Implementations are currently placeholders.
- **Model objects**: canonical data types live in `models/structures.py` — use `FrameData` and `Event` dataclasses as the project's standard interchange format.
- **Analysis pipeline**: `analysis/` modules assemble detections into timeline data (`frame_data.py`), detect events (`events.py`), and summarize into insights (`insights.py`). Output is written to `output/results.json`.

## Data flow / conventions (use exact names)
- Use `FrameData` from [models/structures.py](models/structures.py#L1) as the per-frame record: fields include `frame_id`, `timestamp`, `p1_state`, `p2_state`, `p1_bbox`, `p2_bbox`, `life_p1`, `life_p2`.
- Event objects use the `Event` dataclass with `type`, `frame_id`, `attacker`, `defender`.
- Bounding boxes are tuples (x, y, w, h) across the codebase; stick to this format when producing or consuming detections.
- Frame identifiers are integer `frame_id` and timestamps are floats in seconds.

## File responsibilities (quick reference)
- `video/extract_frames.py`: frame extraction and naming/ordering. Producers should return ordered frames and `frame_id` metadata.
- `vision/*`: detectors should expose a simple function `detect(frame) -> dict` or `detect_batch(frames) -> list[dict]` returning bboxes and labels.
- `analysis/frame_data.py`: assemble `FrameData` instances from vision outputs.
- `analysis/events.py`: convert sequences of `FrameData` into `Event` instances.
- `analysis/insights.py`: aggregate events into higher-level summaries and persist to `output/results.json`.

## Project-specific patterns & expectations
- The repo currently contains many empty modules; do not assume implementations exist — check each file before editing.
- Keep APIs minimal and explicit: prefer functions that accept a frame or frame list and return plain Python datatypes (dict, list, dataclass instances). Example: `def detect(frame) -> dict: return {"bboxes": [...], "labels": [...], "conf": [...]}`.
- When creating new public helpers, add them to the relevant module (eg. vision helpers under `vision/`) rather than a new top-level file.

## Running / debugging notes (developer must confirm)
- There is no runnable `main.py` nor recorded `requirements.txt`. Before running, ask the maintainer for the intended entrypoint and required packages.
- Recommended workflow to validate changes locally (confirm with owner):
```powershell
# extract frames
python -m video.extract_frames --input path/to/video --out frames/
# run detectors on frames
python -m vision.state_detection --frames frames/ --out detections.json
# assemble analysis
python -m analysis.frame_data --detections detections.json --out frames_data.json
python -m analysis.events --frames frames_data.json --out events.json
python -m analysis.insights --events events.json --out output/results.json
```

## Integration points & checks before edits
- Always import and use `FrameData` when passing per-frame state between modules.
- Before modifying files under `analysis/` or `vision/`, run a quick grep for `FrameData` usage to ensure compatibility.
- If adding external dependencies, update `requirements.txt` and confirm with the maintainer — the repo currently contains no declared dependencies.

## Examples (copyable)
- Construct a frame record:
```python
from models.structures import FrameData
fd = FrameData(frame_id=123, timestamp=4.1, p1_state='neutral', p2_state='hit', p1_can_act=True,
               p2_can_act=False, p1_bbox=(10,20,50,80), p2_bbox=(200,20,50,80), life_p1=950, life_p2=900)
```
- Emit an event:
```python
from models.structures import Event
e = Event(type='hit', frame_id=123, attacker='p1', defender='p2')
```

## When unsure, ask the maintainer
- Confirm runtime commands, expected input video formats, and preferred detector interfaces before implementing heavy changes.

---
If any section is unclear or you want more detail (environment, intended CLI, or expected detector outputs), tell me which part to expand and I will iterate.

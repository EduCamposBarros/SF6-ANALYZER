import cv2
import numpy as np
from typing import Optional, Tuple
from .tracker import get_manager


TEMPLATE_MATCH_THRESHOLD = 0.42


def _clamp(v, a, b):
    return max(a, min(b, v))


def detect_characters(frame: np.ndarray, prev_frame: Optional[np.ndarray] = None, prev_bboxes: Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]] = None):
    """
    Detecta ou rastreia os personagens no `frame`.

    - Se `prev_frame` e `prev_bboxes` forem fornecidos, tenta rastrear cada bbox por
      template-matching local (procura a melhor correspondência próxima ao bbox anterior).
    - Caso contrário, retorna bboxes aproximadas fixas baseado na resolução.

    Retorna `(p1_bbox, p2_bbox)` como tuplas (x, y, w, h).
    """

    h, w = frame.shape[:2]

    # Default fixed boxes (fallback) in (x,y,w,h)
    default_p1_xywh = (int(w * 0.1), int(h * 0.5), int(w * 0.15), int(h * 0.4))
    default_p2_xywh = (int(w * 0.75), int(h * 0.5), int(w * 0.15), int(h * 0.4))

    # helpers to convert between formats
    def xywh_to_xyxy(b):
        x, y, ww, hh = map(int, b)
        return (x, y, x + ww, y + hh)

    def xyxy_to_xywh(b):
        x1, y1, x2, y2 = map(int, b)
        return (x1, y1, x2 - x1, y2 - y1)

    mgr = get_manager()

    # If no history provided, initialize tracker and return defaults (as xyxy)
    if prev_frame is None or prev_bboxes is None:
        mgr.initialize(frame, xywh_to_xyxy(default_p1_xywh), xywh_to_xyxy(default_p2_xywh))
        return xywh_to_xyxy(default_p1_xywh), xywh_to_xyxy(default_p2_xywh)

    # Try tracker update first but only accept it if trackers are actually active.
    # `mgr.update()` may return `last_bboxes` even when trackers are None; in that
    # case we must fall back to template-matching/detection to avoid stuck boxes.
    tb1, tb2 = mgr.update(frame)
    if (mgr.trackers.get("p1") is not None or mgr.trackers.get("p2") is not None) and tb1 is not None and tb2 is not None:
        return tb1, tb2

    out_bboxes = []
    # For each previous bbox, try to find it in the new frame using template matching
    for prev_bbox in prev_bboxes:
        try:
            # prev_bbox is expected in (x,y,w,h) from caller; convert to xyxy
            if len(prev_bbox) == 4:
                # support callers passing either format
                if prev_bbox[2] > prev_bbox[0] and prev_bbox[3] > prev_bbox[1]:
                    # likely xyxy
                    px1, py1, px2, py2 = map(int, prev_bbox)
                    pw = px2 - px1
                    ph = py2 - py1
                    px, py = px1, py1
                else:
                    px, py, pw, ph = map(int, prev_bbox)
            else:
                px, py, pw, ph = map(int, prev_bbox)
            # extract template from previous frame, clamp coords
            px = _clamp(px, 0, prev_frame.shape[1] - 1)
            py = _clamp(py, 0, prev_frame.shape[0] - 1)
            pw = max(4, pw)
            ph = max(4, ph)
            tpl = prev_frame[py:py + ph, px:px + pw]
            if tpl.size == 0:
                out_bboxes.append(prev_bbox)
                continue

            # search region in current frame: expand previous bbox by 1.5x
            cx = int(px + pw / 2)
            cy = int(py + ph / 2)
            sw = int(pw * 1.5)
            sh = int(ph * 1.5)
            sx = _clamp(cx - sw // 2, 0, w - 1)
            sy = _clamp(cy - sh // 2, 0, h - 1)
            sx2 = _clamp(sx + sw, 0, w)
            sy2 = _clamp(sy + sh, 0, h)

            search = frame[sy:sy2, sx:sx2]
            if search.size == 0:
                out_bboxes.append(prev_bbox)
                continue

            tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
            search_gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)

            # matchTemplate requires template smaller than search region
            if tpl_gray.shape[0] > search_gray.shape[0] or tpl_gray.shape[1] > search_gray.shape[1]:
                out_bboxes.append(prev_bbox)
                continue

            res = cv2.matchTemplate(search_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # threshold for accepting match; if low confidence, keep previous bbox
            if max_val < TEMPLATE_MATCH_THRESHOLD:
                out_bboxes.append(prev_bbox)
                continue

            top_left = (sx + max_loc[0], sy + max_loc[1])
            # return bbox as xyxy
            new_bbox = (top_left[0], top_left[1], top_left[0] + pw, top_left[1] + ph)
            out_bboxes.append(new_bbox)
        except Exception:
            out_bboxes.append(prev_bbox)

    # if we obtained matches, re-init trackers with the new bboxes (ensure xyxy)
    if len(out_bboxes) >= 2:
        mgr.initialize(frame, out_bboxes[0], out_bboxes[1])
        return out_bboxes[0], out_bboxes[1]
    elif len(out_bboxes) == 1:
        mgr.initialize(frame, out_bboxes[0], xywh_to_xyxy(default_p2_xywh))
        return out_bboxes[0], xywh_to_xyxy(default_p2_xywh)
    else:
        # nothing found: fall back to last known from manager or defaults
        last1, last2 = mgr.last_bboxes.get("p1"), mgr.last_bboxes.get("p2")
        return (last1 if last1 is not None else xywh_to_xyxy(default_p1_xywh), last2 if last2 is not None else xywh_to_xyxy(default_p2_xywh))

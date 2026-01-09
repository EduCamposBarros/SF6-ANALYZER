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

    # Default fixed boxes (fallback)
    default_p1 = (int(w * 0.1), int(h * 0.5), int(w * 0.15), int(h * 0.4))
    default_p2 = (int(w * 0.75), int(h * 0.5), int(w * 0.15), int(h * 0.4))

    mgr = get_manager()

    # If no history provided, return default and initialize tracker
    if prev_frame is None or prev_bboxes is None:
        mgr.initialize(frame, default_p1, default_p2)
        return default_p1, default_p2

    # Try tracker update first
    tb1, tb2 = mgr.update(frame)
    if tb1 is not None and tb2 is not None:
        return tb1, tb2

    out_bboxes = []
    # For each previous bbox, try to find it in the new frame using template matching
    for prev_bbox in prev_bboxes:
        try:
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
            new_bbox = (top_left[0], top_left[1], pw, ph)
            out_bboxes.append(new_bbox)
        except Exception:
            out_bboxes.append(prev_bbox)

    # if we obtained matches, re-init trackers with the new bboxes
    if len(out_bboxes) >= 2:
        mgr.initialize(frame, out_bboxes[0], out_bboxes[1])
        return out_bboxes[0], out_bboxes[1]
    elif len(out_bboxes) == 1:
        mgr.initialize(frame, out_bboxes[0], default_p2)
        return out_bboxes[0], default_p2
    else:
        # nothing found: fall back to last known from manager or defaults
        last1, last2 = mgr.last_bboxes.get("p1"), mgr.last_bboxes.get("p2")
        return (last1 if last1 is not None else default_p1, last2 if last2 is not None else default_p2)

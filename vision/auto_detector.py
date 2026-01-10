import cv2
import numpy as np
from typing import Optional, Tuple, List

from .tracker import get_manager


class AutoDetector:
    """Detector simples baseado em background subtraction que inicializa trackers.

    - usa MOG2 para detectar regiões em movimento
    - escolhe as duas maiores regiões como jogadores
    - inicializa `vision.tracker.TrackerManager` automaticamente
    - usa trackers para retornar bboxes confiáveis a cada frame
    """

    def __init__(self, min_area: int = 800, reinit_interval: int = 30):
        self.backsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.min_area = min_area
        self.reinit_interval = reinit_interval
        self.frame_count = 0
        self.mgr = get_manager()

    def _detect_moving(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        # apply background subtractor
        fg = self.backsub.apply(frame)
        # cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

        # threshold and find contours
        _, thresh = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        h, w = frame.shape[:2]
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            # clamp
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + ww)
            y2 = min(h, y + hh)
            boxes.append((x1, y1, x2, y2))

        # return boxes sorted by area desc
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        return boxes

    def process(self, frame: np.ndarray) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """Processa um frame e retorna `(p1_bbox, p2_bbox)` em formato xyxy.

        Garante que, sempre que possível, os `trackers` serão usados para produzir
        bboxes em todos os frames. Re-detecta movimentos periodicamente para
        corrigir drift.
        """
        self.frame_count += 1

        # if trackers exist, prefer tracker update
        tb1, tb2 = self.mgr.update(frame)
        # If trackers are alive, return their boxes (mgr.update falls back to last known)
        if self.mgr.trackers.get("p1") is not None or self.mgr.trackers.get("p2") is not None:
            return tb1, tb2

        # Otherwise attempt detection
        boxes = self._detect_moving(frame)
        # If we found at least two moving regions, take top two
        if len(boxes) >= 2:
            b1, b2 = boxes[0], boxes[1]
            # initialize trackers with detected boxes
            try:
                self.mgr.initialize(frame, b1, b2)
            except Exception:
                pass
            return b1, b2

        # If only one found, pair with a heuristic opposite box
        h, w = frame.shape[:2]
        if len(boxes) == 1:
            b1 = boxes[0]
            # heuristic for opposite side: mirror horizontally
            x1, y1, x2, y2 = b1
            bw = x2 - x1
            hb = y2 - y1
            # try to place second bbox on the opposite side center
            nb_x1 = max(0, w - x2 - bw)
            nb_x2 = min(w, nb_x1 + bw)
            nb_y1 = y1
            nb_y2 = min(h, y1 + hb)
            b2 = (nb_x1, nb_y1, nb_x2, nb_y2)
            try:
                self.mgr.initialize(frame, b1, b2)
            except Exception:
                pass
            return b1, b2

        # last resort: use manager last known bboxes or center defaults
        last1 = self.mgr.last_bboxes.get("p1")
        last2 = self.mgr.last_bboxes.get("p2")
        if last1 is not None and last2 is not None:
            return last1, last2

        # fall back to fixed positions (xyxy)
        default_p1 = (int(w * 0.1), int(h * 0.5), int(w * 0.25), int(h * 0.9))
        default_p2 = (int(w * 0.75), int(h * 0.5), int(w * 0.9), int(h * 0.9))
        # initialize trackers with defaults
        try:
            self.mgr.initialize(frame, default_p1, default_p2)
        except Exception:
            pass
        return default_p1, default_p2


def detect_characters_auto(frame, prev_frame=None, prev_bboxes=None):
    """Compatibility wrapper similar to `vision.character_detection.detect_characters`.

    Returns `(p1_bbox, p2_bbox)` in xyxy format.
    """
    # simple singleton detector
    if not hasattr(detect_characters_auto, "_inst"):
        detect_characters_auto._inst = AutoDetector()
    return detect_characters_auto._inst.process(frame)

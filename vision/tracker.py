"""Gerencia trackers por jogador usando OpenCV CSRT quando disponível.

Fornece inicialização e atualização simples para obter bboxes estáveis entre frames.
Se o CSRT não estiver disponível ou o tracker falhar, o manager retorna None
para aquela bbox e o chamador pode aplicar fallback (ex.: template-matching).
"""
from typing import Tuple, Optional
import cv2


class TrackerManager:
    def __init__(self):
        # trackers por jogador: 'p1', 'p2'
        self.trackers = {"p1": None, "p2": None}
        self.last_bboxes = {"p1": None, "p2": None}
        # failure counters to track consecutive update failures
        self._fail_counts = {"p1": 0, "p2": 0}
        self._max_fail = 6
        # store last frame for template-based recovery
        self._last_frame = None

        # criar factory de tracker com fallback
        try:
            self._create = cv2.TrackerCSRT_create
            # note: algumas builds usam cv2.legacy.TrackerCSRT_create
        except Exception:
            try:
                self._create = cv2.legacy.TrackerCSRT_create  # type: ignore[attr-defined]
            except Exception:
                self._create = None

    def initialize(self, frame, p1_bbox, p2_bbox):
        """Inicializa trackers para ambos os jogadores com as bboxes (x,y,w,h)."""
        # Expect incoming bboxes in (x1,y1,x2,y2) format; convert to (x,y,w,h) for tracker
        def to_xywh(b):
            if b is None:
                return None
            x1, y1, x2, y2 = map(int, b)
            return (x1, y1, max(4, x2 - x1), max(4, y2 - y1))

        if self._create is None:
            # tracker não disponível: store last_bboxes as xyxy
            self.trackers = {"p1": None, "p2": None}
            self.last_bboxes = {"p1": p1_bbox, "p2": p2_bbox}
            return

        try:
            t1 = self._create()
            t2 = self._create()
            p1_xywh = to_xywh(p1_bbox)
            p2_xywh = to_xywh(p2_bbox)
            if p1_xywh:
                t1.init(frame, tuple(map(int, p1_xywh)))
            if p2_xywh:
                t2.init(frame, tuple(map(int, p2_xywh)))
            self.trackers["p1"] = t1
            self.trackers["p2"] = t2
            # store last_bboxes in xyxy format
            self.last_bboxes["p1"] = p1_bbox
            self.last_bboxes["p2"] = p2_bbox
            # store last frame for future template matching
            try:
                self._last_frame = frame.copy()
            except Exception:
                self._last_frame = None
        except Exception:
            # se init falhar, deixa trackers em None e salva bboxes
            self.trackers = {"p1": None, "p2": None}
            self.last_bboxes = {"p1": p1_bbox, "p2": p2_bbox}

    def update(self, frame) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
        """Atualiza ambos os trackers; retorna bboxes (x,y,w,h) ou None para cada jogador."""
        out1 = None
        out2 = None
        try:
            t1 = self.trackers.get("p1")
            t2 = self.trackers.get("p2")
            if t1 is not None:
                ok, box = t1.update(frame)
                if ok:
                    # box is (x,y,w,h) from tracker -> convert to xyxy
                    bx, by, bw, bh = map(int, box)
                    out1 = (bx, by, bx + bw, by + bh)
                    self.last_bboxes["p1"] = out1
                    self._fail_counts["p1"] = 0
                else:
                    self._fail_counts["p1"] += 1
                    if self._fail_counts["p1"] >= self._max_fail:
                        # mark tracker as dead
                        self.trackers["p1"] = None
                    else:
                        # quick recovery attempt using template-match from last_frame
                        recovered = self._attempt_recover("p1", frame)
                        if recovered is not None:
                            out1 = recovered
                            self.last_bboxes["p1"] = out1
                            self._fail_counts["p1"] = 0
                            try:
                                # re-init tracker with recovered bbox
                                if self._create is not None:
                                    tnew = self._create()
                                    x1, y1, x2, y2 = map(int, out1)
                                    tnew.init(frame, (x1, y1, x2 - x1, y2 - y1))
                                    self.trackers["p1"] = tnew
                            except Exception:
                                pass
            if t2 is not None:
                ok2, box2 = t2.update(frame)
                if ok2:
                    bx2, by2, bw2, bh2 = map(int, box2)
                    out2 = (bx2, by2, bx2 + bw2, by2 + bh2)
                    self.last_bboxes["p2"] = out2
                    self._fail_counts["p2"] = 0
                else:
                    self._fail_counts["p2"] += 1
                    if self._fail_counts["p2"] >= self._max_fail:
                        self.trackers["p2"] = None
                    else:
                        recovered2 = self._attempt_recover("p2", frame)
                        if recovered2 is not None:
                            out2 = recovered2
                            self.last_bboxes["p2"] = out2
                            self._fail_counts["p2"] = 0
                            try:
                                if self._create is not None:
                                    tnew2 = self._create()
                                    x1, y1, x2, y2 = map(int, out2)
                                    tnew2.init(frame, (x1, y1, x2 - x1, y2 - y1))
                                    self.trackers["p2"] = tnew2
                            except Exception:
                                pass
        except Exception:
            return self.last_bboxes.get("p1"), self.last_bboxes.get("p2")

        # detect and correct large sudden jumps (smooth positions)
        def smooth(prev_box, new_box):
            if prev_box is None or new_box is None:
                return new_box
            px1, py1, px2, py2 = map(int, prev_box)
            nx1, ny1, nx2, ny2 = map(int, new_box)
            pcx = (px1 + px2) / 2.0
            pcy = (py1 + py2) / 2.0
            ncx = (nx1 + nx2) / 2.0
            ncy = (ny1 + ny2) / 2.0
            # distance in pixels
            dist = ((ncx - pcx) ** 2 + (ncy - pcy) ** 2) ** 0.5
            size = max(1.0, (px2 - px1 + py2 - py1) / 2.0)
            # threshold scales with size
            max_step = max(20.0, size * 0.6)
            if dist > max_step:
                # clamp movement vector
                ratio = max_step / dist
                cx = pcx + (ncx - pcx) * ratio
                cy = pcy + (ncy - pcy) * ratio
                # keep dimensions of new_box
                w = nx2 - nx1
                h = ny2 - ny1
                nx1c = int(cx - w / 2)
                ny1c = int(cy - h / 2)
                nx2c = nx1c + w
                ny2c = ny1c + h
                return (nx1c, ny1c, nx2c, ny2c)
            return new_box

        # fallback to last known if tracker failed for one side (last_bboxes stored as xyxy)
        if out1 is None:
            out1 = self.last_bboxes.get("p1")
        if out2 is None:
            out2 = self.last_bboxes.get("p2")

        # apply smoothing to prevent tracker "jumps"
        out1 = smooth(self.last_bboxes.get("p1"), out1)
        out2 = smooth(self.last_bboxes.get("p2"), out2)

        # update last_frame for next iteration
        try:
            self._last_frame = frame.copy()
        except Exception:
            self._last_frame = None

        return out1, out2

    def _attempt_recover(self, side: str, frame):
        """Attempt to find the last bbox in the current frame via template matching.

        Uses the stored `_last_frame` and `last_bboxes[side]` as template.
        Returns a xyxy bbox if successful, otherwise None.
        """
        try:
            prev = self._last_frame
            if prev is None:
                return None
            last_bbox = self.last_bboxes.get(side)
            if last_bbox is None:
                return None
            x1, y1, x2, y2 = map(int, last_bbox)
            tw = max(4, x2 - x1)
            th = max(4, y2 - y1)
            tpl = prev[y1:y1 + th, x1:x1 + tw]
            if tpl.size == 0:
                return None

            h, w = frame.shape[:2]
            cx = int(x1 + tw / 2)
            cy = int(y1 + th / 2)
            sw = int(tw * 2.5)  # larger search area to handle fast motion
            sh = int(th * 2.5)
            sx = max(0, cx - sw // 2)
            sy = max(0, cy - sh // 2)
            sx2 = min(w, sx + sw)
            sy2 = min(h, sy + sh)
            search = frame[sy:sy2, sx:sx2]
            if search.size == 0:
                return None

            tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
            search_gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
            if tpl_gray.shape[0] > search_gray.shape[0] or tpl_gray.shape[1] > search_gray.shape[1]:
                return None

            res = cv2.matchTemplate(search_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            # require decent confidence
            if max_val < 0.35:
                return None

            top_left = (sx + max_loc[0], sy + max_loc[1])
            new_bbox = (top_left[0], top_left[1], top_left[0] + tw, top_left[1] + th)
            return new_bbox
        except Exception:
            return None


# singleton para uso simples pelo pipeline
_MANAGER: Optional[TrackerManager] = None


def get_manager() -> TrackerManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = TrackerManager()
    return _MANAGER

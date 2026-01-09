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
        if self._create is None:
            # tracker não disponível
            self.trackers = {"p1": None, "p2": None}
            self.last_bboxes = {"p1": p1_bbox, "p2": p2_bbox}
            return

        try:
            t1 = self._create()
            t2 = self._create()
            t1.init(frame, tuple(map(int, p1_bbox)))
            t2.init(frame, tuple(map(int, p2_bbox)))
            self.trackers["p1"] = t1
            self.trackers["p2"] = t2
            self.last_bboxes["p1"] = p1_bbox
            self.last_bboxes["p2"] = p2_bbox
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
                    out1 = tuple(map(int, box))
                    self.last_bboxes["p1"] = out1
                    self._fail_counts["p1"] = 0
                else:
                    self._fail_counts["p1"] += 1
                    if self._fail_counts["p1"] >= self._max_fail:
                        # mark tracker as dead
                        self.trackers["p1"] = None
            if t2 is not None:
                ok2, box2 = t2.update(frame)
                if ok2:
                    out2 = tuple(map(int, box2))
                    self.last_bboxes["p2"] = out2
                    self._fail_counts["p2"] = 0
                else:
                    self._fail_counts["p2"] += 1
                    if self._fail_counts["p2"] >= self._max_fail:
                        self.trackers["p2"] = None
        except Exception:
            return self.last_bboxes.get("p1"), self.last_bboxes.get("p2")

        # fallback to last known if tracker failed for one side
        if out1 is None:
            out1 = self.last_bboxes.get("p1")
        if out2 is None:
            out2 = self.last_bboxes.get("p2")

        return out1, out2


# singleton para uso simples pelo pipeline
_MANAGER: Optional[TrackerManager] = None


def get_manager() -> TrackerManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = TrackerManager()
    return _MANAGER

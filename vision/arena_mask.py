"""Utilities to produce an arena mask that ignores background characters and UI.

Provides `ArenaMask`, a small helper around OpenCV MOG2 + heuristics to build
binary masks that keep moving fighters and remove static background characters
and HUD regions (top/bottom). Designed to be lightweight and easily integrated
into existing detectors (`vision.auto_detector`, `vision.character_detection`).
"""
from typing import Optional, Tuple
import cv2
import numpy as np


class ArenaMask:
    """Creates and updates an arena mask.

    Usage:
      am = ArenaMask()
      mask = am.update(frame)            # updates internal background model and returns mask
      masked = am.apply(frame)           # returns frame * mask (3-channel)

    Heuristics applied:
    - background subtraction (MOG2) to detect moving regions
    - morphological cleanup
    - remove top/bottom UI bands (configurable)
    - optional minimum area filtering to ignore small speckles
    """

    def __init__(
        self,
        history: int = 300,
        var_threshold: int = 16,
        detect_shadows: bool = True,
        min_area: int = 600,
        top_crop: float = 0.12,
        bottom_crop: float = 0.08,
    ):
        self.backsub = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=detect_shadows
        )
        self.min_area = min_area
        self.top_crop = top_crop
        self.bottom_crop = bottom_crop

    def _postprocess(self, fgmask: np.ndarray) -> np.ndarray:
        # remove shadows (if any) and threshold
        _, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        # morphological open + close
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

        # remove small contours
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(th)
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            cv2.drawContours(mask, [c], -1, 255, -1)

        return mask

    def update(self, frame: np.ndarray) -> np.ndarray:
        """Update background model with `frame` and return a binary mask (uint8 0/255).

        The mask has top/bottom UI bands zeroed-out to avoid HUD influence.
        """
        if frame is None:
            raise ValueError("frame is required")

        fg = self.backsub.apply(frame)
        mask = self._postprocess(fg)

        # zero top/bottom HUD regions
        h, w = mask.shape[:2]
        top_h = int(h * self.top_crop)
        bot_h = int(h * self.bottom_crop)
        if top_h > 0:
            mask[0:top_h, :] = 0
        if bot_h > 0:
            mask[h - bot_h : h, :] = 0

        return mask

    def apply(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply mask to `frame`. If `mask` is None, update() will be called first."""
        if mask is None:
            mask = self.update(frame)
        # make 3-channel mask
        if len(mask.shape) == 2:
            mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            mask3 = mask
        out = cv2.bitwise_and(frame, mask3)
        return out


def quick_mask_from_frames(frames, **kwargs) -> np.ndarray:
    """Utility: build an ArenaMask from a list/iterator of frames by warming the BG model.

    Returns the last mask produced. Useful to initialize the model before processing.
    """
    am = ArenaMask(**kwargs)
    last_mask = None
    for f in frames:
        last_mask = am.update(f)
    return last_mask

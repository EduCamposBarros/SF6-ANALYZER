import cv2
from typing import Optional


def detect_game_state(frame, prev_frame, frame_data, prev_state: Optional[str] = None) -> Optional[str]:
    """Detecta estado de jogo simples: 'FIGHT', 'KO', 'REPLAY' ou None.

    Heurísticas usadas:
    - 'KO' quando `life_p1` ou `life_p2` é 0.
    - 'FIGHT' quando ambos têm vida quase cheia e há um overlay de alto contraste
      na região superior-central (heurística para texto 'FIGHT').
    - 'REPLAY' quando o estado anterior era 'KO' e o frame atual é muito semelhante
      ao anterior (pequena diferença de pixels), sugerindo um replay estático.

    Esta função é intencionalmente conservadora e baseada em heurísticas simples.
    """
    # KO: vida zerada
    try:
        if frame_data.life_p1 == 0 or frame_data.life_p2 == 0:
            return "KO"
    except Exception:
        pass

    # FIGHT: ambos com vida quase cheia e presença de alto-contraste na região superior-central
    try:
        if frame is not None and frame_data.life_p1 is not None and frame_data.life_p2 is not None:
            if frame_data.life_p1 >= 95 and frame_data.life_p2 >= 95:
                h, w = frame.shape[:2]
                cx0, cx1 = w // 4, 3 * w // 4
                cy0, cy1 = h // 12, h // 3
                region = frame[cy0:cy1, cx0:cx1]
                if region.size > 0:
                    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    total_area = sum(cv2.contourArea(c) for c in contours)
                    region_area = max(1, region.shape[0] * region.shape[1])
                    if (total_area / region_area) > 0.02:
                        return "FIGHT"
    except Exception:
        pass

    # REPLAY: if previous state was KO and frames are nearly identical
    try:
        if prev_state == "KO" and prev_frame is not None and frame is not None:
            diff = cv2.absdiff(frame, prev_frame)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            nonzero = cv2.countNonZero(gray)
            h, w = frame.shape[:2]
            # if less than 1% of pixels changed, assume replay/static screen
            if nonzero < (h * w * 0.01):
                return "REPLAY"
    except Exception:
        pass

    return None

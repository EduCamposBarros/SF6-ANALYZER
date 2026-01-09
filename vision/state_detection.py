import numpy as np

def detect_state(frame, bbox):
    """
    Classifica o estado do personagem baseado em heurísticas visuais.
    NÃO identifica golpes específicos — apenas estados.
    """

    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]  # Região do personagem

    area = roi.shape[0] * roi.shape[1]
    mean_color = np.mean(roi)

    # Heurística: tons escuros/azulados → bloqueio
    if mean_color < 40:
        return "block"

    # Heurística: bounding box grande → ataque ativo
    if area > 20000:
        return "attack_active"

    return "neutral"


def can_act(state):
    """
    Define se o personagem pode agir neste frame.
    Base para cálculo de frame advantage.
    """
    return state not in ["block", "attack_active", "hitstun"]

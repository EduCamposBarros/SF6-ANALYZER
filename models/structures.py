from dataclasses import dataclass
from typing import Optional

# Representa o estado completo de UM frame da partida
@dataclass
class FrameData:
    frame_id: int              # Número do frame (0, 1, 2...)
    timestamp: float           # Tempo em segundos (frame_id / FPS)

    p1_state: str              # Estado do Player 1 (neutral, attack_active, block, etc)
    p2_state: str              # Estado do Player 2

    p1_can_act: bool           # Player 1 pode agir neste frame?
    p2_can_act: bool           # Player 2 pode agir neste frame?

    p1_bbox: tuple             # Bounding box do P1 (x1, y1, x2, y2)
    p2_bbox: tuple             # Bounding box do P2

    life_p1: int               # Vida do Player 1 (placeholder no MVP)
    life_p2: int               # Vida do Player 2


# Representa um EVENTO relevante detectado no jogo
@dataclass
class Event:
    type: str                  # Tipo do evento (attack_start, block, hit, etc)
    frame_id: int              # Frame onde ocorreu
    attacker: Optional[str] = None
    defender: Optional[str] = None

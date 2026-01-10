from dataclasses import dataclass
from typing import Optional, Tuple


# Representa o estado completo de UM frame da partida
@dataclass
class FrameData:
    """
    Estrutura de dados que representa o estado de um único frame.

    Campos principais:
    - `frame_id`: identificador inteiro do frame (sequencial)
    - `timestamp`: tempo em segundos
    - `p1_state`, `p2_state`: strings com o estado atual do jogador
    - `p1_bbox`, `p2_bbox`: bounding boxes (x,y,w,h)
    - `life_p1`, `life_p2`: valores de vida (inteiros) quando disponíveis
    - `p1_can_act`, `p2_can_act`: flags que indicam se o jogador pode agir
    - `p1_action`, `p2_action`: rótulos de ação detectada (opcional)
    """
    frame_id: int  # Número do frame (0, 1, 2...)
    timestamp: float  # Tempo em segundos (frame_id / FPS)

    p1_state: str  # Estado do Player 1 (neutral, attack_active, block, jump, etc)
    p2_state: str  # Estado do Player 2

    p1_can_act: bool  # Player 1 pode agir neste frame?
    p2_can_act: bool  # Player 2 pode agir neste frame?

    p1_bbox: Tuple[int, int, int, int]  # Bounding box do P1 (x1, y1, x2, y2)
    p2_bbox: Tuple[int, int, int, int]  # Bounding box do P2

    life_p1: int  # Vida do Player 1 (placeholder no MVP)
    life_p2: int  # Vida do Player 2

    # Ações/opcional labels detectadas (jump, drive, etc.)
    p1_action: Optional[str] = None
    p2_action: Optional[str] = None
    # Game-wide state: 'FIGHT', 'KO', 'REPLAY' or None
    game_state: Optional[str] = None


# Representa um EVENTO relevante detectado no jogo
@dataclass
class Event:
    """
    Representa um evento discreto detectado na timeline.

    - `type`: string com o tipo do evento (ex.: 'hit', 'block', 'attack_start')
    - `frame_id`: frame onde o evento foi detectado
    - `attacker`, `defender`: identificadores opcionais ('p1'/'p2' ou 'P1'/'P2')
    """
    type: str  # Tipo do evento (attack_start, block, hit, jump, etc)
    frame_id: int  # Frame onde ocorreu
    attacker: Optional[str] = None
    defender: Optional[str] = None

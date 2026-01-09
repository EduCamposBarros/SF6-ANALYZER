from models.structures import Event


def detect_events(current, previous):
    """
    Detecta eventos discretos comparando `current` com `previous`.

    Eventos extraídos:
    - `attack_start`: quando um jogador passa a estar em `attack_active`.
    - `block`: quando o defensor está em estado `block`.
    - `hit`: quando a vida do defensor diminui entre frames.
    - `jump_start` / `jump_land`: mudanças de estado de salto.
    - `drive_impact`: heurística para detectar impacto causado por 'drive' (se houve perda de vida
      e o atacante estava em estado `drive`).

    Retorna uma lista de instâncias `Event` (possivelmente vazia).
    """

    events = []

    if not previous:
        return events

    # Início de ataque (P1)
    if previous.p1_state != "attack_active" and current.p1_state == "attack_active":
        events.append(Event("attack_start", current.frame_id, attacker="P1"))

    # Ataque bloqueado (detecção simples quando p2 está em block)
    if current.p2_state == "block":
        events.append(Event("block", current.frame_id, attacker="P1", defender="P2"))

    # Hit: detectado por queda na vida do defensor
    if current.life_p2 < previous.life_p2:
        events.append(Event("hit", current.frame_id, attacker="P1", defender="P2"))

        # Jump start / land (p1)
        if previous.p1_state != "jump" and current.p1_state == "jump":
            events.append(Event("jump_start", current.frame_id, attacker="P1"))

        if previous.p1_state == "jump" and current.p1_state != "jump":
            events.append(Event("jump_land", current.frame_id, attacker="P1"))

        # Jump start / land (p2)
        if previous.p2_state != "jump" and current.p2_state == "jump":
            events.append(Event("jump_start", current.frame_id, attacker="P2"))

        if previous.p2_state == "jump" and current.p2_state != "jump":
            events.append(Event("jump_land", current.frame_id, attacker="P2"))

        # Drive impact: heurística baseada no estado `drive` imediatamente antes do hit
        if current.life_p2 < previous.life_p2 and previous.p1_state == "drive":
            events.append(Event("drive_impact", current.frame_id, attacker="P1", defender="P2"))

        if current.life_p1 < previous.life_p1 and previous.p2_state == "drive":
            events.append(Event("drive_impact", current.frame_id, attacker="P2", defender="P1"))

    return events

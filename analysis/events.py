from models.structures import Event

def detect_events(current, previous):
    """
    Compara o frame atual com o anterior
    para identificar eventos discretos.
    """

    events = []

    if not previous:
        return events

    # Início de ataque
    if previous.p1_state != "attack_active" and current.p1_state == "attack_active":
        events.append(Event("attack_start", current.frame_id, attacker="P1"))

    # Ataque bloqueado
    if current.p2_state == "block":
        events.append(Event("block", current.frame_id, attacker="P1", defender="P2"))

    # Hit confirmado (vida caiu)
    if current.life_p2 < previous.life_p2:
        events.append(Event("hit", current.frame_id, attacker="P1", defender="P2"))

    return events
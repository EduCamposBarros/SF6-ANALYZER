def calculate_frame_data(timeline):
    """
    Analisa a timeline frame a frame e constrói
    janelas de ataque para calcular vantagem.
    """

    windows = []
    current_attack = None

    for frame in timeline:

        # Detecta início de ataque
        if frame.p1_state == "attack_active" and not current_attack:
            current_attack = {
                "start": frame.frame_id,
                "attacker_free": None,
                "defender_free": None
            }

        if current_attack:
            # Primeiro frame em que o atacante pode agir
            if frame.p1_can_act and not current_attack["attacker_free"]:
                current_attack["attacker_free"] = frame.frame_id

            # Primeiro frame em que o defensor pode agir
            if frame.p2_can_act and not current_attack["defender_free"]:
                current_attack["defender_free"] = frame.frame_id

            # Quando ambos recuperaram o controle, calculamos vantagem
            if current_attack["attacker_free"] and current_attack["defender_free"]:
                adv = current_attack["attacker_free"] - current_attack["defender_free"]

                windows.append({
                    "start": current_attack["start"],
                    "on_block_adv": adv
                })

                current_attack = None

    return windows

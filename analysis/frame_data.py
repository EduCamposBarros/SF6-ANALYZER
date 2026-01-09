def calculate_frame_data(timeline, events=None):
    """
    Analisa a timeline (lista de `FrameData`) e eventos para extrair métricas de frame data.

    Produz:
    - `windows`: janelas de ataque detectadas (inicio, fim, vantagem on-block ou marcações de whiff)
    - `summary`: agregações úteis para gerar insights (contagens de plus/minus, pulos puníveis, drive impacts,
        e whiff_punishes)

    A função percorre a timeline construindo janelas a partir de estados `attack_active`, mas também
    deriva janelas a partir de eventos discretos (ex.: `hit`, `block`) quando disponíveis.
    """

    if events is None:
        events = []

    windows = []

    # Suporta rastrear ataques para ambos os jogadores
    current = {"p1": None, "p2": None}

    for frame in timeline:
        # Verifica início de ataque para P1
        if frame.p1_state == "attack_active" and current["p1"] is None:
            current["p1"] = {"start": frame.frame_id, "attacker_free": None, "defender_free": None}

        # Verifica início de ataque para P2
        if frame.p2_state == "attack_active" and current["p2"] is None:
            current["p2"] = {"start": frame.frame_id, "attacker_free": None, "defender_free": None}

        # Atualiza janelas abertas
        for attacker, attack in list(current.items()):
            if attack is None:
                continue

            # Define quem é atacante/defensor
            if attacker == "p1":
                atk_can_act = frame.p1_can_act
                def_can_act = frame.p2_can_act
            else:
                atk_can_act = frame.p2_can_act
                def_can_act = frame.p1_can_act

            if atk_can_act and attack["attacker_free"] is None:
                attack["attacker_free"] = frame.frame_id

            if def_can_act and attack["defender_free"] is None:
                attack["defender_free"] = frame.frame_id

            # Ambos recuperaram controle -> calcula vantagem
            if attack["attacker_free"] is not None and attack["defender_free"] is not None:
                adv = attack["attacker_free"] - attack["defender_free"]
                windows.append({"attacker": attacker, "start": attack["start"], "end": frame.frame_id, "on_block_adv": adv})
                current[attacker] = None

            # Fallback: ataque terminou sem contato (whiff)
            elif attack["attacker_free"] is not None and frame.frame_id - attack["start"] > 30:
                # Whiff fallback: mark as whiff and record end frame
                end_frame = frame.frame_id
                # check if opponent can act within short window after end
                opp = "p2" if attacker == "p1" else "p1"
                opp_can_punish = False
                for off in range(0, 4):
                    idx = end_frame + off
                    if idx < 0 or idx >= len(timeline):
                        continue
                    if getattr(timeline[idx], f"{opp}_can_act"):
                        opp_can_punish = True
                        break

                windows.append({"attacker": attacker, "start": attack["start"], "end": end_frame, "on_block_adv": -999, "whiff": True, "punishable": opp_can_punish})
                current[attacker] = None

    # (aggregation moved down after event-derived windows)

    # Detecta pulos puníveis: procura sequências de 'jump' por jogador
    punishable_jumps = []
    for player in ("p1", "p2"):
        in_jump = False
        jump_start = None
        for frame in timeline:
            state = getattr(frame, f"{player}_state")
            opp = "p2" if player == "p1" else "p1"

            if not in_jump and state == "jump":
                in_jump = True
                jump_start = frame.frame_id

            if in_jump and state != "jump":
                landing_frame = frame.frame_id
                # verifica se o oponente pode punir no landing ou nos próximos 2 frames (janela reduzida)
                landing_index = landing_frame
                opp_can_punish = False
                # busca frame por id (timeline index == frame_id assumido sequencial)
                for offset in range(0, 3):
                    idx = landing_index + offset
                    if idx < 0 or idx >= len(timeline):
                        continue
                    opp_can = getattr(timeline[idx], f"{opp}_can_act")
                    if opp_can:
                        opp_can_punish = True
                        break

                punishable_jumps.append({
                    "player": player,
                    "start": jump_start,
                    "land": landing_frame,
                    "punishable": opp_can_punish,
                })

                in_jump = False
                jump_start = None

    # Drive impacts a partir da lista de eventos
    drive_impacts = [e for e in (events or []) if e.type == "drive_impact"]

    # Também derive janelas a partir de eventos discretos (hit, block, attack_start)
    for e in (events or []):
        if e.type not in ("hit", "block", "attack_start", "drive_impact"):
            continue

        attacker = None
        defender = None
        if e.attacker == "P1" or e.attacker == "p1":
            attacker = "p1"
            defender = "p2"
        elif e.attacker == "P2" or e.attacker == "p2":
            attacker = "p2"
            defender = "p1"
        else:
            continue

        start = e.frame_id
        attacker_free = None
        defender_free = None

        # procura, até 60 frames à frente, o primeiro frame em que cada um pode agir
        for fid in range(start, min(start + 60, len(timeline))):
            f = timeline[fid]
            if attacker == "p1":
                if attacker_free is None and f.p1_can_act:
                    attacker_free = f.frame_id
                if defender_free is None and f.p2_can_act:
                    defender_free = f.frame_id
            else:
                if attacker_free is None and f.p2_can_act:
                    attacker_free = f.frame_id
                if defender_free is None and f.p1_can_act:
                    defender_free = f.frame_id

            if attacker_free is not None and defender_free is not None:
                adv = attacker_free - defender_free
                # evita duplicatas (mesmo start e attacker)
                if not any(w["start"] == start and w["attacker"] == attacker for w in windows):
                    windows.append({"attacker": attacker, "start": start, "on_block_adv": adv})
                break
        else:
            # se não encontrou janelas reais, use heurística por tipo de evento
            if e.type == "hit":
                heur_adv = 10
            elif e.type == "block":
                heur_adv = 2
            elif e.type == "drive_impact":
                heur_adv = 0
            else:
                heur_adv = 0

            if not any(w["start"] == start and w["attacker"] == attacker for w in windows):
                windows.append({"attacker": attacker, "start": start, "on_block_adv": heur_adv})


    # Agregações (incluindo janelas derivadas de eventos)
    valid_advs = [w["on_block_adv"] for w in windows if w["on_block_adv"] != -999]
    plus_on_block = sum(1 for a in valid_advs if a > 0)
    minus_on_block = sum(1 for a in valid_advs if a < 0)
    avg_on_block = sum(valid_advs) / len(valid_advs) if valid_advs else 0

    summary = {
        "plus_on_block": plus_on_block,
        "minus_on_block": minus_on_block,
        "avg_on_block": avg_on_block,
        "punishable_jumps": punishable_jumps,
        "drive_impacts": [
            {"frame_id": d.frame_id, "attacker": d.attacker, "defender": d.defender} for d in drive_impacts
        ],
        "whiff_punishes": [
            {"attacker": w.get("attacker"), "start": w.get("start"), "end": w.get("end"), "punishable": w.get("punishable", False)}
            for w in windows if w.get("whiff")
        ],
    }

    return {"windows": windows, "summary": summary}


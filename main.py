"""
Pipeline principal do analisador.

Este módulo orquestra a execução das etapas principais:
- leitura de frames do vídeo
- detecção de personagens (`vision.character_detection`)
- detecção de efeitos visuais (`vision.effects_detection`)
- classificação de estado por frame (`vision.state_detection`)
- detecção de eventos discretos (`analysis.events`)
- agregação de frame-data (`analysis.frame_data`)
- geração de insights (`analysis.insights`)

O resultado é escrito em `output/results.json` e inclui uma fatia
de `debug_timeline` para inspeção rápida.
"""

import cv2
import json

from models.structures import FrameData
from vision.character_detection import detect_characters
from vision.state_detection import detect_state, can_act
from analysis.events import detect_events
from analysis.frame_data import calculate_frame_data
from analysis.insights import generate_insights


def run(video_path):
    """
    Pipeline principal:
    vídeo → frames → estados → eventos → frame data → insights
    """

    cap = cv2.VideoCapture(video_path)

    timeline = []  # Linha do tempo completa da partida
    events = []  # Eventos relevantes

    prev = None
    frame_id = 0

    # estados persistentes
    prev_frame = None
    prev_p1_bbox = None
    prev_p2_bbox = None
    life_p1 = 100
    life_p2 = 100

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detecta/rastra posição dos personagens (usa dados do frame anterior para estabilidade)
        prev_bboxes = (prev_p1_bbox, prev_p2_bbox) if (prev_p1_bbox is not None and prev_p2_bbox is not None) else None
        p1_bbox, p2_bbox = detect_characters(frame, prev_frame, prev_bboxes)

        # Detecta estado de cada jogador usando histórico para detectar jump/drive
        p1_state = detect_state(frame, p1_bbox, prev_frame, prev_p1_bbox)
        p2_state = detect_state(frame, p2_bbox, prev_frame, prev_p2_bbox)

        # Detecta efeitos entre frames (hitsparks)
        from vision.effects_detection import detect_effects
        from config import DAMAGE_PER_HIT

        effects = detect_effects(frame, prev_frame, p1_bbox, p2_bbox)

        # Atualiza vida/ações com base em efeitos
        p1_action = None
        p2_action = None
        for eff in effects:
            if eff.get("type") == "hitspark":
                target = eff.get("target")
                if target == "p2":
                    life_p2 = max(0, life_p2 - DAMAGE_PER_HIT)
                    p2_action = "hit"
                    # marca atacante como em ataque ativo (melhora janelas)
                    p1_state = "attack_active"
                    p1_action = "attack"
                elif target == "p1":
                    life_p1 = max(0, life_p1 - DAMAGE_PER_HIT)
                    p1_action = "hit"
                    p2_state = "attack_active"
                    p2_action = "attack"

        # Cria o FrameData
        data = FrameData(
            frame_id=frame_id,
            timestamp=frame_id / 60,
            p1_state=p1_state,
            p2_state=p2_state,
            p1_can_act=can_act(p1_state),
            p2_can_act=can_act(p2_state),
            p1_bbox=p1_bbox,
            p2_bbox=p2_bbox,
            life_p1=life_p1,
            life_p2=life_p2,
            p1_action=p1_action,
            p2_action=p2_action,
        )

        timeline.append(data)

        # Detecta eventos com base no frame anterior
        events.extend(detect_events(data, prev))

        prev = data
        prev_frame = frame.copy() if frame is not None else None
        prev_p1_bbox = p1_bbox
        prev_p2_bbox = p2_bbox
        frame_id += 1
        

    cap.release()

    # Calcula frame advantage e outras métricas a partir da timeline e eventos
    frame_data_result = calculate_frame_data(timeline, events)

    # Gera insights de gameplay
    insights = generate_insights(frame_data_result)

    # Exporta resultados estruturados
    with open("output/results.json", "w") as f:
        json.dump(
            {
                "frame_data": frame_data_result,
                "insights": insights,
                "events": [e.__dict__ for e in events],
                "debug_timeline": [
                    {
                        "frame_id": fd.frame_id,
                        "p1_state": fd.p1_state,
                        "p2_state": fd.p2_state,
                        "p1_can_act": fd.p1_can_act,
                        "p2_can_act": fd.p2_can_act,
                    }
                    for fd in timeline[:200]
                ],
            },
            f,
            indent=2,
        )

if __name__ == "__main__":
    run("match.mp4")

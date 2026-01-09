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
    events = []    # Eventos relevantes

    prev = None
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detecta posição dos personagens
        p1_bbox, p2_bbox = detect_characters(frame)

        # Detecta estado de cada jogador
        p1_state = detect_state(frame, p1_bbox)
        p2_state = detect_state(frame, p2_bbox)

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
            life_p1=100,   # Placeholder
            life_p2=100    # Placeholder
        )

        timeline.append(data)

        # Detecta eventos com base no frame anterior
        events.extend(detect_events(data, prev))

        prev = data
        frame_id += 1

    cap.release()

    # Calcula frame advantage
    frame_data = calculate_frame_data(timeline)

    # Gera insights de gameplay
    insights = generate_insights(frame_data)

    # Exporta resultados
    with open("output/results.json", "w") as f:
        json.dump({
            "frame_data": frame_data,
            "insights": insights
        }, f, indent=2)


if __name__ == "__main__":
    run("match.mp4")

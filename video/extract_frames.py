import cv2
import os


def extract_frames(video_path, output_dir, fps=60):
    """
    Extrai frames do vídeo na taxa desejada.
    Cada frame representa uma unidade de tempo para cálculo de frame data.
    """

    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)

    frame_id = 0
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # Define quantos frames pular para manter 60fps
    frame_interval = int(video_fps / fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Salva apenas os frames necessários
        if frame_id % frame_interval == 0:
            cv2.imwrite(f"{output_dir}/frame_{frame_id}.png", frame)

        frame_id += 1

    cap.release()

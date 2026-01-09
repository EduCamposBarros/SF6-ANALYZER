def detect_characters(frame):
    """
    Detecta onde estão os personagens.
    No MVP usamos bounding boxes fixas.
    Em versões futuras: YOLO / tracking.
    """

    h, w, _ = frame.shape

    # Bounding box aproximada do Player 1
    p1_bbox = (int(w*0.1), int(h*0.5), int(w*0.25), int(h*0.9))

    # Bounding box aproximada do Player 2
    p2_bbox = (int(w*0.75), int(h*0.5), int(w*0.9), int(h*0.9))

    return p1_bbox, p2_bbox

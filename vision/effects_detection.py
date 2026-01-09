"""Detecção simples de efeitos visuais (impact sparks, hitsparks, etc.).

Este módulo contém stubs leves usados pelo pipeline MVP. Em versões
futuras, substitua por um detector baseado em visão (CNN, template match, etc.).
"""

from typing import List, Dict, Optional, Tuple
import numpy as np


def detect_effects(
    frame: np.ndarray,
    prev_frame: Optional[np.ndarray] = None,
    p1_bbox: Optional[Tuple[int, int, int, int]] = None,
    p2_bbox: Optional[Tuple[int, int, int, int]] = None,
    mean_diff_thresh: float = 10.0,
    blur_ksize: Optional[int] = None,
    morph_kernel: Optional[int] = None,
    binary_thresh: Optional[int] = None,
) -> List[Dict]:
    """
    Detecta efeitos visuais (hitsparks) por diferenciação entre `frame` e `prev_frame`.

    Retorna lista de dicionários: {"type": "hitspark", "target": "p1"/"p2", "confidence": float}
    """

    results: List[Dict] = []
    if prev_frame is None:
        return results


    # converte para grayscale para simplificar e aplica pré-processamento opcional
    try:
        import cv2

        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        if blur_ksize and blur_ksize > 1:
            cur_gray = cv2.GaussianBlur(cur_gray, (blur_ksize, blur_ksize), 0)
            prev_gray = cv2.GaussianBlur(prev_gray, (blur_ksize, blur_ksize), 0)

        diff = cv2.absdiff(cur_gray, prev_gray)

        if binary_thresh is not None:
            _, diff = cv2.threshold(diff, binary_thresh, 255, cv2.THRESH_BINARY)

        if morph_kernel and morph_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
            diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
    except Exception:
        # Fallback simples com numpy (se cv2 não disponível)
        cur_gray = np.mean(frame, axis=2).astype(np.uint8)
        prev_gray = np.mean(prev_frame, axis=2).astype(np.uint8)
        diff = np.abs(cur_gray - prev_gray)

    def analyze_bbox(bbox: Tuple[int, int, int, int], target: str):
        x1, y1, x2, y2 = bbox
        roi_diff = diff[y1:y2, x1:x2]
        if roi_diff.size == 0:
            return
        # média de diferença como proxy para efeito
        mean_diff = float(np.mean(roi_diff))
        if mean_diff > mean_diff_thresh:  # threshold ajustável
            results.append({"type": "hitspark", "target": target, "confidence": mean_diff})

    if p1_bbox:
        analyze_bbox(p1_bbox, "p1")
    if p2_bbox:
        analyze_bbox(p2_bbox, "p2")

import cv2
import numpy as np
from typing import Optional, Tuple


"""
Detecção simples de efeitos visuais (ex.: hitsparks) por diferença entre frames.

Este módulo fornece uma função leve `detect_effects` que compara a região de
cada personagem entre dois frames consecutivos e retorna uma medida de
confiança (mean absolute diff) quando encontra alteração visual forte.

Parâmetros ajustáveis:
- `mean_diff_thresh`: limite mínimo de média absoluta para considerar um efeito.
- `blur_ksize`, `binary_thresh`, `morph_kernel`: opções de pré-processamento
  para reduzir ruído e focar em pixels relevantes.

Retorno:
- Dicionário `{ "conf": <valor> }` quando detecta efeito em alguma bbox;
- `None` caso não detecte nada.
"""


def detect_effects(cur_frame, prev_frame, p1_bbox, p2_bbox, mean_diff_thresh=10.0, blur_ksize=None, morph_kernel=None, binary_thresh=None):
    """
    Detecta alterações visuais entre `prev_frame` e `cur_frame` nas ROIs fornecidas.

    - Calcula a diferença absoluta média (MAD) por ROI.
    - Aplica pré-processamento opcional para reduzir falsos positivos.

    Uso típico: passar os bboxes de ambos personagens; a função retorna o
    primeiro efeito detectado (prioriza `p1_bbox`) ou `None`.
    """

    # função helper: prepara ROI (grayscale, blur, thresh, morph)
    def preprocess(img, blur_k, bin_th, morph_k):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if blur_k:
            g = cv2.GaussianBlur(g, (blur_k, blur_k), 0)
        if bin_th is not None:
            _, g = cv2.threshold(g, bin_th, 255, cv2.THRESH_BINARY)
        if morph_k:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
            g = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel)
        return g

    # segurança: precisa de ambos os frames para calcular diferença
    if cur_frame is None or prev_frame is None:
        return []

    # aplica pré-processamento se configurado
    if blur_ksize or binary_thresh or morph_kernel:
        cur_p = preprocess(cur_frame, blur_ksize, binary_thresh, morph_kernel)
        prev_p = preprocess(prev_frame, blur_ksize, binary_thresh, morph_kernel)
    else:
        cur_p = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        prev_p = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # calcula diff e média nas regiões dos personagens
    results = []
    for bbox in (p1_bbox, p2_bbox):
        try:
            x, y, w, h = map(int, bbox)
            roi_cur = cur_p[y:y+h, x:x+w]
            roi_prev = prev_p[y:y+h, x:x+w]
            if roi_cur.size == 0 or roi_prev.size == 0:
                results.append(None)
                continue
            mad = float(np.mean(np.abs(roi_cur.astype(float) - roi_prev.astype(float))))
            if mad > mean_diff_thresh:
                results.append({"conf": mad})
            else:
                results.append(None)
        except Exception:
            results.append(None)

    # Converte os resultados em uma lista uniforme de efeitos compatível com o pipeline
    out = []
    if results[0]:
        out.append({"type": "hitspark", "target": "p2", "confidence": results[0].get("conf")})
    if results[1]:
        out.append({"type": "hitspark", "target": "p1", "confidence": results[1].get("conf")})
    return out

"""
Detector simples de estado do personagem.

Este módulo contém heurísticas leves para inferir estados de personagem
a partir de frames e bounding boxes (ex.: 'attack_active', 'jump', 'drive', 'block').
As decisões são baseadas em tamanho do bbox, cor média e diferenças
entre frames consecutivos (movimento local).

O objetivo não é ser perfeito, mas fornecer sinais suficientes para a
etapa de análise que agrupa janelas de ataque e gera eventos.
"""

import json
import os
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class StateDetectorConfig:
    area_attack_threshold: int = 20000
    area_attack_fallback: int = 5000
    mean_color_block_threshold: float = 40.0
    jump_cy_delta: float = 0.1  # fraction of bbox height
    jump_cy_min: int = 5
    drive_cx_delta_factor: float = 0.2  # fraction of bbox width
    drive_cx_min: int = 10
    motion_thresh: float = 5.0  # mean absolute diff threshold within bbox to consider motion


def detect_state(frame, bbox, prev_frame=None, prev_bbox=None, config: Optional[StateDetectorConfig] = None):
    """
    Determina o estado do personagem em um frame.

    Parâmetros
    - frame: frame atual (imagem BGR numpy)
    - bbox: tupla (x1,y1,x2,y2) definindo a região do personagem
    - prev_frame: frame anterior (opcional), usado para calcular movimento
    - prev_bbox: bbox anterior (opcional), usado para detectar mudanças de posição
    - config: instância de `StateDetectorConfig` com thresholds de leitura

    Retorna
    - string representando o estado: 'neutral', 'attack_active', 'jump', 'drive', 'block'
    """

    if config is None:
        # prefer a persisted tuned config if available
        try:
            from vision import tuned_state_config

            config = tuned_state_config.get_default_config()
        except Exception:
            # fallback to reading the tuning report at runtime
            cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "tuning_report.json")
            try:
                with open(cfg_path, "r") as f:
                    data = json.load(f)
                top = data.get("results", [])[0] if data.get("results") else None
                if top and "config" in top:
                    c = top["config"]
                    config = StateDetectorConfig(
                        area_attack_threshold=c.get("area_t", StateDetectorConfig.area_attack_threshold),
                        area_attack_fallback=c.get("area_fb", StateDetectorConfig.area_attack_fallback),
                        mean_color_block_threshold=c.get("mean_c", StateDetectorConfig.mean_color_block_threshold),
                        jump_cy_delta=c.get("jump_delta", StateDetectorConfig.jump_cy_delta),
                        drive_cx_delta_factor=c.get("drive_factor", StateDetectorConfig.drive_cx_delta_factor),
                    )
                else:
                    config = StateDetectorConfig()
            except Exception:
                config = StateDetectorConfig()

    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]  # Região do personagem

    # Safety: se ROI inválida/ vazia, retorna neutral
    if roi.size == 0:
        return "neutral"

    h = y2 - y1
    w = x2 - x1
    area = h * w
    mean_color = float(np.mean(roi))

    # Detect jump via comparação de posição vertical do bbox com o anterior
    if prev_bbox is not None:
        _, py1, _, py2 = prev_bbox
        prev_cy = (py1 + py2) / 2
        cur_cy = (y1 + y2) / 2
        if prev_cy - cur_cy > max(config.jump_cy_min, config.jump_cy_delta * h):
            return "jump"

    # Heurísticas simples (ordenadas por sinal claro)
    # Requer movimento local (diferença em relação ao prev_frame) para reduzir falsos positivos
    if area > config.area_attack_threshold:
        if prev_frame is not None and prev_bbox is not None:
            try:
                # compute mean absolute diff in ROI
                prev_x1, prev_y1, prev_x2, prev_y2 = prev_bbox
                prev_roi = prev_frame[prev_y1:prev_y2, prev_x1:prev_x2]
                # align sizes (in case bbox changed slightly)
                if prev_roi.shape == roi.shape:
                    mad = float(np.mean(np.abs(roi.astype(float) - prev_roi.astype(float))))
                else:
                    mad = float(np.mean(np.abs(roi.astype(float) - np.mean(prev_roi, axis=(0, 1)))))
            except Exception:
                mad = 0.0
        else:
            mad = 0.0

        if mad > config.motion_thresh:
            return "attack_active"

    if mean_color < config.mean_color_block_threshold:
        return "block"

    # Heurística simples para 'drive' (movimento horizontal brusco)
    if prev_bbox is not None:
        px1, _, px2, _ = prev_bbox
        prev_cx = (px1 + px2) / 2
        cur_cx = (x1 + x2) / 2
        if abs(cur_cx - prev_cx) > max(config.drive_cx_min, config.drive_cx_delta_factor * w):
            return "drive"

    # DEBUG fallback: força ataque para validar pipeline (use com cuidado)
    if area > config.area_attack_fallback and prev_frame is None:
        return "attack_active"

    return "neutral"


def can_act(state):
    return state == "neutral"

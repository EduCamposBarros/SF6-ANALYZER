"""Helpers utilitários para operações com vídeo/frames.

Contém funções pequenas para manter o pipeline limpo.
"""

import os
from typing import Any


def ensure_dir(path: str) -> None:
    """Cria o diretório `path` se não existir."""

    os.makedirs(path, exist_ok=True)


def write_frame(path: str, frame: Any) -> None:
    """Escreve um frame no caminho fornecido.

    Implementação mínima para manter dependências de IO fora de módulos
    que importam `cv2` diretamente.
    """

    # Import local para evitar exigir cv2 no import do módulo
    try:
        import cv2

        cv2.imwrite(path, frame)
    except Exception:
        # Falha silenciosa no MVP; chamador decide como lidar com erros
        pass

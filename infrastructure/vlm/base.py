import base64
import json
import cv2
import numpy as np
from domain.ports import IVlmAgent

PROMPT = (
    "Analise este veículo e responda SOMENTE em JSON com as chaves: "
    "marca, modelo, ano_estimado, cor. "
    'Exemplo: {"marca": "Toyota", "modelo": "Corolla", "ano_estimado": "2020", "cor": "prata"}. '
    "Se não souber, use null."
)

_EMPTY = {"marca": None, "modelo": None, "ano_estimado": None, "cor": None}


class BaseVlmAgent(IVlmAgent):
    def encode_image(self, image: np.ndarray) -> str:
        _, buffer = cv2.imencode(".jpg", image)
        return base64.b64encode(buffer).decode("utf-8")

    def parse_response(self, text: str) -> dict:
        try:
            start, end = text.find("{"), text.rfind("}") + 1
            return json.loads(text[start:end])
        except Exception:
            return _EMPTY.copy()

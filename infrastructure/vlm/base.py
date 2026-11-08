import base64
import json
import cv2
import numpy as np
from domain.ports import IVlmAgent

PROMPT = (
    "Analise este veículo e responda SOMENTE em JSON com as chaves: "
    "marca, modelo, ano_estimado, cor, placa. "
    'Exemplo: {"marca": "Toyota", "modelo": "Corolla", "ano_estimado": "2020", "cor": "prata", "placa": "ABC1D23"}. '
    "Para placa: informe somente se visível na imagem, caso contrário use null. "
    "Se não souber qualquer outro campo, use null."
)

_EMPTY = {"marca": None, "modelo": None, "ano_estimado": None, "cor": None, "placa": None}
_STR_KEYS = {"marca", "modelo", "ano_estimado", "cor", "placa"}


class BaseVlmAgent(IVlmAgent):
    def encode_image(self, image: np.ndarray) -> str:
        _, buffer = cv2.imencode(".jpg", image)
        return base64.b64encode(buffer).decode("utf-8")

    def parse_response(self, text: str) -> dict:
        try:
            start, end = text.find("{"), text.rfind("}") + 1
            raw = json.loads(text[start:end])
            return self._sanitize(raw)
        except Exception:
            return _EMPTY.copy()

    @staticmethod
    def _sanitize(data: dict) -> dict:
        """Garante que todos os campos de texto sejam str ou None."""
        result = _EMPTY.copy()
        for key in _STR_KEYS:
            val = data.get(key)
            if val is None:
                result[key] = None
            elif isinstance(val, dict):
                # VLM retornou objeto — pega o primeiro valor string
                result[key] = next((str(v) for v in val.values() if v), None)
            elif isinstance(val, list):
                result[key] = ", ".join(str(v) for v in val if v) or None
            else:
                cleaned = str(val).strip()
                # descarta frases longas como "Não é possível ver a placa"
                result[key] = cleaned if len(cleaned) <= 30 else None
        return result

import io
import cv2
import numpy as np
from PIL import Image
from google import genai
from google.genai import types
from config import get_settings
from infrastructure.vlm.base import BaseVlmAgent, PROMPT


class GeminiAgent(BaseVlmAgent):
    def __init__(self):
        cfg = get_settings()
        self._client = genai.Client(api_key=cfg.gemini_api_key)
        self._model = cfg.vlm_gemini_model

    def analyze(self, image: np.ndarray) -> dict:
        _, buffer = cv2.imencode(".jpg", image)
        pil_image = Image.open(io.BytesIO(buffer.tobytes()))
        response = self._client.models.generate_content(
            model=self._model,
            contents=[PROMPT, pil_image],
        )
        return self.parse_response(response.text)

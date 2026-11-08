import io
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai
from config import get_settings
from infrastructure.vlm.base import BaseVlmAgent, PROMPT


class GeminiAgent(BaseVlmAgent):
    def __init__(self):
        cfg = get_settings()
        genai.configure(api_key=cfg.gemini_api_key)
        self._model = genai.GenerativeModel(cfg.vlm_gemini_model)

    def analyze(self, image: np.ndarray) -> dict:
        _, buffer = cv2.imencode(".jpg", image)
        pil_image = Image.open(io.BytesIO(buffer.tobytes()))
        response = self._model.generate_content([PROMPT, pil_image])
        return self.parse_response(response.text)

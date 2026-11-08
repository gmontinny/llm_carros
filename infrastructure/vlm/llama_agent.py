import cv2
import numpy as np
import ollama
from config import get_settings
from infrastructure.vlm.base import BaseVlmAgent, PROMPT


class LlamaAgent(BaseVlmAgent):
    def __init__(self):
        self._model = get_settings().llama_model

    def analyze(self, image: np.ndarray) -> dict:
        _, buffer = cv2.imencode(".jpg", image)
        response = ollama.chat(
            model=self._model,
            messages=[{
                "role": "user",
                "content": PROMPT,
                "images": [buffer.tobytes()],
            }],
        )
        return self.parse_response(response["message"]["content"])

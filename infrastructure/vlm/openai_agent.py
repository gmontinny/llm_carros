import numpy as np
from openai import OpenAI
from config import get_settings
from infrastructure.vlm.base import BaseVlmAgent, PROMPT


class OpenAiAgent(BaseVlmAgent):
    def __init__(self):
        cfg = get_settings()
        if not cfg.openai_api_key:
            raise ValueError("OPENAI_API_KEY não configurada no .env")
        self._client = OpenAI(api_key=cfg.openai_api_key)
        self._model = cfg.vlm_openai_model

        # Valida a chave antes de processar qualquer frame
        try:
            self._client.models.list()
        except Exception as e:
            raise ValueError(f"Chave OpenAI inválida: {e}") from e

    def analyze(self, image: np.ndarray) -> dict:
        b64 = self.encode_image(image)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            max_tokens=200,
        )
        return self.parse_response(response.choices[0].message.content)

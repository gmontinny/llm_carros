import re
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from infrastructure.vlm.base import BaseVlmAgent
from config import get_settings

_MODEL_ID = "microsoft/Florence-2-large"
_TASK = "<MORE_DETAILED_CAPTION>"
_PROMPT = (
    "Analise este veículo e responda SOMENTE em JSON com as chaves: "
    "marca, modelo, ano_estimado, cor, placa. "
    'Exemplo: {"marca": "Toyota", "modelo": "Corolla", "ano_estimado": "2020", "cor": "prata", "placa": "ABC1D23"}. '
    "Se não souber, use null."
)


class FlorenceAgent(BaseVlmAgent):
    def __init__(self):
        device = get_settings().device
        self._device = device
        self._processor = AutoProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device)
        self._model.eval()

    def analyze(self, image: np.ndarray) -> dict:
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self._processor(
            text=_PROMPT, images=pil, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
            )

        text = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return self.parse_response(text)

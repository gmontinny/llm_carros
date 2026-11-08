import re
from typing import Optional
import cv2
import numpy as np
import easyocr
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from domain.ports import IOcrReader
from config import get_settings

_PLATE_PATTERN = re.compile(r"[A-Z]{3}[\s-]?[0-9][A-Z0-9][0-9]{2}", re.IGNORECASE)
_MODEL_ID = "nickmuchi/yolos-small-finetuned-license-plate-detection"


class YolosPlateOcrReader(IOcrReader):
    def __init__(self):
        self._device = get_settings().device
        self._processor = AutoImageProcessor.from_pretrained(_MODEL_ID)
        self._model = AutoModelForObjectDetection.from_pretrained(_MODEL_ID).to(self._device)
        self._model.eval()
        self._ocr = easyocr.Reader(["pt", "en"], gpu=self._device == "cuda")

    def extract_plate(self, image: np.ndarray) -> Optional[str]:
        plate_crop = self._detect_plate_region(image)
        if plate_crop is not None:
            result = self._read_text(plate_crop)
            if result:
                return result

        # fallback: faixa inferior + imagem inteira
        h, w = image.shape[:2]
        lower = image[int(h * 0.65):h, int(w * 0.1):int(w * 0.9)]
        for region in (lower, image):
            result = self._read_text(region)
            if result:
                return result
        return None

    def _detect_plate_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self._processor(images=pil, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([pil.size[::-1]])
        results = self._processor.post_process_object_detection(
            outputs, threshold=0.5, target_sizes=target_sizes
        )[0]

        if len(results["boxes"]) == 0:
            return None

        # pega a detecção com maior score
        best = results["scores"].argmax().item()
        x1, y1, x2, y2 = map(int, results["boxes"][best].tolist())

        # margem de 4px
        h, w = image.shape[:2]
        x1, y1 = max(0, x1 - 4), max(0, y1 - 4)
        x2, y2 = min(w, x2 + 4), min(h, y2 + 4)

        return image[y1:y2, x1:x2]

    def _read_text(self, image: np.ndarray) -> Optional[str]:
        preprocessed = self._preprocess(image)
        results = self._ocr.readtext(preprocessed, detail=0, paragraph=False)
        text = "".join(results).upper()
        match = _PLATE_PATTERN.search(text)
        return match.group(0) if match else None

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale = max(1, 200 // max(h, w, 1))
        if scale > 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        return cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

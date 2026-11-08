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

# Mercosul: ABC1D23 — antigo: ABC1234
_PLATE_PATTERN = re.compile(r"[A-Z]{3}[\s\-]?[0-9][A-Z0-9][0-9]{2}", re.IGNORECASE)
_MODEL_ID = "nickmuchi/yolos-small-finetuned-license-plate-detection"
_TARGET_W = 400   # largura alvo para ampliar o crop da placa antes do OCR
_YOLOS_THRESHOLD = 0.3   # reduzido de 0.5 para capturar placas com baixa confiança


class YolosPlateOcrReader(IOcrReader):
    def __init__(self):
        self._device = get_settings().device
        self._processor = AutoImageProcessor.from_pretrained(_MODEL_ID)
        self._model = AutoModelForObjectDetection.from_pretrained(_MODEL_ID).to(self._device)
        self._model.eval()
        self._ocr = easyocr.Reader(["pt", "en"], gpu=self._device == "cuda")

    def extract_plate(self, image: np.ndarray) -> Optional[str]:
        # 1ª tentativa: YOLOS localiza a placa com precisão
        plate_crop = self._detect_plate_region(image)
        if plate_crop is not None:
            result = self._read_text(plate_crop)
            if result:
                return result

        # 2ª tentativa: faixa inferior do crop (heurística posicional)
        h, w = image.shape[:2]
        lower = image[int(h * 0.60):h, int(w * 0.05):int(w * 0.95)]
        result = self._read_text(lower)
        if result:
            return result

        # 3ª tentativa: imagem inteira
        return self._read_text(image)

    def _detect_plate_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self._processor(images=pil, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([pil.size[::-1]])
        results = self._processor.post_process_object_detection(
            outputs, threshold=_YOLOS_THRESHOLD, target_sizes=target_sizes
        )[0]

        if len(results["boxes"]) == 0:
            return None

        best = results["scores"].argmax().item()
        x1, y1, x2, y2 = map(int, results["boxes"][best].tolist())

        # margem de 6px
        h, w = image.shape[:2]
        x1, y1 = max(0, x1 - 6), max(0, y1 - 6)
        x2, y2 = min(w, x2 + 6), min(h, y2 + 6)

        crop = image[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _read_text(self, image: np.ndarray) -> Optional[str]:
        preprocessed = self._preprocess(image)
        results = self._ocr.readtext(preprocessed, detail=0, paragraph=False)
        text = "".join(results).upper().replace(" ", "").replace("-", "")
        match = _PLATE_PATTERN.search(text)
        return match.group(0) if match else None

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if w < _TARGET_W:
            scale = _TARGET_W / max(w, 1)
            image = cv2.resize(
                image, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 11, 75, 75)
        return cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 13, 2
        )

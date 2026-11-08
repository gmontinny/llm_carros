import re
from typing import Optional
import cv2
import numpy as np
import easyocr
from domain.ports import IOcrReader
from config import get_settings

# Mercosul: ABC1D23 — antigo: ABC1234
_PLATE_PATTERN = re.compile(r"[A-Z]{3}[\s-]?[0-9][A-Z0-9][0-9]{2}", re.IGNORECASE)


class EasyOcrReader(IOcrReader):
    def __init__(self):
        device = get_settings().device
        self._reader = easyocr.Reader(["pt", "en"], gpu=device == "cuda")

    def extract_plate(self, image: np.ndarray) -> Optional[str]:
        # Tenta primeiro na região inferior do crop (onde a placa costuma estar)
        plate_region = self._crop_plate_region(image)

        for region in (plate_region, image):  # fallback para imagem inteira
            result = self._read(region)
            if result:
                return result
        return None

    def _read(self, image: np.ndarray) -> Optional[str]:
        preprocessed = self._preprocess(image)
        results = self._reader.readtext(preprocessed, detail=0, paragraph=False)
        text = "".join(results).upper()
        match = _PLATE_PATTERN.search(text)
        return match.group(0) if match else None

    @staticmethod
    def _crop_plate_region(image: np.ndarray) -> np.ndarray:
        """Recorta a faixa inferior do veículo onde a placa geralmente aparece."""
        h, w = image.shape[:2]
        return image[int(h * 0.65):h, int(w * 0.1):int(w * 0.9)]

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        """Amplia, converte para cinza e aplica threshold para melhorar leitura OCR."""
        scale = max(1, 200 // max(image.shape[:2], default=1))
        if scale > 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        return cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

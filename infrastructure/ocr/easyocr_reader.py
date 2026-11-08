import re
from typing import Optional
import numpy as np
import easyocr
from domain.ports import IOcrReader

_PLATE_PATTERN = re.compile(r"[A-Z]{3}[\s-]?[0-9][A-Z0-9][0-9]{2}", re.IGNORECASE)


class EasyOcrReader(IOcrReader):
    def __init__(self):
        self._reader = easyocr.Reader(["pt", "en"], gpu=False)

    def extract_plate(self, image: np.ndarray) -> Optional[str]:
        results = self._reader.readtext(image, detail=0, paragraph=False)
        text = "".join(results).upper()
        match = _PLATE_PATTERN.search(text)
        return match.group(0) if match else None

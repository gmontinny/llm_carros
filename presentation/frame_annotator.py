import cv2
import numpy as np
from domain.models import VehicleAnalysis

_COLOR = (0, 200, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class FrameAnnotator:
    def annotate(self, frame: np.ndarray, analyses: list[VehicleAnalysis]) -> np.ndarray:
        out = frame.copy()
        for a in analyses:
            self._draw(out, a)
        return out

    def _draw(self, frame: np.ndarray, a: VehicleAnalysis) -> None:
        x1, y1, x2, y2 = a.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), _COLOR, 2)

        lines = [
            f"{a.category} {a.confidence:.0%}",
            f"{a.marca or '?'} {a.modelo or ''}".strip(),
            f"Placa: {a.placa or '-'}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x1, y1 - 10 - i * 18), _FONT, 0.55, _COLOR, 2)

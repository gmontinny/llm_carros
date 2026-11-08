from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Detection:
    class_id: int
    category: str
    confidence: float
    bbox: tuple[int, int, int, int]          # (x1, y1, x2, y2)
    crop: np.ndarray = field(repr=False)


@dataclass
class VehicleAnalysis:
    category: str
    confidence: float
    bbox: tuple[int, int, int, int]
    marca: Optional[str]
    modelo: Optional[str]
    ano_estimado: Optional[str]
    cor: Optional[str]
    placa: Optional[str]

    def __str__(self) -> str:
        return (
            f"[{self.category}] {self.confidence:.0%} | "
            f"{self.marca or '?'} {self.modelo or '?'} {self.ano_estimado or '?'} | "
            f"Cor: {self.cor or '?'} | Placa: {self.placa or '-'}"
        )

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
    agente: str = ""
    frame: int = 0
    video: str = ""

    def __str__(self) -> str:
        return (
            f"[{self.category}] {self.confidence:.0%} | "
            f"{self.marca or '?'} {self.modelo or '?'} {self.ano_estimado or '?'} | "
            f"Cor: {self.cor or '?'} | Placa: {self.placa or '-'}"
        )

    def to_dict(self) -> dict:
        return {
            "video": self.video,
            "frame": self.frame,
            "agente": self.agente,
            "categoria": self.category,
            "confianca": round(self.confidence, 4),
            "marca": self.marca,
            "modelo": self.modelo,
            "ano_estimado": self.ano_estimado,
            "cor": self.cor,
            "placa": self.placa,
            "bbox": list(self.bbox),
        }

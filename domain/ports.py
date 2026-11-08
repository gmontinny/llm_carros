from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from domain.models import Detection, VehicleAnalysis


class IDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        ...


class IOcrReader(ABC):
    @abstractmethod
    def extract_plate(self, image: np.ndarray) -> Optional[str]:
        ...


class IVlmAgent(ABC):
    @abstractmethod
    def analyze(self, image: np.ndarray) -> dict:
        ...


class IReportExporter(ABC):
    @abstractmethod
    def export(self, records: list[VehicleAnalysis], output_dir, stem: str) -> list:
        """Exporta os registros e retorna lista de caminhos gerados."""
        ...

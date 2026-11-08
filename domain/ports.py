from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from domain.models import Detection


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

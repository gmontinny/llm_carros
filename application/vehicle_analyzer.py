from domain.models import Detection, VehicleAnalysis
from domain.ports import IDetector, IOcrReader, IVlmAgent


class VehicleAnalyzer:
    def __init__(self, detector: IDetector, ocr: IOcrReader, vlm: IVlmAgent):
        self._detector = detector
        self._ocr = ocr
        self._vlm = vlm

    def analyze_frame(self, frame) -> list[VehicleAnalysis]:
        detections = self._detector.detect(frame)
        return [self._analyze_detection(d) for d in detections]

    def _analyze_detection(self, detection: Detection) -> VehicleAnalysis:
        vlm_result = self._vlm.analyze(detection.crop)
        plate = self._ocr.extract_plate(detection.crop)

        return VehicleAnalysis(
            category=detection.category,
            confidence=detection.confidence,
            bbox=detection.bbox,
            marca=vlm_result.get("marca"),
            modelo=vlm_result.get("modelo"),
            ano_estimado=vlm_result.get("ano_estimado"),
            cor=vlm_result.get("cor"),
            placa=plate,
        )

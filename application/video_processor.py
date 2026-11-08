from pathlib import Path
import cv2
from application.vehicle_analyzer import VehicleAnalyzer
from domain.ports import IReportExporter
from presentation.frame_annotator import FrameAnnotator
from config import get_settings


class VideoProcessor:
    def __init__(self, analyzer: VehicleAnalyzer, annotator: FrameAnnotator, exporter: IReportExporter):
        self._analyzer = analyzer
        self._annotator = annotator
        self._exporter = exporter
        self._frame_interval = get_settings().video_frame_interval

    def process(self, video_path: Path, output_dir: Path) -> Path:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Não foi possível abrir: {video_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{video_path.stem}_annotated.mp4"
        writer = self._create_writer(cap, out_path)

        frame_idx = 0
        last_analyses = []
        all_records = []

        print(f"\n▶ {video_path.name}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self._frame_interval == 0:
                    try:
                        last_analyses = self._analyzer.analyze_frame(
                            frame, frame_idx, video_path.name
                        )
                        all_records.extend(last_analyses)
                        for a in last_analyses:
                            print(f"  frame {frame_idx}: {a}")
                    except Exception as e:
                        print(f"  [AVISO] frame {frame_idx}: {e}")

                writer.write(self._annotator.annotate(frame, last_analyses))
                frame_idx += 1
        finally:
            cap.release()
            writer.release()

        print(f"✔ Vídeo: {out_path}")

        if all_records:
            provider = get_settings().vlm_provider.lower()
            export_stem = f"{video_path.stem}_{provider}"
            self._exporter.export(all_records, output_dir, export_stem)

        return out_path

    @staticmethod
    def _create_writer(cap: cv2.VideoCapture, path: Path) -> cv2.VideoWriter:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

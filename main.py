import sys
from pathlib import Path

from config import get_settings
from infrastructure.detection.yolo_detector import YoloDetector
from infrastructure.ocr.yolos_plate_reader import YolosPlateOcrReader
from infrastructure.vlm.factory import create_vlm_agent
from infrastructure.export.json_exporter import JsonExporter
from infrastructure.export.xlsx_exporter import XlsxExporter
from infrastructure.export.composite_exporter import CompositeExporter
from application.vehicle_analyzer import VehicleAnalyzer
from application.video_processor import VideoProcessor
from presentation.frame_annotator import FrameAnnotator

VIDEOS_DIR = Path(__file__).parent / "videos"
OUTPUT_DIR = Path(__file__).parent / "output"
_VIDEO_EXTENSIONS = (".mp4", ".mov", ".mts", ".avi", ".mkv")


def build_processor() -> VideoProcessor:
    detector = YoloDetector()
    ocr = YolosPlateOcrReader()
    vlm = create_vlm_agent()
    analyzer = VehicleAnalyzer(detector, ocr, vlm)
    annotator = FrameAnnotator()
    exporter = CompositeExporter([JsonExporter(), XlsxExporter()])
    return VideoProcessor(analyzer, annotator, exporter)


def resolve_videos(args: list[str]) -> list[Path]:
    all_videos = [
        v for v in VIDEOS_DIR.iterdir()
        if v.suffix.lower() in _VIDEO_EXTENSIONS
    ]
    if not all_videos:
        print(f"Nenhum vídeo encontrado em {VIDEOS_DIR} (extensões: {_VIDEO_EXTENSIONS})")
        sys.exit(1)

    if not args:
        return all_videos

    filtered = [v for v in all_videos if args[0] in v.name]
    if not filtered:
        print(f"Vídeo '{args[0]}' não encontrado.")
        sys.exit(1)

    return filtered


def main() -> None:
    cfg = get_settings()
    videos = resolve_videos(sys.argv[1:])
    processor = build_processor()

    print(f"Provedor VLM: {cfg.vlm_provider.upper()}")
    print(f"Vídeos: {[v.name for v in videos]}")

    for video in videos:
        processor.process(video, OUTPUT_DIR)

    print("\n✅ Concluído.")


if __name__ == "__main__":
    main()

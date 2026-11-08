import json
from pathlib import Path
from domain.models import VehicleAnalysis
from domain.ports import IReportExporter


class JsonExporter(IReportExporter):
    def export(self, records: list[VehicleAnalysis], output_dir: Path, stem: str) -> list[Path]:
        path = output_dir / f"{stem}_report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in records], f, ensure_ascii=False, indent=2)
        print(f"  📄 JSON: {path}")
        return [path]

from pathlib import Path
from domain.models import VehicleAnalysis
from domain.ports import IReportExporter


class CompositeExporter(IReportExporter):
    def __init__(self, exporters: list[IReportExporter]):
        self._exporters = exporters

    def export(self, records: list[VehicleAnalysis], output_dir: Path, stem: str) -> list[Path]:
        paths = []
        for exporter in self._exporters:
            paths.extend(exporter.export(records, output_dir, stem))
        return paths

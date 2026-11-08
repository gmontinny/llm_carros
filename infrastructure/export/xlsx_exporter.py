from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from domain.models import VehicleAnalysis
from domain.ports import IReportExporter

_HEADERS = ["Video", "Frame", "Agente", "Categoria", "Confiança", "Marca", "Modelo", "Ano", "Cor", "Placa", "BBox"]
_HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
_HEADER_FONT = Font(bold=True, color="FFFFFF")


class XlsxExporter(IReportExporter):
    def export(self, records: list[VehicleAnalysis], output_dir: Path, stem: str) -> list[Path]:
        path = output_dir / f"{stem}_report.xlsx"
        wb = Workbook()
        ws = wb.active
        ws.title = "Detecções"

        self._write_header(ws)
        for row_idx, r in enumerate(records, start=2):
            d = r.to_dict()
            ws.append([
                d["video"], d["frame"], d["agente"], d["categoria"],
                f"{d['confianca']:.0%}", d["marca"], d["modelo"],
                d["ano_estimado"], d["cor"], d["placa"],
                str(d["bbox"]),
            ])
            if row_idx % 2 == 0:
                for cell in ws[row_idx]:
                    cell.fill = PatternFill("solid", fgColor="D6E4F0")

        self._adjust_columns(ws)
        wb.save(path)
        print(f"  📊 XLSX: {path}")
        return [path]

    @staticmethod
    def _write_header(ws) -> None:
        ws.append(_HEADERS)
        for cell in ws[1]:
            cell.font = _HEADER_FONT
            cell.fill = _HEADER_FILL
            cell.alignment = Alignment(horizontal="center")

    @staticmethod
    def _adjust_columns(ws) -> None:
        for col in ws.columns:
            max_len = max((len(str(c.value or "")) for c in col), default=10)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 4, 40)

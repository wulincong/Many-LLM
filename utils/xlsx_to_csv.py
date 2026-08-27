import argparse
import csv
import posixpath
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "office_rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
RID = f"{{{NS['office_rel']}}}id"


def column_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref.upper())
    if not letters:
        return 0

    value = 0
    for char in letters.group(0):
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []

    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values = []
    for item in root.findall("main:si", NS):
        texts = [node.text or "" for node in item.findall(".//main:t", NS)]
        values.append("".join(texts))
    return values


def workbook_rels(zf: zipfile.ZipFile) -> dict[str, str]:
    root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rels = {}
    for rel in root.findall("rel:Relationship", NS):
        rel_id = rel.attrib["Id"]
        target = rel.attrib["Target"]
        if target.startswith("/"):
            path = target.lstrip("/")
        else:
            path = posixpath.normpath(posixpath.join("xl", target))
        rels[rel_id] = path
    return rels


def worksheet_path(zf: zipfile.ZipFile, sheet_name: str | None = None) -> str:
    root = ET.fromstring(zf.read("xl/workbook.xml"))
    sheets = root.findall("main:sheets/main:sheet", NS)
    if not sheets:
        raise ValueError("Workbook has no worksheets")

    selected = sheets[0]
    if sheet_name is not None:
        selected = next(
            (sheet for sheet in sheets if sheet.attrib.get("name") == sheet_name),
            None,
        )
        if selected is None:
            raise ValueError(f"Worksheet not found: {sheet_name}")

    rel_id = selected.attrib[RID]
    return workbook_rels(zf)[rel_id]


def cell_value(cell: ET.Element, shared: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//main:is//main:t", NS))

    value = cell.find("main:v", NS)
    if value is None or value.text is None:
        return ""

    text = value.text
    if cell_type == "s":
        return shared[int(text)]
    if cell_type == "b":
        return "TRUE" if text == "1" else "FALSE"
    return text


def worksheet_rows(zf: zipfile.ZipFile, path: str, shared: list[str]) -> list[list[str]]:
    root = ET.fromstring(zf.read(path))
    rows = []
    max_width = 0
    for row in root.findall(".//main:sheetData/main:row", NS):
        values = []
        for cell in row.findall("main:c", NS):
            index = column_index(cell.attrib.get("r", ""))
            if index >= len(values):
                values.extend([""] * (index - len(values) + 1))
            values[index] = cell_value(cell, shared)
        max_width = max(max_width, len(values))
        rows.append(values)

    for row in rows:
        row.extend([""] * (max_width - len(row)))
    return rows


def convert_xlsx_to_csv(
    xlsx_path: str | Path,
    csv_path: str | Path | None = None,
    sheet_name: str | None = None,
    encoding: str = "utf-8-sig",
) -> Path:
    xlsx_path = Path(xlsx_path)
    csv_path = Path(csv_path) if csv_path is not None else xlsx_path.with_suffix(".csv")

    with zipfile.ZipFile(xlsx_path) as zf:
        shared = shared_strings(zf)
        sheet_path = worksheet_path(zf, sheet_name)
        rows = worksheet_rows(zf, sheet_path, shared)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding=encoding, newline="") as f:
        csv.writer(f).writerows(rows)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an .xlsx worksheet to CSV.")
    parser.add_argument("xlsx_path", help="Input .xlsx file")
    parser.add_argument("csv_path", nargs="?", help="Output .csv file")
    parser.add_argument("--sheet", dest="sheet_name", help="Worksheet name")
    args = parser.parse_args()

    output = convert_xlsx_to_csv(args.xlsx_path, args.csv_path, args.sheet_name)
    print(output)


if __name__ == "__main__":
    main()

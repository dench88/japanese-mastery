"""Utility to convert a PDF to a UTF-8 text file using pypdf.

Usage:
  python -m scripts.pdf_to_txt input.pdf [output.txt]

If output.txt is omitted, a .txt sibling will be created next to the PDF.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader


def convert_pdf_to_txt(pdf_path: Path, txt_path: Path | None = None) -> Path:
    """Extract text from `pdf_path` and write to `txt_path` (UTF-8)."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if txt_path is None:
        # default: place output in content/source_materials with same stem
        project_root = Path(__file__).resolve().parents[1]
        out_dir = project_root / "content" / "source_materials"
        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path = out_dir / f"{pdf_path.stem}.txt"

    reader = PdfReader(pdf_path)
    texts: list[str] = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")

    output = "\n\n".join(texts)
    txt_path.write_text(output, encoding="utf-8")
    return txt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PDF to UTF-8 text file.")
    parser.add_argument("pdf", type=Path, help="Path to input PDF")
    parser.add_argument("txt", type=Path, nargs="?", help="Optional output .txt path")
    args = parser.parse_args()

    out_path = convert_pdf_to_txt(args.pdf, args.txt)
    print(f"Wrote text to {out_path}")


if __name__ == "__main__":
    main()

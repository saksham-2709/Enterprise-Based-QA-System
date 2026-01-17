from pathlib import Path
from typing import List, Dict

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError("PyPDF2 not installed")
    pages = []
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for p in reader.pages:
            pages.append(p.extract_text() or "")
    return "\n".join(pages)


def load_documents_from_folder(folder: str, exts=(".txt", ".pdf")) -> List[Dict]:
    docs = []
    folder = Path(folder)
    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() not in exts:
            continue

        if p.suffix.lower() == ".txt":
            text = read_txt(p)
        else:
            text = read_pdf(p)

        docs.append({"path": str(p), "text": text})
    return docs

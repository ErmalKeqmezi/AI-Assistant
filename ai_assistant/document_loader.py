from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Document:
    text: str
    source: str


def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _load_pdf(path)
    elif suffix in {".txt", ".md"}:
        text = _load_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return Document(text=text, source=str(path))


def load_documents(directory: str) -> list[Document]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"No such directory: {directory}")

    paths = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return [load_document(p) for p in paths]

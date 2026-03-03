import os
import fitz  # PyMuPDF

SUPPORTED_EXTENSIONS = [".pdf", ".txt"]

def load_document(file_path: str) -> str:
    """
    Load a document (PDF or TXT) and return extracted raw text.
    """
    ext = os.path.splitext(file_path)[-1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    if ext == ".pdf":
        return _load_pdf(file_path)

    if ext == ".txt":
        return _load_txt(file_path)


def _load_pdf(file_path: str) -> str:
    text = []
    doc = fitz.open(file_path)

    for page in doc:
        page_text = page.get_text()
        if page_text:
            text.append(page_text)

    return "\n".join(text)


def _load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

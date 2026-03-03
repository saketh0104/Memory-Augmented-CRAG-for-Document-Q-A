import re

def clean_text(text: str) -> str:
    """
    Clean extracted document text.
    """
    # Remove excessive newlines
    text = re.sub(r"\n{2,}", "\n", text)

    # Remove page numbers (simple heuristic)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    # Normalize spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()

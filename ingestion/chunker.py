def chunk_text(text: str, chunk_size: int = 180, overlap: int = 40):
    #Split text into overlapping chunks for enterprise documents.
    # Preserves local financial and governance context.
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    
    return chunks
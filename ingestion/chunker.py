import re
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def chunk_text(
    text: str,
    embedder,
    max_tokens: int = 200,
    similarity_threshold: float = 0.50) -> List[str]:

    sentences = split_into_sentences(text)

    if not sentences:
        return []

    # Embed all sentences
    embeddings = embedder.model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]
    current_tokens = estimate_tokens(sentences[0])

    for i in range(1, len(sentences)):

        sim = cosine_similarity(
            [embeddings[i - 1]],
            [embeddings[i]]
        )[0][0]

        sent = sentences[i]
        sent_tokens = estimate_tokens(sent)

        # 🚨 Break conditions
        if (
            sim < similarity_threshold or
            current_tokens + sent_tokens > max_tokens
        ):
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0

        current_chunk.append(sent)
        current_tokens += sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
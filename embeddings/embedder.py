from sentence_transformers import SentenceTransformer
import sys
import os
import contextlib

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

class Embedder:
    _model = None

    def __init__(self):
        if Embedder._model is None:
            Embedder._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.model = Embedder._model

    def embed_query(self, text):
        return self.model.encode(text).tolist()

    def embed_texts(self, texts):
        return self.model.encode(texts).tolist()


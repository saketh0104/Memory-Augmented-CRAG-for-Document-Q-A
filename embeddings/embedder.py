from sentence_transformers import SentenceTransformer

class Embedder:
    _model = None

    def __init__(self):
        if Embedder._model is None:
            Embedder._model = SentenceTransformer("models/all-MiniLM-L6-v2")
        self.model = Embedder._model

    def embed_query(self, text):
        return self.model.encode(text).tolist()

    def embed_texts(self, texts):
        return self.model.encode(texts).tolist()
import os
import numpy as np
from typing import List

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import openai
except Exception:
    openai = None


class LocalEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True).astype("float32")


class OpenAIEmbedder:
    def __init__(self, model="text-embedding-3-small"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.dim = 1536

    def embed(self, texts: List[str]) -> np.ndarray:
        out = []
        for t in texts:
            r = openai.Embedding.create(model=self.model, input=t)
            out.append(r["data"][0]["embedding"])
        return np.array(out, dtype="float32")

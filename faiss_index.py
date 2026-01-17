import numpy as np

try:
    import faiss
except Exception:
    faiss = None


def build_index(embeddings: np.ndarray, use_cosine=True):
    if faiss is None:
        raise RuntimeError("faiss is not installed. pip install faiss-cpu")

    if use_cosine:
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])

    index.add(embeddings)
    return index


def save_index(index, path: str):
    faiss.write_index(index, path)


def load_index(path: str):
    return faiss.read_index(path)

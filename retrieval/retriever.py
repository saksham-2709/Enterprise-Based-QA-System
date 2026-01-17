import faiss
from typing import List, Tuple


def retrieve(index, embed_fn, query: str, k=4) -> Tuple[List[int], List[float]]:
    qv = embed_fn([query])
    faiss.normalize_L2(qv)
    scores, ids = index.search(qv, k)
    return ids[0].tolist(), scores[0].tolist()

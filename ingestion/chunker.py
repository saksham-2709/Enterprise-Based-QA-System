from typing import List, Dict, Tuple
import re

try:
    import nltk
    from nltk import sent_tokenize
except Exception:
    sent_tokenize = None


def ensure_nltk():
    global sent_tokenize
    if sent_tokenize is None:
        import nltk
        nltk.download("punkt")
        from nltk import sent_tokenize as st
        sent_tokenize = st


def sentence_chunk(text: str, max_chars=1000, overlap_chars=200) -> List[str]:
    ensure_nltk()
    sents = sent_tokenize(text)
    chunks, cur = [], ""

    for s in sents:
        if len(cur) + len(s) <= max_chars:
            cur += " " + s
        else:
            chunks.append(cur.strip())
            cur = cur[-overlap_chars:] + " " + s

    if cur.strip():
        chunks.append(cur.strip())

    return chunks


def block_chunk(text: str) -> List[str]:
    blocks = text.split("\n\n")
    return [b.strip() for b in blocks if len(b.strip()) > 30]


def extract_command(text: str) -> str:
    match = re.search(r"db\.\w+\.\w+\(.*?\)", text, re.DOTALL)
    return match.group() if match else text


def doc_to_chunks(docs: List[Dict], max_chars=1000, overlap_chars=200) -> Tuple[List[str], List[Dict]]:
    chunks, metas = [], []
    chunk_id = 0

    for doc in docs:
        text = doc["text"]
        path = doc["path"]

        if path.endswith(".txt"):
            doc_chunks = block_chunk(text)
        else:
            doc_chunks = sentence_chunk(text, max_chars, overlap_chars)

        for i, c in enumerate(doc_chunks):
            chunks.append(c)
            metas.append({
                "chunk_id": chunk_id,
                "source": path,
                "chunk_index_in_doc": i,
                "text_preview": c[:200].replace("\n", " ")
            })
            chunk_id += 1

    return chunks, metas

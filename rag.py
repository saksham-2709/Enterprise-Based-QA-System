"""
rag_helper.py

Single script to:
- load text & PDF documents from a folder
- chunk text (sentence-based with overlap)
- embed chunks (sentence-transformers or OpenAI)
- index embeddings in FAISS (inner product with L2-normalization for cosine)
- retrieve top-k chunks for a query
- optionally call OpenAI ChatCompletion to generate a final answer using retrieved chunks

Usage:
  python rag_helper.py build   --data_dir data/ --index_path my_index.index --meta_path meta.json --embedder local
  python rag_helper.py query   --index_path my_index.index --meta_path meta.json --embedder local --k 4
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import re

# Optional imports
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import openai
except Exception:
    openai = None

try:
    import faiss
except Exception:
    faiss = None

try:
    import nltk
    from nltk import sent_tokenize
except Exception:
    nltk = None
    sent_tokenize = None

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


def ensure_nltk():
    global sent_tokenize
    if sent_tokenize is None:
        import nltk as _nltk
        _nltk.download("punkt")
        from nltk import sent_tokenize as _st
        sent_tokenize = _st

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_pdf(path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError("PyPDF2 not installed. pip install PyPDF2")
    text_pages = []
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for p in reader.pages:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            text_pages.append(t)
    return "\n".join(text_pages)

def load_documents_from_folder(folder: str, exts=(".txt", ".pdf")) -> List[Dict]:
    folder = Path(folder)
    docs = []
    for p in folder.iterdir():
        if not p.is_file(): continue
        if p.suffix.lower() not in exts: continue
        if p.suffix.lower() == ".txt":
            text = read_txt(p)
        elif p.suffix.lower() == ".pdf":
            text = read_pdf(p)
        else:
            continue
        docs.append({"path": str(p), "text": text})
    return docs

# ----------------------------
# Chunking
# ----------------------------
def sentence_chunk(text: str, max_chars: int = 1000, overlap_chars: int = 200) -> List[str]:
    """
    Sentence-based chunking: accumulate sentences until reaching max_chars, then create a chunk.
    Keep a small overlap (by characters) to avoid cutting important context.
    """
    ensure_nltk()
    sents = sent_tokenize(text)
    chunks = []
    cur = ""
    for s in sents:
        if len(cur) + len(s) <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur.strip())
            # prepare new chunk with overlap: last overlap_chars of cur + current sentence
            if overlap_chars > 0 and len(cur) > 0:
                tail = cur[-overlap_chars:]
                cur = (tail + " " + s).strip()
            else:
                cur = s.strip()
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

import re

def extract_command(text):
    pattern = r"db\.\w+\.\w+\(.*?\)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group() if match else text

def block_chunk(text):
    blocks = text.split("\n\n")  # split by empty line
    return [b.strip() for b in blocks if len(b.strip()) > 30]

def doc_to_chunks(docs: List[Dict], max_chars=1000, overlap_chars=200) -> Tuple[List[str], List[Dict]]:
    """
    Returns:
      - chunks: list of chunk texts
      - metas: list of metadata dicts aligned with chunks
    """
    chunks = []
    metas = []
    chunk_id = 0

    for doc in docs:
        path = doc.get("path", "<unknown>")
        text = doc.get("text", "")
        if not text.strip():
            continue

        if path.lower().endswith(".txt"):
            doc_chunks = block_chunk(text)   # for MongoDB / code notes
        else:
            doc_chunks = sentence_chunk(
                text,
                max_chars=max_chars,
                overlap_chars=overlap_chars
            )

        for i, c in enumerate(doc_chunks):
            chunks.append(c)
            metas.append({
                "chunk_id": chunk_id,
                "source": path,
                "chunk_index_in_doc": i,
                "text_preview": c[:200].replace("\n", " "),
            })
            chunk_id += 1

    return chunks, metas

def rerank_with_llm(query, chunks):
    prompt = f"""
Given the question and candidate answers,
return ONLY the most precise answer.

Question:
{query}

Candidates:
{chunks}
"""
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )
    return resp["choices"][0]["message"]["content"]

# ----------------------------
# Embeddings
# ----------------------------
class LocalEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")
        self.model = SentenceTransformer(model_name)
        # SentenceTransformer returns numpy arrays by default if convert_to_numpy=True
        self.dim = self.model.get_sentence_embedding_dimension()
    def embed(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embs.astype("float32")

class OpenAIEmbedder:
    def __init__(self, model="text-embedding-3-small"):
        if openai is None:
            raise RuntimeError("openai package not installed. pip install openai")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set OPENAI_API_KEY environment variable to use OpenAI embedder")
        openai.api_key = key
        self.model = model
        # dimension for 'text-embedding-3-small' is 1536 (check OpenAI docs)
        self.dim = 1536
    def embed(self, texts: List[str]) -> np.ndarray:
        # Note: This naive batch approach may hit rate limits for large lists.
        out = []
        BATCH = 16
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i+BATCH]
            resp = openai.Embedding.create(model=self.model, input=batch)
            for r in resp["data"]:
                out.append(r["embedding"])
        arr = np.array(out, dtype="float32")
        return arr

# ----------------------------
# FAISS index helpers
# ----------------------------
def build_faiss_index(embs: np.ndarray, use_cosine=True):
    if faiss is None:
        raise RuntimeError("faiss not installed. pip install faiss-cpu")
    if use_cosine:
        # normalize for cosine similarity, then use IndexFlatIP (inner product)
        faiss.normalize_L2(embs)
        index = faiss.IndexFlatIP(embs.shape[1])
    else:
        index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)
    return index

def save_faiss_index(index, path: str):
    faiss.write_index(index, path)

def load_faiss_index(path: str):
    return faiss.read_index(path)

# ----------------------------
# Querying & LLM answer generation
# ----------------------------
def retrieve(index, embed_fn, query: str, k: int = 4, use_cosine=True) -> Tuple[List[int], List[float]]:
    qv = embed_fn([query]).astype("float32")
    if use_cosine:
        faiss.normalize_L2(qv)
    scores, ids = index.search(qv, k)
    return ids[0].tolist(), scores[0].tolist()

def build_prompt_with_context(question: str, retrieved_texts: List[str]) -> str:
    prompt = "You are a helpful assistant. Use the provided document chunks to answer the question. Cite chunk indices if helpful.\n\n"
    for i, t in enumerate(retrieved_texts, 1):
        prompt += f"=== Document chunk {i} ===\n{t}\n\n"
    prompt += f"Question: {question}\nAnswer concisely and cite the chunk numbers used."
    return prompt

def call_openai_chat(prompt: str, model: str = "gpt-4o-mini", max_tokens=300):
    if openai is None:
        raise RuntimeError("openai package not installed. pip install openai")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY environment variable to use OpenAI LLM")
    openai.api_key = key
    # Use ChatCompletion or Chat API depending on your openai library; simple completion shown
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return resp["choices"][0]["message"]["content"]

# ----------------------------
# CLI actions
# ----------------------------
def action_build(args):
    print(f"[+] Loading documents from {args.data_dir} ...")
    docs = load_documents_from_folder(args.data_dir)
    print(f"[+] Loaded {len(docs)} documents.")
    print("[+] Chunking documents ...")
    chunks, metas = doc_to_chunks(docs, max_chars=args.chunk_size, overlap_chars=args.overlap)
    print(f"[+] Produced {len(chunks)} chunks.")

    # choose embedder
    if args.embedder == "local":
        emb = LocalEmbedder(model_name=args.local_model)
    elif args.embedder == "openai":
        emb = OpenAIEmbedder(model=args.openai_model)
    else:
        raise ValueError("embedder must be 'local' or 'openai'")

    print("[+] Creating embeddings (this may take a while) ...")
    embs = emb.embed(chunks)  # numpy array (N, d)
    print(f"[+] Embeddings shape: {embs.shape}")

    print("[+] Building FAISS index ...")
    index = build_faiss_index(embs, use_cosine=True)

    print(f"[+] Saving FAISS index to {args.index_path}")
    save_faiss_index(index, args.index_path)

    print(f"[+] Saving metadata to {args.meta_path}")
    with open(args.meta_path, "w", encoding="utf-8") as f:
        json.dump({"metas": metas, "chunks": chunks}, f)

    print("[+] Build complete.")

def action_query(args):
    if not os.path.exists(args.index_path) or not os.path.exists(args.meta_path):
        print("Index or metadata not found. Run build first.")
        return

    print("[+] Loading FAISS index ...")
    index = load_faiss_index(args.index_path)

    print("[+] Loading metadata ...")
    with open(args.meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metas = data["metas"]
    chunks = data["chunks"]

    # choose embedder
    if args.embedder == "local":
        emb = LocalEmbedder(model_name=args.local_model)
    elif args.embedder == "openai":
        emb = OpenAIEmbedder(model=args.openai_model)
    else:
        raise ValueError("embedder must be 'local' or 'openai'")

    print("\n=== Enter interactive query mode (type 'exit' to quit) ===")

    try:
        while True:
            q = input("\nQuery> ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                break

            # 1️⃣ FAISS retrieval
            ids, scores = retrieve(index, emb.embed, q, k=args.k)

            print("\nTop chunks returned (idx, score, source, preview):")

            retrieved_texts = []

            for rank, (i, s) in enumerate(zip(ids, scores), 1):
                if i < 0:
                    continue

                meta = metas[i]
                raw_txt = chunks[i]

                # 2️⃣ Extract precise command (post-retrieval parsing)
                clean_txt = extract_command(raw_txt)
                retrieved_texts.append(clean_txt)

                print(
                    f"{rank}. id={i}  score={float(s):.4f}  "
                    f"source={meta['source']}  preview='{meta['text_preview']}'"
                )

            # Optional: show retrieved chunks
            show_more = input("Show full retrieved chunks? (y/N) ").strip().lower()
            if show_more == "y":
                for idx, t in enumerate(retrieved_texts, 1):
                    print(f"\n--- chunk {idx} ---\n{t}\n")

            # 3️⃣ LLM re-ranking (PRECISION step)
            if args.use_llm:
                print("\n[+] Re-ranking retrieved chunks to find the most precise answer...")
                best_answer = rerank_with_llm(q, retrieved_texts)

                print("\n=== Final Precise Answer ===\n")
                print(best_answer)

    except KeyboardInterrupt:
        print("\nExiting interactive mode.")

def get_arg_parser():
    p = argparse.ArgumentParser(description="All-in-one RAG helper (build & query FAISS index)")
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("build", help="Build FAISS index from documents")
    b.add_argument("--data_dir", required=True)
    b.add_argument("--index_path", default="faiss.index")
    b.add_argument("--meta_path", default="meta.json")
    b.add_argument("--embedder", choices=["local", "openai"], default="local")
    b.add_argument("--local_model", default="all-MiniLM-L6-v2")
    b.add_argument("--openai_model", default="text-embedding-3-small")
    b.add_argument("--chunk_size", type=int, default=1000)
    b.add_argument("--overlap", type=int, default=200)

    q = sub.add_parser("query", help="Query the index interactively")
    q.add_argument("--index_path", default="faiss.index")
    q.add_argument("--meta_path", default="meta.json")
    q.add_argument("--embedder", choices=["local", "openai"], default="local")
    q.add_argument("--local_model", default="all-MiniLM-L6-v2")
    q.add_argument("--openai_model", default="text-embedding-3-small")
    q.add_argument("--k", type=int, default=4)
    q.add_argument("--use_llm", action="store_true", help="Call OpenAI chat to produce a final answer using retrieved chunks")
    q.add_argument("--llm_model", default="gpt-4o-mini")
    q.add_argument("--max_tokens", type=int, default=300)

    return p

# ----------------------------
# Entry point
# ----------------------------
def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    if args.cmd == "build":
        action_build(args)
    elif args.cmd == "query":
        action_query(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

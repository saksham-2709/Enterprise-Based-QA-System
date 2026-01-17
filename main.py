import json
import argparse

from faiss_index import build_index, save_index, load_index
from ingestion.loader import load_documents_from_folder
from ingestion.chunker import doc_to_chunks
from ingestion.embedder import LocalEmbedder, OpenAIEmbedder

from orchestration.pipeline import RAGPipeline


def build(args):
    docs = load_documents_from_folder(args.data_dir)
    chunks, metas = doc_to_chunks(docs)

    embedder = LocalEmbedder() if args.embedder == "local" else OpenAIEmbedder()
    embeddings = embedder.embed(chunks)

    index = build_index(embeddings)
    save_index(index, args.index_path)

    with open(args.meta_path, "w") as f:
        json.dump({"chunks": chunks, "metas": metas}, f)

    print("Index built successfully.")


def parse_mode_and_query(user_input: str):
    if ":" in user_input:
        mode, query = user_input.split(":", 1)
        return mode.strip().lower(), query.strip()
    return "decision", user_input.strip()


def query(args):
    index = load_index(args.index_path)
    data = json.load(open(args.meta_path))

    pipeline = RAGPipeline(
        index=index,
        chunks=data["chunks"],
        metas=data["metas"],
        embedder_type=args.embedder,
        k=args.k
    )

    print("\n=== Interactive Mode (type exit to quit) ===")

    while True:
        user_input = input("\nQuery> ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        mode, query = parse_mode_and_query(user_input)
        if user_input == query:
            mode = None  # Let the pipeline guess the mode

        result = pipeline.run(query, mode=mode)
        rmode = result.get("mode")
        if result.get("status") == "REFUSED":
            print(result["message"])
            continue

        if rmode == "decision":
            print("\n=== Decision ===")
            print(result["decision"])

            print("\nReasons:")
            for r in result["reasons"]:
                print("-", r)

            print(f"\nConfidence: {result['confidence']}")

        elif rmode == "qa":
            print("\n=== Answer ===")
            print(result["answer"])
            print(f"\nConfidence: {result['confidence']}")

        elif rmode == "search":
            print("\n=== Search Results ===")
            for r in result["results"]:
                print(f"- ({r['score']:.3f}) {r['source']}")
                print(f"  {r['text']}...")



def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    b = sub.add_parser("build")
    b.add_argument("--data_dir", required=True)
    b.add_argument("--index_path", default="faiss.index")
    b.add_argument("--meta_path", default="meta.json")
    b.add_argument("--embedder", default="local")

    q = sub.add_parser("query")
    q.add_argument("--index_path", default="faiss.index")
    q.add_argument("--meta_path", default="meta.json")
    q.add_argument("--embedder", default="local")
    q.add_argument("--k", type=int, default=4)

    args = parser.parse_args()

    if args.cmd == "build":
        build(args)
    elif args.cmd == "query":
        query(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

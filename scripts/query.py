import argparse, json
from rag.store import VectorStore
from rag.retriever import Retriever
from rag.rerank import Reranker
from rag import config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("--topk", type=int, default=config.TOPK)
    ap.add_argument("--rerank-topk", type=int, default=config.RERANK_TOPK)
    args = ap.parse_args()
    q = " ".join(args.question)

    store = VectorStore(config.INDEX_DIR); store.load()
    retriever = Retriever(store)
    reranker = Reranker()
    initial = retriever.query(q, topk=args.topk)
    final = reranker.rerank(q, initial, topk=args.rerank_topk)

    print("\n=== ANSWER (stitched from top hits) ===\n")
    for i, r in enumerate(final, 1):
        print(f"[{i}] {r['path']}:{r['start_line']}-{r['end_line']}")
        print(r["text"].strip())
        print()

    print("=== JSON ===")
    print(json.dumps(final, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

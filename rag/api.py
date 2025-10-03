from fastapi import FastAPI, Query
from pydantic import BaseModel
from .store import VectorStore
from .retriever import Retriever
from .rerank import Reranker
from . import config

app = FastAPI(title="AOSP RAG PoC")

store = VectorStore(config.INDEX_DIR)
store.load()
retriever = Retriever(store)
reranker = Reranker()

class Answer(BaseModel):
    answer: str
    citations: list

@app.get("/search")
def search(q: str = Query(...), topk: int = config.TOPK, rerank_topk: int = config.RERANK_TOPK):
    initial = retriever.query(q, topk=topk)
    reranked = reranker.rerank(q, initial, topk=rerank_topk)
    # Simple stitched answer (no LLM step for PoC)
    answer_lines = []
    cites = []
    for r in reranked:
        path = r["path"]; s=r["start_line"]; e=r["end_line"]
        answer_lines.append(r["text"])
        cites.append({"path": path, "lines": [s,e]})
    return {"answer": "\n\n---\n\n".join(answer_lines), "citations": cites}

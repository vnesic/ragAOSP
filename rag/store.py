from pathlib import Path
import faiss, numpy as np
from rank_bm25 import BM25Okapi
import json, os

class VectorStore:
    def __init__(self, index_dir: str):
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.faiss_index = None
        self.embeddings = None
        self.meta = []   # [{path, start_line, end_line}]
        self.bm25 = None
        self.tokenized = None

    def save(self):
        faiss.write_index(self.faiss_index, str(self.dir / "vectors.faiss"))
        np.save(self.dir / "emb.npy", self.embeddings)
        with open(self.dir / "meta.json","w",encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False)
        # BM25
        with open(self.dir / "bm25.json","w",encoding="utf-8") as f:
            json.dump({"corpus":[m["text"] for m in self.meta]}, f)

    def load(self):
        self.faiss_index = faiss.read_index(str(self.dir / "vectors.faiss"))
        self.embeddings = np.load(self.dir / "emb.npy")
        with open(self.dir / "meta.json","r",encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(self.dir / "bm25.json","r",encoding="utf-8") as f:
            corpus = json.load(f)["corpus"]
        self.tokenized = [c.split() for c in corpus]
        self.bm25 = BM25Okapi(self.tokenized)

    def build(self, vectors: np.ndarray, metas: list, bm25_corpus: list):
        self.embeddings = vectors.astype("float32")
        index = faiss.IndexFlatIP(self.embeddings.shape[1])
        faiss.normalize_L2(self.embeddings)
        index.add(self.embeddings)
        self.faiss_index = index
        self.meta = metas
        self.tokenized = [c.split() for c in bm25_corpus]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, q_vec, q_text: str, topk: int = 12, alpha: float = 0.6):
        # dense
        qv = q_vec.reshape(1,-1).astype("float32")
        faiss.normalize_L2(qv)
        sims, ids = self.faiss_index.search(qv, topk*3)
        dense = {int(i): float(s) for i,s in zip(ids[0], sims[0])}

        # sparse
        scores = self.bm25.get_scores(q_text.split())
        sparse = {i: float(s) for i,s in enumerate(scores)}

        # fuse
        keys = set(dense.keys()) | set(sparse.keys())
        fused = [(i, alpha*dense.get(i,0.0)+(1-alpha)*sparse.get(i,0.0)) for i in keys]
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:topk]

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from . import config

class Reranker:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(config.RERANK_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.RERANK_MODEL)

    @torch.no_grad()
    def rerank(self, query: str, passages: list, topk: int):
        pairs = [(query, p["text"]) for p in passages]
        batch = self.tok.batch_encode_plus(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
        scores = self.model(**batch).logits.squeeze(-1)
        order = torch.argsort(scores, descending=True).tolist()
        return [passages[i] | {"rerank": float(scores[i])} for i in order[:topk]]

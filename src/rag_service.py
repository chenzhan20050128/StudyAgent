from typing import Any, Dict, List, Optional

from pymilvus import AnnSearchRequest, WeightedRanker
from sqlalchemy.orm import Session

from .llm_client import LLMClient
from .models import RAGQueryLog
from .vector_store import MilvusVectorStore


class RAGService:
    def __init__(self, vec: MilvusVectorStore, llm: LLMClient) -> None:
        self._vec = vec
        self._llm = llm

    def query(
        self,
        db: Session,
        user_id: int,
        question: str,
        doc_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        dense = self._llm.embed([question])[0]
        filter_expr = f'user_id == "{user_id}"'
        if doc_ids is not None:
            safe_doc_ids = sorted({int(d) for d in doc_ids})
            if len(safe_doc_ids) == 0:
                return {"answer": "没有匹配到符合条件的文档。", "sources": []}
            # Milvus 表达式 `in` 需要使用列表语法，如 [84, 85]
            doc_ids_expr = "[" + ", ".join(str(d) for d in safe_doc_ids) + "]"
            filter_expr += f" and doc_id in {doc_ids_expr}"

        dense_req = AnnSearchRequest(
            data=[dense],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=50,
            expr=filter_expr,
        )
        sparse_req = AnnSearchRequest(
            data=[question],
            anns_field="sparse_bm25",
            param={"metric_type": "BM25"},
            limit=50,
            expr=filter_expr,
        )
        ranker = WeightedRanker(0.8, 0.2)
        res = self._vec.client.hybrid_search(
            collection_name=self._vec.collection,
            reqs=[dense_req, sparse_req],
            ranker=ranker,
            limit=50,
            output_fields=["doc_id", "content", "section_title"],
        )
        hits = res[0] if res else []
        docs = [h["entity"]["content"] for h in hits]
        rerank = self._llm.rerank(question, docs, top_n=10)
        picked = []
        for item in rerank:
            ent = hits[item["index"]]["entity"]
            picked.append(
                {
                    "doc_id": ent["doc_id"],
                    "section_title": ent["section_title"],
                    "content": ent["content"],
                    "score": item["relevance_score"],
                }
            )

        context = ""
        for i, c in enumerate(picked, 1):
            header = (
                f"\n--- 片段{i} (doc_id={c['doc_id']}, "
                f"标题={c['section_title']}) ---\n"
            )
            context += header + f"{c['content']}\n"
        prompt = (
            "你是学习助手，请基于给定片段回答用户问题。如果依据不足也尽量回答\n"
            f"片段:\n{context}\n\n问题:{question}"
        )
        answer = self._llm.chat(prompt)
        log = RAGQueryLog(
            user_id=user_id,
            query=question,
            answer=answer,
            meta=None,
        )
        db.add(log)
        return {"answer": answer, "sources": picked}

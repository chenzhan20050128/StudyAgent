from typing import Any, Dict, List, Optional

from pymilvus import AnnSearchRequest, WeightedRanker
from sqlalchemy.orm import Session

from .llm_client import LLMClient
from .models import RAGQueryLog
from .vector_store import MilvusVectorStore


class RAGService:
    def __init__(self, vec: MilvusVectorStore, llm: LLMClient) -> None:
        # vec: 向量库（hybrid_search/collection/client）
        # llm: 大模型客户端（embedding、rerank、chat）
        self._vec = vec
        self._llm = llm

    def query(
        self,
        db: Session,
        user_id: int,
        question: str,
        doc_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """RAG 问答主入口。

        业务链路：
        1) 将问题 embedding 成稠密向量
        2) 构造 dense + sparse(BM25) 两路检索请求，并按权重融合得到候选片段（top50）
        3) 对候选片段做 rerank 精排，挑 top10 作为最终上下文
        4) 拼接上下文后调用 chat 模型生成答案
        5) 将 query/answer 写入 rag_queries 日志表，便于追踪与复盘
        """
        dense = self._llm.embed([question])[0]
        # 过滤表达式：按 user_id 做数据隔离；doc_ids 可选做白名单预过滤
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
        # 加权融合：dense 偏语义，sparse 偏关键词命中
        ranker = WeightedRanker(0.8, 0.2)
        res = self._vec.client.hybrid_search(
            collection_name=self._vec.collection,
            reqs=[dense_req, sparse_req],
            ranker=ranker,
            limit=50,
            output_fields=["doc_id", "content", "section_title"],
        )
        hits = res[0] if res else []
        # 候选内容列表：供 rerank 使用（只传文本，降低网络负载）
        docs = [h["entity"]["content"] for h in hits]
        # rerank：把“召回正确但排序不稳”的候选进一步精排，提升最终上下文质量
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

        # 将 top10 片段拼成上下文，并带上 doc_id/标题便于溯源
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

        # 生成式回答：模型只看到“片段上下文 + 问题”，天然具备跨文档聚合能力
        answer = self._llm.chat(prompt)

        # 落库：保留问答日志，便于统计与调试（可扩展存检索元信息 meta）
        log = RAGQueryLog(
            user_id=user_id,
            query=question,
            answer=answer,
            meta=None,
        )
        db.add(log)
        return {"answer": answer, "sources": picked}

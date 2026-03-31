from typing import Any, Dict, List, Optional

from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    WeightedRanker,
)

from .config import settings


class MilvusVectorStore:
    """封装 Milvus：collection 定义 + 插入 + 混合检索"""

    def __init__(self) -> None:
        self.client = MilvusClient(uri=settings.milvus.uri)
        self.collection = settings.milvus.collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.has_collection(self.collection):
            return

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(
            "chunk_id",
            DataType.VARCHAR,
            is_primary=True,
            max_length=128,
        )
        schema.add_field("user_id", DataType.BOOL)
        schema.add_field("doc_id", DataType.INT64)
        schema.add_field(
            "content",
            DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
            enable_match=True,
        )
        schema.add_field("section_title", DataType.VARCHAR, max_length=256)
        schema.add_field("metadata", DataType.JSON)
        schema.add_field(
            "embedding",
            DataType.FLOAT_VECTOR,
            dim=settings.milvus.embedding_dim,
        )
        schema.add_field("sparse_bm25", DataType.SPARSE_FLOAT_VECTOR)

        bm25 = Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names="sparse_bm25",
        )
        schema.add_function(bm25)

        index = self.client.prepare_index_params()
        index.add_index(
            "embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 64},
        )
        index.add_index(
            "sparse_bm25",
            index_type="SPARSE_WAND",
            metric_type="BM25",
        )

        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index,
        )

    def insert(self, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        res = self.client.insert(self.collection, rows)
        return int(res.get("insert_count", 0))

    def hybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        user_id: int | str,
        doc_ids: Optional[List[int]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """混合检索，返回带 chunk_id 的命中结果。"""

        filter_expr = f'user_id == "{user_id}"'
        if doc_ids:
            filter_expr += f" and doc_id in {tuple(doc_ids)}"

        dense_req = AnnSearchRequest(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=limit,
        )
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse_bm25",
            param={"metric_type": "BM25"},
            limit=limit,
        )
        ranker = WeightedRanker(0.8, 0.2)
        result = self.client.hybrid_search(
            collection_name=self.collection,
            reqs=[dense_req, sparse_req],
            ranker=ranker,
            limit=limit,
            filter=filter_expr,
            output_fields=[
                "chunk_id",
                "doc_id",
                "content",
                "section_title",
                "metadata",
            ],
        )
        hits = result[0] if result else []
        parsed: List[Dict[str, Any]] = []
        for h in hits:
            ent = h.get("entity", {})
            parsed.append(
                {
                    "chunk_id": ent.get("chunk_id"),
                    "doc_id": ent.get("doc_id"),
                    "section_title": ent.get("section_title"),
                    "content": ent.get("content"),
                    "metadata": ent.get("metadata") or {},
                    "score": h.get("score"),
                }
            )
        return parsed

    def close(self) -> None:
        self.client.close()

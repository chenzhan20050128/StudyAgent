import os
import uuid
from dataclasses import dataclass
from typing import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from .llm_client import LLMClient
from .models import Document, DocumentTag
from .parsers import DocumentParser, WebPageParser
from .vector_store import MilvusVectorStore


@dataclass
class ParsedDocument:
    doc: Document
    chunks: Sequence[str]


class DocumentService:
    """文档上传 / 解析 / 入库 + 写向量库"""

    def __init__(
        self,
        base_dir: str,
        splitter: RecursiveCharacterTextSplitter,
    ) -> None:
        self._base_dir = base_dir
        self._splitter = splitter
        self._doc_parser = DocumentParser(chunk_size=800, overlap=200)
        self._web_parser = WebPageParser()

    def _ensure_user_dir(self, user_id: int) -> str:
        path = os.path.join(self._base_dir, str(user_id))
        os.makedirs(path, exist_ok=True)
        return path

    def create_document(
        self,
        db: Session,
        user_id: int,
        title: str,
        source_type: str,
        file_path: str | None = None,
        source_url: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> Document:
        doc = Document(
            user_id=user_id,
            title=title,
            source_type=source_type,
            file_path=file_path,
            source_url=source_url,
            status="uploaded",
        )
        db.add(doc)
        db.flush()

        if tags:
            for t in tags:
                db.add(DocumentTag(doc_id=doc.id, tag=t))
        return doc

    def save_uploaded_file(
        self,
        user_id: int,
        filename: str,
        data: bytes,
    ) -> str:
        user_dir = self._ensure_user_dir(user_id)
        ext = os.path.splitext(filename)[1]
        name = f"{uuid.uuid4().hex}{ext}"
        path = os.path.join(user_dir, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def parse_file_to_text(self, path: str) -> str:
        """将 pdf/word/txt 统一转为纯文本。"""
        ext = os.path.splitext(path)[1].lower()
        if ext in {".pdf", ".doc", ".docx"}:
            return self._doc_parser.parse(path)
        # 其他情况按纯文本读取（例如 .txt / .md 等）
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def parse_url_to_text(self, url: str) -> str:
        """使用 WebPageParser 把网页正文抽取出来。"""
        return self._web_parser.parse(url)

    def parse_and_chunk(self, doc: Document, raw_text: str) -> ParsedDocument:
        chunks = self._splitter.split_text(raw_text)
        return ParsedDocument(doc=doc, chunks=chunks)

    def index_to_milvus(
        self,
        vec: MilvusVectorStore,
        llm: LLMClient,
        parsed: ParsedDocument,
    ) -> int:
        texts = list(parsed.chunks)
        embeds = llm.embed(texts)
        rows = []
        for i, (t, e) in enumerate(zip(texts, embeds)):
            rows.append(
                {
                    "chunk_id": uuid.uuid4().hex,
                    "user_id": str(parsed.doc.user_id),
                    "doc_id": parsed.doc.id,
                    "content": t,
                    "section_title": f"{parsed.doc.title}-{i+1}",
                    "metadata": {"len": len(t)},
                    "embedding": e,
                }
            )
        return vec.insert(rows)

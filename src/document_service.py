import json
import os
import random
import uuid
from dataclasses import dataclass
from typing import Iterable, Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from .llm_client import LLMClient
from .models import Document, DocumentTag
from .parsers import DocumentParser, WebPageParser
from .vector_store import MilvusVectorStore


@dataclass
class ParsedDocument:
    # 解析后的中间态：
    # - doc: 数据库中的 Document 元数据对象（含 doc_id/user_id 等）
    # - chunks: 对原始纯文本进行分块后的片段序列（后续用于 embedding/写 Milvus/打标抽样）
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
        # 注意：DocumentParser 当前只负责把 pdf/word 转成纯文本；
        # 真正的“分块”由 self._splitter（LangChain TextSplitter）统一完成。
        self._doc_parser = DocumentParser(chunk_size=800, overlap=200)
        # 网页解析器：内置微信/知乎/通用网页的不同正文抽取策略
        self._web_parser = WebPageParser()

    def _ensure_user_dir(self, user_id: int) -> str:
        # 用户私有目录：不同 user_id 保存到不同子目录，避免文件名冲突与跨用户混用
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
        # 创建一条 documents 元数据记录（先入库拿到 doc_id，后续分块/向量库写入都靠它关联）
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
            # tags 持久化：写入 document_tags 表（多对多/一对多关系）
            for t in tags:
                norm = self._normalize_tag(t)
                if norm:
                    db.add(DocumentTag(doc_id=doc.id, tag=norm))
        return doc

    def save_uploaded_file(
        self,
        user_id: int,
        filename: str,
        data: bytes,
    ) -> str:
        # 上传文件落盘：保留扩展名，文件名用 uuid 防止冲突
        user_dir = self._ensure_user_dir(user_id)
        ext = os.path.splitext(filename)[1]
        name = f"{uuid.uuid4().hex}{ext}"
        path = os.path.join(user_dir, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def parse_file_to_text(self, path: str) -> str:
        """将 pdf/word/txt 统一转为纯文本。"""
        # 解析策略：
        # - pdf/word 走专用解析器（抽取正文、表格等）
        # - 其他（txt/md 等）按纯文本读取，忽略编码错误
        ext = os.path.splitext(path)[1].lower()
        if ext in {".pdf", ".doc", ".docx"}:
            return self._doc_parser.parse(path)
        # 其他情况按纯文本读取（例如 .txt / .md 等）
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def parse_url_to_text(self, url: str) -> str:
        """使用 WebPageParser 把网页正文抽取出来。"""
        # WebPageParser 内部会根据域名选择解析策略（微信/知乎/通用网页）
        return self._web_parser.parse(url)

    def parse_and_chunk(self, doc: Document, raw_text: str) -> ParsedDocument:
        # 分块策略交给上层统一配置的 TextSplitter：
        # - chunk_size / chunk_overlap 在 main.py 初始化
        # - 这里不关心文本来源（pdf/word/url/笔记），只要 raw_text 即可
        chunks = self._splitter.split_text(raw_text)
        return ParsedDocument(doc=doc, chunks=chunks)

    # ===== 共享/复制相关 =====

    def _rebuild_raw_text(self, doc: Document) -> str | None:
        """根据已有存储重新获得文档原始文本。

        优先使用本地文件，其次是可抓取的 URL，最后退回 source_url 文本本身。
        若全部缺失，则返回 None 以便上层标记失败。
        """
        # 这是“共享/复制(Fork)”能力的关键：目标用户需要一份独立的向量数据。
        # 因此系统会尽量复原源文档的原始文本，再重新分块/embedding/写入 Milvus。

        if doc.file_path and os.path.exists(doc.file_path):
            try:
                return self.parse_file_to_text(doc.file_path)
            except Exception:
                return None

        if doc.source_url:
            if doc.source_url.startswith(("http://", "https://")):
                try:
                    return self.parse_url_to_text(doc.source_url)
                except Exception:
                    return None
            # 旧实现会把非 http 文本直接塞在 source_url 字段里，这里复用即可
            return doc.source_url

        return None

    def fork_document_to_user(
        self,
        db: Session,
        vec: MilvusVectorStore,
        llm: LLMClient,
        src_doc: Document,
        target_user_id: int,
    ) -> dict:
        """复制单个文档到目标用户，返回执行结果。

        返回结构：
        {
            "from_doc_id": ...,
            "to_doc_id": ...,
            "status": "parsed" | "failed",
            "reason": ...,
        }
        """
        # Fork 复制策略（安全分享的核心设计）：
        # - documents 表里创建一条新记录（归属 target_user_id）
        # - 同步复制标签
        # - 重新构建 raw_text -> 分块 -> embedding -> 写入 Milvus
        # 这样目标用户拥有“独立向量资产”，避免跨用户共用同一份向量数据带来的权限风险。

        new_doc = Document(
            user_id=target_user_id,
            title=src_doc.title,
            source_type=src_doc.source_type,
            file_path=src_doc.file_path,
            source_url=src_doc.source_url,
            status="processing",
        )
        db.add(new_doc)
        db.flush()  # 获取新 doc_id

        # 同步复制标签
        tags = (
            db.query(DocumentTag)
            .filter(DocumentTag.doc_id == src_doc.id)
            .order_by(DocumentTag.id.asc())
            .all()
        )
        for t in tags:
            db.add(DocumentTag(doc_id=new_doc.id, tag=t.tag))

        result: dict = {
            "from_doc_id": src_doc.id,
            "to_doc_id": new_doc.id,
            "status": "processing",
        }

        raw_text = self._rebuild_raw_text(src_doc)
        if not raw_text:
            # 找不到内容：可能源文件丢失、URL 无法抓取或源记录缺失
            new_doc.status = "failed"
            db.add(new_doc)
            result.update({"status": "failed", "reason": "content_not_found"})
            return result

        try:
            # 复制链路的“重建式入库”：分块与写向量库
            parsed = self.parse_and_chunk(new_doc, raw_text)
            self.index_to_milvus(vec, llm, parsed)
            new_doc.status = "parsed"
            db.add(new_doc)
            result["status"] = "parsed"
        except Exception as exc:  # noqa: BLE001
            # 任一步骤异常都视为复制失败，但不影响其它文档的 fork 继续执行
            new_doc.status = "failed"
            db.add(new_doc)
            result.update({"status": "failed", "reason": str(exc)})

        return result

    # ===== 标签相关业务逻辑 =====

    @staticmethod
    def _normalize_tag(tag: str | None) -> str | None:
        """标签规范化：去除首尾空格、处理全角空格、限制长度，并按英文字符转小写。"""
        # 目标：
        # - 过滤噪声标签（太短/太长）
        # - 统一中英文空格与大小写
        # - 便于去重与检索
        if tag is None:
            return None
        t = str(tag).replace("\u3000", " ").strip()
        if not t:
            return None
        # 过滤过短或过长的标签（通常是噪声），建议长度 2~10
        if len(t) < 2 or len(t) > 10:
            return None
        if all(ord(c) < 128 for c in t):
            t = t.lower()
        return t

    def _dedup_tags(self, tags: Iterable[str]) -> list[str]:
        """对传入的标签列表进行规范化并去重，同时保留出现顺序。"""
        # “保序去重”：让用户手工输入的顺序尽量被保留
        seen: set[str] = set()
        result: list[str] = []
        for t in tags:
            norm = self._normalize_tag(t)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            result.append(norm)
        return result

    def _build_tagging_sample(
        self,
        chunks: Sequence[str],
        front_k: int = 3,
        random_k: int = 2,
        max_chars: int = 8000,
    ) -> str:
        """为打标构建文本样本：通过抓取文档的前几块以及随机抽取几块，确保覆盖全篇主题。"""
        # 抽样策略：
        # - 短文档：全取
        # - 长文档：取开头 front_k + 从剩余随机抽 random_k
        # 目的：控制 prompt 长度，同时尽量覆盖文档主题分布
        if not chunks:
            return ""
        chunks_list = list(chunks)
        if len(chunks_list) <= 5:
            selected = chunks_list
        else:
            selected = list(chunks_list[:front_k])
            remain = chunks_list[front_k:]
            if remain and random_k > 0:
                # 使用 chunk 数作为随机种子，使相同文档打标相对稳定
                rnd = random.Random(len(chunks_list))
                k = min(random_k, len(remain))
                selected.extend(rnd.sample(remain, k))
        sample = "\n\n".join(selected)
        return sample[:max_chars]

    _TAG_SCHEMA = {
        "name": "auto_tags",
        "schema": {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 3,
                }
            },
            "required": ["tags"],
            "additionalProperties": False,
        },
        "strict": False,
    }

    def _generate_ai_tags(self, llm: LLMClient, parsed: ParsedDocument) -> list[str]:
        """通过 LLM 分析文档采样内容，自动提取 2~3 个核心主题标签。"""
        # 这里使用 json_schema 约束 LLM 输出为结构化 JSON，避免“自由发挥”导致解析失败
        sample_text = self._build_tagging_sample(parsed.chunks)
        if not sample_text:
            return []

        prompt = (
            "你是一个学习资料分类助手。\n"
            "下面是用户一篇学习资料的部分内容，请仅基于这些内容为整篇资料生成 2~3 个核心标签。\n"
            "要求：\n"
            "- 标签应能概括资料的学科或主题，例如：计算机网络、Python、操作系统、考研 408 等。\n"
            "- 每个标签不超过 6 个汉字，或不超过 4 个英语单词。\n"
            "- 不要输出'文章'、'资料'、'文件'这类没有意义的标签。\n"
            "- 只在你确信与内容相关时才输出标签，不要胡乱猜测。\n"
            "输出格式：必须使用约定好的 JSON schema。\n"
            "---------------- 内容片段开始 ----------------\n"
            f"{sample_text}\n"
            "---------------- 内容片段结束 ----------------"
        )

        try:
            raw = llm.chat_with_json_schema(prompt, self._TAG_SCHEMA)
            data = json.loads(raw)
            tags = data.get("tags") or []
            return [t for t in tags if isinstance(t, str)]
        except Exception:
            # AI 打标失败不应中断处理流程
            return []

    def auto_tag_document(
        self,
        db: Session,
        llm: LLMClient,
        parsed: ParsedDocument,
        manual_tags: Sequence[str] | None = None,
    ) -> list[str]:
        """对文档进行自动打标：整合已有标签、手动输入与 AI 生成标签，并写入数据库持久化。"""
        # 合并策略：
        # - existing：历史已存在的标签
        # - manual：本次上传/调用传入的手动标签
        # - ai_tags：从文本采样得到的 LLM 标签
        # 最终会规范化、去重，并限制总量，防止标签泛滥
        existing_rows = (
            db.query(DocumentTag)
            .filter(DocumentTag.doc_id == parsed.doc.id)
            .order_by(DocumentTag.id.asc())
            .all()
        )
        existing = [row.tag for row in existing_rows]
        manual = list(manual_tags or [])

        ai_tags = self._generate_ai_tags(llm, parsed)

        merged = self._dedup_tags([*existing, *manual, *ai_tags])
        # 文档级标签保持精炼，只记录前 6 个
        final_tags = merged[:6]

        existing_norm = {
            self._normalize_tag(t) for t in existing if self._normalize_tag(t)
        }
        for t in final_tags:
            norm = self._normalize_tag(t)
            if not norm or norm in existing_norm:
                continue
            # 创建新的多对多关系记录
            db.add(DocumentTag(doc_id=parsed.doc.id, tag=norm))
            existing_norm.add(norm)

        return final_tags

    def resolve_doc_ids_by_tags(
        self, db: Session, user_id: int, tags: Sequence[str]
    ) -> list[int]:
        """将选定的若干标签反查为其关联的所有文档 ID，作为 RAG 检索的预过滤条件。"""
        # 典型用法：前端选标签 -> 后端先把标签映射为 doc_ids -> 再把 doc_ids 作为 Milvus 过滤条件
        normalized = self._dedup_tags(tags)
        if not normalized:
            return []
        rows = (
            db.query(DocumentTag.doc_id)
            .join(Document, Document.id == DocumentTag.doc_id)
            .filter(
                Document.user_id == user_id,
                DocumentTag.tag.in_(normalized),
            )
            .all()
        )
        # 去重并排序返回文档 ID 集合
        doc_ids = {int(r[0]) for r in rows}
        return sorted(doc_ids)

    def index_to_milvus(
        self,
        vec: MilvusVectorStore,
        llm: LLMClient,
        parsed: ParsedDocument,
    ) -> int:
        # 向量化入库：
        # - texts: 文本 chunk
        # - embeds: 由 embedding 模型生成的稠密向量
        # - 每条记录写入 Milvus，并携带 doc_id/user_id 等元信息用于过滤与溯源
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

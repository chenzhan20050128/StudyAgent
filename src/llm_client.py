"""LLM 客户端封装：统一 embedding / rerank / chat / 结构化输出。

需求
- 提供一个轻量、可复用的 LLM 访问层，供 RAG/计划/测验/意图识别等模块调用。
- 兼容 DashScope 的 OpenAI-compatible 接口（base_url + API Key 环境变量）。
- 降低成本与延迟：对 embedding 做本地缓存，并对远程请求做必要的重试/限流。

业务逻辑
- embedding：输入一批文本，优先走本地 sha256 缓存；未命中部分再按批次请求远程 embedding。
- rerank：对召回候选片段做相关性重排，给上层返回更靠前的片段（用于 RAG/出题）。
- chat：对话生成/结构化输出（上层可传入 JSON Schema 约束，减少格式漂移）。

主要实现方式
- 使用官方 OpenAI Python SDK（通过 base_url 指向 DashScope compatible-mode）。
- embedding_cache.json：落盘在项目根目录，用“文本 hash -> 向量”做缓存。
- 关键配置来源于 `src/config.py` 的 settings，并通过环境变量注入密钥。
"""

import hashlib
import json
import os
import logging
import time
from pathlib import Path
from typing import Dict, List

import requests
from openai import OpenAI

from .config import settings


EMBEDDING_BATCH_SIZE = 10


class LLMClient:
    def __init__(self) -> None:
        # LLM/API Key 读取：使用环境变量，避免把密钥写进代码或仓库
        api_key = os.getenv(settings.llm.api_key_env)
        if not api_key:
            raise RuntimeError(f"env {settings.llm.api_key_env} not set")
        self._client = OpenAI(api_key=api_key, base_url=settings.llm.base_url)

        self._logger = logging.getLogger(__name__)

        # embedding 缓存：
        # - 以文本内容 sha256 做 key
        # - 落盘到项目根目录 embedding_cache.json
        # 目的：降低重复入库/重复检索的 embedding 成本与时延
        root = Path(__file__).resolve().parents[2]
        self._cache_path = root / "embedding_cache.json"
        self._embedding_cache: Dict[str, List[float]] = self._load_embedding_cache()

    def _load_embedding_cache(self) -> Dict[str, List[float]]:
        if not self._cache_path.exists():
            return {}
        try:
            with self._cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # 确保 key 为 str，value 为 list[float]
            return {str(k): v for k, v in data.items()}
        except Exception:
            # 读取失败时忽略旧缓存，重新开始
            return {}

    def _save_embedding_cache(self) -> None:
        try:
            with self._cache_path.open("w", encoding="utf-8") as f:
                json.dump(self._embedding_cache, f)
        except Exception:
            # 缓存落盘失败不影响主流程
            return

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

    def embed(self, texts: List[str]) -> List[List[float]]:
        """对一批文本做向量化，带本地缓存且遵守 batch<=10 的限制。"""

        if not texts:
            return []

        # 先根据 hash 在缓存中命中已有向量：命中则跳过远程 embedding 请求
        hashes = [self._text_hash(t) for t in texts]
        vectors: List[List[float] | None] = [None] * len(texts)
        to_call_inputs: List[str] = []
        to_call_indices: List[int] = []

        cached_total = 0
        for idx, h in enumerate(hashes):
            cached = self._embedding_cache.get(h)
            if cached is not None:
                vectors[idx] = cached
                cached_total += 1
            else:
                to_call_indices.append(idx)
                to_call_inputs.append(texts[idx])

        # 如果本次全部命中缓存，也打一条日志，方便观察
        if not to_call_inputs:
            print(
                "[EMBED] all from cache, "
                f"cached_total={cached_total}, total_inputs={len(texts)}"
            )

        # 对需要远程调用的文本按批次处理，防止超过服务限制
        for i in range(0, len(to_call_inputs), EMBEDDING_BATCH_SIZE):
            batch_inputs = to_call_inputs[i : i + EMBEDDING_BATCH_SIZE]
            if not batch_inputs:
                continue
            # 日志：本批次 embedding 数量 & 缓存命中数量
            batch_total = len(batch_inputs)
            print(
                f"[EMBED] batch size={batch_total}, "
                f"cached_total={cached_total}, total_inputs={len(texts)}"
            )
            resp = self._client.embeddings.create(
                model=settings.llm.embedding_model,
                input=batch_inputs,
                dimensions=settings.milvus.embedding_dim,
            )
            batch_vectors = [d.embedding for d in resp.data]

            for j, vec in enumerate(batch_vectors):
                global_idx = to_call_indices[i + j]
                vectors[global_idx] = vec
                self._embedding_cache[hashes[global_idx]] = vec

        # 同步一次缓存到磁盘（失败不影响主流程）
        self._save_embedding_cache()

        # 类型断言：此时 vectors 中应该没有 None
        return [v for v in vectors if v is not None]

    def rerank(
        self,
        query: str,
        docs: List[str],
        top_n: int = 10,
    ) -> List[dict]:
        # rerank：把候选文本片段按 query 相关性重新排序。
        # 使用 DashScope 的 compatible rerank endpoint。
        if not docs:
            return []
        api_key = os.getenv(settings.llm.api_key_env)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.llm.rerank_model,
            "query": query,
            "documents": docs,
            "top_n": top_n,
        }
        r = requests.post(
            "https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
            headers=headers,
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("results", [])

    def chat(self, prompt: str) -> str:
        # 通用对话：用于 RAG 生成回答、计划/测验等自然语言输出
        t0 = time.perf_counter()
        prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
        self._logger.info(
            "llm.chat start model=%s prompt_len=%d preview=%s",
            settings.llm.chat_model,
            len(prompt),
            prompt_preview,
        )
        resp = self._client.chat.completions.create(
            model=settings.llm.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            extra_body={"enable_thinking": False},
        )
        content = resp.choices[0].message.content or ""
        self._logger.info(
            "llm.chat answer_full=%s",
            content,
        )
        self._logger.info(
            "llm.chat done model=%s cost=%.2fs output_len=%d",
            settings.llm.chat_model,
            time.perf_counter() - t0,
            len(content),
        )
        return content

    def chat_0_6B(self, prompt: str) -> str:
        """使用小模型 qwen3-0.6b 进行快速意图识别。"""
        # 小模型的目标：降低成本与延迟，用于 intent/slot 等轻量任务
        t0 = time.perf_counter()
        prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
        self._logger.info(
            "llm.chat_0_6B start model=qwen3-0.6b prompt_len=%d preview=%s",
            len(prompt),
            prompt_preview,
        )
        resp = self._client.chat.completions.create(
            model="qwen3-0.6b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            extra_body={"enable_thinking": False},
        )
        content = resp.choices[0].message.content or ""
        self._logger.info(
            "llm.chat_0_6B answer_full=%s",
            content,
        )
        self._logger.info(
            "llm.chat_0_6B done model=qwen3-0.6b cost=%.2fs output_len=%d",
            time.perf_counter() - t0,
            len(content),
        )
        return content

    def chat_with_json_schema(
        self,
        prompt: str,
        json_schema: dict,
    ) -> str:
        """调用 LLM 进行结构化输出（JSON Schema 模式）。

        Args:
            prompt: 提示词，需明确指示返回 JSON
            json_schema: JSON Schema 定义，定义输出结构

        Returns:
            符合 json_schema 的 JSON 字符串
        """
        # 结构化输出：用于
        # - 自动打标 tags
        # - 学习大纲 syllabus
        # - 测验出题 question
        # - 主观题批改 grading
        # 通过 schema 约束降低“输出不可解析”的风险
        # 部分服务商要求 prompt 中显式包含 “json” 以启用 json_schema/response_format
        if "json" not in prompt.lower():
            prompt = "请严格按 JSON 输出。" + prompt
        t0 = time.perf_counter()
        prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
        self._logger.info(
            "llm.chat_json_schema start model=%s prompt_len=%d preview=%s",
            settings.llm.chat_model,
            len(prompt),
            prompt_preview,
        )
        resp = self._client.chat.completions.create(
            model=settings.llm.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={  # type: ignore
                "type": "json_schema",
                "json_schema": json_schema,
            },
            extra_body={"enable_thinking": False},
        )
        content = resp.choices[0].message.content or ""
        self._logger.info(
            "llm.chat_json_schema prompt_full=%s",
            prompt,
        )
        self._logger.info(
            "llm.chat_json_schema answer_full=%s",
            content,
        )
        self._logger.info(
            "llm.chat_json_schema done model=%s cost=%.2fs output_len=%d",
            settings.llm.chat_model,
            time.perf_counter() - t0,
            len(content),
        )
        return content

    def chat_0_6B_with_json_schema(
        self,
        prompt: str,
        json_schema: dict,
    ) -> str:
        """使用小模型 qwen3-0.6b 进行结构化输出（JSON Schema 模式）。"""
        # 同 chat_with_json_schema，但使用小模型做低成本结构化输出
        # 部分服务商要求 prompt 中显式包含 "json" 以启用 json_schema/response_format
        if "json" not in prompt.lower():
            prompt = "请严格按 JSON 输出。" + prompt
        t0 = time.perf_counter()
        prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
        self._logger.info(
            "llm.chat_0_6B_with_json_schema start "
            "model=qwen3-0.6b prompt_len=%d preview=%s",
            len(prompt),
            prompt_preview,
        )
        resp = self._client.chat.completions.create(
            model="qwen3-0.6b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={  # type: ignore
                "type": "json_schema",
                "json_schema": json_schema,
            },
            extra_body={"enable_thinking": False},
        )
        content = resp.choices[0].message.content or ""
        self._logger.info(
            "llm.chat_0_6B_with_json_schema prompt_full=%s",
            prompt,
        )
        self._logger.info(
            "llm.chat_0_6B_with_json_schema answer_full=%s",
            content,
        )
        self._logger.info(
            "llm.chat_0_6B_with_json_schema done "
            "model=qwen3-0.6b cost=%.2fs output_len=%d",
            time.perf_counter() - t0,
            len(content),
        )
        return content

    def chat_stream(self, prompt: str):
        """调用 LLM 进行流式输出。

        Args:
            提示词

        Yields:
            流式数据块，每个块包含增量文本内容
        """
        t0 = time.perf_counter()
        prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
        self._logger.info(
            "llm.chat_stream start model=%s prompt_len=%d preview=%s",
            settings.llm.chat_model,
            len(prompt),
            prompt_preview,
        )

        try:
            completion = self._client.chat.completions.create(
                model=settings.llm.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                stream=True,
                stream_options={"include_usage": True},
                extra_body={"enable_thinking": False},
            )

            for chunk in completion:
                # 直接返回每个数据块的 JSON，方便调用方处理
                yield chunk.model_dump_json()

            self._logger.info(
                "llm.chat_stream done model=%s cost=%.2fs",
                settings.llm.chat_model,
                time.perf_counter() - t0,
            )
        except Exception as e:
            self._logger.error(
                "llm.chat_stream error model=%s error=%s",
                settings.llm.chat_model,
                str(e),
            )
            raise

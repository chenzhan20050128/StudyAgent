"""阶段三：测验生成与批改服务。

设计目标：
- 与阶段二保持一致：确定性工作流（QuizWorkflow）+ 服务层（QuizService）。
- 出题采用 Hybrid RAG：top50 检索 -> rerank top5。
- 送入 LLM 出题时使用 top5 的完整 chunk 内容（不截断）。
- generate 接口不暴露标准答案；标准答案存入 MySQL，submit 时读取评分。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, TypedDict, cast

from sqlalchemy.orm import Session

from .llm_client import LLMClient
from .models import Quiz, QuizAttempt, WeakPoint
from .review_service import ReviewService
from .vector_store import MilvusVectorStore

try:  # 可选：langgraph 可用时编排节点
    from langgraph.graph import END, START, StateGraph
except Exception:  # noqa: BLE001
    END = None
    START = None
    StateGraph = None


logger = logging.getLogger(__name__)


class QuizState(TypedDict, total=False):
    user_id: int
    description: str
    question_type: str
    doc_ids: Optional[List[int]]

    candidates: List[Dict[str, Any]]
    picked_chunks: List[Dict[str, Any]]
    source_chunks: Dict[str, Any]
    question_json: Dict[str, Any]


@dataclass
class QuizInput:
    user_id: int
    description: str
    question_type: str
    doc_ids: Optional[List[int]] = None


class QuizWorkflow:
    def __init__(self, vec: MilvusVectorStore, llm: LLMClient) -> None:
        self._vec = vec
        self._llm = llm
        self._graph = self._build_graph()

    def _build_graph(self):
        if not StateGraph or START is None or END is None:
            return None
        g = StateGraph(QuizState)
        g.add_node("retrieve", self._retrieve_candidates)
        g.add_node("pick", self._pick_top5)
        g.add_node("compose", self._compose_question)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "pick")
        g.add_edge("pick", "compose")
        g.add_edge("compose", END)
        return g.compile()

    def _retrieve_candidates(self, state: QuizState) -> QuizState:
        description = state.get("description", "")
        user_id = state.get("user_id")
        if user_id is None:
            raise ValueError("user_id is required")

        dense = self._llm.embed([description])[0]
        hits = self._vec.hybrid_search(
            query_vector=dense,
            query_text=description,
            user_id=user_id,
            doc_ids=state.get("doc_ids"),
            limit=50,
        )
        state["candidates"] = hits
        return state

    def _pick_top5(self, state: QuizState) -> QuizState:
        description = state.get("description", "")
        candidates = state.get("candidates", [])
        docs = [str(h.get("content") or "") for h in candidates]
        ranked = self._llm.rerank(description, docs, top_n=5) if docs else []

        picked: List[Dict[str, Any]] = []
        for r in ranked[:5]:
            idx = int(r.get("index", -1))
            if idx < 0 or idx >= len(candidates):
                continue
            hit = candidates[idx]
            picked.append(
                {
                    "chunk_id": hit.get("chunk_id"),
                    "doc_id": hit.get("doc_id"),
                    "section_title": hit.get("section_title"),
                    "content": hit.get("content") or "",
                    "score": r.get("relevance_score"),
                }
            )

        # rerank 失败兜底：直接拿前 5
        if not picked:
            for h in candidates[:5]:
                picked.append(
                    {
                        "chunk_id": h.get("chunk_id"),
                        "doc_id": h.get("doc_id"),
                        "section_title": h.get("section_title"),
                        "content": h.get("content") or "",
                        "score": h.get("score"),
                    }
                )

        state["picked_chunks"] = picked
        state["source_chunks"] = {
            "items": [
                {
                    "chunk_id": p.get("chunk_id"),
                    "doc_id": p.get("doc_id"),
                    "snippet": str(p.get("content") or "")[:200],
                }
                for p in picked
            ]
        }
        return state

    def _compose_question(self, state: QuizState) -> QuizState:
        question_type = str(state.get("question_type") or "single_choice")
        description = str(state.get("description") or "")
        picked = state.get("picked_chunks", [])

        context_parts: List[str] = []
        for i, p in enumerate(picked, 1):
            context_parts.append(
                f"[片段{i}] doc_id={p.get('doc_id')} "
                f"chunk_id={p.get('chunk_id')}\n"
                f"{p.get('content') or ''}"
            )
        context = "\n\n".join(context_parts)

        schema = {
            "name": "quiz_question",
            "schema": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "stem": {"type": "string"},
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "answer": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["type", "stem", "answer"],
                "additionalProperties": False,
            },
            "strict": False,
        }

        prompt = (
            "你是一个严谨的出题老师，请根据用户资料片段生成一道题。\n"
            f"目标题型必须是: {question_type}\n"
            "要求：\n"
            "1) 题目必须基于给定片段，不可编造片段外事实；\n"
            "2) single_choice 必须有4个选项，answer 为 A/B/C/D 之一；\n"
            "3) multi_choice answer 用 A|C 形式；\n"
            "4) fill_blank answer 可用 ans1||ans2 表示多空；\n"
            "5) short_answer answer 写参考答案摘要；\n"
            "6) 题干字段名必须使用 stem（不要用 question）；\n"
            "7) 只输出一个题目对象，不要附加解释。\n\n"
            f"用户需求: {description}\n\n"
            f"资料片段（top5，完整内容）:\n{context}\n"
        )

        raw = self._llm.chat_with_json_schema(prompt, schema)
        question = self._safe_parse_question(raw, question_type)
        state["question_json"] = question
        return state

    @staticmethod
    def _safe_parse_question(raw: str, question_type: str) -> Dict[str, Any]:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                stem = str(data.get("stem") or data.get("question") or "")
                q = {
                    "type": str(data.get("type") or question_type),
                    "stem": stem,
                    "options": data.get("options") or [],
                    "answer": str(data.get("answer") or ""),
                    "explanation": str(data.get("explanation") or ""),
                }
                # 保底修正题型
                q["type"] = question_type
                if q["type"] in {"single_choice", "multi_choice"} and not q["options"]:
                    q["options"] = [
                        "A. 选项A",
                        "B. 选项B",
                        "C. 选项C",
                        "D. 选项D",
                    ]
                if not q["stem"]:
                    q["stem"] = "请根据资料内容作答。"
                return q
        except Exception:  # noqa: BLE001
            pass

        return {
            "type": question_type,
            "stem": "请根据学习资料回答该问题。",
            "options": (
                ["A. 选项A", "B. 选项B", "C. 选项C", "D. 选项D"]
                if question_type in {"single_choice", "multi_choice"}
                else []
            ),
            "answer": "",
            "explanation": "",
        }

    def run(self, data: QuizInput) -> QuizState:
        state: QuizState = {
            "user_id": data.user_id,
            "description": data.description,
            "question_type": data.question_type,
            "doc_ids": data.doc_ids,
        }
        if self._graph:
            return cast(QuizState, self._graph.invoke(state))

        for fn in (
            self._retrieve_candidates,
            self._pick_top5,
            self._compose_question,
        ):
            state = fn(state)
        return state


class QuizService:
    def __init__(
        self,
        vec: MilvusVectorStore,
        llm: LLMClient,
        review_service: Optional[ReviewService] = None,
    ) -> None:
        self._workflow = QuizWorkflow(vec, llm)
        self._llm = llm
        self._reviews = review_service

    def generate_quiz(
        self,
        db: Session,
        user_id: int,
        description: str,
        question_type: str,
        doc_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        state = self._workflow.run(
            QuizInput(
                user_id=user_id,
                description=description,
                question_type=question_type,
                doc_ids=doc_ids,
            )
        )

        qj = dict(state.get("question_json") or {})
        quiz = Quiz(
            user_id=user_id,
            description=description,
            question_type=question_type,
            question_json=qj,
            source_chunks=state.get("source_chunks"),
        )
        db.add(quiz)
        db.flush()

        qj["id"] = f"q-{quiz.id}"
        qj["type"] = question_type
        quiz.question_json = qj
        db.add(quiz)

        logger.info(
            "quiz.generate persisted quiz_id=%s type=%s picked=%d",
            quiz.id,
            question_type,
            len((state.get("source_chunks") or {}).get("items", [])),
        )

        # 立即提交，确保后续会话可查询
        try:
            db.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning("quiz.generate_commit_fail err=%s", exc)
            db.rollback()

        # 不回传 answer，避免前端直接看到标准答案
        return {
            "quiz_id": quiz.id,
            "question": {
                "id": qj.get("id"),
                "type": qj.get("type"),
                "stem": qj.get("stem"),
                "options": qj.get("options", []),
                "explanation": qj.get("explanation", ""),
            },
        }

    def submit_answer(
        self,
        db: Session,
        user_id: int,
        quiz_id: int,
        answer: Any,
    ) -> Dict[str, Any]:
        quiz = db.get(Quiz, quiz_id)
        if quiz is None or quiz.user_id != user_id:
            raise ValueError("quiz not found")

        qj = quiz.question_json or {}
        q_type = str(qj.get("type") or quiz.question_type)
        std_answer = str(qj.get("answer") or "")

        score, comment, weak_desc = self._grade_answer(
            q_type,
            qj,
            std_answer,
            answer,
        )

        attempt = QuizAttempt(
            user_id=user_id,
            quiz_id=quiz_id,
            answer_json={"raw": answer},
            score=score,
            comment=comment,
        )
        db.add(attempt)
        db.flush()

        weak_point_id: Optional[int] = None
        if score < 4:
            source_items = (quiz.source_chunks or {}).get("items") or []
            related_doc_id = source_items[0].get("doc_id") if source_items else None
            related_chunk_ids = [
                i.get("chunk_id") for i in source_items if i.get("chunk_id")
            ]
            description = (
                weak_desc or f"对本题知识点掌握不牢：{str(qj.get('stem') or '')[:40]}"
            )
            level = "high" if score <= 2 else "medium"

            wp = WeakPoint(
                user_id=user_id,
                quiz_id=quiz_id,
                attempt_id=attempt.id,
                description=description,
                related_doc_id=related_doc_id,
                related_chunk_ids={"chunk_ids": related_chunk_ids},
                level=level,
            )
            db.add(wp)
            db.flush()
            weak_point_id = wp.id

            # 自动为薄弱点生成复习排期（艾宾浩斯间隔）
            if self._reviews:
                self._reviews.create_for_weak_point(
                    db,
                    user_id=user_id,
                    weak_point_id=wp.id,
                    base_date=date.today(),
                )

        # 持久化作答与薄弱点
        try:
            db.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning("quiz.submit_commit_fail err=%s", exc)
            db.rollback()

        return {
            "score": score,
            "comment": comment,
            "weak_point_id": weak_point_id,
        }

    def _grade_answer(
        self,
        question_type: str,
        question_json: Dict[str, Any],
        std_answer: str,
        user_answer: Any,
    ) -> tuple[float, str, str]:
        if question_type == "single_choice":
            user = str(user_answer).strip().upper()
            std = std_answer.strip().upper()
            if user == std and std:
                return 5.0, "回答正确，知识点掌握扎实。", ""
            return 0.0, f"回答错误，正确答案是 {std or '（未设置）'}。", ""

        if question_type == "multi_choice":
            user_set = self._parse_choice_set(user_answer)
            std_set = self._parse_choice_set(std_answer)
            if user_set and user_set == std_set:
                return 5.0, "多选答案完全正确。", ""
            return (
                0.0,
                (
                    f"多选答案不正确，标准答案是 {'|'.join(sorted(std_set)) or '（未设置）'}。"
                ),
                "",
            )

        if question_type == "fill_blank":
            std_parts = [x.strip().lower() for x in std_answer.split("||") if x.strip()]
            user_parts = self._normalize_fill_answer(user_answer)
            ok = bool(std_parts) and user_parts == std_parts
            if ok:
                return 5.0, "填空答案正确。", ""
            return 0.0, "填空答案与标准答案不一致。", ""

        # short_answer 及其他主观题：LLM 批改
        return self._grade_with_llm(question_json, std_answer, user_answer)

    @staticmethod
    def _parse_choice_set(value: Any) -> set[str]:
        if isinstance(value, list):
            items = value
        else:
            raw = str(value or "")
            sep_norm = raw.replace(",", "|").replace("，", "|")
            items = sep_norm.split("|")
        return {str(x).strip().upper() for x in items if str(x).strip()}

    @staticmethod
    def _normalize_fill_answer(value: Any) -> List[str]:
        if isinstance(value, list):
            arr = value
        else:
            raw = str(value or "")
            arr = raw.split("||")
        return [str(x).strip().lower() for x in arr if str(x).strip()]

    def _grade_with_llm(
        self,
        question_json: Dict[str, Any],
        std_answer: str,
        user_answer: Any,
    ) -> tuple[float, str, str]:
        schema = {
            "name": "quiz_grading",
            "schema": {
                "type": "object",
                "properties": {
                    "score": {"type": "number"},
                    "comment": {"type": "string"},
                    "weak_point": {"type": "string"},
                },
                "required": ["score", "comment"],
                "additionalProperties": False,
            },
            "strict": False,
        }
        prompt = (
            "你是严格但友好的阅卷老师。请给这道简答题评分。\n"
            "满分 5 分，按覆盖关键点程度打分。\n"
            f"题目：{question_json.get('stem', '')}\n"
            f"标准答案：{std_answer}\n"
            f"学生答案：{user_answer}\n"
            "返回 JSON，包含 score/comment，可选 weak_point。"
        )
        raw = self._llm.chat_with_json_schema(prompt, schema)
        try:
            data = json.loads(raw)
            score = float(data.get("score", 0.0))
            score = max(0.0, min(5.0, score))
            comment = str(data.get("comment") or "已完成批改。")
            weak = str(data.get("weak_point") or "")
            return score, comment, weak
        except Exception:  # noqa: BLE001
            return 0.0, "批改失败，按 0 分计入，请稍后重试。", ""

"""
quiz_chat_query
基于学习内容自动生成测验题目（选择题、填空题、问答题）
支持即时答题和批改
记录答题情况，识别薄弱环节
自然语言槽位填充 + QuizService 调用。

设计目标：
- LLM 槽位填充，必要时结合规则兜底；
- 支持“出题 -> 作答 -> 批改反馈”的命令行多轮对话；
- LangGraph 可用时走图编排，不可用时顺序降级。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, TypedDict, cast

from sqlalchemy.orm import Session

from .llm_client import LLMClient
from .models import Quiz, WeakPoint
from .quiz_service import QuizService

try:  # 可选：langgraph 可用时编排节点
    from langgraph.graph import END, START, StateGraph
except Exception:  # noqa: BLE001
    END = None
    START = None
    StateGraph = None


logger = logging.getLogger(__name__)


class QuizSlots(TypedDict, total=False):
    description: str
    question_type: str
    doc_ids: List[int]
    answer: str


class QuizChatState(TypedDict, total=False):
    text: str
    intent: str
    slots: QuizSlots
    reply: str


class QuizChatAgent:
    def __init__(self, quiz_service: QuizService, llm: LLMClient) -> None:
        self._quiz_service = quiz_service
        self._llm = llm
        self._pending_quiz_id: Optional[int] = None
        self._pending_question: Dict[str, Any] = {}
        self._active_db: Optional[Session] = None
        self._graph = self._build_graph()

    def has_pending_quiz(self) -> bool:
        return self._pending_quiz_id is not None

    # ===== 公开入口 =====
    def handle_message(self, text: str, db: Session) -> str:
        cleaned = text.strip()
        if not cleaned:
            return "你可以这样说：\n" "1) 帮我出一道 Python 单选题\n" "2) 我的答案是 A"

        logger.info("quiz_chat.handle_message text=%s", cleaned)
        state: QuizChatState = {"text": cleaned}
        self._active_db = db
        try:
            if self._graph:
                state = cast(QuizChatState, self._graph.invoke(state))
            else:
                for fn in (self._detect_intent, self._parse_slots, self._act):
                    state = fn(state)
        finally:
            self._active_db = None

        return str(state.get("reply") or "我没理解你的意思，请再说具体一点。")

    # ===== 图编排 =====
    def _build_graph(self):
        if not StateGraph or START is None or END is None:
            return None

        g = StateGraph(QuizChatState)
        g.add_node("detect_intent", self._detect_intent)
        g.add_node("parse_slots", self._parse_slots)
        g.add_node("act", self._act)
        g.add_edge(START, "detect_intent")
        g.add_edge("detect_intent", "parse_slots")
        g.add_edge("parse_slots", "act")
        g.add_edge("act", END)
        return g.compile()

    # ===== 节点实现 =====
    def _detect_intent(self, state: QuizChatState) -> QuizChatState:
        text = str(state.get("text") or "")
        intent = self._llm_detect_intent(text)
        if not intent:
            intent = self._rule_detect_intent(text)
        state["intent"] = intent
        logger.info("quiz_chat.intent=%s", intent)
        return state

    def _parse_slots(self, state: QuizChatState) -> QuizChatState:
        text = str(state.get("text") or "")
        slots = self._parse_slots_with_llm(text)

        # 规则补强：题型
        if not slots.get("question_type"):
            q_type = self._infer_question_type(text)
            if q_type:
                slots["question_type"] = q_type

        # 规则补强：答案
        if not slots.get("answer"):
            extracted = self._extract_answer_text(text)
            if extracted:
                slots["answer"] = extracted

        state["slots"] = slots
        return state

    def _act(self, state: QuizChatState) -> QuizChatState:
        db = self._active_db
        if db is None:
            state["reply"] = "系统状态异常：数据库会话不可用，请重试。"
            return state

        intent = str(state.get("intent") or "unknown")
        text = str(state.get("text") or "")
        slots = state.get("slots") or {}

        if intent == "list_weak_points":
            state["reply"] = self._list_weak_points(db)
            return state

        if intent == "list_history_quizzes":
            state["reply"] = self._list_history_quizzes(db)
            return state

        if intent == "generate_quiz":
            question_type = str(
                slots.get("question_type")
                or self._infer_question_type(text)
                or "single_choice"
            )
            description = str(slots.get("description") or text)
            doc_ids = slots.get("doc_ids")

            result = self._quiz_service.generate_quiz(
                db,
                user_id=1,
                description=description,
                question_type=question_type,
                doc_ids=doc_ids,
            )

            quiz_id = int(result.get("quiz_id") or 0)
            question = cast(Dict[str, Any], result.get("question") or {})
            self._pending_quiz_id = quiz_id if quiz_id > 0 else None
            self._pending_question = question

            state["reply"] = self._format_question_reply(quiz_id, question)
            return state

        if intent == "submit_answer":
            if not self._pending_quiz_id:
                state["reply"] = "当前没有进行中的题目，请先说“帮我出一道题”。"
                return state

            question_type = str(self._pending_question.get("type") or "single_choice")
            if question_type in {"fill_blank", "short_answer"}:
                answer_source = text
            else:
                answer_source = str(slots.get("answer") or text)
            answer = self._extract_answer_by_type(answer_source, question_type)
            if not answer:
                state["reply"] = "我还没收到你的答案，请直接回复答案内容。"
                return state

            result = self._quiz_service.submit_answer(
                db,
                user_id=1,
                quiz_id=self._pending_quiz_id,
                answer=answer,
            )

            score = float(result.get("score", 0.0))
            comment = str(result.get("comment") or "")
            weak_point_id = result.get("weak_point_id")

            weak_msg = (
                f"\n- 薄弱点记录ID：{weak_point_id}"
                if weak_point_id is not None
                else "\n- 本次未生成薄弱点"
            )
            raw_reply = (
                "批改完成：\n"
                f"- 得分：{score:.1f}/5\n"
                f"- 点评：{comment}"
                f"{weak_msg}\n\n"
            )

            # 调用 LLM 润色批改反馈，使其更加自然流畅
            polished_reply = self._polish_reply(raw_reply)
            state["reply"] = polished_reply

            # 作答完成后清空进行中题目，避免下一轮误判
            self._pending_quiz_id = None
            self._pending_question = {}
            return state

        hint = "你可以说：帮我出一道 Python 简答题；或在题目后直接回复你的答案。"
        if self._pending_quiz_id:
            pending_type = str(self._pending_question.get("type") or "single_choice")
            if pending_type == "single_choice":
                hint = "你可以直接提交答案，例如：我的答案是 A。"
            elif pending_type == "multi_choice":
                hint = "你可以直接提交答案，例如：我的答案是 A|C。"
            elif pending_type == "fill_blank":
                hint = "你可以直接提交答案，例如：我的答案是 关键术语。"
            else:
                hint = "你可以直接提交答案，例如：我的答案是 23。"
        state["reply"] = hint
        return state

    def _polish_reply(self, raw_reply: str) -> str:
        """用 LLM 将批改反馈改写得更口语、亲切，保持所有事实不丢失。"""
        prompt = (
            "你是对话助手，请把下面的批改反馈改写得更口语、亲切、鼓励性，"
            "但必须保留所有信息（得分、点评、薄弱点ID等），"
            "使用简短中文回应，不要添加新内容。\n"
            f"原始反馈:\n{raw_reply}"
        )
        try:
            return self._llm.chat(prompt)
        except Exception:
            return raw_reply

    def _polish_weak_points_reply(self, raw_reply: str) -> str:
        """用 LLM 将薄弱点列表改写得更有针对性和激励性。"""
        prompt = (
            "你是一位鼓励性的学习教练，请把下面的薄弱点列表改写得更具激励性、"
            "同时更有针对性地指出用户可以在哪些方面改进。风格要亲切、正向、"
            "但也要坦诚指出问题。必须保留所有信息（ID、等级、题号等），"
            "用简短中文回应，不要添加虚构的内容。\n"
            f"原始列表:\n{raw_reply}"
        )
        try:
            return self._llm.chat(prompt)
        except Exception:
            return raw_reply

    def _polish_history_quizzes_reply(self, raw_reply: str) -> str:
        """用 LLM 将题目历史记录改写得更生动，鼓励用户复习和继续练习。"""
        prompt = (
            "你是学习助手，请把下面的题目历史记录改写得更生动、鼓励性强，"
            "帮助用户回顾已做过的题目，激发复习的热情。风格要活泼但专业，"
            "必须保留所有信息（题目ID、题型、题干等），用简短中文回应，"
            "不要添加虚构的题目或信息。\n"
            f"原始记录:\n{raw_reply}"
        )
        try:
            return self._llm.chat(prompt)
        except Exception:
            return raw_reply

    def _llm_detect_intent(self, text: str) -> str:
        schema = {
            "name": "quiz_intent",
            "schema": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                },
                "required": ["intent"],
                "additionalProperties": False,
            },
            "strict": False,
        }
        prompt = (
            "你是测验对话系统的意图判定器，只输出 JSON。\n"
            "intent 仅允许：generate_quiz / submit_answer / list_weak_points / "
            "list_history_quizzes / unknown。\n"
            "判定规则（严格执行）：\n"
            "1) 仅当用户明确表达‘要出题/再来一道/生成题目/来一道X题’时，"
            "才判定为 generate_quiz；\n"
            "2) 若 has_pending_quiz=true，则除非用户明确要求新出题或查询薄弱点/"
            "历史题，否则一律判定为 submit_answer；\n"
            "3) 在 has_pending_quiz=true 时，用户给出长文本、数字、代码、"
            "选项字母，都视为 submit_answer；\n"
            "4) 只有确实无法判断时才返回 unknown。\n"
            f"has_pending_quiz={bool(self._pending_quiz_id)}\n"
            f"用户输入：{text}"
        )
        try:
            raw = self._llm.chat_0_6B_with_json_schema(prompt, schema)
            data = json.loads(raw)
            intent = str((data or {}).get("intent") or "").strip().lower()
            if intent in {
                "generate_quiz",
                "submit_answer",
                "list_weak_points",
                "list_history_quizzes",
                "unknown",
            }:
                return intent
        except Exception as exc:  # noqa: BLE001
            logger.warning("quiz_chat.llm_detect_intent_fail err=%s", exc)
        return ""

    def _rule_detect_intent(self, text: str) -> str:
        lowered = text.lower()

        weak_keys = ["薄弱点", "弱项", "weak point", "weakpoint"]
        if any(k in lowered for k in weak_keys):
            return "list_weak_points"

        history_keys = [
            "历史题",
            "之前的题",
            "旧题",
            "做过的题",
            "历史记录",
            "history quiz",
        ]
        if any(k in lowered for k in history_keys):
            return "list_history_quizzes"

        generate_keys = [
            "出题",
            "来一道",
            "再来一道",
            "生成题目",
            "重新出题",
            "quiz",
            "考我",
        ]
        if any(k in lowered for k in generate_keys):
            return "generate_quiz"

        submit_keys = ["答案", "我选", "提交", "我的回答", "答：", "answer"]
        if any(k in lowered for k in submit_keys):
            return "submit_answer"

        # 有进行中题目时，默认将用户后续输入视为作答
        if self._pending_quiz_id:
            return "submit_answer"

        return "unknown"

    def _list_weak_points(self, db: Session) -> str:
        rows = (
            db.query(WeakPoint)
            .filter(WeakPoint.user_id == 1)
            .order_by(WeakPoint.id.desc())
            .limit(5)
            .all()
        )
        if not rows:
            return "暂无薄弱点记录。你可以先做题，我会在低分时记录薄弱点。"

        lines = ["最近的薄弱点："]
        for row in rows:
            desc = (row.description or "").strip()
            lines.append(
                f"- ID={row.id} 等级={row.level} 关联quiz_id={row.quiz_id} "
                f"描述={desc[:80]}"
            )
        lines.append("\n你可以说：再来一道题，继续练习。")
        raw_reply = "\n".join(lines)

        # 调用 LLM 润色薄弱点列表，使其更具激励性和针对性
        polished_reply = self._polish_weak_points_reply(raw_reply)
        return polished_reply

    def _list_history_quizzes(self, db: Session) -> str:
        rows = (
            db.query(Quiz)
            .filter(Quiz.user_id == 1)
            .order_by(Quiz.id.desc())
            .limit(5)
            .all()
        )
        if not rows:
            return "暂无历史题目记录。你可以说：帮我出一道题。"

        lines = ["最近的题目记录："]
        for row in rows:
            qj = row.question_json or {}
            stem = str(qj.get("stem") or row.description or "").strip()
            lines.append(
                f"- quiz_id={row.id} 类型={row.question_type} 题干={stem[:80]}"
            )
        lines.append("\n你可以继续提交答案或说：再出一道题。")
        raw_reply = "\n".join(lines)

        # 调用 LLM 润色题目历史记录，使其更生动和鼓励用户复习
        polished_reply = self._polish_history_quizzes_reply(raw_reply)
        return polished_reply

    def _parse_slots_with_llm(self, text: str) -> QuizSlots:
        schema = {
            "name": "quiz_slots",
            "schema": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "question_type": {"type": "string"},
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "answer": {"type": "string"},
                },
                "required": [],
                "additionalProperties": False,
            },
            "strict": False,
        }

        prompt = (
            "请从用户输入提取测验槽位，并仅输出 JSON：\n"
            "- description: 出题需求文本（用于生成题目）；\n"
            "- question_type: "
            "single_choice/multi_choice/fill_blank/short_answer；\n"
            "- doc_ids: 若出现文档ID列表则提取；\n"
            "- answer: 作答内容（如 A、A|C 或简答文本）。\n"
            "若无法提取，字段留空或不返回。\n"
            f"用户输入：{text}"
        )

        try:
            raw = self._llm.chat_with_json_schema(prompt, schema)
            data = json.loads(raw)
            if not isinstance(data, dict):
                return QuizSlots()

            slots: QuizSlots = {}
            if data.get("description"):
                slots["description"] = str(data["description"])
            if data.get("question_type"):
                slots["question_type"] = self._normalize_question_type(
                    str(data["question_type"])
                )
            if isinstance(data.get("doc_ids"), list):
                slots["doc_ids"] = [int(x) for x in data["doc_ids"] if str(x).isdigit()]
            if data.get("answer"):
                slots["answer"] = str(data["answer"])
            return slots
        except Exception as exc:  # noqa: BLE001
            logger.warning("quiz_chat.parse_slots_fail err=%s", exc)
            return QuizSlots()

    @staticmethod
    def _normalize_question_type(raw: str) -> str:
        v = raw.strip().lower()
        if v in {
            "single_choice",
            "multi_choice",
            "fill_blank",
            "short_answer",
        }:
            return v

        mapping = {
            "单选": "single_choice",
            "多选": "multi_choice",
            "填空": "fill_blank",
            "简答": "short_answer",
            "问答": "short_answer",
        }
        for k, q_type in mapping.items():
            if k in raw:
                return q_type
        return "single_choice"

    def _infer_question_type(self, text: str) -> Optional[str]:
        t = text.lower()
        if "多选" in t:
            return "multi_choice"
        if "填空" in t:
            return "fill_blank"
        if "简答" in t or "问答" in t:
            return "short_answer"
        if "单选" in t or "选择" in t:
            return "single_choice"
        return None

    @staticmethod
    def _extract_answer_text(text: str) -> str:
        s = text.strip()
        s = re.sub(r"^(我的答案是|答案是|我选|提交答案|回答是|答：|答案：)\s*", "", s)
        return s.strip()

    def _extract_answer_by_type(self, text: str, question_type: str) -> str:
        s = self._extract_answer_text(text)
        if not s:
            return ""

        if question_type in {"single_choice", "multi_choice"}:
            picks = re.findall(r"(?<![A-Za-z])[A-Da-d](?![A-Za-z])", s)
            if not picks:
                return ""
            if question_type == "single_choice":
                return picks[0].upper()

            ordered: List[str] = []
            for p in picks:
                u = p.upper()
                if u not in ordered:
                    ordered.append(u)
            return "|".join(ordered)

        # 填空题与简答题：按整段文本提交答案
        return s

    @staticmethod
    def _format_question_reply(quiz_id: int, question: Dict[str, Any]) -> str:
        stem = str(question.get("stem") or "请作答。")
        q_type = str(question.get("type") or "single_choice")
        options = question.get("options") or []

        lines = [
            f"题目已生成（quiz_id={quiz_id}）：",
            f"- 题型：{q_type}",
            f"- 题干：{stem}",
        ]

        if isinstance(options, dict):
            for key in sorted(options.keys()):
                lines.append(f"  {key}. {options[key]}")
        elif isinstance(options, list):
            for item in options:
                lines.append(f"  - {item}")

        lines.append("\n请直接回复你的答案。")
        return "\n".join(lines)

"""主对话 Agent：路由 RAG / 计划 / 测验 / 复习，单入口对话体验。

设计目标：
- 对用户“无感知”的多 Agent 整合：一个 handle_message 入口即可完成五大核心功能。
- 轻量规则 + 可选 LLM 意图识别，避免强依赖外部服务时的脆弱。
- 复用现有子 Agent / Service，不重复业务逻辑。
"""

from __future__ import annotations

from typing import Any, Dict, List
import json

from sqlalchemy.orm import Session

from .llm_client import LLMClient
from .rag_service import RAGService
from .chat_agent import PlanChatAgent
from .quiz_chat_agent import QuizChatAgent
from .review_service import ReviewService


Intent = str


class MainChatAgent:
    def __init__(
        self,
        llm: LLMClient,
        rag_service: RAGService,
        plan_agent: PlanChatAgent,
        quiz_agent: QuizChatAgent,
        review_service: ReviewService,
        user_id: int = 1,
    ) -> None:
        self._llm = llm
        self._rag = rag_service
        self._plan_agent = plan_agent
        self._quiz_agent = quiz_agent
        self._review_service = review_service
        self._user_id = user_id

    # ===== public =====
    def handle_message(self, text: str, db: Session) -> Dict[str, Any]:
        text = text.strip()
        if not text:
            return {"reply": "请告诉我你要做什么，比如提问、制定计划或做测验。"}

        intent = self._detect_intent(text)
        quiz_intents = {
            "quiz_now",
            "quiz_generate",
            "quiz_answer",
            "quiz_list_weak_points",
            "quiz_list_history",
        }

        if intent in {"create_plan", "adjust_plan", "show_plan"}:
            reply = self._plan_agent.handle_message(text, db)
        elif intent in quiz_intents:
            reply = self._quiz_agent.handle_message(text, db)
        elif intent == "review_today":
            reply = self._handle_review_today(db)
        else:  # 默认走 RAG
            reply = self._handle_rag(text, db)

        # 针对列表类意图，追加一次 LLM 口语化润色，避免生硬拼接
        if intent in {
            "quiz_list_history",
            "quiz_list_weak_points",
            "review_today",
        }:
            reply = self._polish_reply(intent, reply)

        return {"reply": reply, "intent": intent}

    # ===== intent detection =====
    def _detect_intent(self, text: str) -> Intent:
        """意图识别：优先 LLM，多枚举，规则兜底，最终回退 RAG。"""

        pending_quiz = getattr(self._quiz_agent, "has_pending_quiz", lambda: False)()

        # ===== 1) LLM 优先判定（覆盖所有子 Agent 的意图枚举） =====
        llm_schema = {
            "name": "detect_intent",
            "schema": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": [
                            # 通用 / 知识检索
                            "rag_query",
                            # 计划
                            "create_plan",
                            "adjust_plan",
                            "show_plan",
                            # 测验（细分，以覆盖子 agent 的所有分支）
                            "quiz_generate",
                            "quiz_answer",
                            "quiz_list_weak_points",
                            "quiz_list_history",
                            # 复习
                            "review_today",
                        ],
                    }
                },
                "required": ["intent"],
                "additionalProperties": False,
            },
            "strict": False,
        }

        llm_prompt = (
            "你是意图分类器，只输出 JSON，不要解释。intent 取值：\n"
            "- rag_query: 纯粹问问题/知识查询/带问号的提问。\n"
            "- create_plan: 让你制定/生成/安排/规划学习计划。\n"
            "- adjust_plan: 调整/修改/重做/重新规划/延期/缩短学习计划。\n"
            "- show_plan: 查看/回顾/展示当前学习计划内容。\n"
            "- quiz_generate: 让你出题/测验/再来一道题/考考我/生成题目。\n"
            "- quiz_answer: 用户在作答或提交答案（含'我的答案是'、选项、代码、数字等）。\n"
            "- quiz_list_weak_points: 查询薄弱点/弱项/weak point 列表。\n"
            "- quiz_list_history: 查询历史题目/做过的题/历史记录。\n"
            "- review_today: 询问今天/当前需要复习什么、今日复习任务。\n"
            '请严格返回 JSON，例如 {"intent": "quiz_generate"}。\n'
            f"用户输入：{text}"
        )

        try:
            raw = self._llm.chat_0_6B_with_json_schema(llm_prompt, llm_schema)
            data = json.loads(raw)
            intent = str(data.get("intent") or "").strip()
            if intent:
                if pending_quiz and intent == "rag_query":
                    return "quiz_answer"
                return intent
        except Exception:
            pass

        # ===== 2) 规则兜底（关键字覆盖常见说法；无则再看问号） =====
        lower = text.lower()

        # 计划类
        if any(
            k in lower
            for k in [
                "学习计划",
                "复习计划",
                "安排学习",
                "帮我规划",
                "帮我安排",
                "制定计划",
                "做一个计划",
                "计划一下",
                "study plan",
            ]
        ):
            return "create_plan"
        if any(k in lower for k in ["查看计划", "看看计划", "看一下计划", "show plan"]):
            return "show_plan"
        if any(k in lower for k in ["调整计划", "改下计划", "改一下计划", "重新规划"]):
            return "adjust_plan"
        if any(k in lower for k in ["计划", "安排"]) and any(
            t in lower for t in ["天", "周", "每天", "分钟", "小时", "days", "weeks"]
        ):
            return "create_plan"

        # 测验/出题/答题/弱点/历史
        if any(
            k in lower
            for k in [
                "测验",
                "出题",
                "出一道",
                "出一题",
                "出个题",
                "来一道题",
                "来一题",
                "考考我",
                "再来一道",
                "做题",
                "考试",
                "quiz",
                "test me",
                "question please",
            ]
        ):
            return "quiz_generate"

        if any(
            k in lower
            for k in ["答案", "我选", "提交", "回答是", "答：", "答案是", "answer"]
        ):
            return "quiz_answer"

        if any(k in lower for k in ["薄弱点", "弱项", "weak point", "weakpoint"]):
            return "quiz_list_weak_points"

        if any(
            k in lower
            for k in [
                "历史题",
                "之前的题",
                "旧题",
                "做过的题",
                "历史记录",
                "history quiz",
            ]
        ):
            return "quiz_list_history"

        # 复习
        if any(
            k in lower
            for k in [
                "今天复习什么",
                "今天要复习什么",
                "今天要学什么",
                "复习任务",
                "今日复习",
                "today review",
            ]
        ):
            return "review_today"

        if pending_quiz:
            return "quiz_answer"

        # 问句默认 RAG
        if "?" in text or "？" in text:
            return "rag_query"

        # 最终回退
        return "rag_query"

    # ===== handlers =====
    def _handle_rag(self, text: str, db: Session) -> str:
        result = self._rag.query(db, user_id=self._user_id, question=text, doc_ids=None)
        answer = result.get("answer", "")
        return answer or "我没有检索到足够的依据来回答这个问题。"

    def _handle_review_today(self, db: Session) -> str:
        items = self._review_service.list_today_reviews(db, user_id=self._user_id)
        if not items:
            return "今天没有待复习任务。"
        lines: List[str] = ["今日复习任务："]
        for i, it in enumerate(items, 1):
            tgt = it.get("target", {}) or {}
            if it.get("target_type") == "daily_task":
                lines.append(
                    f"{i}. 计划任务 {tgt.get('title')} (日期 {tgt.get('date')})"
                )
            else:
                lines.append(f"{i}. 薄弱点：{tgt.get('description') or '未知'}")
        return "\n".join(lines)

    def _polish_reply(self, intent: Intent, raw_reply: str) -> str:
        """用 LLM 将列表式回复口语化，保持事实不丢失。"""

        prompt = (
            "你是对话助手，请把下面的列表式回复改写得更口语、亲切，"
            "但必须保留所有事实、顺序和编号，使用简短中文回应，不要添加新内容。\n"
            f"意图: {intent}\n"
            f"原始回复:\n{raw_reply}"
        )
        try:
            return self._llm.chat(prompt)
        except Exception:
            return raw_reply

"""
、学习计划agent
设计目标：
- 不修改底层 PlanWorkflow/PlanService 逻辑；
- 通过 LLM 或规则尽量从自然语言提取槽位，缺失则追问；
- 支持“调整计划”意图：将旧计划 pending 任务标记为 cancelled，重跑生成新计划。
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, TypedDict, Tuple, cast

from sqlalchemy.orm import Session

from .llm_client import LLMClient
from .models import DailyTask, LearningPlan
from .plan_service import PlanService


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PlanSlots(TypedDict, total=False):
    goal_description: str
    start_date: str
    end_date: str
    target_days: int
    daily_minutes: int
    doc_ids: List[int]


class PlanChatAgent:
    def __init__(self, plan_service: PlanService, llm: LLMClient) -> None:
        self._plan_service = plan_service
        self._llm = llm
        self._slots: PlanSlots = {}
        self._current_plan_id: Optional[int] = None
        self._intent: str = "create_plan"

    # ===== 公开入口 =====
    def handle_message(self, text: str, db: Session) -> str:
        text = text.strip()
        if not text:
            return "请告诉我你的学习目标、时间范围和每天可用时间。"

        logger.info("chat_agent.handle_message start text=%s", text)
        self._intent = self._detect_intent(text)
        logger.info("chat_agent.intent=%s", self._intent)

        # 查看计划详情的快速通道
        if self._intent == "show_plan":
            plan_id = self._current_plan_id
            if not plan_id:
                plan_id = self._load_latest_plan_id(db)
                self._current_plan_id = plan_id

            if plan_id:
                return self._describe_plan(db, plan_id)
            return "当前没有可查看的计划，请先创建一个学习计划。"

        parsed = self._parse_slots_with_llm(text)
        print(f"[SLOT] parsed_slots={parsed}")
        self._merge_slots(parsed)

        # ===== 在有历史计划的情况下，优先复用旧 goal =====
        # - adjust_plan 场景：如果本轮没明确给出新 goal，则从旧计划沿用 goal_text
        # - create_plan 场景：若完全没有历史 plan，可将本句自然语言作为默认目标
        if self._intent == "adjust_plan" and self._current_plan_id:
            # 尝试从数据库读取当前计划，复用其 goal_text
            if "goal_description" not in self._slots or not self._slots.get(
                "goal_description"
            ):
                try:
                    plan = db.get(LearningPlan, self._current_plan_id)
                    if plan and plan.goal_text:
                        self._slots["goal_description"] = plan.goal_text
                        print(
                            f"[SLOT] adjust_plan_reuse_goal plan_id={plan.id} "
                            f"goal={plan.goal_text}"
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "chat_agent.adjust_plan_reuse_goal_fail plan_id=%s " "err=%s",
                        self._current_plan_id,
                        exc,
                    )

        # 对于首次创建计划，且没有历史 goal，可用当前这句话作为默认目标
        goal_missing = "goal_description" not in self._slots or not self._slots.get(
            "goal_description"
        )
        if goal_missing and text and self._intent == "create_plan":
            if self._is_goal_like(text):
                self._slots["goal_description"] = text
            else:
                print(
                    "[SLOT] goal_description rejected: need a concrete skill/"
                    "knowledge goal"
                )
        print(f"[SLOT] merged_slots={self._slots}")
        print(f"[SLOT] slots_current={self._slots}")
        print(f"[SLOT] merged_slots={self._slots}")
        print(f"[SLOT] slots_current={self._slots}")

        
        missing = self._missing_fields(self._slots)
        if missing:
            print(f"[SLOT] missing_fields={missing}")
            
            # 将缺失字段映射为用户友好的中文描述
            field_mapping = {
                "goal_description(学习目标)": "学习目标（比如：想学什么、要达到什么水平）",
                "daily_minutes(每天时长)": "每天可投入的学习时间（比如：每天1小时、30分钟等）",
                "start/end 或 target_days": "时间安排（比如：总共几天、从哪天到哪天）"
            }
            
            friendly_missing = []
            for field in missing:
                if field in field_mapping:
                    friendly_missing.append(field_mapping[field])
                elif field.endswith("(学习目标)"):
                    friendly_missing.append("学习目标")
                elif field.endswith("(每天时长)"):
                    friendly_missing.append("每天学习时长")
                else:
                    friendly_missing.append(field)
            
            if len(friendly_missing) == 1:
                return f"要给你制定完美的学习计划，我还需要知道{friendly_missing[0]}哦～"
            elif len(friendly_missing) == 2:
                return f"为了让计划更合你心意，需要补充这两个信息：{friendly_missing[0]}和{friendly_missing[1]}"
            else:
                missing_str = "、".join(friendly_missing[:-1]) + f"和{friendly_missing[-1]}"
                return f"咱们就差最后几步啦！请告诉我{missing_str}，马上为你量身打造学习计划～"

        try:
            start_date, end_date = self._finalize_dates(self._slots)
        except ValueError as exc:  # 无法确定日期
            logger.warning(
                "chat_agent.finalize_dates_failed slots=%s err=%s",
                self._slots,
                exc,
            )
            return str(exc)

        logger.info(
            "chat_agent.finalized goal=%s start=%s end=%s daily=%s doc_ids=%s",
            self._slots.get("goal_description"),
            start_date,
            end_date,
            self._slots.get("daily_minutes"),
            self._slots.get("doc_ids"),
        )

        daily_minutes = int(self._slots.get("daily_minutes", 0))
        doc_ids = self._slots.get("doc_ids")
        goal = self._slots.get("goal_description", "")

        # 调整计划：先软取消旧 pending 任务
        cancelled = 0
        old_plan_id = None
        if self._intent == "adjust_plan" and self._current_plan_id:
            old_plan_id = self._current_plan_id
            cancelled = self._cancel_pending(db, self._current_plan_id)
            logger.info(
                "chat_agent.adjust_plan cancelled_pending count=%s " "old_plan_id=%s",
                cancelled,
                old_plan_id,
            )

        result = self._plan_service.create_plan(
            db,
            user_id=1,
            goal_description=goal,
            start_date=start_date,
            end_date=end_date,
            daily_minutes=daily_minutes,
            doc_ids=doc_ids,
        )
        self._current_plan_id = result.get("plan_id")
        logger.info(
            "chat_agent.plan_created plan_id=%s syllabus=%d daily_plan=%d"
            " not_covered=%d",
            self._current_plan_id,
            len(result.get("syllabus", [])),
            len(result.get("daily_plan", [])),
            len(result.get("not_covered", [])),
        )
        self._slots = {}  # 重置槽位，下一轮重新填

        daily_plan = result.get("daily_plan", [])
        syllabus = result.get("syllabus", [])
        not_covered = result.get("not_covered", [])

        summary = self._build_plan_detail(
            self._current_plan_id, daily_plan, syllabus, not_covered
        )
        if old_plan_id:
            summary = (
                f"已基于新的约束重建计划，旧计划 {old_plan_id} 的未完成任务已标记为取消（{cancelled} 条）；"
                + summary
            )
        return summary

    # ===== 内部：意图与槽处理 =====
    def _detect_intent(self, text: str) -> str:
        intent = self._llm_detect_intent(text)
        if intent:
            return intent
        if any(k in text for k in ["改", "调整", "重新", "换", "延期", "缩短", "延长"]):
            return "adjust_plan"
        if any(k in text for k in ["查看", "详情", "详细", "看看", "show", "看计划"]):
            return "show_plan"
        return "create_plan"

    def _llm_detect_intent(self, text: str) -> str:
        try:
            prompt = f"""
你是一个学习计划的意图识别助手，请根据用户输入准确判断其意图。用户意图分为三类：
1. create_plan: 创建新的学习计划
2. adjust_plan: 调整/修改/变更现有的学习计划
3. show_plan: 查看/展示/询问现有的学习计划

请参考以下示例：

示例1:
用户输入: "我想创建一个学习Python的计划"
意图: create_plan

示例2:
用户输入: "我需要一个7天学完机器学习的计划"
意图: create_plan

示例3:
用户输入: "帮我定个计划，每天学1小时"
意图: create_plan

示例4:
用户输入: "能不能把计划改一下，增加每天学习时间到2小时"
意图: adjust_plan

示例5:
用户输入: "我的学习进度有点慢，能重新规划一下吗"
意图: adjust_plan

示例6:
用户输入: "我想调整一下学习安排"
意图: adjust_plan

示例7:
用户输入: "看看我现在的计划"
意图: show_plan

示例8:
用户输入: "我的学习安排是什么样的"
意图: show_plan

示例9:
用户输入: "今天要学什么"
意图: show_plan

请仔细分析用户输入，注意以下关键词：
- 创建/制定/定/新建/生成 → create_plan
- 调整/修改/改/重新/变更 → adjust_plan
- 查看/显示/看/当前/现在 → show_plan

如果意图不明显，按以下优先级判断：
1. 先判断是否有调整的关键词
2. 再判断是否有创建的关键词
3. 最后判断是否有查看的关键词
4. 都无则根据用户是否有明确目标判断

用户输入: {text}
意图:"""
            resp = self._llm.chat_0_6B(prompt)
            resp_l = (resp or "").lower()
            for key in ("adjust_plan", "show_plan", "create_plan"):
                if key in resp_l:
                    return key
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat_agent.llm_intent_fail err=%s", exc)
        return ""

    def _parse_slots_with_llm(self, text: str) -> PlanSlots:
        schema = {
            "name": "plan_slots",
            "schema": {
                "type": "object",
                "properties": {
                    "goal_description": {"type": "string"},
                    "start_date": {
                        "type": "string",
                        "description": "格式 YYYY-MM-DD",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "格式 YYYY-MM-DD",
                    },
                    "target_days": {"type": "integer"},
                    "daily_minutes": {"type": "integer"},
                    "doc_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": [],
                "additionalProperties": False,
            },
            "strict": False,
        }
        prompt = (
            "请从用户输入中提取学习计划槽位，严格以 JSON 返回，能提取就填，缺失留空或 null：\n"
            "- goal_description: 具体的知识点或技能目标\n"
            "  （比如“掌握 Python 爬虫原理”、“熟练使用 HTTP 请求”），不要把时间、约束或感叹当作 goal。\n"
            "  如果语句只是说‘我只有两天’、‘每天 50 分钟’、‘帮帮我’，就不要填 goal。\n"
            "- start_date / end_date: 若缺其中之一，可不填；\n"
            "- target_days: 若说到几天内完成；\n"
            "- daily_minutes: 每天多少分钟/小时（转整数分钟）；\n"
            "- doc_ids: 若出现数字 ID 列表则填；\n"
            f"用户输入：{text}"
        )
        raw = self._llm.chat_with_json_schema(prompt, schema)
        try:
            data = json.loads(raw)
            print(f"[SLOT] slots_parsed raw={data}")
            if isinstance(data, dict):
                filtered = {k: v for k, v in data.items() if v not in (None, "")}
                return cast(PlanSlots, filtered)
        except Exception:  # noqa: BLE001
            return PlanSlots()
        return PlanSlots()

    def _merge_slots(self, new_slots: PlanSlots) -> None:
        for k, v in new_slots.items():
            if v is None or v == "":
                continue
            self._slots[k] = v  # 覆盖为最新输入

    def _load_latest_plan_id(self, db: Session) -> Optional[int]:
        try:
            row = (
                db.query(LearningPlan.id)
                .filter(LearningPlan.user_id == 1)
                .order_by(LearningPlan.id.desc())
                .first()
            )
            if row:
                return int(row[0])
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat_agent.load_latest_plan_fail err=%s", exc)
        return None

    @staticmethod
    def _missing_fields(slots: PlanSlots) -> List[str]:
        missing: List[str] = []
        if not slots.get("goal_description"):
            missing.append("goal_description(学习目标)")
        if not slots.get("daily_minutes"):
            missing.append("daily_minutes(每天时长)")
        has_dates = slots.get("start_date") and slots.get("end_date")
        has_days = slots.get("target_days")
        if not has_dates and not has_days:
            missing.append("start/end 或 target_days")
        return missing

    @staticmethod
    def _finalize_dates(slots: PlanSlots) -> Tuple[date, date]:
        if slots.get("start_date") and slots.get("end_date"):
            return (
                date.fromisoformat(str(slots.get("start_date"))),
                date.fromisoformat(str(slots.get("end_date"))),
            )
        if slots.get("target_days"):
            start = date.today()
            days = int(slots.get("target_days") or 1)
            end = start + timedelta(days=max(1, days) - 1)
            return start, end
        raise ValueError("需要提供 start/end 或 target_days 才能生成计划。")

    @staticmethod
    def _is_goal_like(text: Optional[str]) -> bool:
        if not text:
            return False
        normalized = text.strip().lower()
        if len(normalized) < 4:
            return False
        keywords = [
            "学",
            "掌握",
            "熟悉",
            "了解",
            "积累",
            "能力",
            "技能",
            "实践",
            "应用",
            "构建",
        ]
        for k in keywords:
            if k in normalized:
                return True
        english = ["learn", "master", "skill", "practice", "build"]
        for k in english:
            if k in normalized:
                return True
        return False

    def _cancel_pending(self, db: Session, plan_id: int) -> int:
        q = db.query(DailyTask).filter(
            DailyTask.plan_id == plan_id, DailyTask.status == "pending"
        )
        count = q.count()
        q.update({"status": "cancelled"})
        return count

    def _describe_plan(self, db: Session, plan_id: int) -> str:
        plan = db.get(LearningPlan, plan_id)
        if not plan:
            return "未找到计划，请先创建一个新的学习计划。"
        tasks = self._plan_service.list_daily_tasks(
            db, plan_id=plan_id, user_id=plan.user_id
        )
        lines = [
            (
                f"计划 {plan.id} | 目标: {plan.goal_text} | 时间: {plan.start_date}"
                f" ~ {plan.end_date} | 每日 {plan.daily_minutes} 分钟 | 状态:"
                f" {plan.status}"
            )
        ]
        for t in tasks:
            outline = t.get("outline", {})
            outline_items = (
                outline.get("outline", []) if isinstance(outline, dict) else []
            )
            lines.append(
                f"- {t.get('date')} {t.get('title')} "
                f"(主题数: {len(outline_items)}) 状态: {t.get('status')}"
            )
            if outline_items:
                lines.append(
                    "  主题详情: "
                    + json.dumps(outline_items, ensure_ascii=False, indent=2)
                )
        lines.append("=== 任务原始 JSON（含单元/主题） ===")
        lines.append(json.dumps(tasks, ensure_ascii=False, indent=2))
        # llm_view = self._llm_describe_plan(tasks)
        # if llm_view:
        #     lines.append("=== LLM 描述 ===")
        #     lines.append(llm_view)
        return "\n".join(lines)

    def _llm_describe_plan(self, tasks: List[Dict[str, Any]]) -> str:
        try:
            prompt = (
                "你是一位学习规划顾问，现在需要向学生讲解他的学习计划。\n"
                "请用自然、友好的语气，像在和朋友对话一样，把下面的学习日程串联起来讲一遍。\n"
                "重点：一定要提到每天要学的所有细致知识点（topic_name），不能漏掉任何一个，\n"
                "并且按照日期顺序讲解，让学生清楚知道自己每天要学什么。\n\n"
                "学习计划如下：\n"
                f"{json.dumps(tasks, ensure_ascii=False)}"
            )
            resp = self._llm.chat(prompt)
            return resp.strip() if resp else ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat_agent.llm_describe_plan_fail err=%s", exc)
            return ""

    def _build_plan_detail(
        self,
        plan_id: Optional[int],
        daily_plan: List[Dict[str, Any]],
        syllabus: List[Dict[str, Any]],
        not_covered: List[str],
    ) -> str:
        fallback = self._build_brief_plan_detail(
            plan_id, daily_plan, syllabus, not_covered
        )
        try:
            prompt = (
                "你是一位学习规划顾问，现在需要向学生讲解他的学习计划。\n"
                "请用自然、友好的语气，像在和朋友对话一样，把下面的学习日程串联起来讲一遍。\n"
                "重点：一定要提到每天要学的所有细致知识点（topic_name），不能漏掉任何一个，\n"
                "并且按照日期顺序讲解，让学生清楚知道自己每天要学什么。\n\n"
                "语气要积极向上，给学生信心和动力，但信息必须准确、完整。\n\n"
                f"计划编号: {plan_id}\n"
                f"学习单元: {json.dumps(syllabus, ensure_ascii=False)}\n"
                f"每日安排: {json.dumps(daily_plan, ensure_ascii=False)}\n"
                f"暂无覆盖: {json.dumps(not_covered, ensure_ascii=False)}"
            )
            resp = self._llm.chat(prompt)
            if resp:
                return resp.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat_agent.plan_detail_llm_fail err=%s", exc)
        return fallback

    @staticmethod
    def _build_brief_plan_detail(
        plan_id: Optional[int],
        daily_plan: List[Dict[str, Any]],
        syllabus: List[Dict[str, Any]],
        not_covered: List[str],
    ) -> str:
        days = len(daily_plan)
        day_preview = ""
        if daily_plan:
            d0 = daily_plan[0]
            day_preview = (
                f"Day1: {d0.get('title')}，包含 {len(d0.get('outline', []))} 个主题。"
            )
        syllabus_units = len(syllabus)
        miss = f" 未覆盖: {', '.join(not_covered)}" if not_covered else ""
        return (
            f"已生成计划 {plan_id}，共 {days} 天，{syllabus_units} 个单元。"
            f"{(' ' + day_preview) if day_preview else ''}{miss}"
        )

"""学习计划：LangGraph 风格的规划工作流 + 落库服务。

阶段二目标：
- 基于 LLM 的学习大纲生成（仅依赖用户知识库的真实内容）。
- LangGraph Plan Workflow：依照“扩展意图 -> RAG 检索 -> 大纲生成 -> 日计划拆分”节点串联。
- learning_plans / daily_tasks 写入与查询接口。

设计取舍：
- 为兼容无 langgraph 环境，工作流节点按顺序执行；若检测到 langgraph，可动态编译 StateGraph，但不会阻塞主流程。
- 所有 JSON 解析均提供兜底策略，避免 LLM 输出格式偏差导致计划生成中断。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, TypedDict, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from .llm_client import LLMClient
from .models import DailyTask, LearningPlan
from .vector_store import MilvusVectorStore

try:  # 可选使用 langgraph，缺失时走顺序执行
    from langgraph.graph import StateGraph, START, END
except Exception:  # noqa: BLE001
    StateGraph = None
    START = None
    END = None


logger = logging.getLogger(__name__)


class PlanState(TypedDict, total=False):
    user_id: int
    goal_description: str
    start_date: date
    end_date: date
    daily_minutes: int
    doc_ids: Optional[List[int]]

    topics: List[str]
    topic_hits: Dict[str, List[Dict[str, Any]]]
    not_covered: List[str]
    syllabus: List[Dict[str, Any]]
    daily_plan: List[Dict[str, Any]]
    error: Optional[str]


@dataclass
class PlanInput:
    user_id: int
    goal_description: str
    start_date: date
    end_date: date
    daily_minutes: int
    doc_ids: Optional[List[int]] = None


class PlanWorkflow:
    def __init__(self, vec: MilvusVectorStore, llm: LLMClient) -> None:
        self._vec = vec
        self._llm = llm
        self._graph = self._build_graph()

    # ===== 图编排（可选） =====
    def _build_graph(self):
        if not StateGraph or START is None or END is None:
            return None
        g = StateGraph(PlanState)
        g.add_node("expand_goal", self._expand_goal)
        g.add_node("retrieve", self._retrieve_topics)
        g.add_node("syllabus", self._build_syllabus)
        g.add_node("allocate", self._allocate_daily_plan)
        g.add_edge(START, "expand_goal")
        g.add_edge("expand_goal", "retrieve")
        g.add_edge("retrieve", "syllabus")
        g.add_edge("syllabus", "allocate")
        g.add_edge("allocate", END)
        return g.compile()

    # ===== 节点实现 =====
    def _expand_goal(self, state: PlanState) -> PlanState:
        goal = state.get("goal_description", "")
        t0 = time.perf_counter()
        prompt = (
            "你是学习规划助手。请根据用户目标拆分 3-6 个子主题，不要返回描述，"
            '仅返回 JSON 数组，例如: ["Python 基础", "数据结构"].\n'
            f"用户目标: {goal}"
        )
        raw = self._llm.chat(prompt)
        topics = self._safe_parse_list(raw)
        if not topics:
            topics = [goal]
        state["topics"] = topics[:6]
        logger.info(
            "plan.expand_goal topics=%s cost=%.2fs",
            topics[:4],
            time.perf_counter() - t0,
        )
        return state

    def _retrieve_topics(self, state: PlanState) -> PlanState:
        topic_hits: Dict[str, List[Dict[str, Any]]] = {}
        not_covered: List[str] = []
        t0 = time.perf_counter()
        goal = state.get("goal_description", "")
        user_id = state.get("user_id")
        if user_id is None:
            raise ValueError("user_id is required for plan workflow")

        for topic in state.get("topics", []):
            q = f"{goal} | {topic}"
            dense = self._llm.embed([q])[0]
            hits = self._vec.hybrid_search(
                query_vector=dense,
                query_text=q,
                user_id=user_id,
                doc_ids=state.get("doc_ids"),
                limit=12,
            )
            if not hits:
                not_covered.append(topic)
            else:
                topic_hits[topic] = hits
        state["topic_hits"] = topic_hits
        state["not_covered"] = not_covered
        logger.info(
            "plan.retrieve topics=%d covered=%d uncovered=%d cost=%.2fs",
            len(state.get("topics", [])),
            len(topic_hits),
            len(not_covered),
            time.perf_counter() - t0,
        )
        return state

    def _build_syllabus(self, state: PlanState) -> PlanState:
        # 收集上下文，限制总量防止 prompt 过长
        snippets: List[str] = []
        for topic, hits in state.get("topic_hits", {}).items():
            for h in hits[:3]:
                content = (h.get("content") or "")[:400].replace("\n", " ")
                snippets.append(
                    f"[topic:{topic}] doc:{h.get('doc_id')} "
                    f"chunk:{h.get('chunk_id')} {content}"
                )
        joined = "\n".join(snippets[:30])

        # 计算时间预算
        start = state.get("start_date")
        end = state.get("end_date")
        daily_minutes = state.get("daily_minutes") or 60
        if start and end:
            days = (end - start).days + 1
        else:
            days = 3
        total_budget = daily_minutes * days

        topics_cnt = max(1, len(state.get("topics", [])))
        min_items = (
            max(2, min(topics_cnt, days or topics_cnt)) if days else max(2, topics_cnt)
        )
        max_items = max(
            min_items,
            min(topics_cnt + 1, max(days, topics_cnt)),
        )

        # 定义 JSON Schema —— 完全匹配示例：
        # [ {"unit_id":"U1","name":"单元名","estimated_minutes":整数,
        #    "topics":[{"topic_id":"U1-T1","name":"主题",
        #    "chunk_ids":["c1","c2"]}]} ]
        json_schema = {
            "name": "learning_syllabus",
            "schema": {
                "type": "array",
                "minItems": min_items,
                "maxItems": max_items,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "unit_id": {"type": "string"},
                        "name": {"type": "string"},
                        "estimated_minutes": {"type": "integer"},
                        "topics": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "topic_id": {"type": "string"},
                                    "name": {"type": "string"},
                                    "chunk_ids": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": [
                                    "topic_id",
                                    "name",
                                    "chunk_ids",
                                ],
                            },
                        },
                    },
                    "required": [
                        "unit_id",
                        "name",
                        "estimated_minutes",
                        "topics",
                    ],
                },
            },
            "strict": True,
        }

        t0 = time.perf_counter()
        prompt = (
            "你是学习计划设计师。仅基于下列真实片段归纳学习大纲，"
            "确保所有主题都能从片段溯源，不要凭空生成。\n"
            f"用户目标: {state.get('goal_description', '')}\n"
            f"片段（带 chunk_id ）:\n{joined}\n"
            f"总学习天数: {days} 天，每天 {daily_minutes} 分钟，总时长预算 <= "
            f"{total_budget} 分钟。\n"
            "请确保所有单元的 estimated_minutes 求和不超过总预算，并尽量均衡。\n"
            f"至少输出 {min_items} 个单元，不要只返回一个单元。你的单元可以拆分的细粒度一些。\n"
            "每个单元下的name不允许跟时间有关，只是知识点的总结。\n"
            "topic的name必须很详细 详细地介绍这个topic应该学习的知识点。\n"
            "直接输出数组本身，不要包含 schema 或字段说明，不要再包裹任何键名。\n"
            "请严格按照以下 JSON Schema 输出（顶层为数组，每个元素为单元）：\n"
            f"{json.dumps(json_schema['schema'], ensure_ascii=False)}"
        )
        raw = self._llm.chat_with_json_schema(prompt, json_schema)

        # 解析 JSON 响应，直接期望为数组
        try:
            syllabus = json.loads(raw)
            if isinstance(syllabus, dict):
                if isinstance(syllabus.get("items"), list):
                    syllabus = syllabus["items"]
                elif {"unit_id", "name", "estimated_minutes"} <= set(syllabus.keys()):
                    syllabus = [syllabus]
                else:
                    syllabus = []
            elif not isinstance(syllabus, list):
                syllabus = []
        except Exception:  # noqa: BLE001
            syllabus = []

        # 若 LLM 返回的总时长超预算，按比例缩放（保底 20 分钟）
        if syllabus and total_budget > 0:
            sum_est = sum(int(u.get("estimated_minutes") or 0) for u in syllabus)
            if sum_est <= 0:
                avg = max(20, total_budget // max(len(syllabus), 1))
                for u in syllabus:
                    u["estimated_minutes"] = avg
            elif sum_est > total_budget:
                scale = total_budget / sum_est
                total_after = 0
                for u in syllabus:
                    est = int(u.get("estimated_minutes") or 0)
                    est = max(20, int(est * scale))
                    u["estimated_minutes"] = est
                    total_after += est
                # 若因取整超出预算，压缩最后一个单元
                if total_after > total_budget and syllabus:
                    overflow = total_after - total_budget
                    last = syllabus[-1]
                    last_est = int(last.get("estimated_minutes") or 0)
                    last["estimated_minutes"] = max(20, last_est - overflow)

        # 若 LLM 输出为空，按检索结果兜底构造
        if not syllabus:
            syllabus = self._fallback_syllabus(state)
        state["syllabus"] = syllabus
        logger.info(
            "plan.syllabus units=%d cost=%.2fs",
            len(syllabus),
            time.perf_counter() - t0,
        )
        return state

    def _allocate_daily_plan(self, state: PlanState) -> PlanState:
        t0 = time.perf_counter()
        topics_queue = self._flatten_topics(state.get("syllabus", []))
        start = state.get("start_date")
        end = state.get("end_date")
        daily_minutes = state.get("daily_minutes") or 60
        if not start or not end:
            raise ValueError("start_date/end_date is required for plan allocation")
        days = (end - start).days + 1
        dates = [start + timedelta(days=i) for i in range(days)]
        daily_plan: List[Dict[str, Any]] = []
        idx = 0
        for day_idx, d in enumerate(dates):
            minutes_left = daily_minutes
            today_items: List[Dict[str, Any]] = []
            while idx < len(topics_queue):
                item = topics_queue[idx]
                need = item["minutes"]
                # 最后一天：无论剩余多少，都塞进来，避免漏掉内容
                if day_idx == len(dates) - 1:
                    today_items.append(item)
                    idx += 1
                    continue
                if need > minutes_left and today_items:
                    break
                if need > minutes_left and not today_items:
                    today_items.append(item)
                    idx += 1
                    break
                today_items.append(item)
                minutes_left -= need
                idx += 1
            if not today_items and idx >= len(topics_queue):
                today_items.append(
                    {
                        "unit_id": "review",
                        "unit_name": "复习巩固",
                        "topic_id": "review",
                        "topic_name": "自由复盘/整理笔记/查漏补缺",
                        "chunk_ids": [],
                        "minutes": max(10, daily_minutes // 3),
                    }
                )
            outline_items = [
                {
                    "unit_id": t["unit_id"],
                    "unit_name": t["unit_name"],
                    "topic_id": t["topic_id"],
                    "topic_name": t["topic_name"],
                    "chunk_ids": t.get("chunk_ids", []),
                    "estimated_minutes": t["minutes"],
                }
                for t in today_items
            ]
            title = self._build_title(day_idx, today_items)
            daily_plan.append(
                {
                    "date": d.isoformat(),
                    "title": title,
                    "outline": outline_items,
                }
            )
        state["daily_plan"] = daily_plan
        logger.info(
            "plan.allocate days=%d topics=%d cost=%.2fs",
            len(dates),
            len(topics_queue),
            time.perf_counter() - t0,
        )
        return state

    # ===== 公共入口 =====
    def run(self, data: PlanInput) -> PlanState:
        state: PlanState = {
            "user_id": data.user_id,
            "goal_description": data.goal_description,
            "start_date": data.start_date,
            "end_date": data.end_date,
            "daily_minutes": data.daily_minutes,
            "doc_ids": data.doc_ids,
        }
        logger.info(
            "plan.run start goal=%s start=%s end=%s daily=%s doc_ids=%s",
            data.goal_description,
            data.start_date,
            data.end_date,
            data.daily_minutes,
            data.doc_ids,
        )
        if self._graph:
            return cast(PlanState, self._graph.invoke(state))
        # 顺序执行兜底
        for fn in (
            self._expand_goal,
            self._retrieve_topics,
            self._build_syllabus,
            self._allocate_daily_plan,
        ):
            state = fn(state)
        return state

    # ===== 工具方法 =====
    @staticmethod
    def _safe_parse_list(raw: str) -> List[str]:
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(x) for x in data if str(x).strip()]
        except Exception:  # noqa: BLE001
            pass
        # 尝试从文本中提取类似 [..]
        if "[" in raw and "]" in raw:
            try:
                frag = raw[raw.index("[") : raw.rindex("]") + 1]
                data = json.loads(frag)
                if isinstance(data, list):
                    return [str(x) for x in data if str(x).strip()]
            except Exception:  # noqa: BLE001
                return []
        return []

    @staticmethod
    def _safe_parse_syllabus(raw: str) -> List[Dict[str, Any]]:
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
        except Exception:  # noqa: BLE001
            pass
        # 容错：查找首尾 [] 片段
        if "[" in raw and "]" in raw:
            try:
                frag = raw[raw.index("[") : raw.rindex("]") + 1]
                data = json.loads(frag)
                if isinstance(data, list):
                    return data
            except Exception:  # noqa: BLE001
                return []
        return []

    @staticmethod
    def _fallback_syllabus(state: PlanState) -> List[Dict[str, Any]]:
        units: List[Dict[str, Any]] = []
        for idx, (topic, hits) in enumerate(
            state.get("topic_hits", {}).items(),
            1,
        ):
            units.append(
                {
                    "unit_id": f"U{idx}",
                    "name": topic,
                    "estimated_minutes": 90,
                    "topics": [
                        {
                            "topic_id": f"U{idx}-T1",
                            "name": topic,
                            "chunk_ids": [h.get("chunk_id") for h in hits[:3]],
                        }
                    ],
                }
            )
        return units

    @staticmethod
    def _flatten_topics(
        syllabus: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for unit in syllabus:
            topics = unit.get("topics", []) or []
            est = int(unit.get("estimated_minutes") or 60)
            default_minutes = max(20, min(90, est // max(len(topics), 1)))
            for topic in topics:
                items.append(
                    {
                        "unit_id": unit.get("unit_id"),
                        "unit_name": unit.get("name"),
                        "topic_id": topic.get("topic_id"),
                        "topic_name": topic.get("name"),
                        "chunk_ids": topic.get("chunk_ids") or [],
                        "minutes": int(
                            topic.get("estimated_minutes") or default_minutes
                        ),
                    }
                )
        return items

    @staticmethod
    def _build_title(day_idx: int, items: List[Dict[str, Any]]) -> str:
        if not items:
            return f"Day{day_idx + 1}: 待定"
        names: List[str] = []
        for it in items:
            name = it.get("unit_name") or it.get("topic_name")
            if name and name not in names:
                names.append(name)
        title = " / ".join(names) if names else "待定"
        return f"Day{day_idx + 1}: {title}"


class PlanService:
    def __init__(self, vec: MilvusVectorStore, llm: LLMClient) -> None:
        self._workflow = PlanWorkflow(vec, llm)

    def create_plan(
        self,
        db: Session,
        user_id: int,
        goal_description: str,
        start_date: date,
        end_date: date,
        daily_minutes: int,
        doc_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        state = self._workflow.run(
            PlanInput(
                user_id=user_id,
                goal_description=goal_description,
                start_date=start_date,
                end_date=end_date,
                daily_minutes=daily_minutes,
                doc_ids=doc_ids,
            )
        )
        plan = LearningPlan(
            user_id=user_id,
            goal_text=goal_description,
            start_date=start_date,
            end_date=end_date,
            daily_minutes=daily_minutes,
            status="active",
        )
        db.add(plan)
        db.flush()

        tasks_to_add: List[DailyTask] = []
        for item in state.get("daily_plan", []):
            tasks_to_add.append(
                DailyTask(
                    plan_id=plan.id,
                    user_id=user_id,
                    task_date=date.fromisoformat(item["date"]),
                    title=item.get("title") or "学习任务",
                    outline_json={
                        "goal": goal_description,
                        "outline": item.get("outline", []),
                        "not_covered": state.get("not_covered", []),
                    },
                    status="pending",
                )
            )
        db.add_all(tasks_to_add)
        logger.info(
            "plan.create persisted plan_id=%s tasks=%d not_covered=%d",
            plan.id,
            len(tasks_to_add),
            len(state.get("not_covered", [])),
        )
        return {
            "plan_id": plan.id,
            "syllabus": state.get("syllabus", []),
            "daily_plan": state.get("daily_plan", []),
            "not_covered": state.get("not_covered", []),
        }

    @staticmethod
    def list_daily_tasks(
        db: Session, plan_id: int, user_id: int
    ) -> List[Dict[str, Any]]:
        stmt = (
            select(DailyTask)
            .where(DailyTask.plan_id == plan_id, DailyTask.user_id == user_id)
            .order_by(DailyTask.task_date)
        )
        tasks = db.scalars(stmt).all()
        return [
            {
                "id": t.id,
                "date": t.task_date.isoformat(),
                "title": t.title,
                "outline": t.outline_json,
                "status": t.status,
            }
            for t in tasks
        ]

    @staticmethod
    def complete_daily_task(
        db: Session,
        user_id: int,
        task_id: int,
        status: str = "done",
    ) -> Optional[DailyTask]:
        """标记 daily_task 完成/跳过，返回任务对象或 None。"""
        task = db.get(DailyTask, task_id)
        if task is None or task.user_id != user_id:
            return None
        task.status = status
        db.add(task)
        return task

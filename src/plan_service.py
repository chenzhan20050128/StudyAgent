"""规划工作流 + 落库服务。

- 基于 LLM 的学习大纲生成（仅依赖用户知识库的真实内容）。
- LangGraph Plan Workflow：依照“扩展意图 -> RAG 检索 -> 大纲生成 -> 日计划拆分”节点串联。
- learning_plans / daily_tasks 写入与查询接口。

- 为兼容无 langgraph 环境，工作流节点按顺序执行；若检测到 langgraph，可动态编译 StateGraph，但不会阻塞主流程。
- 所有 JSON 解析均提供兜底策略，避免 LLM 输出格式偏差导致计划生成中断。

需求
- 把“用户目标 + 时间范围 + 每日时长 + 可选文档范围”转成可执行的学习计划。
- 输出并落库两层结构：LearningPlan（总计划）+ DailyTask（每日任务）。

业务逻辑
- 目标扩展：将 goal_description 拆成更细的主题列表（topics），便于检索与组织。
- 知识覆盖检查：对每个主题做向量库检索，产出 topic_hits，并标记 not_covered（命中为空的主题）。
- 大纲生成：基于“检索到的真实片段 + 目标约束”让 LLM 生成 syllabus（单元/主题/预计时长）。
- 日计划拆分：把 syllabus 的预计时长按天数与 daily_minutes 约束切分成 daily_tasks。
- 可靠性兜底：
    - 没有 langgraph 时顺序执行节点；有 langgraph 时编译为 StateGraph（不影响主流程）。
    - LLM 输出解析失败时提供默认/降级策略，避免整条链路中断。

主要实现方式
- LLM：通过 `LLMClient` 做 chat 与结构化 JSON 输出（带解析兜底）。
- 检索：通过 `MilvusVectorStore.hybrid_search()` 对 topic 做召回（必要时可用 doc_ids 限定范围）。
- 持久化：使用 SQLAlchemy Session 写入 `LearningPlan` / `DailyTask` 并提供查询能力。
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

    # ===== 输入字段（用户提供） =====

    # 用户唯一标识，用于数据隔离和知识库检索
    user_id: int

    # 学习目标描述文本(例如: "掌握Python机器学习")
    goal_description: str

    # 学习计划的开始日期
    start_date: date

    # 学习计划的结束日期
    end_date: date

    # 每日学习时长，单位: 分钟
    daily_minutes: int

    # 可选：用户已上传的文档ID列表，用于限定知识库检索范围
    # 如果为None，则检索该用户的所有文档
    doc_ids: Optional[List[int]]

    # ===== 中间字段（由节点生成） =====

    # expand_goal 节点输出
    # 将用户目标拆分后的子主题列表(例如: ["Python基础", "数据结构", "算法"])
    topics: List[str]

    # retrieve_topics 节点输出
    # 每个主题检索到的相关文档片段
    # 格式: {
    #   "Python基础": [
    #     {"doc_id": 1, "chunk_id": "c1", "content": "..."},
    #     {"doc_id": 1, "chunk_id": "c2", "content": "..."}
    #   ],
    #   "数据结构": [...]
    # }
    topic_hits: Dict[str, List[Dict[str, Any]]]

    # retrieve_topics 节点输出
    # 在知识库中未覆盖的主题列表(即无法检索到相关内容的主题)
    # 这些主题需要用户补充资料或自主学习
    not_covered: List[str]

    # build_syllabus 节点输出
    # 学习大纲：按单元(unit)组织，每个单元包含多个主题(topic)
    # 格式示例: [
    #   {
    #     "unit_id": "U1",
    #     "name": "Python基础",
    #     "estimated_minutes": 300,
    #     "topics": [
    #       {
    #         "topic_id": "U1-T1",
    #         "name": "变量、数据类型、运算符",
    #         "chunk_ids": ["c1", "c2", "c3"]
    #       }
    #     ]
    #   }
    # ]
    syllabus: List[Dict[str, Any]]

    # allocate_daily_plan 节点输出
    # 按日期分配的学习计划
    # 格式示例: [
    #   {
    #     "date": "2024-01-01",
    #     "title": "Day1: Python基础 / 变量类型",
    #     "outline": [
    #       {
    #         "unit_id": "U1",
    #         "unit_name": "Python基础",
    #         "topic_id": "U1-T1",
    #         "topic_name": "变量、数据类型、运算符",
    #         "chunk_ids": ["c1", "c2"],
    #         "estimated_minutes": 90
    #       }
    #     ]
    #   },
    #   ...
    # ]
    daily_plan: List[Dict[str, Any]]

    # ===== 错误处理 =====

    # 如果规划过程中发生错误，记录错误信息
    # 正常情况下为 None
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
    """
    学习规划工作流

    四个节点顺序执行：
    1. expand_goal: 将用户目标拆分为子主题
    2. retrieve_topics: 从知识库检索每个子主题的相关内容
    3. build_syllabus: 组织成结构化的学习大纲
    4. allocate_daily_plan: 按日期分配学习任务
    """

    def __init__(self, vec: MilvusVectorStore, llm: LLMClient) -> None:
        # vec: 用户知识库检索入口（hybrid_search）
        # llm: 目标拆分/大纲生成所需的 LLM
        self._vec = vec
        self._llm = llm
        # 可选：若安装了 langgraph，则编译成状态图；否则走顺序执行兜底
        self._graph = self._build_graph()

    # ===== 图编排（可选） =====
    def _build_graph(self):
        """
        构建 LangGraph 工作流图（如果 langgraph 库可用）
        - 若 langgraph 不可用，返回 None，后续走顺序执行兜底
        - 若可用，编译图以支持并行执行和状态管理
        """
        if not StateGraph or START is None or END is None:
            # 缺少 langgraph 时不阻塞主流程；后续 run() 会按顺序执行节点
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
        """
        节点1：目标拆分

        将用户的学习目标自然语言描述拆分为 3-6 个具体的子主题。
        例如："掌握Python机器学习" -> ["Python基础", "NumPy/Pandas", "算法原理", ...]
        """
        # 节点 1：把用户的“学习大目标”拆成若干可检索的小主题（3~6 个）
        goal = state.get("goal_description", "")
        t0 = time.perf_counter()
        prompt = (
            "你是学习规划助手。请根据用户目标拆分 3-6 个子主题，不要返回描述，"
            '仅返回 JSON 数组，例如: ["Python 基础", "数据结构"].\n'
            f"用户目标: {goal}"
        )
        # 输出约束为 JSON 数组，避免模型输出散文导致解析失败
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
        """
        节点2：知识库检索

        对每个子主题进行向量+文本混合检索，从用户的知识库中找到相关学习资料。
        - 结果分为两部分：
          * topic_hits: 找到相关资料的主题及其片段
          * not_covered: 知识库中未涵盖的主题（需要用户补充或自学）
        - 每个主题最多检索 12 个相关片段
        """
        topic_hits: Dict[str, List[Dict[str, Any]]] = {}
        not_covered: List[str] = []
        t0 = time.perf_counter()
        goal = state.get("goal_description", "")
        user_id = state.get("user_id")
        if user_id is None:
            raise ValueError("user_id is required for plan workflow")

        # 节点 2：对每个子主题做检索，确保计划“有真实资料支撑”
        for topic in state.get("topics", []):
            # 组合目标和子主题作为查询，提高检索精度
            q = f"{goal} | {topic}"
            # 使用 LLM 生成查询的向量表示
            dense = self._llm.embed([q])[0]
            # 执行混合搜索（向量 + 关键词）
            hits = self._vec.hybrid_search(
                query_vector=dense,
                query_text=q,
                user_id=user_id,
                doc_ids=state.get("doc_ids"),
                limit=12,
            )
            if not hits:
                # 知识库中没有覆盖该主题：后续会回传 not_covered，提示用户补充资料
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
        """
        节点3：生成学习大纲

        核心逻辑：
        1. 从检索结果中提取文本片段，作为 LLM 的上下文
        2. 计算总学习时长预算（天数 × 每日分钟数）
        3. 提示 LLM 生成结构化大纲（JSON 格式），约束条件：
           - 大纲单元数量在合理范围内
           - 所有单元的学习时长总和不超过预算
           - 所有主题都有真实的文档片段支撑（不凭空生成）
        4. 对 LLM 输出进行容错处理，确保能解析出有效大纲
        5. 若大纲总时长超预算，按比例缩放
        6. 若 LLM 输出为空，使用检索结果作兜底方案
        """
        # 节点 3：生成学习大纲（syllabus）
        # 核心原则：
        # - 只基于检索到的真实片段归纳（不允许凭空编造）
        # - 使用 JSON Schema 约束输出结构可解析
        # - 强制控制总学习时长 <= 预算（天数 * daily_minutes），并做兜底校准

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

        # 计算时间预算：用户指定的学习周期和每日时间
        start = state.get("start_date")
        end = state.get("end_date")
        daily_minutes = state.get("daily_minutes") or 60
        if start and end:
            days = (end - start).days + 1
        else:
            days = 3
        total_budget = daily_minutes * days

        # 计算大纲单元数的合理范围
        topics_cnt = max(1, len(state.get("topics", [])))
        min_items = (
            max(2, min(topics_cnt, days or topics_cnt)) if days else max(2, topics_cnt)
        )
        max_items = max(
            min_items,
            min(topics_cnt + 1, max(days, topics_cnt)),
        )

        # 定义严格的 JSON Schema，确保 LLM 输出结构化且可解析
        # 顶层为数组，每个元素代表一个学习单元(unit)
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

        # 调用 LLM 生成大纲（使用 JSON Schema 确保结构化输出）
        t0 = time.perf_counter()
        prompt = (
            "你是学习计划设计师。仅基于下列真实片段归纳学习大纲，"
            "确保所有主题都能从片段溯源，不要凭空生成。\n"
            f"用户目标: {state.get('goal_description', '')}\n"
            f"片段（带 chunk_id ）:\n{joined}\n"
            f"总学习天数: {days} 天，每天 {daily_minutes} 分钟，总时长预算 <= "
            f"{total_budget} 分钟。\n"
            "请确保所有单元的 estimated_minutes 求和不超过总预算，"
            "并尽量均衡。\n"
            f"至少输出 {min_items} 个单元，"
            "不要只返回一个单元。你的单元可以拆分的细粒度一些。\n"
            "每个单元下的name不允许跟时间有关，只是知识点的总结。\n"
            "topic的name必须很详细 详细地介绍这个topic应该学习"
            "的知识点。\n"
            "直接输出数组本身，不要包含 schema 或字段说明，"
            "不要再包裹任何键名。\n"
            "请严格按照以下 JSON Schema 输出"
            "（顶层为数组，每个元素为单元）：\n"
            f"{json.dumps(json_schema['schema'], ensure_ascii=False)}"
        )
        raw = self._llm.chat_with_json_schema(prompt, json_schema)

        # 容错解析：处理 LLM 可能返回的各种格式
        # - 直接返回数组
        # - 返回 {"items": [...]} 格式
        # - 返回单个对象
        try:
            syllabus = json.loads(raw)
            if isinstance(syllabus, dict):
                if isinstance(syllabus.get("items"), list):
                    syllabus = syllabus["items"]
                elif {
                    "unit_id",
                    "name",
                    "estimated_minutes",
                } <= set(syllabus.keys()):
                    syllabus = [syllabus]
                else:
                    syllabus = []
            elif not isinstance(syllabus, list):
                syllabus = []
        except Exception:  # noqa: BLE001
            syllabus = []

        # 时长预算校准：若 LLM 输出的总时长超预算，按比例缩放
        # 目的：确保生成的计划可在指定时间内完成
        if syllabus and total_budget > 0:
            sum_est = sum(int(u.get("estimated_minutes") or 0) for u in syllabus)
            if sum_est <= 0:
                # 若所有单元时长都为 0，平均分配预算
                avg = max(20, total_budget // max(len(syllabus), 1))
                for u in syllabus:
                    u["estimated_minutes"] = avg
            elif sum_est > total_budget:
                # 按比例缩放，保证每个单元至少 20 分钟
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

        # 若 LLM 输出为空，按检索结果兜底构造（保证主流程不被模型波动卡死）
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
        """
        节点4：生成日计划

        核心逻辑：
        1. 将大纲中的所有主题展开成一个队列（保持顺序）
        2. 按日期遍历，每天根据剩余时间来分配主题
        3. 贪心策略：
           - 优先分配能完整放入当天的主题
           - 若某主题超过剩余时间但当天无内容，仍强制分配
           - 最后一天：将所有剩余主题都塞进去，确保没有遗漏
        4. 若某天最后无内容（全部分配完毕），添加"复习巩固"任务
        5. 生成日标题（显示当天的主题名称）
        """
        t0 = time.perf_counter()
        # 节点 4：把大纲拆成“每日学习任务”
        # 算法是典型的“队列 + 贪心分配”，并带三个关键边界规则：
        # 1) 当天已有内容时，遇到超时 topic 就停止（留给明天）
        # 2) 当天无内容时，即使超时也强塞一个，避免空日
        # 3) 最后一天无条件把剩余全部塞进去，确保不遗漏

        # 展开大纲为主题队列，方便按顺序分配
        topics_queue = self._flatten_topics(state.get("syllabus", []))
        start = state.get("start_date")
        end = state.get("end_date")
        daily_minutes = state.get("daily_minutes") or 60
        if not start or not end:
            msg = "start_date/end_date is required for plan allocation"
            raise ValueError(msg)

        # 生成日期列表
        days = (end - start).days + 1
        dates = [start + timedelta(days=i) for i in range(days)]
        daily_plan: List[Dict[str, Any]] = []
        idx = 0

        # 按日期分配主题
        for day_idx, d in enumerate(dates):
            minutes_left = daily_minutes
            today_items: List[Dict[str, Any]] = []

            # 贪心分配：尽量多地加入今天的日程
            while idx < len(topics_queue):
                item = topics_queue[idx]
                need = item["minutes"]

                # 最后一天：无论剩余多少，都塞进来，避免漏掉内容
                if day_idx == len(dates) - 1:
                    today_items.append(item)
                    idx += 1
                    continue

                # 若当天已有内容，新项超时则停止
                if need > minutes_left and today_items:
                    break

                # 若当天还无内容，即使超时也强制加入
                if need > minutes_left and not today_items:
                    today_items.append(item)
                    idx += 1
                    break

                # 正常情况：加入并更新剩余时间
                today_items.append(item)
                minutes_left -= need
                idx += 1

            # 若当天无内容（所有主题都已分配），添加复习任务
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

            # 组织当天的学习大纲
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
        """
        执行完整的规划工作流

        执行顺序：
        1. 若 langgraph 可用，使用并行图执行（高效）
        2. 否则，顺序执行四个节点（兼容性好）
        """
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
    """学习规划服务 - 对外API接口"""

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
        """
        创建学习计划（完整工作流 + 落库）

        流程：
        1. 执行规划工作流，生成 syllabus（大纲）和 daily_plan（日计划）
        2. 将计划写入 learning_plans 表
        3. 将每日任务写入 daily_tasks 表
        4. 返回计划ID、大纲、日计划和未覆盖的主题
        """
        # 对外服务入口：执行工作流得到 syllabus/daily_plan/not_covered，然后落库
        # 这里的“落库”保证：计划生成一次即可持久化，后续可查看/统计/复习联动

        # 执行工作流
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

        # 创建学习计划记录（learning_plans）
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

        # 创建日常任务记录（daily_tasks）：每天一条，outline_json 保存结构化大纲与溯源信息
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
        """
        查询某个学习计划的所有日常任务

        按日期排序返回该计划下的所有日常任务，包括：
        - 任务ID、日期、标题、学习大纲、完成状态
        """
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
        status: str = "completed",
    ) -> Optional[DailyTask]:
        """
        标记日常任务为完成/跳过

        参数：
        - status: "completed"(完成) 或 "skipped"(跳过)

        返回：
        - 更新后的任务对象，如果任务不存在或权限不符则返回 None
        """
        task = db.get(DailyTask, task_id)
        if task is None or task.user_id != user_id:
            return None

        # 统一状态枚举，只允许使用规范值，防止出现 "done" 等临时文案
        allowed_status = {"pending", "in_progress", "completed", "skipped"}
        if status not in allowed_status:
            # 非法状态一律按 completed 处理，避免前端/旧版本传入脏数据
            status = "completed"

        task.status = status
        db.add(task)
        return task

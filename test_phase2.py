"""阶段二：学习计划端到端自测。

用例覆盖：
- 上传示例文档 -> 构建 Milvus 向量。
- 调用 /api/plans/create 生成学习计划（大纲 + 每日任务）。
- 查询每日任务接口，验证持久化成功。

说明：
- 依赖真实 LLM / Milvus / MySQL，若缺少 DASHSCOPE_API_KEY 则跳过。
"""

from datetime import date, timedelta
import os
import time

import pytest
from fastapi.testclient import TestClient

from main import app
from src.db import SessionLocal
from src.models import LearningPlan, DailyTask


pytestmark = pytest.mark.skipif(
    not os.getenv("DASHSCOPE_API_KEY"),
    reason="需要 DASHSCOPE_API_KEY 才能跑阶段二端到端自测",
)

client = TestClient(app)


def _print_section(title: str) -> None:
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def test_phase2_plan_roundtrip():
    _print_section("阶段二：计划生成")

    # 2) 创建 3 天计划
    start = date.today()
    end = start + timedelta(days=2)
    body = {
        "goal_description": "三天搞定 Python 基础语法与常见算法",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily_minutes": 45,
        "doc_ids": [26],
    }
    t0 = time.perf_counter()
    plan_resp = client.post("/api/plans/create", json=body)
    total_cost = time.perf_counter() - t0
    assert plan_resp.status_code == 200
    payload = plan_resp.json()

    print(f"[TIMING] /api/plans/create total_cost={total_cost:.2f}s")

    assert payload.get("plan_id")
    assert payload.get("daily_plan"), "daily_plan 不能为空"

    # 3) 查询每日任务
    plan_id = payload["plan_id"]
    task_resp = client.get(f"/api/plans/{plan_id}/daily-tasks")
    assert task_resp.status_code == 200
    tasks = task_resp.json().get("tasks", [])
    assert tasks, "应至少有一天任务"
    assert all("outline" in t for t in tasks)

    # 4) MySQL 持久化校验：learning_plans / daily_tasks 是否存在且匹配
    with SessionLocal() as session:
        plan_row = session.get(LearningPlan, plan_id)
        assert plan_row is not None, "learning_plans 中应存在该计划"
        assert plan_row.goal_text == body["goal_description"]
        assert plan_row.user_id == 1

        print(
            "[DB] learning_plans row:",
            {
                "id": plan_row.id,
                "user_id": plan_row.user_id,
                "goal_text": plan_row.goal_text,
                "start_date": plan_row.start_date.isoformat(),
                "end_date": plan_row.end_date.isoformat(),
                "daily_minutes": plan_row.daily_minutes,
                "status": plan_row.status,
            },
        )

        task_rows = (
            session.query(DailyTask)
            .filter(DailyTask.plan_id == plan_id, DailyTask.user_id == 1)
            .all()
        )
        assert task_rows, "daily_tasks 中应有对应任务"
        # 确认日期范围与 API 返回一致
        api_dates = {t["date"] for t in tasks}
        db_dates = {t.task_date.isoformat() for t in task_rows}
        assert api_dates == db_dates

        print("[DB] daily_tasks rows:")
        for t in task_rows:
            print(
                {
                    "id": t.id,
                    "plan_id": t.plan_id,
                    "user_id": t.user_id,
                    "task_date": t.task_date.isoformat(),
                    "title": t.title,
                    "status": t.status,
                    "outline_json": t.outline_json,
                }
            )

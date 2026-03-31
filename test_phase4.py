"""阶段四：复习与闭环自测。

覆盖点：
- 创建学习计划 -> 生成 daily_tasks。
- 标记 daily_task 完成并指定 completed_date，使今日产生复习排期（艾宾浩斯首日）。
- 查询今日复习任务 -> 校验 pending 任务存在。
- 完成一条复习任务 -> 再次查询，确认已从 pending 列表移除。

依赖：真实 LLM/Milvus/MySQL；无 DASHSCOPE_API_KEY 时跳过。
"""

from datetime import date, timedelta
import os

import pytest
from fastapi.testclient import TestClient

from main import app
from src.db import SessionLocal
from src.models import ReviewSchedule, WeakPoint
from src.review_service import ReviewService

pytestmark = pytest.mark.skipif(
    not os.getenv("DASHSCOPE_API_KEY"),
    reason="需要 DASHSCOPE_API_KEY 才能跑阶段四端到端自测",
)

client = TestClient(app)


def _print_section(title: str) -> None:
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def test_phase4_review_roundtrip() -> None:
    _print_section("阶段四：复习排期与闭环")
    log = "[PHASE4_TEST]"
    review_service = ReviewService()

    start = date.today() - timedelta(days=1)
    end = start + timedelta(days=1)
    plan_body = {
        "goal_description": "两天完成 Python 复盘",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily_minutes": 30,
        "doc_ids": [26],
    }
    plan_resp = client.post("/api/plans/create", json=plan_body)
    assert plan_resp.status_code == 200
    plan_id = plan_resp.json().get("plan_id")
    assert plan_id
    print(f"{log} plan created id={plan_id}")

    tasks_resp = client.get(f"/api/plans/{plan_id}/daily-tasks")
    assert tasks_resp.status_code == 200
    tasks = tasks_resp.json().get("tasks", [])
    assert tasks, "应该至少生成一个日任务"
    first_task_id = tasks[0]["id"]
    print(f"{log} first_task_id={first_task_id}")

    # 将完成日期设为昨天，使首个复习间隔 (1 天) 落在今天
    yesterday = date.today() - timedelta(days=1)
    complete_resp = client.post(
        f"/api/daily-tasks/{first_task_id}/complete",
        json={"status": "done", "completed_date": yesterday.isoformat()},
    )
    assert complete_resp.status_code == 200
    created_reviews = complete_resp.json().get("created_reviews")
    assert created_reviews and created_reviews == len(review_service.EB_INTERVALS)
    print(
        f"{log} created_reviews={created_reviews} "
        f"intervals={review_service.EB_INTERVALS}"
    )

    with SessionLocal() as db:
        schedules = (
            db.query(ReviewSchedule)
            .filter(
                ReviewSchedule.target_type == "daily_task",
                ReviewSchedule.target_id == first_task_id,
            )
            .order_by(ReviewSchedule.stage)
            .all()
        )
        assert len(schedules) == len(review_service.EB_INTERVALS)
        base = yesterday
        for sch, days in zip(schedules, review_service.EB_INTERVALS):
            assert sch.scheduled_date == base + timedelta(days=days)
        print(
            f"{log} daily_task schedule dates="
            f"{[s.scheduled_date.isoformat() for s in schedules]}"
        )

    today_resp = client.get("/api/reviews/today")
    assert today_resp.status_code == 200
    items = today_resp.json().get("items", [])
    assert items, "今天应有至少一条待复习记录"
    first_review_id = items[0]["review_id"]
    print(
        f"{log} today reviews count={len(items)} " f"first_review_id={first_review_id}"
    )

    # 完成一条复习，pending 列表应减少
    complete_review_resp = client.post(
        f"/api/reviews/{first_review_id}/complete",
        json={"status": "completed"},
    )
    assert complete_review_resp.status_code == 200

    with SessionLocal() as db:
        row = db.get(ReviewSchedule, first_review_id)
        assert row is not None and row.status == "completed"
        assert row.completed_at is not None
        print(
            f"{log} history captured review_id={row.id} "
            f"status={row.status} completed_at={row.completed_at}"
        )

    after_resp = client.get("/api/reviews/today")
    assert after_resp.status_code == 200
    after_items = after_resp.json().get("items", [])
    assert len(after_items) <= len(items) - 1, "完成后待办应减少"
    print(f"{log} today reviews after complete: {len(after_items)}")

    # 覆盖薄弱点复习排期
    with SessionLocal() as db:
        wp = WeakPoint(
            user_id=1,
            quiz_id=9999,
            attempt_id=9999,
            description="weak point placeholder",
            related_doc_id=None,
            related_chunk_ids=None,
            level="high",
        )
        db.add(wp)
        db.commit()
        db.refresh(wp)

        created_wp_reviews = review_service.create_for_weak_point(
            db,
            user_id=1,
            weak_point_id=wp.id,
            base_date=date.today() - timedelta(days=1),
        )
        db.commit()
        assert created_wp_reviews == len(review_service.EB_INTERVALS)
        today_items = review_service.list_today_reviews(
            db, user_id=1, today=date.today()
        )
        has_wp_today = any(
            it.get("target_type") == "weak_point"
            and it.get("target", {}).get("id") == wp.id
            for it in today_items
        )
        assert has_wp_today, "薄弱点应产生今日复习"
        print(
            f"{log} weak_point schedule count={created_wp_reviews} "
            f"today_total={len(today_items)}"
        )

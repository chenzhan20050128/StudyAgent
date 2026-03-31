"""复习提醒服务：基于艾宾浩斯间隔生成复习计划，查询与完成状态。

设计原则：
- 只在 Service 层做持久化与业务判断，不做 LLM 调用（留接口扩展）。
- 避免重复生成：同一 target_type/target_id/stage 不重复写入。
- 记录历史：review_schedules.status + completed_at 即为复习历史。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from .models import DailyTask, ReviewSchedule, WeakPoint


class ReviewService:
    EB_INTERVALS = [1, 2, 4, 7, 15, 30]

    def create_for_daily_task(
        self,
        db: Session,
        user_id: int,
        task_id: int,
        completed_date: Optional[date] = None,
    ) -> int:
        """为已完成的 daily_task 生成复习排期，返回新建条数。"""
        completed_date = completed_date or date.today()
        created = 0
        for idx, days in enumerate(self.EB_INTERVALS, start=1):
            scheduled_date = completed_date + timedelta(days=days)
            exists = (
                db.query(ReviewSchedule)
                .filter(
                    ReviewSchedule.user_id == user_id,
                    ReviewSchedule.target_type == "daily_task",
                    ReviewSchedule.target_id == task_id,
                    ReviewSchedule.stage == idx,
                )
                .first()
            )
            if exists:
                continue
            db.add(
                ReviewSchedule(
                    user_id=user_id,
                    target_type="daily_task",
                    target_id=task_id,
                    stage=idx,
                    scheduled_date=scheduled_date,
                    status="pending",
                )
            )
            created += 1
        return created

    def create_for_weak_point(
        self,
        db: Session,
        user_id: int,
        weak_point_id: int,
        base_date: Optional[date] = None,
    ) -> int:
        """针对薄弱点生成复习排期，返回新建条数。"""
        base_date = base_date or date.today()
        created = 0
        for idx, days in enumerate(self.EB_INTERVALS, start=1):
            scheduled_date = base_date + timedelta(days=days)
            exists = (
                db.query(ReviewSchedule)
                .filter(
                    ReviewSchedule.user_id == user_id,
                    ReviewSchedule.target_type == "weak_point",
                    ReviewSchedule.target_id == weak_point_id,
                    ReviewSchedule.stage == idx,
                )
                .first()
            )
            if exists:
                continue
            db.add(
                ReviewSchedule(
                    user_id=user_id,
                    target_type="weak_point",
                    target_id=weak_point_id,
                    stage=idx,
                    scheduled_date=scheduled_date,
                    status="pending",
                )
            )
            created += 1
        return created

    def list_today_reviews(
        self, db: Session, user_id: int, today: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """查询今日需要复习的任务（pending）。"""
        today = today or date.today()
        rows: List[ReviewSchedule] = (
            db.query(ReviewSchedule)
            .filter(
                ReviewSchedule.user_id == user_id,
                ReviewSchedule.status == "pending",
                ReviewSchedule.scheduled_date == today,
            )
            .all()
        )
        results: List[Dict[str, Any]] = []
        for row in rows:
            detail: Dict[str, Any] = {
                "review_id": row.id,
                "target_type": row.target_type,
                "stage": row.stage,
                "scheduled_date": row.scheduled_date.isoformat(),
                "status": row.status,
            }
            if row.target_type == "daily_task":
                task = db.get(DailyTask, row.target_id)
                if task and task.user_id == user_id:
                    detail["target"] = {
                        "id": task.id,
                        "title": task.title,
                        "date": task.task_date.isoformat(),
                        "outline": task.outline_json,
                        "status": task.status,
                    }
            elif row.target_type == "weak_point":
                wp = db.get(WeakPoint, row.target_id)
                if wp and wp.user_id == user_id:
                    detail["target"] = {
                        "id": wp.id,
                        "description": wp.description,
                        "related_doc_id": wp.related_doc_id,
                        "related_chunk_ids": wp.related_chunk_ids,
                        "level": wp.level,
                    }
            results.append(detail)
        return results

    def complete_review(
        self,
        db: Session,
        user_id: int,
        review_id: int,
        status: str = "completed",
    ) -> None:
        row = db.get(ReviewSchedule, review_id)
        if not row or row.user_id != user_id:
            raise ValueError("review not found")
        row.status = status
        row.completed_at = datetime.utcnow()
        db.add(row)

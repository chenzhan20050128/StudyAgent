"""复习提醒服务：基于艾宾浩斯间隔生成复习计划，查询与完成状态。

设计原则：
- 只在 Service 层做持久化与业务判断，不做 LLM 调用（留接口扩展）。
- 避免重复生成：同一 target_type/target_id/stage 不重复写入。
- 记录历史：review_schedules.status + completed_at 即为复习历史。

需求
- 当用户完成学习任务或暴露薄弱点后，自动生成后续复习提醒。
- 支持查询“今天要复习什么”以及标记完成，形成可追溯的复习记录。

业务逻辑
- 排期：使用固定间隔表 EB_INTERVALS（近似艾宾浩斯/间隔重复），计算 scheduled_date。
- 幂等：同一 (user_id, target_type, target_id, stage) 只生成一次，避免重复插入。
- 查询：按 scheduled_date/today 过滤，返回待复习条目。
- 完成：更新 status 与 completed_at，作为历史与统计依据。

主要实现方式
- 纯 Service：仅做 SQLAlchemy 读写与业务判断，不做 LLM 调用。
- 目标抽象：target_type 支持 daily_task 与 weak_point 两类，便于扩展更多复习来源。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from .models import DailyTask, ReviewSchedule, WeakPoint


class ReviewService:
    # 固定的“间隔重复”间隔（天）：用离散表实现艾宾浩斯/间隔记忆的近似排期
    EB_INTERVALS = [1, 2, 4, 7, 15, 30]

    def create_for_daily_task(
        self,
        db: Session,
        user_id: int,
        task_id: int,
        completed_date: Optional[date] = None,
    ) -> int:
        """为已完成的 daily_task 生成复习排期，返回新建条数。

        设计要点：
        - 每个 daily_task 生成 6 条复习记录（stage=1..6）
        - scheduled_date = completed_date + interval
        - 幂等：同一 (target_type, target_id, stage) 不重复生成
        """
        completed_date = completed_date or date.today()
        created = 0
        for idx, days in enumerate(self.EB_INTERVALS, start=1):
            scheduled_date = completed_date + timedelta(days=days)
            # 幂等去重：避免重复调用导致同一阶段的复习任务被重复插入
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
        """针对薄弱点生成复习排期，返回新建条数。

        薄弱点（weak_point）来源通常是测验低分触发：
        - 一旦记录薄弱点，就立刻生成后续复习日程
        - 与 daily_task 一样，遵循幂等去重
        """
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
        """查询今日需要复习的任务（pending）。

        说明：当前实现是“拉取式提醒”：
        - 前端/对话层询问“今天要复习什么”，后端按 scheduled_date==today 返回列表
        - 若未来需做推送，可在此基础上加定时任务/通知渠道
        """
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
                # 复习来源是日任务：回填任务标题/日期/大纲，便于用户复盘
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
                # 复习来源是薄弱点：回填描述、关联 doc/chunk 等，便于针对性巩固
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
        # 完成复习：仅更新状态并写入完成时间戳；历史查询可据此回溯复习轨迹
        row = db.get(ReviewSchedule, review_id)
        if not row or row.user_id != user_id:
            raise ValueError("review not found")
        row.status = status
        row.completed_at = datetime.utcnow()
        db.add(row)

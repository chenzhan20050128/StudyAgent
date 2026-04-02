"""学习数据统计与仪表盘口径。

需求
- 提供面向“展示/面试讲解”的统计摘要：学习时长、任务完成趋势、测验得分、薄弱点数量等。

业务逻辑
- 以最近 7 天为窗口构造趋势（last_7_days_trend），并汇总：
    - total_learning_minutes：按“完成任务天数 * plan.daily_minutes”的近似口径
    - completed_task_days：近 7 天完成过任务的天数
    - average_mastery_score：测验平均分（内部 score * 20 映射到 0~100）
    - active_weak_points：未解决薄弱点数量
- 统计偏“稳定口径”而非绝对精确：优先保证可解释与可复现。

主要实现方式
- 直接用 SQLAlchemy query + 简单聚合（必要处用 func）。
- 输出结构为字典，供 API 层直接序列化返回。
"""

from datetime import date, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
from src.models import LearningPlan, DailyTask, QuizAttempt, WeakPoint


class StatsService:
    # StatsService：仪表盘统计
    # 本文件的统计是“展示导向”：以近 7 天趋势 + 总量指标为主，口径稳定即可。
    def get_dashboard_stats(self, db: Session, user_id: int):
        # dashboard 统计输出字段：
        # - total_learning_minutes：近似累计学习分钟（按完成任务天数 * plan.daily_minutes）
        # - completed_task_days：近 7 天中完成过任务的天数
        # - average_mastery_score：测验平均分映射到 0~100（score*20）
        # - active_weak_points：未解决薄弱点按等级计数
        # - last_7_days_trend：近 7 天的分钟数与状态（completed/pending/skipped/none）
        # - recent_weak_points：最近 10 条薄弱点摘要
        tasks = (
            db.query(DailyTask, LearningPlan)
            .join(LearningPlan, DailyTask.plan_id == LearningPlan.id)
            .filter(DailyTask.user_id == user_id)
            .all()
        )

        # 这里使用“按任务完成天累计 daily_minutes”的口径，简单但稳定
        total_learning_minutes = 0
        completed_task_dates = set()

        today = date.today()
        last_7_days = [today - timedelta(days=i) for i in range(6, -1, -1)]

        tasks_by_date = {d: [] for d in last_7_days}

        for task, plan in tasks:
            if task.status == "completed":
                total_learning_minutes += plan.daily_minutes
                completed_task_dates.add(task.task_date)

            if task.task_date in tasks_by_date:
                tasks_by_date[task.task_date].append((task, plan))

        completed_task_days = len(completed_task_dates)

        last_7_days_trend = []
        for d in last_7_days:
            day_tasks = tasks_by_date[d]
            if not day_tasks:
                last_7_days_trend.append(
                    {"date": d.isoformat(), "minutes": 0, "status": "none"}
                )
            else:
                has_completed = False
                day_minutes = 0
                day_status = "skipped"
                is_pending = False
                for t, p in day_tasks:
                    if t.status == "completed":
                        has_completed = True
                        day_minutes += p.daily_minutes
                    elif t.status == "pending":
                        is_pending = True

                if has_completed:
                    day_status = "completed"
                elif is_pending:
                    day_status = "pending"

                last_7_days_trend.append(
                    {
                        "date": d.isoformat(),
                        "minutes": day_minutes,
                        "status": day_status,
                    }
                )

        # 测验均分（数据库聚合，不拉全量 attempts）
        avg_score = (
            db.query(func.avg(QuizAttempt.score))
            .filter(QuizAttempt.user_id == user_id)
            .scalar()
        )
        average_mastery_score = 0.0
        if avg_score is not None:
            average_mastery_score = round(float(avg_score) * 20, 1)

        # 未解决薄弱点分布
        weak_points = (
            db.query(WeakPoint.level, func.count(WeakPoint.id))
            .filter(WeakPoint.user_id == user_id, WeakPoint.status == "pending")
            .group_by(WeakPoint.level)
            .all()
        )

        active_weak_points = {"high": 0, "medium": 0}
        for level, count in weak_points:
            if level in active_weak_points:
                active_weak_points[level] = count
            else:
                active_weak_points[level] = count

        recent_wps = (
            db.query(WeakPoint)
            .filter(WeakPoint.user_id == user_id, WeakPoint.status == "pending")
            .order_by(WeakPoint.created_at.desc())
            .limit(10)
            .all()
        )

        recent_weak_points = [
            {
                "id": wp.id,
                "level": wp.level,
                "description": wp.description,
                "created_at": wp.created_at.isoformat(),
            }
            for wp in recent_wps
        ]

        return {
            "total_learning_minutes": total_learning_minutes,
            "completed_task_days": completed_task_days,
            "average_mastery_score": average_mastery_score,
            "active_weak_points": active_weak_points,
            "last_7_days_trend": last_7_days_trend,
            "recent_weak_points": recent_weak_points,
        }

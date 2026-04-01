from datetime import date, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
from src.models import LearningPlan, DailyTask, QuizAttempt, WeakPoint


class StatsService:
    def get_dashboard_stats(self, db: Session, user_id: int):
        tasks = (
            db.query(DailyTask, LearningPlan)
            .join(LearningPlan, DailyTask.plan_id == LearningPlan.id)
            .filter(DailyTask.user_id == user_id)
            .all()
        )

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

        avg_score = (
            db.query(func.avg(QuizAttempt.score))
            .filter(QuizAttempt.user_id == user_id)
            .scalar()
        )
        average_mastery_score = 0.0
        if avg_score is not None:
            average_mastery_score = round(float(avg_score) * 20, 1)

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

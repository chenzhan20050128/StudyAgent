"""
数据模型定义模块

该模块定义了学习平台的所有核心数据表结构:
- Document: 学习文档管理
- LearningPlan: 学习计划管理
- DailyTask: 日常任务管理
- DocumentTag: 文档标签
- RAGQueryLog: 知识问答记录
- Quiz: 测试题目
- QuizAttempt: 用户答题记录
- WeakPoint: 薄弱点追踪
- ReviewSchedule: 复习计划安排
"""

from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Integer,
    String,
    Text,
    JSON,
    Float,
)
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class Document(Base):
    """
    文档表 - 管理用户上传的所有学习文档

    表名: documents
    用途: 存储PDF、Word等文档的元数据和访问路径
    """

    __tablename__ = "documents"

    # 文档唯一标识，自动递增主键
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )
    # 上传用户ID，用于多用户隔离
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 文档标题/名称
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    # 来源类型: "pdf"、"docx"、"txt"等
    source_type: Mapped[str] = mapped_column(String(16), nullable=False)
    # 本地存储路径(如果是上传的文件)
    file_path: Mapped[Optional[str]] = mapped_column(String(512))
    # 源网址(如果是网页抓取)
    source_url: Mapped[Optional[str]] = mapped_column(String(512))
    # 文档状态: "uploaded"、"processing"、"indexed"等
    status: Mapped[str] = mapped_column(String(16), default="uploaded")
    # 创建时间，自动设置为UTC当前时间
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )


class LearningPlan(Base):
    """
    学习计划表 - 用户的学习目标和计划

    表名: learning_plans
    用途: 记录用户的学习目标、周期、每日学习时间等
    """

    __tablename__ = "learning_plans"

    # 计划ID，自动递增主键
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )
    # 用户ID，用于多用户隔离
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 学习目标的文本描述(例如: "掌握机器学习算法基础")
    goal_text: Mapped[str] = mapped_column(Text, nullable=False)
    # 计划开始日期
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    # 计划结束日期
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    # 每日学习时间(分钟)
    daily_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    # 计划状态: "active"(激活中)、"completed"(已完成)、"paused"(暂停)等
    status: Mapped[str] = mapped_column(String(16), default="active")
    # 创建时间
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )


class DailyTask(Base):
    """
    日常任务表 - 每日学习任务的大纲和进度

    表名: daily_tasks
    用途: 记录每天的学习任务主题、学习大纲、进度状态
    关键字段: outline_json 存储AI生成的学习大纲结构化信息
    """

    __tablename__ = "daily_tasks"

    # 任务ID，自动递增主键
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )
    # 关联的学习计划ID
    plan_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 用户ID，用于快速查询用户的所有任务
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 任务对应的日期
    task_date: Mapped[date] = mapped_column(Date, nullable=False)
    # 任务标题/主题
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    # 学习大纲JSON格式
    # 结构示例: {"chapters": [{"title": "...", "points": [...]}]}
    outline_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    # 任务状态: "pending"(待学习)、"in_progress"(学习中)、"completed"(已完成)、"skipped"(已跳过)
    # 实际存储值统一使用上述四种之一，避免出现 "done" 这类自定义文案
    status: Mapped[str] = mapped_column(String(16), default="pending")
    # 创建时间
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )


class DocumentTag(Base):
    """
    文档标签表 - 为文档添加分类标签

    表名: document_tags
    用途: 多对多关系，实现文档的灵活分类和检索
    """

    __tablename__ = "document_tags"

    # 标签ID，自动递增主键
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )
    # 文档ID(外键，指向Document表)
    doc_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 标签内容(例如: "机器学习"、"Python"、"重点"等)
    tag: Mapped[str] = mapped_column(String(64), nullable=False)
    # 创建时间
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )


class RAGQueryLog(Base):
    """
    RAG查询日志表 - 记录用户与知识库的交互

    表名: rag_queries
    用途: 记录用户提问、系统回答、查询元数据等，用于改进和分析
    RAG: Retrieval-Augmented Generation，检索增强生成
    """

    __tablename__ = "rag_queries"

    # 查询ID，自动递增主键
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )
    # 用户ID
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 用户的提问内容
    query: Mapped[str] = mapped_column(Text, nullable=False)
    # 系统的回答内容
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    # 创建时间(查询时间)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    # 元数据JSON(可选)
    # 内容可能包括: 检索到的文档ID、相关度评分、响应时间等
    meta: Mapped[Optional[dict]] = mapped_column(JSON)


class Quiz(Base):
    """
    测试题目表 - 存储AI生成的测试题

    表名: quizzes
    用途: 记录为用户生成的测试题及其配置信息
    """

    __tablename__ = "quizzes"

    # 题目ID，自动递增主键
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )
    # 用户ID
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 题目描述/上下文信息
    description: Mapped[str] = mapped_column(Text, nullable=False)
    # 题目类型: "single_choice"(单选)、"multiple_choice"(多选)、"short_answer"(简答)等
    question_type: Mapped[str] = mapped_column(String(32), nullable=False)
    # 题目的完整JSON数据
    # 结构示例: {"question": "...", "options": [...], "answer": "..."}
    question_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    # 生成这道题的源文档块信息(可选)
    # 用于题目溯源，便于用户回顾相关学习内容
    source_chunks: Mapped[Optional[dict]] = mapped_column(JSON)
    # 创建时间
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )


class QuizAttempt(Base):
    """
    答题记录表 - 记录用户对每道题的作答情况

    表名: quiz_attempts
    用途: 追踪用户的学习效果，记录答题、评分、反馈等
    """

    __tablename__ = "quiz_attempts"

    # 答题记录ID，自动递增主键
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )
    # 用户ID
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 题目ID
    quiz_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 用户的答案JSON格式
    # 结构示例: {"selected": "A"}、{"answer": "用户的简答内容"}
    answer_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    # 题目评分(0.0-1.0或0-100)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    # 题目反馈/讲解，可能包含为什么这个答案是正确或错误的
    comment: Mapped[str] = mapped_column(Text, nullable=False)
    # 答题时间
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )


class WeakPoint(Base):
    """
    薄弱点表 - 追踪用户的知识薄弱环节

    表名: weak_points
    用途: 识别用户答题错误的原因，关联相关文档，用于针对性复习
    """

    __tablename__ = "weak_points"

    # 薄弱点ID，自动递增主键
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )
    # 用户ID
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 题目ID
    quiz_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 答题记录ID
    attempt_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 薄弱点描述(例如: "不理解递归算法的终止条件")
    description: Mapped[str] = mapped_column(Text, nullable=False)
    # 关联的文档ID(可选)，用于链接到相关学习材料
    related_doc_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    # 相关的文档块ID列表(可选)
    # 结构示例: {"chunk_ids": [1, 2, 3]}
    related_chunk_ids: Mapped[Optional[dict]] = mapped_column(JSON)
    # 薄弱点严重程度: "low"(轻微)、"medium"(中等)、"high"(严重)
    level: Mapped[str] = mapped_column(String(16), default="medium")
    # 薄弱点状态: "pending"(待解决)、"resolved"(已解决)、"ignored"(已忽略)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    # 创建时间
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )


class ReviewSchedule(Base):
    """
    复习计划表 - 安排用户的复习任务

    表名: review_schedules
    用途: 实现间隔重复算法，安排最优的复习时间
    """

    __tablename__ = "review_schedules"

    # 复习计划ID，自动递增主键
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )
    # 用户ID
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 目标类型: "daily_task"(日任务)或"weak_point"(薄弱点)
    target_type: Mapped[str] = mapped_column(String(16), nullable=False)
    # 目标ID: 可能指向DailyTask或WeakPoint表的记录
    target_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 复习阶段(第几次复习): 1, 2, 3...
    # 用于间隔重复算法，不同阶段的间隔时间不同
    stage: Mapped[int] = mapped_column(Integer, nullable=False)
    # 计划复习日期
    scheduled_date: Mapped[date] = mapped_column(Date, nullable=False)
    # 复习状态: "pending"(待复习)、"completed"(已完成)、"skipped"(已跳过)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    # 创建时间(复习计划生成时间)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    # 完成时间(可选，用户完成复习时自动填充)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

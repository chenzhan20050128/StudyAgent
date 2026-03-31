from datetime import date, timedelta
from pathlib import Path
import logging
from typing import Generator, List, Optional

from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from src.db import Base, engine, get_session
from src.document_service import DocumentService
from src.llm_client import LLMClient
from src.rag_service import RAGService
from src.plan_service import PlanService
from src.quiz_service import QuizService
from src.review_service import ReviewService
from src.models import WeakPoint
from src.vector_store import MilvusVectorStore
from src.chat_agent import PlanChatAgent
from src.quiz_chat_agent import QuizChatAgent
from src.main_chat_agent import MainChatAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    force=True,
)

app = FastAPI(title="StudyAgent 基础能力")

# 前端静态页挂载
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/", response_class=FileResponse)
async def index():
    return FileResponse("frontend/index2.html")


Base.metadata.create_all(bind=engine)

_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
_docs = DocumentService(str(Path("data/docs")), _splitter)
_vec = MilvusVectorStore()
_llm = LLMClient()
_rag = RAGService(_vec, _llm)
_plans = PlanService(_vec, _llm)
_reviews = ReviewService()
_quizzes = QuizService(_vec, _llm, _reviews)
_plan_chat = PlanChatAgent(_plans, _llm)
_quiz_chat = QuizChatAgent(_quizzes, _llm)
_main_chat = MainChatAgent(
    _llm,
    _rag,
    _plan_chat,
    _quiz_chat,
    _reviews,
)


def get_db() -> Generator[Session, None, None]:
    with get_session() as s:
        yield s


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile | None = File(default=None),
    url: str | None = Form(default=None),
    title: str | None = Form(default=None),
    tags: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
):
    user_id = 1
    if not file and not url:
        return JSONResponse({"error": "file or url required"}, status_code=400)

    tag_list: List[str] | None = None
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    # 情况一：用户通过 multipart/form-data 真正上传了文件。
    # 支持 pdf / word / txt / 其他纯文本文件，解析逻辑在 DocumentService 内部完成。
    if file is not None:
        data = await file.read()
        filename = file.filename or "upload.bin"
        path = _docs.save_uploaded_file(user_id, filename, data)
        doc = _docs.create_document(
            db,
            user_id=user_id,
            title=(title or filename or "uploaded"),
            source_type="file",
            file_path=path,
            tags=tag_list,
        )
        text = _docs.parse_file_to_text(path)

    # 情况二：没有上传文件，而是传入 url 字段。
    # 这里 url 既可能是真实的 URL，也可能是一段“看起来像 URL 的文本”；
    # 约定：
    #   - 如果以 http/https 开头，则作为网页链接抓取正文；
    #   - 否则视为一段普通文本，直接入库。
    else:
        doc = _docs.create_document(
            db,
            user_id=user_id,
            title=(title or url or "web"),
            source_type="web",
            source_url=url,
            tags=tag_list,
        )
        if url and url.startswith(("http://", "https://")):
            text = _docs.parse_url_to_text(url)
        else:
            text = url or ""

    parsed = _docs.parse_and_chunk(doc, text)
    _docs.index_to_milvus(_vec, _llm, parsed)
    doc.status = "parsed"
    db.add(doc)
    return {"doc_id": doc.id, "status": doc.status}


@app.post("/api/rag/query")
async def rag_query(
    body: dict,
    db: Session = Depends(get_db),
):
    user_id = 1
    question = body.get("query")
    doc_ids = body.get("doc_ids")
    if not question:
        return JSONResponse({"error": "query required"}, status_code=400)
    result = _rag.query(
        db,
        user_id=user_id,
        question=question,
        doc_ids=doc_ids,
    )
    return result


@app.post("/api/plans/create")
async def create_plan(
    body: dict,
    db: Session = Depends(get_db),
):
    user_id = 1
    goal = body.get("goal_description")
    start = body.get("start_date")
    end = body.get("end_date")
    target_days = body.get("target_days")
    daily_minutes = body.get("daily_minutes")
    doc_ids = body.get("doc_ids")

    if not goal:
        return JSONResponse(
            {"error": "goal_description required"},
            status_code=400,
        )
    if not daily_minutes:
        return JSONResponse(
            {"error": "daily_minutes required"},
            status_code=400,
        )

    if start and end:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    elif target_days:
        start_date = date.today()
        end_date = start_date + timedelta(days=int(target_days) - 1)
    else:
        return JSONResponse(
            {"error": "start/end or target_days required"}, status_code=400
        )

    result = _plans.create_plan(
        db,
        user_id=user_id,
        goal_description=goal,
        start_date=start_date,
        end_date=end_date,
        daily_minutes=int(daily_minutes),
        doc_ids=doc_ids,
    )
    return result


@app.get("/api/plans/{plan_id}/daily-tasks")
async def list_daily_tasks(
    plan_id: int,
    db: Session = Depends(get_db),
):
    user_id = 1
    tasks = _plans.list_daily_tasks(db, plan_id=plan_id, user_id=user_id)
    return {"plan_id": plan_id, "tasks": tasks}


@app.get("/api/daily-tasks/today")
async def list_today_daily_tasks(db: Session = Depends(get_db)):
    """查询当前用户今天的日常学习任务。

    简化实现：
    - 直接在 daily_tasks 表中过滤 user_id + task_date == today
    - 返回结构与 /api/plans/{plan_id}/daily-tasks 中的单条任务结构保持一致
    """
    from datetime import date as _date_mod

    from src.models import DailyTask as _DailyTaskMod

    user_id = 1
    today = _date_mod.today()
    rows = (
        db.query(_DailyTaskMod)
        .filter(
            _DailyTaskMod.user_id == user_id,
            _DailyTaskMod.task_date == today,
        )
        .order_by(_DailyTaskMod.task_date)
        .all()
    )
    tasks = [
        {
            "id": t.id,
            "date": t.task_date.isoformat(),
            "title": t.title,
            "outline": t.outline_json,
            "status": t.status,
        }
        for t in rows
    ]
    return {"tasks": tasks}


# 向后兼容旧路径 `/complete`，推荐使用 `/status`
@app.post("/api/daily-tasks/{task_id}/status")
@app.post("/api/daily-tasks/{task_id}/complete")
async def complete_daily_task(
    task_id: int,
    body: dict | None = None,
    db: Session = Depends(get_db),
):
    """标记每日任务完成/跳过，并生成复习排期。"""
    user_id = 1
    # 新语义统一使用 completed/skipped，仍兼容历史的 done
    status = (body or {}).get("status", "completed")
    if status == "done":
        status = "completed"
    completed_date_raw = (body or {}).get("completed_date")
    completed_date = None
    if completed_date_raw:
        completed_date = date.fromisoformat(str(completed_date_raw))
    task = _plans.complete_daily_task(
        db, user_id=user_id, task_id=task_id, status=status
    )
    if task is None:
        return JSONResponse({"error": "task not found"}, status_code=404)

    created = _reviews.create_for_daily_task(
        db,
        user_id=user_id,
        task_id=task.id,
        completed_date=completed_date or date.today(),
    )
    return {
        "task_id": task.id,
        "status": task.status,
        "created_reviews": created,
    }


@app.post("/api/quizzes/generate")
async def generate_quiz(
    body: dict,
    db: Session = Depends(get_db),
):
    user_id = 1
    description = body.get("description")
    question_type = body.get("question_type")
    doc_ids = body.get("doc_ids")

    if not description:
        return JSONResponse({"error": "description required"}, status_code=400)
    if not question_type:
        return JSONResponse(
            {"error": "question_type required"},
            status_code=400,
        )

    result = _quizzes.generate_quiz(
        db,
        user_id=user_id,
        description=str(description),
        question_type=str(question_type),
        doc_ids=doc_ids,
    )
    return result


@app.post("/api/quizzes/{quiz_id}/submit")
async def submit_quiz(
    quiz_id: int,
    body: dict,
    db: Session = Depends(get_db),
):
    user_id = 1
    if "answer" not in body:
        return JSONResponse({"error": "answer required"}, status_code=400)

    try:
        result = _quizzes.submit_answer(
            db,
            user_id=user_id,
            quiz_id=quiz_id,
            answer=body.get("answer"),
        )
    except ValueError:
        return JSONResponse({"error": "quiz not found"}, status_code=404)
    return result


@app.post("/api/chat")
async def main_chat(
    body: dict,
    db: Session = Depends(get_db),
):
    """统一主对话入口：一条自然语言即可路由到 RAG / 计划 / 测验 / 复习。

    请求体示例：{"query": "帮我出一道 17 世纪历史的简答题"}
    返回：{"reply": "...", "intent": "quiz_now"}
    """

    text = str(body.get("query") or "").strip()
    if not text:
        return JSONResponse({"error": "query required"}, status_code=400)
    result = _main_chat.handle_message(text, db)
    return result


@app.get("/api/reviews/today")
async def list_today_reviews(db: Session = Depends(get_db)):
    user_id = 1
    items = _reviews.list_today_reviews(
        db,
        user_id=user_id,
        today=date.today(),
    )
    return {"items": items}


@app.post("/api/reviews/{review_id}/complete")
async def complete_review(
    review_id: int,
    body: dict | None = None,
    db: Session = Depends(get_db),
):
    user_id = 1
    status = (body or {}).get("status", "completed")
    try:
        _reviews.complete_review(
            db, user_id=user_id, review_id=review_id, status=status
        )
    except ValueError:
        return JSONResponse({"error": "review not found"}, status_code=404)
    return {"review_id": review_id, "status": status}


@app.get("/api/weak-points")
async def list_weak_points(
    status: str | None = None,
    db: Session = Depends(get_db),
):
    """列出当前用户的薄弱点，可按状态过滤。"""
    user_id = 1
    query = db.query(WeakPoint).filter(WeakPoint.user_id == user_id)
    if status:
        query = query.filter(WeakPoint.status == status)
    rows = query.order_by(WeakPoint.created_at.desc()).all()
    items = [
        {
            "id": wp.id,
            "description": wp.description,
            "level": wp.level,
            "status": wp.status,
            "related_doc_id": wp.related_doc_id,
            "related_chunk_ids": wp.related_chunk_ids,
            "created_at": wp.created_at.isoformat(),
        }
        for wp in rows
    ]
    return {"items": items}


@app.post("/api/weak-points/{weak_point_id}/status")
async def update_weak_point_status(
    weak_point_id: int,
    body: dict,
    db: Session = Depends(get_db),
):
    """更新薄弱点状态：pending / resolved / ignored。"""
    user_id = 1
    new_status = str(body.get("status") or "").strip() or "pending"
    allowed = {"pending", "resolved", "ignored"}
    if new_status not in allowed:
        return JSONResponse({"error": "invalid status"}, status_code=400)

    wp = db.get(WeakPoint, weak_point_id)
    if not wp or wp.user_id != user_id:
        return JSONResponse({"error": "weak_point not found"}, status_code=404)

    wp.status = new_status
    db.add(wp)
    return {
        "id": wp.id,
        "status": wp.status,
    }

"""命令行入口：通过 MainChatAgent 访问统一主接口。

用法示例：
    python test_main.py
然后在提示符下直接输入自然语言，例如：
    帮我规划一个关于17世纪欧洲历史的学习计划，这两天内学完，每天50分钟。
    帮我出一道关于17世纪欧洲历史的简答题，考考我。
    我今天要复习什么？
输入 exit / quit 可退出。
"""

from __future__ import annotations

from src.db import Base, engine, SessionLocal
from src.llm_client import LLMClient
from src.rag_service import RAGService
from src.plan_service import PlanService
from src.quiz_service import QuizService
from src.review_service import ReviewService
from src.vector_store import MilvusVectorStore
from src.chat_agent import PlanChatAgent
from src.quiz_chat_agent import QuizChatAgent
from src.main_chat_agent import MainChatAgent


def main() -> None:
    # 初始化与 main.py 一致的依赖（便于“脱离 HTTP”在 CLI 下复现问题）
    # - Base.metadata.create_all: 确保本地 MySQL 表存在
    # - vec/llm/rag/plans/quizzes/reviews: 与后端服务使用同一套对象
    Base.metadata.create_all(bind=engine)

    vec = MilvusVectorStore()
    llm = LLMClient()
    rag = RAGService(vec, llm)
    plans = PlanService(vec, llm)
    reviews = ReviewService()
    quizzes = QuizService(vec, llm, reviews)
    plan_chat = PlanChatAgent(plans, llm)
    quiz_chat = QuizChatAgent(quizzes, llm)
    agent = MainChatAgent(llm, rag, plan_chat, quiz_chat, reviews)

    # 交互式循环：每输入一句自然语言，就走一次 agent.handle_message 路由
    print("StudyAgent 主对话 CLI 已启动。输入 exit/quit 可退出。\n")

    while True:
        try:
            text = input("你：").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+C / Ctrl+D
            print("\n再见！")
            break

        if not text:
            continue
        if text.lower() in {"exit", "quit", "q"}:
            print("再见！")
            break

        with SessionLocal() as db:
            try:
                # 这里复用 MainChatAgent 的意图识别 + 路由：
                # - create_plan/quiz/review_today/rag_query 等都会在内部自动分发
                result = agent.handle_message(text, db)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] 处理出错：{exc}")
                continue

        reply = result.get("reply") if isinstance(result, dict) else result
        intent = result.get("intent") if isinstance(result, dict) else None
        if intent:
            print(f"[intent={intent}] 助手：{reply}")
        else:
            print(f"助手：{reply}")


if __name__ == "__main__":
    main()

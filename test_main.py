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

from typing import NoReturn

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


def main() -> NoReturn:
    # 初始化与 main.py 一致的依赖
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

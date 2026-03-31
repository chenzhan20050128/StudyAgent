"""命令行多轮对话脚本：阶段三测验 Agent。

用法：
    python test_phase3_chat.py

说明：
- 复用 QuizService，不在 Agent 内重复实现业务逻辑；
- 对话中支持“出题 -> 作答 -> 批改”完整闭环；
- 依赖真实 LLM / Milvus / MySQL。
"""

from __future__ import annotations

from src.db import SessionLocal
from src.llm_client import LLMClient
from src.quiz_chat_agent import QuizChatAgent
from src.quiz_service import QuizService
from src.vector_store import MilvusVectorStore


def main() -> int:
    vec = MilvusVectorStore()
    llm = LLMClient()
    quiz_service = QuizService(vec, llm)
    agent = QuizChatAgent(quiz_service, llm)

    print("阶段三 Quiz 对话测试已启动（输入 exit 退出）")
    print("示例：帮我出一道 Python 处理 Excel 的简答题")

    while True:
        try:
            user = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return 0

        if user.lower() in {"exit", "quit"}:
            print("已退出。")
            return 0

        with SessionLocal() as session:
            reply = agent.handle_message(user, session)
            session.commit()

        print(f"[PHASE3_CHAT] Agent> {reply}")


if __name__ == "__main__":
    raise SystemExit(main())

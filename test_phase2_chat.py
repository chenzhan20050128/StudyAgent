"""命令行多轮对话脚本：用自然语言与计划 Agent 交互。

用法：
    python test_phase2_chat.py

特点：
- 不改动 test_phase2.py 现有逻辑；
- 复用现有 PlanService / LangGraph 工作流；
- 对话状态在进程内保持，可多轮调整计划。
"""

from __future__ import annotations

import sys

from src.chat_agent import PlanChatAgent
from src.db import SessionLocal
from src.llm_client import LLMClient
from src.plan_service import PlanService
from src.vector_store import MilvusVectorStore


def main() -> None:
    vec = MilvusVectorStore()
    llm = LLMClient()
    plan_service = PlanService(vec, llm)
    agent = PlanChatAgent(plan_service, llm)

    print("请输入学习需求（exit 退出）：")
    while True:
        try:
            user = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return
        if user.lower() in {"exit", "quit"}:
            print("已退出。")
            return

        with SessionLocal() as session:
            reply = agent.handle_message(user, session)
            session.commit()
        print(f"Agent> {reply}")


if __name__ == "__main__":
    sys.exit(main())

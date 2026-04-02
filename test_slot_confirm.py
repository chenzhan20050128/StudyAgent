"""槽位确认交互（占位测试文件）。

说明：
- 这个仓库里“槽位确认/追问”的核心逻辑目前在 `src/chat_agent.py`（PlanChatAgent）
  和 `src/quiz_chat_agent.py`（QuizChatAgent）的 handle_message 流程内体现：
  当缺少必要槽位时，会返回一条追问提示，引导用户补齐信息。

为什么这个文件为空：
- POC 阶段更偏向“端到端脚本”验证（phase2_chat/phase3_chat），
  这里预留一个专门的 unit test 文件，后续可把“缺槽位→追问→补齐→执行”的对话
  通过假 LLM / 固定解析结果来做确定性单测。
"""

"""开发期便捷脚本：读取并打印一个测试文件内容。

需求
- 在开发/调试时快速查看 `test_slot_confirm.py` 的当前内容，便于确认注释或占位说明是否已写入。

业务逻辑
- 直接打开仓库根目录下的 `test_slot_confirm.py` 并打印到标准输出。
- 该脚本不参与任何业务流程，不会影响计划/测验/RAG 等主链路。

主要实现方式
- 使用相对路径读取文件并打印（没有命令行参数、没有异常兜底，适合本地临时使用）。
"""

with open("../test_slot_confirm.py", "r", encoding="utf-8") as f:
    print(f.read())

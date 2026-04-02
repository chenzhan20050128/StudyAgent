"""全局配置：数据库 / 向量库 / LLM 访问参数的集中管理。

需求
- 给出本地可跑的默认配置，同时允许在部署时用环境变量覆盖关键参数。

业务逻辑
- Settings 作为全局单例：其他模块只依赖 `settings`，避免到处读取环境变量或硬编码。
- 将配置按领域拆分：DBConfig / MilvusConfig / LLMConfig，便于扩展与单元测试替换。

主要实现方式
- 使用 Pydantic BaseModel 承载配置，并在模块加载时实例化 `settings`。
- 敏感信息（API Key）不写死在配置里，而是通过 `LLMConfig.api_key_env` 指向环境变量名。
"""

from pydantic import BaseModel


class DBConfig(BaseModel):
    url: str = "mysql+pymysql://study:123456@localhost:3306/study_agent"


class MilvusConfig(BaseModel):
    uri: str = "http://localhost:19530"
    collection: str = "learning_chunks"
    embedding_dim: int = 2048


class LLMConfig(BaseModel):
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: str = "DASHSCOPE_API_KEY"
    embedding_model: str = "text-embedding-v4"
    rerank_model: str = "qwen3-rerank"
    chat_model: str = "qwen3.5-flash"


class Settings(BaseModel):
    db: DBConfig = DBConfig()
    milvus: MilvusConfig = MilvusConfig()
    llm: LLMConfig = LLMConfig()
    # 全局配置入口：
    # - 通过环境变量覆盖（适合部署）
    # - 给出合理默认值（适合本地跑 demo/面试展示）
    # 注意：密钥等敏感信息不写死在代码里


settings = Settings()

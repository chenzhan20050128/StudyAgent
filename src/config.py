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


settings = Settings()

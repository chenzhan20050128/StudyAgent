"""数据库基础设施：创建 engine 与 Session 工厂，并提供请求级会话作用域。

需求
- 为全项目提供统一的数据库连接与会话管理，供各 Service/Agent 读写 MySQL。

业务逻辑
- 启动时创建 SQLAlchemy engine（带 pool_pre_ping，减少断线带来的故障）。
- 提供 SessionLocal 工厂，并通过 `get_session()` 返回一个 contextmanager：
    - 正常结束自动 commit
    - 异常自动 rollback
    - 最终确保 close

主要实现方式
- 使用 `src/config.py` 的 settings.db.url 作为连接串来源。
- 统一 DeclarativeBase（Base）供 `models.py` 声明 ORM 表。
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from .config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(settings.db.url, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_session():
    from contextlib import contextmanager

    @contextmanager
    def _session_scope():
        session = SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return _session_scope()
    # FastAPI dependency：yield 一个 session，确保请求结束后正确 close

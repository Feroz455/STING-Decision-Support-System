from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Boolean, DateTime, Text, Integer
from datetime import datetime

from app.core.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(128), default="")
    hashed_password: Mapped[str] = mapped_column(String(256))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    role: Mapped[str] = mapped_column(String(32), default="clinician")  # clinician | admin
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Session(Base):
    """Tracks a DSS analysis session — links user → results across tabs."""
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int]
    session_name: Mapped[str] = mapped_column(String(128), default="")
    status: Mapped[str] = mapped_column(String(32), default="tab1_pending")
    result_dir: Mapped[str] = mapped_column(String(256), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ActivityLog(Base):
    """Kullanıcı aktivite logu — her önemli işlem kaydedilir."""
    __tablename__ = "activity_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    username: Mapped[str] = mapped_column(String(64), default="")
    tab: Mapped[str] = mapped_column(String(32), default="")       # tab1, tab2, tab3, tab4
    action: Mapped[str] = mapped_column(String(128), default="")   # run_ode, run_ga, predict, etc.
    summary: Mapped[str] = mapped_column(Text, default="")         # JSON string — key metrics
    detail: Mapped[str] = mapped_column(Text, default="")          # Ek detay (ilaçlar, dozlar vb.)
    status: Mapped[str] = mapped_column(String(32), default="success")  # success | error
    duration_sec: Mapped[float] = mapped_column(default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

from __future__ import annotations

from contextlib import contextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from ..config import DATABASE_URL


class Base(DeclarativeBase):
    pass


if DATABASE_URL.startswith('sqlite'):
    engine = create_engine(
        DATABASE_URL, 
        echo=False, 
        future=True,
        connect_args={"timeout": 30}  # SQLite timeout
    )
else:
    engine = create_engine(
        DATABASE_URL, 
        echo=False, 
        future=True,
        pool_size=2,         # Smaller pool per process
        max_overflow=3,      # Limited overflow
        pool_pre_ping=True,  # Test connections before using
        pool_recycle=300,    # Recycle connections every 5 min
        connect_args={
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
            "connect_timeout": 10,
        }
    )

if DATABASE_URL.startswith('sqlite'):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=30000")  # 30 second timeout
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

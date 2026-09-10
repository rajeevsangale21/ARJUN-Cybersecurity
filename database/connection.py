from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker


# ---------------------------------------------------------
# Database location
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_DIR = (
    BASE_DIR / "data" / "database"
)

DATABASE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DATABASE_PATH = (
    DATABASE_DIR / "arjun.db"
)


DATABASE_URL = (
    "sqlite:///"
    + DATABASE_PATH.as_posix()
)


# ---------------------------------------------------------
# SQLAlchemy
# ---------------------------------------------------------

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False
    },
)


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


Base = declarative_base()


# ---------------------------------------------------------
# Database initialization
# ---------------------------------------------------------

def init_database():
    """
    Create all ARJUN database tables.
    """

    # Import models before create_all.
    from database import models  # noqa: F401

    Base.metadata.create_all(
        bind=engine
    )


# ---------------------------------------------------------
# Session dependency
# ---------------------------------------------------------

def get_db():
    """
    Provide a database session.
    """

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

def database_health():
    """
    Check whether SQLite is accessible.
    """

    try:

        with engine.connect() as connection:

            connection.execute(
                text("SELECT 1")
            )

        return "connected"

    except Exception:

        return "disconnected"
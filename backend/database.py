import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(_ROOT, 'notes.db')}")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


if DATABASE_URL.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _tune_sqlite(dbapi_connection, _connection_record):
        """Tune SQLite for the app's concurrent read-while-writing workload.

        The backend serves HTTP reads while background tasks (transcription,
        summarization) write. WAL lets readers proceed during a write instead
        of hitting "database is locked"; busy_timeout absorbs brief contention;
        synchronous=NORMAL is durable under WAL and much faster than FULL.
        """
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

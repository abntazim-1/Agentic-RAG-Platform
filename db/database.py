from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from db.models import Base

# Path to the database file
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "metrics.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

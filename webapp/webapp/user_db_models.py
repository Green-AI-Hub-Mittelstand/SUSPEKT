import datetime
import os

from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()

# Projektverzeichnis ermitteln und Datenbankpfad erstellen
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'system180.db')}"

# Engine und Session erstellen
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


def init_db():
    """Datenbank initialisieren"""
    Base.metadata.create_all(bind=engine)


# Dependency für Datenbankverbindung
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Datenbank initialisieren wenn Datei direkt ausgeführt wird
if __name__ == "__main__":
    init_db()

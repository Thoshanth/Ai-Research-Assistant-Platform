from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./research_assistant.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class DocumentRecord(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    file_type = Column(String)
    page_count = Column(Integer, nullable=True)
    word_count = Column(Integer)
    file_size_kb = Column(Float)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    extracted_text = Column(Text)

def init_db():
    Base.metadata.create_all(bind=engine)
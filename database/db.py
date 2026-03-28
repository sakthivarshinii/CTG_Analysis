from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os

DATABASE_URL = "sqlite:///./fetal_health.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PatientRecord(Base):
    __tablename__ = "patient_records"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)
    # Features
    LB = Column(Float)
    ASTV = Column(Float)
    AC = Column(Float)
    DL = Column(Float)
    UC = Column(Float)
    # Target Results
    prediction = Column(String) # Normal, Suspect, Pathological
    confidence = Column(Float)
    risk_level = Column(String) # Stable, Monitor, High Risk
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

def init_db():
    # Create the db tables
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime
from pydantic import BaseModel
from typing import Optional

from app.database.connection import Base

# SQLAlchemy Model
class DBHistory(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(String, index=True)
    intent = Column(String)
    expression = Column(String)
    result_text = Column(String)
    result_latex = Column(String)
    result_spoken = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Pydantic Schemas

# Request model for manual text input calculations
class CalculationRequest(BaseModel):
    query: str

# Model returned by LLM parsing
class LLMIntentParse(BaseModel):
    intent: str
    expression: str
    variables: list[str] = ["x"]
    parameters: dict = {}
    explanation: str
    is_valid: bool = True

# Full calculation response schema (returned to frontend)
class CalculationResult(BaseModel):
    id: Optional[int] = None
    query: str
    intent: str
    expression: str
    result_text: str
    result_latex: str
    result_spoken: str
    explanation: Optional[str] = None
    timestamp: Optional[datetime] = None

    class Config:
        from_attributes = True

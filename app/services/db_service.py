from sqlalchemy.orm import Session
from sqlalchemy import desc, or_
from app.models.schemas import DBHistory, CalculationResult
from datetime import datetime

def create_history_entry(
    db: Session,
    query: str,
    intent: str,
    expression: str,
    result_text: str,
    result_latex: str,
    result_spoken: str
) -> DBHistory:
    """
    Saves a calculation transaction into the database history.
    """
    db_entry = DBHistory(
        query=query,
        intent=intent,
        expression=expression,
        result_text=result_text,
        result_latex=result_latex,
        result_spoken=result_spoken,
        timestamp=datetime.utcnow()
    )
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    return db_entry

def get_history(
    db: Session,
    limit: int = 50,
    offset: int = 0,
    search_query: str = None
) -> list[DBHistory]:
    """
    Retrieves execution logs, ordered by newest first, with optional search text filter.
    """
    query = db.query(DBHistory)
    if search_query:
        query = query.filter(
            or_(
                DBHistory.query.ilike(f"%{search_query}%"),
                DBHistory.result_text.ilike(f"%{search_query}%"),
                DBHistory.intent.ilike(f"%{search_query}%")
            )
        )
    return query.order_by(desc(DBHistory.timestamp)).offset(offset).limit(limit).all()

def delete_history_entry(db: Session, entry_id: int) -> bool:
    """
    Deletes a single calculation history entry.
    """
    db_entry = db.query(DBHistory).filter(DBHistory.id == entry_id).first()
    if db_entry:
        db.delete(db_entry)
        db.commit()
        return True
    return False

def clear_all_history(db: Session) -> None:
    """
    Wipes the entire calculation history.
    """
    db.query(DBHistory).delete()
    db.commit()

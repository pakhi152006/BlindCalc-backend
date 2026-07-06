from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional

from app.database.connection import get_db
from app.models.schemas import CalculationResult
from app.services import db_service
from app.config import logger

router = APIRouter(
    prefix="/api",
    tags=["history"]
)

@router.get("/history", response_model=List[CalculationResult], status_code=status.HTTP_200_OK)
def fetch_history(
    q: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Fetches the history of calculations.
    Supports a 'q' parameter for fuzzy text matching/searching.
    """
    logger.info(f"Fetching history: query='{q}', limit={limit}, offset={offset}")
    try:
        records = db_service.get_history(db=db, limit=limit, offset=offset, search_query=q)
        return records
    except Exception as e:
        logger.error(f"Failed to fetch history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query database: {str(e)}"
        )

@router.delete("/history/{entry_id}", status_code=status.HTTP_200_OK)
def delete_entry(
    entry_id: int,
    db: Session = Depends(get_db)
):
    """
    Deletes a specific calculation from history logs.
    """
    logger.info(f"Deleting history entry ID: {entry_id}")
    success = db_service.delete_history_entry(db=db, entry_id=entry_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"History entry with ID {entry_id} not found."
        )
    return {"message": f"Successfully deleted history entry {entry_id}."}

@router.delete("/history", status_code=status.HTTP_200_OK)
def clear_history(
    db: Session = Depends(get_db)
):
    """
    Wipes the SQLite database calculation history log clean.
    """
    logger.info("Clearing all history entries")
    try:
        db_service.clear_all_history(db=db)
        return {"message": "All calculation logs successfully wiped."}
    except Exception as e:
        logger.error(f"Failed to clear history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear database logs: {str(e)}"
        )

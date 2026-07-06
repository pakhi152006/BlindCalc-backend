from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database.connection import get_db
from app.models.schemas import CalculationRequest, CalculationResult
from app.services.llm_service import LLMService
from app.services.math_service import MathService
from app.services.db_service import create_history_entry
from app.config import logger

router = APIRouter(
    prefix="/api",
tags=["calculations"]
)

@router.post("/calculate", response_model=CalculationResult, status_code=status.HTTP_200_OK)
def calculate_text_query(
    payload: CalculationRequest,
    db: Session = Depends(get_db)
):
    """
    Solves a natural language mathematical calculation.
    Flow: User Input -> LLM parses intent -> SymPy/NumPy solves -> Save to DB -> Return.
    """
    logger.info(f"Received calculation request: '{payload.query}'")
    
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The mathematical query cannot be empty. Please say or type a question."
        )

    try:
        # Step 1: Parse natural language input via LLM
        parsed_intent = LLMService.parse_query(payload.query)
        logger.info(f"Parsed intent structure: {parsed_intent}")
        
        if not parsed_intent.is_valid or not parsed_intent.expression:
            raise ValueError("The query could not be translated into a valid mathematical operation.")

        # Step 2: Solve calculations strictly through Python Math engines
        solver_outputs = MathService.solve_problem(
            intent=parsed_intent.intent,
            expression=parsed_intent.expression,
            parameters=parsed_intent.parameters
        )
        logger.info(f"Math solver completed: {solver_outputs}")
        
        # Step 3: Record transaction in SQLite database
        db_record = create_history_entry(
            db=db,
            query=payload.query,
            intent=parsed_intent.intent,
            expression=parsed_intent.expression,
            result_text=solver_outputs["result_text"],
            result_latex=solver_outputs["result_latex"],
            result_spoken=solver_outputs["result_spoken"]
        )
        
        # Assemble response
        return CalculationResult(
            id=db_record.id,
            query=db_record.query,
            intent=db_record.intent,
            expression=db_record.expression,
            result_text=db_record.result_text,
            result_latex=db_record.result_latex,
            result_spoken=db_record.result_spoken,
            explanation=parsed_intent.explanation,
            timestamp=db_record.timestamp
        )

    except Exception as e:
        logger.error(f"Error solving calculation request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Mathematical parsing or execution failed: {str(e)}"
        )

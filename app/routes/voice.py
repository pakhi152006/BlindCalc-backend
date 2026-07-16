from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import io

from app.database.connection import get_db
from app.models.schemas import CalculationResult
from app.services.speech_service import SpeechService
from app.services.llm_service import LLMService
from app.services.math_service import MathService
from app.services.db_service import create_history_entry
from app.services.tts_service import TTSService
from app.config import logger

router = APIRouter(
    prefix="/api",
    tags=["voice"]
)

speech_service = SpeechService()

@router.post("/voice-calculate", response_model=CalculationResult, status_code=status.HTTP_200_OK)
async def calculate_voice_query(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Handles audio uploaded from the microphone.
    Transcribes audio with local Whisper, processes the text mathematically,
    saves the calculation, and returns the result.
    """
    logger.info(f"Received audio file upload: name={file.filename}, type={file.content_type}")
    
    # Read audio bytes
    audio_data = await file.read()
    if not audio_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded audio file is empty. Please speak again."
        )

    try:
        # Step 1: Transcribe audio to text via Groq Whisper API
        transcription = await speech_service.transcribe(audio_data)
        
        if not transcription.strip():
            raise ValueError("No speech could be recognized. Please speak louder and clearer.")

        # Step 2: Parse text query via local LLM / Fallback
        parsed_intent = LLMService.parse_query(transcription)
        logger.info(f"Voice parsed intent: {parsed_intent}")
        
        if not parsed_intent.is_valid or not parsed_intent.expression:
            raise ValueError("The recognized query could not be translated into a valid mathematical operation.")

        # Step 3: Solve calculations strictly through Python Math engines
        solver_outputs = MathService.solve_problem(
            intent=parsed_intent.intent,
            expression=parsed_intent.expression,
            parameters=parsed_intent.parameters
        )
        logger.info(f"Voice math solver completed: {solver_outputs}")
        
        # Step 4: Record calculation in SQLite database
        db_record = create_history_entry(
            db=db,
            # We save the exact Whisper transcription as the user query
            query=transcription,
            intent=parsed_intent.intent,
            expression=parsed_intent.expression,
            result_text=solver_outputs["result_text"],
            result_latex=solver_outputs["result_latex"],
            result_spoken=solver_outputs["result_spoken"]
        )
        
        # Return complete details including transcribed text
        return CalculationResult(
            id=db_record.id,
            query=db_record.query, # returns the transcribed speech
            intent=db_record.intent,
            expression=db_record.expression,
            result_text=db_record.result_text,
            result_latex=db_record.result_latex,
            result_spoken=db_record.result_spoken,
            explanation=parsed_intent.explanation,
            timestamp=db_record.timestamp
        )

    except Exception as e:
        logger.error(f"Error executing voice calculation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )

@router.post("/transcribe", status_code=status.HTTP_200_OK)
async def transcribe_voice(
    file: UploadFile = File(...)
):
    """
    Transcribes audio to text via local Whisper without running calculation engines.
    Used for local command processing on the client.
    """
    logger.info(f"Received audio file for transcription: name={file.filename}, type={file.content_type}")
    
    audio_data = await file.read()
    if not audio_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded audio file is empty. Please speak again."
        )

    try:
        # Transcribe audio to text via Groq Whisper API
        transcription = await speech_service.transcribe(audio_data)
        return {"text": transcription}
    except Exception as e:
        logger.error(f"Error during audio transcription: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )

@router.get("/tts")
def speak_result(text: str):
    """
    Synthesizes and returns a streaming WAV file from the provided text.
    If backend TTS is unavailable, returns an empty audio stream with a 204 status.
    """
    logger.info(f"Requested TTS synthesis for text: '{text}'")
    
    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The text parameter cannot be empty."
        )

    audio_bytes = TTSService.generate_speech(text)
    
    if not audio_bytes:
        # Backend TTS is not working (headless/missing drivers). Return 204 No Content
        # Client handles synthesis in browser instead.
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=speech.wav"}
    )

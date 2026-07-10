import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import logging

from app.config import settings, logger
from app.database.connection import engine, Base
from app.routes import calc, history, voice

# Initialize SQLAlchemy Tables
try:
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")
except Exception as e:
    logger.error(f"Database initialization failed: {str(e)}")
# Initialize FastAPI App
app = FastAPI(
    title=settings.APP_NAME,
    description="Full-stack scientific calculation helper designed for visually impaired users.",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(calc.router)
app.include_router(history.router)
app.include_router(voice.router)

# Healthcheck & Landing Page
@app.get("/", status_code=status.HTTP_200_OK)
def read_root():
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "database": "online",
        "models": {
            "stt": settings.WHISPER_MODEL,
            "nlu": settings.OLLAMA_MODEL
        }
    }

# Global HTTP Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception caught on request {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"An unexpected server error occurred: {str(exc)}"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000
    )
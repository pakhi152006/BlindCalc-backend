import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("blindcalc")


class Settings:
    APP_NAME: str = "BlindCalc AI Backend"
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

    # Database
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATABASE_DIR: str = os.path.join(BASE_DIR, "database")
    DATABASE_PATH: str = os.path.join(DATABASE_DIR, "blindcalc.db")
    DATABASE_URL: str = f"sqlite:///{DATABASE_PATH}"

    # Speech to Text (Whisper)
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "tiny")

    # LLM (Ollama legacy - can be removed later)
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "phi3")

    # GROQ (NEW)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing. Please set it in environment variables.")

    # Text to Speech
    TTS_DEFAULT_RATE: int = int(os.getenv("TTS_DEFAULT_RATE", "150"))

    # CORS
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173"
    ]


settings = Settings()

os.makedirs(settings.DATABASE_DIR, exist_ok=True)

logger.info(f"Initialized settings. Database path: {settings.DATABASE_PATH}")
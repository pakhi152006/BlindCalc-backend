import os
import tempfile
import logging
from app.config import settings, logger

# Try importing pyttsx3 and handle import errors
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logger.warn("pyttsx3 library not installed. Backend TTS will be unavailable.")

class TTSService:
    _engine = None
    _failed_init = False

    @classmethod
    def _get_engine(cls):
        if not PYTTSX3_AVAILABLE:
            raise RuntimeError("pyttsx3 library is not available.")
        if cls._failed_init:
            raise RuntimeError("pyttsx3 engine failed to initialize previously.")
            
        if cls._engine is None:
            try:
                # Initialize pyttsx3
                # We specify dummy or standard driver based on OS
                cls._engine = pyttsx3.init()
                cls._engine.setProperty('rate', settings.TTS_DEFAULT_RATE)
            except Exception as e:
                cls._failed_init = True
                logger.error(f"Failed to initialize pyttsx3: {str(e)}")
                raise RuntimeError(f"Failed to initialize local pyttsx3 engine: {str(e)}")
        return cls._engine

    @classmethod
    def generate_speech(cls, text: str) -> bytes:
        """
        Synthesizes text to speech offline and returns the audio bytes (WAV/MP3).
        """
        try:
            engine = cls._get_engine()
        except Exception as err:
            logger.warn(f"Backend TTS unavailable ({str(err)}). Falling back to client-side SpeechSynthesis.")
            # Return empty bytes to indicate backend TTS failure; client will handle speech synthesis.
            return b""

        # Create a temp file to store output speech
        temp_dir = tempfile.gettempdir()
        temp_filename = os.path.join(temp_dir, f"tts_{hash(text)}.wav")
        
        try:
            # pyttsx3 saves to file
            logger.info(f"Synthesizing speech to temp file: {temp_filename}")
            engine.save_to_file(text, temp_filename)
            engine.runAndWait()
            
            # Read audio bytes
            if os.path.exists(temp_filename):
                with open(temp_filename, "rb") as f:
                    audio_bytes = f.read()
                return audio_bytes
            else:
                raise FileNotFoundError("TTS audio output file was not created.")
                
        except Exception as e:
            logger.error(f"Error during pyttsx3 TTS synthesis: {str(e)}")
            return b""
        finally:
            # Clean up temp file
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception as cleanup_err:
                    logger.warn(f"Failed to delete temp TTS file {temp_filename}: {str(cleanup_err)}")

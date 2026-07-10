import os
import tempfile

from app.config import settings, logger


class SpeechService:

    _model = None
    _failed_to_load = False
    _load_error_message = ""

    @classmethod
    def _get_model(cls):
        """
        Lazily loads the Whisper model only when voice feature is used.
        """

        if cls._failed_to_load:
            raise RuntimeError(
                f"Whisper initialization failed earlier: {cls._load_error_message}"
            )

        if cls._model is None:
            logger.info(
                f"Loading local Whisper model: '{settings.WHISPER_MODEL}'..."
            )

            try:
                # Import only when required to reduce startup memory
                import whisper

                cls._model = whisper.load_model(
                    settings.WHISPER_MODEL
                )

                logger.info("Whisper model loaded successfully.")

            except Exception as e:
                cls._failed_to_load = True
                cls._load_error_message = str(e)

                logger.error(
                    f"Failed to load Whisper model: {str(e)}"
                )

                raise RuntimeError(
                    f"Could not load local Whisper STT model. "
                    f"System error: {str(e)}"
                )

        return cls._model


    @classmethod
    def transcribe_audio(cls, audio_content: bytes) -> str:
        """
        Saves audio bytes temporarily and transcribes using Whisper.
        """

        model = cls._get_model()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_audio:

            temp_audio.write(audio_content)
            temp_path = temp_audio.name

        try:
            logger.info(
                f"Transcribing audio file: {temp_path}"
            )

            result = model.transcribe(
                temp_path,
                fp16=False
            )

            transcription = result.get(
                "text",
                ""
            ).strip()

            logger.info(
                f"Transcription result: '{transcription}'"
            )

            return transcription

        except Exception as e:
            logger.error(
                f"Error during audio transcription: {str(e)}"
            )

            raise ValueError(
                f"Whisper speech-to-text transcription failed: {str(e)}"
            )

        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)

                except Exception as cleanup_err:
                    logger.warning(
                        f"Failed to delete temp file {temp_path}: {cleanup_err}"
                    )
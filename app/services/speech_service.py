import os
from groq import Groq


class SpeechService:

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise Exception(
                "GROQ_API_KEY missing"
            )

        self.client = Groq(
            api_key=api_key
        )


    async def transcribe(self, audio_data: bytes):

        try:

            result = self.client.audio.transcriptions.create(
                file=(
                    "audio.webm",
                    audio_data,
                    "audio/webm"
                ),
                model="whisper-large-v3-turbo"
            )


            return result.text


        except Exception as e:

            raise Exception(
                f"Groq Whisper transcription failed: {str(e)}"
            )
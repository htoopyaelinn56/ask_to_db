import os
from typing import Generator, override

from dotenv import load_dotenv
from google import genai

from ai_service.base_ai_service import BaseAIService

load_dotenv()


class GeminiAIService(BaseAIService):
    """AI service implementation using Gemini."""

    def __init__(self):
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_client = genai.Client(api_key=gemini_api_key)
        self.model_name = "gemma-3-27b-it"

    @override
    def generate_content(self, prompt: str):
        response = self.gemini_client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text

    @override
    def generate_content_stream(self, prompt: str):
        response_stream = self.gemini_client.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
        )
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text

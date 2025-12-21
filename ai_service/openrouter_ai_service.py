import os
from typing import override

from dotenv import load_dotenv
from openrouter import OpenRouter

from ai_service.base_ai_service import BaseAIService

load_dotenv()


class OpenRouterAIService(BaseAIService):
    """AI service implementation using OpenRouter."""

    def __init__(self):
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_client = OpenRouter(api_key=openrouter_api_key)
        self.model_name = "google/gemma-3-27b-it:free"

    @override
    def generate_content(self, prompt: str) -> str:
        response = self.openrouter_client.chat.send(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message["content"]

    @override
    def generate_content_stream(self, prompt: str):
        response_stream = self.openrouter_client.chat.send(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        for event in response_stream:
            content = event.choices[0].delta.content if event.choices else None
            if content:
                yield content

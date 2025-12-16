import os

from dotenv import load_dotenv
from openrouter import OpenRouter
from google import genai

load_dotenv()

# temp comment to use Gemini API
_openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
_openrouter_client = OpenRouter(api_key=_openrouter_api_key)
_DEFAULT_MODEL = "google/gemma-3-27b-it:free"

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=gemini_api_key)
DEFAULT_MODEL = "gemma-3-27b-it"

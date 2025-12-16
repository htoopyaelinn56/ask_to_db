import os

from openrouter import OpenRouter
from dotenv import load_dotenv

load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter_client = OpenRouter(api_key=openrouter_api_key)

DEFAULT_MODEL = "google/gemma-3-27b-it:free"


def chat_once(prompt: str, model: str = DEFAULT_MODEL):
    """Send a single prompt and stream the response to stdout."""
    stream = openrouter_client.chat.send(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content if chunk.choices else None
        if content:
            print(content, end="", flush=True)
    print()  # newline after streaming completes


def main():
    print("Interactive chatbot. Type your prompt and press Enter. Type /exit to quit.")
    model = DEFAULT_MODEL
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"/exit", "exit", "quit", "/quit"}:
            print("Goodbye.")
            break

        # Single-turn chat; you can extend this to keep history if needed
        try:
            chat_once(user_input, model=model)
        except Exception as e:
            print(f"[ERROR] Chat failed: {e}")


if __name__ == "__main__":
    main()

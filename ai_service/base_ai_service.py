from abc import ABC

class BaseAIService(ABC):
    """Abstract base class for AI services."""

    def generate_content(self, prompt: str) -> str:
        """Generate text based on the given prompt using the specified model."""
        raise NotImplementedError("Subclasses must implement this method.")

    def generate_content_stream(self, prompt: str):
        """Generate text stream based on the given prompt using the specified model."""
        raise NotImplementedError("Subclasses must implement this method.")
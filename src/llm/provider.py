"""Isolated Gemini LLM Provider interface with robust error handling and secret protection."""

import os
from typing import Optional
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class LLMAuthenticationError(LLMProviderError):
    """Exception raised when API authentication fails or key is missing."""
    pass


class LLMTimeoutError(LLMProviderError):
    """Exception raised when provider request exceeds timeout."""
    pass


class LLMRateLimitError(LLMProviderError):
    """Exception raised when rate limit or quota is exceeded."""
    pass


class GeminiProvider:
    """Isolated Google Gemini provider client using the official google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None
    ):
        """Initialize Gemini provider.

        Args:
            api_key: Optional explicit API key. If omitted, loaded from LLM_API_KEY environment variable.
            default_model: Model name. If omitted, loaded from LLM_MODEL env var or defaults to 'gemini-3.6-flash'.
        """
        self._api_key = api_key or os.getenv("LLM_API_KEY", "").strip()
        self.default_model = default_model or os.getenv("LLM_MODEL", "gemini-3.6-flash").strip()
        self._client = None

    def _get_client(self):
        """Lazy-initialize google-genai Client."""
        if not self._api_key:
            raise LLMAuthenticationError(
                "Gemini API key is not configured. Please set the LLM_API_KEY environment variable."
            )

        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self._api_key)
            except Exception as e:
                # Do not expose raw exception details if they might contain credentials
                raise LLMProviderError(f"Failed to initialize Gemini client: {type(e).__name__}") from None

        return self._client

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        timeout: float = 30.0
    ) -> str:
        """Generate response text from Gemini with grounding instructions.

        Args:
            prompt: User prompt containing grounded context.
            system_instruction: System grounding instructions.
            model: Optional model override.
            temperature: Sampling temperature (0.0 to 1.0).
            timeout: Maximum response timeout in seconds.

        Returns:
            Extracted text response string.

        Raises:
            LLMAuthenticationError, LLMTimeoutError, LLMRateLimitError, LLMProviderError
        """
        if not prompt or not prompt.strip():
            raise LLMProviderError("Prompt content cannot be empty.")

        client = self._get_client()
        target_model = model or self.default_model

        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                temperature=float(temperature),
                system_instruction=system_instruction if system_instruction else None,
            )

            # Generate content using modern official SDK
            response = client.models.generate_content(
                model=target_model,
                contents=prompt,
                config=config
            )

            if not response or not response.text:
                raise LLMProviderError("Gemini provider returned an empty or null response.")

            return response.text.strip()

        except TimeoutError:
            raise LLMTimeoutError(f"Gemini API request timed out after {timeout} seconds.") from None
        except Exception as e:
            err_str = str(e).lower()
            err_type = type(e).__name__

            if "401" in err_str or "unauthenticated" in err_str or "api_key" in err_str or "invalid" in err_str:
                raise LLMAuthenticationError("Gemini API authentication failed. Check your LLM_API_KEY.") from None
            elif "429" in err_str or "quota" in err_str or "rate limit" in err_str or "resource_exhausted" in err_str:
                raise LLMRateLimitError("Gemini API rate limit or quota exceeded. Please retry shortly.") from None
            elif "timeout" in err_str or "deadline" in err_str:
                raise LLMTimeoutError("Gemini API request timed out.") from None
            else:
                raise LLMProviderError(f"Gemini generation error ({err_type}): {err_str[:120]}") from None

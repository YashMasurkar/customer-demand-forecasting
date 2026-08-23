"""Unit and integration tests for Grounded LLM Business Analyst layer (All Gemini API calls mocked)."""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.analytics.schemas import BusinessAnalyticsContext
from src.llm.schemas import BusinessQuestionRequest, BusinessAnswer
from src.llm.prompts import GROUNDED_SYSTEM_INSTRUCTION, build_grounded_prompt
from src.llm.provider import (
    GeminiProvider,
    LLMProviderError,
    LLMAuthenticationError,
    LLMTimeoutError,
    LLMRateLimitError
)
from src.llm.service import LLMBusinessAnalystService, ask_business_question


# ==============================================================================
# 1. PROVIDER UNIT TESTS (Mocked Gemini API)
# ==============================================================================

def test_provider_successful_mocked_response():
    """Verify GeminiProvider successfully extracts generated text from mocked client."""
    provider = GeminiProvider(api_key="mock_test_key_12345")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Demand is projected at 16,266.8 units for 2018, an increase of 30.97%."
    mock_client.models.generate_content.return_value = mock_response

    provider._client = mock_client
    result = provider.generate_text(prompt="What is the forecast?", system_instruction="Be grounded.")

    assert result == "Demand is projected at 16,266.8 units for 2018, an increase of 30.97%."
    assert mock_client.models.generate_content.called


def test_provider_missing_api_key():
    """Verify provider raises LLMAuthenticationError when API key is missing."""
    with patch.dict("os.environ", {}, clear=True):
        provider = GeminiProvider(api_key="")
        with pytest.raises(LLMAuthenticationError, match="Gemini API key is not configured"):
            provider.generate_text(prompt="Test question")


def test_provider_invalid_api_configuration():
    """Verify provider raises LLMAuthenticationError when API returns 401 unauthenticated."""
    provider = GeminiProvider(api_key="mock_invalid_key")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("401 Invalid API_KEY or unauthenticated")
    provider._client = mock_client

    with pytest.raises(LLMAuthenticationError, match="authentication failed"):
        provider.generate_text(prompt="Test question")


def test_provider_timeout_handling():
    """Verify provider raises LLMTimeoutError on timeout exception."""
    provider = GeminiProvider(api_key="mock_key")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = TimeoutError("Request timed out")
    provider._client = mock_client

    with pytest.raises(LLMTimeoutError, match="timed out"):
        provider.generate_text(prompt="Test question")


def test_provider_rate_limit_handling():
    """Verify provider raises LLMRateLimitError when rate limit is exceeded (429)."""
    provider = GeminiProvider(api_key="mock_key")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("429 ResourceExhausted: quota exceeded")
    provider._client = mock_client

    with pytest.raises(LLMRateLimitError, match="rate limit or quota exceeded"):
        provider.generate_text(prompt="Test question")


def test_provider_empty_or_malformed_response():
    """Verify provider raises LLMProviderError on null or empty response."""
    provider = GeminiProvider(api_key="mock_key")
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = ""  # Empty text
    mock_client.models.generate_content.return_value = mock_response
    provider._client = mock_client

    with pytest.raises(LLMProviderError, match="empty or null"):
        provider.generate_text(prompt="Test question")


# ==============================================================================
# 2. REQUEST & PROMPT VALIDATION TESTS
# ==============================================================================

def test_request_empty_and_whitespace_question_validation():
    """Verify input validation rejects empty and whitespace-only questions."""
    with pytest.raises(ValueError, match="Question cannot be empty"):
        BusinessQuestionRequest(question="   ")

    with pytest.raises(ValueError, match="at least 3 characters"):
        BusinessQuestionRequest(question="hi")


def test_request_excessively_long_question_validation():
    """Verify input validation rejects questions over 1000 characters."""
    long_q = "a" * 1001
    with pytest.raises(ValueError, match="exceeds maximum length limit"):
        BusinessQuestionRequest(question=long_q)


def test_grounded_context_construction():
    """Verify grounded prompt wraps JSON context and user query with boundary tags."""
    dummy_json = '{"total_quantity": 37873}'
    prompt = build_grounded_prompt("What is total quantity?", dummy_json)

    assert "=== BEGIN VERIFIED ANALYTICS CONTEXT (JSON) ===" in prompt
    assert dummy_json in prompt
    assert "=== USER QUESTION ===" in prompt
    assert "What is total quantity?" in prompt


def test_grounded_system_prompt_rules():
    """Verify all critical grounding rules are present in the system prompt."""
    sp = GROUNDED_SYSTEM_INSTRUCTION
    assert "NO INVENTED NUMBERS" in sp
    assert "NO INDEPENDENT CALCULATIONS" in sp
    assert "HISTORICAL KPIS & TRENDS" in sp
    assert "HISTORICAL MODEL EVALUATION" in sp
    assert "FORWARD BUSINESS FORECAST" in sp
    assert "NON-CAUSAL DISCIPLINE" in sp
    assert "PROMPT INJECTION DEFENSE" in sp
    assert "SYSTEM & SECRET PROTECTION" in sp


def test_historical_vs_forecast_separation_in_prompt():
    """Verify system instructions explicitly separate 2017 evaluation from 2018 forward forecast."""
    sp = GROUNDED_SYSTEM_INSTRUCTION
    assert "39.02" in sp
    assert "16,266.8" in sp
    assert "Forecast Origin" in sp


# ==============================================================================
# 3. SERVICE & INTEGRATION TESTS (Mocked)
# ==============================================================================

@pytest.fixture
def mock_analytics_context() -> BusinessAnalyticsContext:
    """Load or generate a sample BusinessAnalyticsContext for testing."""
    from src.analytics.engine import build_business_analytics_context
    return build_business_analytics_context()


def test_mocked_end_to_end_business_question_service(mock_analytics_context):
    """Verify complete question flow with mocked GeminiProvider."""
    mock_provider = GeminiProvider(api_key="mock_test_key")
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = (
        "Based on the forward forecast, total demand in 2018 is projected at 16,266.8 units. "
        "Compared with 12,420.0 units actual demand in 2017, this represents an expected growth of +30.97%."
    )
    mock_client.models.generate_content.return_value = mock_response
    mock_provider._client = mock_client

    service = LLMBusinessAnalystService(provider=mock_provider, context=mock_analytics_context)
    answer = service.ask("What is the forecast for next year?")

    assert isinstance(answer, BusinessAnswer)
    assert answer.grounded is True
    assert answer.error is None
    assert "16,266.8" in answer.answer
    assert "+30.97%" in answer.answer
    assert answer.execution_time_seconds is not None


def test_fallback_behavior_when_provider_fails(mock_analytics_context):
    """Verify graceful fallback response when Gemini provider raises an exception."""
    mock_provider = GeminiProvider(api_key="mock_test_key")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API connection dropped")
    mock_provider._client = mock_client

    service = LLMBusinessAnalystService(provider=mock_provider, context=mock_analytics_context)
    answer = service.ask("What are the key business KPIs?")

    assert isinstance(answer, BusinessAnswer)
    assert answer.grounded is False
    assert answer.error is not None
    assert "LLM analysis is currently unavailable" in answer.limitations


def test_fallback_behavior_missing_api_key(mock_analytics_context):
    """Verify service returns clean fallback answer when API key is missing."""
    with patch.dict("os.environ", {}, clear=True):
        mock_provider = GeminiProvider(api_key="")
        service = LLMBusinessAnalystService(provider=mock_provider, context=mock_analytics_context)
        answer = service.ask("Which category sells the most?")

        assert isinstance(answer, BusinessAnswer)
        assert answer.grounded is False
        assert "Authentication Error" in answer.error
        assert "API key" in answer.error


# ==============================================================================
# 4. SECRET-SAFETY & SECURITY TESTS
# ==============================================================================

def test_no_credentials_exposed_in_responses_or_errors():
    """Verify that sensitive API key strings are NEVER present in answer fields or error messages."""
    fake_secret = "AIzaSySecretApiKey123456789XyZ"
    mock_provider = GeminiProvider(api_key=fake_secret)
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception(f"Failed with key {fake_secret}")
    mock_provider._client = mock_client

    service = LLMBusinessAnalystService(provider=mock_provider)
    answer = service.ask("What were the largest demand anomalies?")

    # Ensure secret is NOT in answer, error, limitations, or model fields
    assert fake_secret not in answer.answer
    assert fake_secret not in (answer.error or "")
    assert fake_secret not in (answer.limitations or "")
    assert fake_secret not in answer.model


def test_no_raw_dataset_unrestricted_context_passed(mock_analytics_context):
    """Verify provider receives structured BusinessAnalyticsContext, not raw CSV records."""
    mock_provider = GeminiProvider(api_key="mock_key")
    mock_client = MagicMock()
    mock_provider._client = mock_client

    service = LLMBusinessAnalystService(provider=mock_provider, context=mock_analytics_context)

    with patch.object(mock_provider, "generate_text", return_value="Mock response") as mock_gen:
        service.ask("What is total demand?")
        called_prompt = mock_gen.call_args[1]["prompt"]

        # Context should contain structured JSON keys
        assert "historical_kpis" in called_prompt
        assert "forward_forecast" in called_prompt
        assert "model_evaluation" in called_prompt

        # Context must NOT be raw comma-separated CSV dump
        assert "CA-2016-152156,11/8/2016" not in called_prompt

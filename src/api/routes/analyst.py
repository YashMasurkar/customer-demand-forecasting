"""AI Business Analyst API route serving grounded interpretation via LLMBusinessAnalystService."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.routes.analytics import get_cached_analytics_context
from src.llm.schemas import BusinessQuestionRequest, BusinessAnswer
from src.llm.service import LLMBusinessAnalystService

router = APIRouter(tags=["AI Analyst"])

# Cached singleton service reusing cached analytics context
_ANALYST_SERVICE: LLMBusinessAnalystService | None = None


def get_analyst_service() -> LLMBusinessAnalystService:
    """Retrieve or initialize the singleton LLMBusinessAnalystService."""
    global _ANALYST_SERVICE
    if _ANALYST_SERVICE is None:
        ctx = get_cached_analytics_context()
        _ANALYST_SERVICE = LLMBusinessAnalystService(context=ctx)
    return _ANALYST_SERVICE


@router.post("/ask", response_model=BusinessAnswer)
def ask_analyst_question(request: BusinessQuestionRequest) -> BusinessAnswer:
    """Answer business questions grounded in verified deterministic analytics context."""
    # 1. Validation is handled by BusinessQuestionRequest (raises 422/400 if invalid)
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    service = get_analyst_service()
    answer = service.ask(
        question=request.question,
        model_name=request.model_name,
        temperature=request.temperature
    )

    # If the provider failed due to authentication or service unavailability, raise 503
    if answer.error and not answer.grounded:
        if "Authentication Error" in answer.error or "Provider Error" in answer.error:
            raise HTTPException(
                status_code=503,
                detail=f"LLM Analyst service unavailable: {answer.error}. The underlying deterministic analytics and forecast engines remain fully operational."
            )

    return answer

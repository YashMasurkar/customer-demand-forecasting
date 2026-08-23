"""Pydantic schemas for the Grounded LLM Business Analyst layer."""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class BusinessQuestionRequest(BaseModel):
    """Validated input request for business question answering."""
    question: str = Field(description="Natural language question from the user.")
    model_name: Optional[str] = Field(None, description="Optional LLM model override.")
    temperature: float = Field(default=0.2, ge=0.0, le=1.0, description="Sampling temperature.")

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty or whitespace only.")
        if len(cleaned) < 3:
            raise ValueError("Question must be at least 3 characters long.")
        if len(cleaned) > 1000:
            raise ValueError("Question exceeds maximum length limit of 1000 characters.")
        return cleaned


class BusinessAnswer(BaseModel):
    """Structured response from the Grounded LLM Business Analyst."""
    question: str = Field(description="Original user question.")
    answer: str = Field(description="Human-readable, evidence-grounded answer.")
    model: str = Field(description="LLM model used for interpretation.")
    grounded: bool = Field(default=True, description="Whether the answer is grounded in deterministic analytics.")
    limitations: Optional[str] = Field(None, description="Documented analytical or data limitations if applicable.")
    error: Optional[str] = Field(None, description="Error message if the provider or validation failed.")
    execution_time_seconds: Optional[float] = Field(None, description="Time taken to process and respond.")

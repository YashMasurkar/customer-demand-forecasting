"""Business-facing LLM Analyst service integrating deterministic analytics context and Gemini provider."""

import time
from pathlib import Path
from typing import Optional

from src.analytics.engine import build_business_analytics_context
from src.analytics.schemas import BusinessAnalyticsContext
from src.llm.schemas import BusinessQuestionRequest, BusinessAnswer
from src.llm.prompts import GROUNDED_SYSTEM_INSTRUCTION, build_grounded_prompt
from src.llm.provider import GeminiProvider, LLMProviderError, LLMAuthenticationError


class LLMBusinessAnalystService:
    """Service providing grounded natural-language business answers based on verified analytics."""

    def __init__(
        self,
        provider: Optional[GeminiProvider] = None,
        context: Optional[BusinessAnalyticsContext] = None,
        raw_df_path: Path | str = "data/raw/Sample_Superstore.csv",
        weekly_df_path: Path | str = "data/processed/weekly_demand.csv",
        reports_dir: Path | str = "reports"
    ):
        """Initialize LLM Business Analyst Service.

        Args:
            provider: Optional pre-configured GeminiProvider instance.
            context: Optional pre-computed BusinessAnalyticsContext.
            raw_df_path: Path to raw Superstore CSV.
            weekly_df_path: Path to processed weekly demand CSV.
            reports_dir: Path to reports directory.
        """
        self.provider = provider or GeminiProvider()
        self._context = context
        self.raw_df_path = raw_df_path
        self.weekly_df_path = weekly_df_path
        self.reports_dir = reports_dir

    def get_or_load_context(self) -> BusinessAnalyticsContext:
        """Obtain or compute the deterministic BusinessAnalyticsContext."""
        if self._context is None:
            self._context = build_business_analytics_context(
                raw_df_path=self.raw_df_path,
                weekly_df_path=self.weekly_df_path,
                reports_dir=self.reports_dir
            )
        return self._context

    def ask(
        self,
        question: str,
        model_name: Optional[str] = None,
        temperature: float = 0.2
    ) -> BusinessAnswer:
        """Process user business question using grounded analytics context.

        Args:
            question: User natural-language question.
            model_name: Optional LLM model override.
            temperature: Sampling temperature (default 0.2).

        Returns:
            BusinessAnswer Pydantic model.
        """
        start_time = time.perf_counter()
        target_model = model_name or getattr(self.provider, "default_model", "gemini-3.6-flash")

        # 1. Validate Input Question
        try:
            req = BusinessQuestionRequest(
                question=question,
                model_name=model_name,
                temperature=temperature
            )
        except Exception as e:
            elapsed = round(time.perf_counter() - start_time, 3)
            return BusinessAnswer(
                question=question,
                answer="Invalid question input.",
                model=target_model,
                grounded=False,
                error=str(e),
                execution_time_seconds=elapsed
            )

        # 2. Obtain Verified Analytics Context (Deterministic Python Truth)
        try:
            ctx = self.get_or_load_context()
            ctx_json = ctx.model_dump_json(indent=2)
        except Exception as e:
            elapsed = round(time.perf_counter() - start_time, 3)
            return BusinessAnswer(
                question=req.question,
                answer="Failed to load deterministic analytics context.",
                model=target_model,
                grounded=False,
                error=f"Analytics Engine error: {type(e).__name__}",
                limitations="Underlying analytics calculation failed. Check data pipeline.",
                execution_time_seconds=elapsed
            )

        # 3. Construct Grounded Prompt
        grounded_prompt = build_grounded_prompt(req.question, ctx_json)

        # 4. Invoke LLM Provider with Grounding Instruction
        try:
            raw_answer = self.provider.generate_text(
                prompt=grounded_prompt,
                system_instruction=GROUNDED_SYSTEM_INSTRUCTION,
                model=model_name,
                temperature=temperature
            )
            elapsed = round(time.perf_counter() - start_time, 3)

            return BusinessAnswer(
                question=req.question,
                answer=raw_answer,
                model=target_model,
                grounded=True,
                error=None,
                execution_time_seconds=elapsed
            )

        except LLMAuthenticationError as e:
            elapsed = round(time.perf_counter() - start_time, 3)
            return BusinessAnswer(
                question=req.question,
                answer="LLM analysis is currently unavailable due to an API configuration issue.",
                model=target_model,
                grounded=False,
                error="Authentication Error: Invalid or missing Gemini API key.",
                limitations="LLM analysis is currently unavailable. The underlying analytics and forecast results are still available.",
                execution_time_seconds=elapsed
            )
        except LLMProviderError as e:
            elapsed = round(time.perf_counter() - start_time, 3)
            return BusinessAnswer(
                question=req.question,
                answer="LLM analysis is temporarily unavailable.",
                model=target_model,
                grounded=False,
                error=f"Provider Error: {str(e)[:100]}",
                limitations="LLM analysis is currently unavailable. The underlying analytics and forecast results are still available.",
                execution_time_seconds=elapsed
            )
        except Exception as e:
            elapsed = round(time.perf_counter() - start_time, 3)
            return BusinessAnswer(
                question=req.question,
                answer="An unexpected error occurred during LLM processing.",
                model=target_model,
                grounded=False,
                error=f"Unexpected Error ({type(e).__name__})",
                limitations="LLM analysis is currently unavailable. The underlying analytics and forecast results are still available.",
                execution_time_seconds=elapsed
            )


def ask_business_question(
    question: str,
    model_name: Optional[str] = None,
    context: Optional[BusinessAnalyticsContext] = None
) -> BusinessAnswer:
    """Convenience functional interface to ask business questions with grounded LLM analysis.

    Args:
        question: Natural language question.
        model_name: Optional model override.
        context: Optional pre-loaded BusinessAnalyticsContext.

    Returns:
        BusinessAnswer Pydantic model.
    """
    service = LLMBusinessAnalystService(context=context)
    return service.ask(question=question, model_name=model_name)

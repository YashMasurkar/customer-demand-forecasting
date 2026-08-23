"""Manual smoke test script to test real Gemini API call if LLM_API_KEY is configured.

SECURITY NOTICE:
This script NEVER logs, prints, displays, or exposes the actual API key string.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.service import ask_business_question


def run_smoke_test():
    print("=" * 80)
    print("PHASE 8: GROUNDED LLM BUSINESS ANALYST - GEMINI SMOKE TEST")
    print("=" * 80)

    sample_questions = [
        "What is the demand forecast for next year and how does it compare to 2017?",
        "Which categories generate the most sales and how do their profit margins compare?",
        "How accurate is the forecasting model based on historical evaluation?"
    ]

    for idx, q in enumerate(sample_questions, start=1):
        print(f"\n[Test Query {idx}]: '{q}'")
        print("-" * 65)

        answer_obj = ask_business_question(q)

        print(f"Model:     {answer_obj.model}")
        print(f"Grounded:  {answer_obj.grounded}")
        print(f"Latency:   {answer_obj.execution_time_seconds:.2f}s")
        if answer_obj.error:
            print(f"Error:     {answer_obj.error}")
        if answer_obj.limitations:
            print(f"Caveat:    {answer_obj.limitations}")

        print("\nAnswer Output:")
        print(answer_obj.answer)
        print("=" * 80)


if __name__ == "__main__":
    run_smoke_test()

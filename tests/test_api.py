"""Unit and integration tests for FastAPI backend routes (All LLM provider calls mocked)."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.app import app
from src.llm.schemas import BusinessAnswer
from src.llm.provider import LLMAuthenticationError, LLMProviderError


@pytest.fixture
def client() -> TestClient:
    """Create FastAPI TestClient."""
    return TestClient(app)


# ==============================================================================
# 1. HEALTH ENDPOINT TESTS
# ==============================================================================

def test_get_health_endpoint(client: TestClient):
    """Verify GET /api/health returns 200 OK with expected status payload."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "Demand" in data["app_name"]
    assert data["version"] == "1.0.0"


def test_get_health_v1_alias(client: TestClient):
    """Verify GET /api/v1/health returns 200 OK."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ==============================================================================
# 2. ANALYTICS & DASHBOARD ENDPOINT TESTS
# ==============================================================================

def test_get_analytics_endpoint(client: TestClient):
    """Verify GET /api/analytics returns verified historical KPIs and forward forecast."""
    response = client.get("/api/analytics")
    assert response.status_code == 200
    data = response.json()

    # Verify key structural components
    assert "historical_kpis" in data
    assert data["historical_kpis"]["total_quantity"] == 37873
    assert data["historical_kpis"]["total_sales"] > 2000000.0

    assert "forward_forecast" in data
    assert data["forward_forecast"]["forecast_start_date"] == "2018-01-01"

    assert "model_evaluation" in data
    assert data["model_evaluation"]["test_mae"] == pytest.approx(39.02, abs=0.5)


def test_get_dashboard_summary_endpoint(client: TestClient):
    """Verify GET /api/dashboard returns aggregated executive summary and insights."""
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    data = response.json()

    assert "historical_kpis" in data
    assert "forward_forecast_summary" in data
    assert "category_summary" in data
    assert "regional_summary" in data
    assert "champion_model" in data
    assert "business_insights" in data
    assert len(data["business_insights"]) == 3


# ==============================================================================
# 3. FORECAST ENDPOINT & HORIZON TESTS
# ==============================================================================

def test_get_forecast_default_horizon_52(client: TestClient):
    """Verify GET /api/forecast returns full 52-week forward forecast by default."""
    response = client.get("/api/forecast")
    assert response.status_code == 200
    data = response.json()

    assert data["horizon_weeks"] == 52
    assert len(data["forecast_records"]) == 52
    assert data["forecast_start_date"] == "2018-01-01"
    assert data["forecast_end_date"] == "2018-12-24"
    assert data["total_forecast_quantity"] == pytest.approx(16266.75, abs=10.0)


@pytest.mark.parametrize("horizon", [1, 4, 8, 12, 52])
def test_get_forecast_allowed_horizons(client: TestClient, horizon: int):
    """Verify GET /api/forecast returns exact requested horizon length for allowed values."""
    response = client.get(f"/api/forecast?horizon={horizon}")
    assert response.status_code == 200
    data = response.json()

    assert data["horizon_weeks"] == horizon
    assert len(data["forecast_records"]) == horizon
    assert data["total_forecast_quantity"] > 0


def test_get_forecast_invalid_horizon_rejects_400(client: TestClient):
    """Verify GET /api/forecast rejects unsupported horizon with 400 Bad Request."""
    response = client.get("/api/forecast?horizon=99")
    assert response.status_code == 400
    data = response.json()
    assert "Invalid horizon" in data["detail"]


# ==============================================================================
# 4. MODELS ENDPOINT TESTS
# ==============================================================================

def test_get_models_benchmark_endpoint(client: TestClient):
    """Verify GET /api/models returns comprehensive benchmark comparison and identifies champion."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()

    assert "champion_model" in data
    assert "Holt-Winters" in data["champion_model"]
    assert len(data["models"]) >= 10

    # Ensure Holt-Winters is marked champion and ranked #1
    champion_entry = [m for m in data["models"] if m["is_champion"]][0]
    assert champion_entry["rank"] == 1
    assert champion_entry["mae"] == 39.02


# ==============================================================================
# 5. BUSINESS ANALYST ENDPOINT TESTS (Mocked LLM)
# ==============================================================================

def test_post_ask_valid_question_mocked_success(client: TestClient):
    """Verify POST /api/ask returns grounded answer with mocked LLM service."""
    mock_answer = BusinessAnswer(
        question="What is the forecast for next year?",
        answer="The forward forecast for 2018 is 16,266.8 units (+30.97% vs 2017).",
        model="gemini-3.6-flash",
        grounded=True,
        execution_time_seconds=0.45
    )

    with patch("src.api.routes.analyst.get_analyst_service") as mock_get_svc:
        mock_svc = MagicMock()
        mock_svc.ask.return_value = mock_answer
        mock_get_svc.return_value = mock_svc

        response = client.post("/api/ask", json={"question": "What is the forecast for next year?"})
        assert response.status_code == 200
        data = response.json()
        assert data["grounded"] is True
        assert "16,266.8" in data["answer"]


def test_post_ask_empty_question_returns_400(client: TestClient):
    """Verify POST /api/ask rejects empty or whitespace-only questions with 400."""
    response = client.post("/api/ask", json={"question": "   "})
    assert response.status_code in [400, 422]


def test_post_ask_provider_failure_returns_503(client: TestClient):
    """Verify POST /api/ask returns 503 when LLM provider is unavailable."""
    mock_failed_answer = BusinessAnswer(
        question="What are key KPIs?",
        answer="LLM analysis is temporarily unavailable.",
        model="gemini-3.6-flash",
        grounded=False,
        error="Authentication Error: Invalid API key",
        limitations="LLM analysis is currently unavailable."
    )

    with patch("src.api.routes.analyst.get_analyst_service") as mock_get_svc:
        mock_svc = MagicMock()
        mock_svc.ask.return_value = mock_failed_answer
        mock_get_svc.return_value = mock_svc

        response = client.post("/api/ask", json={"question": "What are key KPIs?"})
        assert response.status_code == 503
        data = response.json()
        assert "LLM Analyst service unavailable" in data["detail"]


# ==============================================================================
# 6. HISTORICAL PERFORMANCE FILTERING TESTS
# ==============================================================================

def test_get_performance_default_unfiltered(client: TestClient):
    """Verify GET /api/performance with defaults returns full historical KPIs."""
    response = client.get("/api/performance")
    assert response.status_code == 200
    data = response.json()

    assert data["active_filters"]["year"] == "All"
    assert data["active_filters"]["category"] == "All"
    assert data["active_filters"]["region"] == "All"
    assert data["record_count"] == 9994
    assert data["filtered_kpis"]["total_quantity"] == 37873
    assert data["filtered_kpis"]["total_sales"] == pytest.approx(2297200.86, abs=1.0)
    assert data["filtered_kpis"]["total_profit"] == pytest.approx(286397.02, abs=1.0)
    assert len(data["category_summary"]) == 3
    assert len(data["regional_summary"]) == 4


def test_get_performance_filter_by_year(client: TestClient):
    """Verify GET /api/performance?year=2017 filters correctly to 2017 transactions."""
    response = client.get("/api/performance?year=2017")
    assert response.status_code == 200
    data = response.json()

    assert data["active_filters"]["year"] == "2017"
    assert data["filtered_kpis"]["total_quantity"] == 12476
    assert data["filtered_kpis"]["total_sales"] == pytest.approx(733215.26, abs=1.0)
    assert data["filtered_kpis"]["total_profit"] == pytest.approx(93439.27, abs=1.0)
    assert data["record_count"] == 3312


def test_get_performance_filter_by_category(client: TestClient):
    """Verify GET /api/performance?category=Technology filters correctly to Technology."""
    response = client.get("/api/performance?category=Technology")
    assert response.status_code == 200
    data = response.json()

    assert data["active_filters"]["category"] == "Technology"
    assert data["filtered_kpis"]["total_quantity"] == 6939
    assert data["filtered_kpis"]["total_sales"] == pytest.approx(836154.03, abs=1.0)
    assert data["filtered_kpis"]["total_profit"] == pytest.approx(145454.95, abs=1.0)

    # In category breakdown, Technology has 100% share while others are 0
    tech_item = [c for c in data["category_summary"] if c["dimension_value"] == "Technology"][0]
    assert tech_item["quantity"] == 6939
    assert tech_item["quantity_share_pct"] == 100.0


def test_get_performance_filter_by_region(client: TestClient):
    """Verify GET /api/performance?region=West filters correctly to West region."""
    response = client.get("/api/performance?region=West")
    assert response.status_code == 200
    data = response.json()

    assert data["active_filters"]["region"] == "West"
    assert data["filtered_kpis"]["total_quantity"] == 12266
    assert data["filtered_kpis"]["total_sales"] == pytest.approx(725457.82, abs=1.0)
    assert data["filtered_kpis"]["total_profit"] == pytest.approx(108418.45, abs=1.0)

    # In regional breakdown, West has 100% share
    west_item = [r for r in data["regional_summary"] if r["dimension_value"] == "West"][0]
    assert west_item["quantity"] == 12266
    assert west_item["quantity_share_pct"] == 100.0


def test_get_performance_combined_filters(client: TestClient):
    """Verify GET /api/performance with combined Year, Category, Region filters."""
    response = client.get("/api/performance?year=2017&category=Technology&region=West")
    assert response.status_code == 200
    data = response.json()

    assert data["active_filters"]["year"] == "2017"
    assert data["active_filters"]["category"] == "Technology"
    assert data["active_filters"]["region"] == "West"
    assert data["filtered_kpis"]["total_quantity"] == 851
    assert data["filtered_kpis"]["total_sales"] == pytest.approx(95959.15, abs=1.0)
    assert data["filtered_kpis"]["total_profit"] == pytest.approx(18983.80, abs=1.0)
    assert data["filtered_kpis"]["total_orders"] == 177
    assert data["record_count"] == 213


def test_get_performance_v1_alias(client: TestClient):
    """Verify /api/v1/performance alias behaves identically."""
    response = client.get("/api/v1/performance?year=2016")
    assert response.status_code == 200
    assert response.json()["active_filters"]["year"] == "2016"


def test_get_performance_invalid_filters_rejects_400(client: TestClient):
    """Verify invalid year, category, or region returns 400 Bad Request."""
    resp1 = client.get("/api/performance?year=1999")
    assert resp1.status_code == 400

    resp2 = client.get("/api/performance?category=InvalidCategory")
    assert resp2.status_code == 400

    resp3 = client.get("/api/performance?region=Antarctica")
    assert resp3.status_code == 400


def test_performance_filtering_does_not_affect_forecast_or_models(client: TestClient):
    """Verify that calling performance filters does not mutate forecast or model benchmarks."""
    # 1. Call filtered performance
    client.get("/api/performance?year=2017&category=Technology")

    # 2. Check forecast is unchanged
    f_resp = client.get("/api/forecast?horizon=52")
    assert f_resp.status_code == 200
    assert f_resp.json()["total_forecast_quantity"] == pytest.approx(16266.75, abs=10.0)

    # 3. Check models benchmark is unchanged
    m_resp = client.get("/api/models")
    assert m_resp.status_code == 200
    champion = [m for m in m_resp.json()["models"] if m["is_champion"]][0]
    assert champion["mae"] == 39.02


# ==============================================================================
# 7. STATIC ASSETS & SECURITY VERIFICATIONS
# ==============================================================================

def test_root_serves_static_dashboard(client: TestClient):
    """Verify GET / returns the HTML dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "DemandIQ" in response.text


def test_no_api_keys_exposed_in_any_endpoint(client: TestClient):
    """Verify that no sensitive environment keys appear in endpoint payloads."""
    for path in ["/api/health", "/api/analytics", "/api/dashboard", "/api/forecast", "/api/models", "/api/performance"]:
        resp = client.get(path)
        assert resp.status_code == 200
        text = resp.text
        assert "AIzaSy" not in text
        assert "LLM_API_KEY" not in text


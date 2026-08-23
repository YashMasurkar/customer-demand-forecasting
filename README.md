# DemandIQ — AI-Powered Customer Demand Forecasting & Business Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20GenAI-gemini--3.6--flash-4285F4.svg)](https://ai.google.dev/)
[![Tests](https://img.shields.io/badge/Tests-110%20Passed-success.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**DemandIQ** is an end-to-end Data Science, Time-Series Forecasting, and Business Intelligence platform. It analyzes historical commercial transactions, benchmarks classical statistical and machine-learning forecasting architectures, generates forward demand projections with analytical prediction intervals, and provides an executive **Grounded AI Business Analyst** powered by Google Gemini.

---

## 🎯 Executive Overview & Architectural Highlights

1. **End-to-End Forecasting Pipeline**: Ingests and processes raw retail transactions into a continuous, Monday-aligned weekly demand time series (208 weeks).
2. **Empirical Model Benchmarking**: Evaluates 12 candidate forecasting architectures across classical statistical methods (Holt-Winters, SARIMA), baselines (Naive, Seasonal Naive), and machine learning models (Ridge, OLS, Random Forest, Gradient Boosting, HistGradientBoosting) using a strict chronological holdout.
3. **Evaluation vs. Forward Forecast Separation**:
   - **Historical Evaluation**: Trains on 2014–2016 (156 weeks) and evaluates out-of-sample accuracy on the **2017 holdout** (52 weeks, MAE = 39.02, RMSE = 52.40).
   - **Forward Production Forecast**: Trains on full historical data through `2017-12-25` to project genuine future demand across **2018** (52 weeks: `2018-01-01` to `2018-12-24`, 16,266.8 units, +30.97% YoY vs. 2017).
4. **Deterministic Analytics Source of Truth**: All numerical aggregations, revenue/profit metrics, and rolling Z-score anomalies are calculated deterministically by Python. The LLM acts strictly as an **interpretation layer**, keeping numerical calculations deterministic and separate from the LLM interpretation layer.
5. **Interactive Full-Stack Web Dashboard**: Built with FastAPI, Vanilla JavaScript, and Chart.js, featuring real-time multi-dimensional performance filters, dynamic forecast planning horizons, and a secure, XSS-sanitized AI Analyst interface.

---

## 📊 Dataset & Statistical Context

- **Source**: Tableau Sample Superstore Dataset (Public commercial demonstration dataset).
- **File**: `data/raw/Sample_Superstore.csv` (2.28 MB, 9,994 transactional rows).
- **Timeframe**: 2014-01-03 to 2017-12-30.
- **Aggregation**: Grouped by Monday-aligned ISO calendar weeks into `data/processed/weekly_demand.csv` (208 continuous observations, zero missing weeks).
- **Why Weekly Aggregation?** Retail sales data at the daily level exhibit extreme day-of-week sparsity and irregular weekend closures. Aggregating to Monday-aligned weeks stabilizes variance and captures the 52-week annual seasonality without artificial noise.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          1. DATA INGESTION & PROCESSING                     │
│   Raw Superstore CSV (9,994 rows) ──► Validation ──► Weekly Monday Demand   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                    2. TIME-SERIES & MACHINE LEARNING PIPELINE               │
│   • Statistical Baselines (Naive, Seasonal Naive lag-52)                    │
│   • Classical Forecasting (Holt-Winters Additive Trend/Season, SARIMA)      │
│   • ML Regressors (Ridge, OLS, Random Forest, GBDT, HistGradientBoosting)  │
│   • Feature Engineering (Lags t-1..t-52, Rolling Stats, Shifted Business)   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                  3. DETERMINISTIC BUSINESS ANALYTICS ENGINE                 │
│   • Historical Baseline KPIs ($2.30M Sales, $286.4K Profit, 12.47% Margin)  │
│   • Multi-Dimensional Slicing (Category, Region, Sub-Category)              │
│   • Rolling Z-Score Anomaly Detection (|Z| >= 2.0)                          │
│   • 2018 Forward Production Forecast (16,266.8 units, 95% Prediction Bounds)│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                         4. APPLICATION & API LAYER                          │
│   FastAPI Backend (`/api/analytics`, `/api/forecast`, `/api/performance`,   │
│                    `/api/models`, `/api/ask`, `/api/health`)                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
         ┌─────────────────────────────┴─────────────────────────────┐
         ▼                                                           ▼
┌──────────────────────────────────┐        ┌────────────────────────────────┐
│   5. INTERACTIVE WEB DASHBOARD   │        │   6. GROUNDED GEMINI ANALYST   │
│   • Executive KPI Overview       │        │   • gemini-3.6-flash           │
│   • Forward Forecast Horizons    │        │   • Verified Analytics Context │
│   • Interactive Perf. Filters    │        │   • Strict JSON Isolation      │
│   • Fixed Model Benchmarks       │        │   • Two-Pass Safe Markdown     │
└──────────────────────────────────┘        └────────────────────────────────┘
```

---

## 🏆 Forecasting Model Benchmark (2017 Chronological Holdout)

The forecasting models were evaluated on an **unseen 52-week chronological test holdout** (2017-01-02 to 2017-12-25) following a 3-year training window (2014–2016). No future data was used in scaling, feature engineering, or model tuning.

| Rank | Model Architecture | Family | MAE (Units) | RMSE (Units) | MAPE (%) | sMAPE (%) | Bias (ME) | Outcome |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 👑 **1** | **Holt-Winters (Add Trend, Add Season, $s=52$)** | **Classical Smoothing** | **39.02** | **52.40** | **19.24%** | **17.40%** | **+3.76** | **Production Champion** |
| 2 | SARIMA(0, 1, 1)(0, 1, 1, 52) | State-Space ARIMA | 46.49 | 59.68 | 20.86% | 20.76% | +20.97 | Strong Seasonal Fit |
| 3 | Ridge Regression ($\alpha=100.0$) | Regularized Linear ML | 50.02 | 66.79 | 22.50% | 21.28% | +14.65 | Top ML Performer |
| 4 | Ridge Regression ($\alpha=10.0$) | Regularized Linear ML | 52.01 | 65.66 | 24.22% | 22.25% | +4.64 | Balanced Regularization |
| 5 | Ridge Regression ($\alpha=1.0$) | Regularized Linear ML | 57.28 | 69.92 | 27.04% | 24.32% | -1.06 | Low Regularization |
| 6 | Linear Regression (OLS) | Linear Regression | 60.35 | 79.84 | 27.52% | 25.07% | +2.92 | Baseline Regression |
| 7 | Gradient Boosting (`depth=3`) | Tree Ensemble | 63.25 | 81.61 | 26.38% | 26.26% | +25.03 | Tree Ensemble |
| 8 | Naive Baseline ($t-1$) | Baseline | 63.75 | 84.39 | 30.93% | 27.86% | +0.25 | Benchmark Baseline |
| 9 | Seasonal Naive ($t-52$) | Seasonal Baseline | 65.63 | 84.66 | 28.03% | 33.41% | +49.10 | Benchmark Baseline |
| 10 | Random Forest (`max_depth=6`) | Tree Ensemble | 67.76 | 86.90 | 27.80% | 28.45% | +32.93 | Tree Ensemble |
| 11 | Random Forest (`max_depth=4`) | Tree Ensemble | 68.10 | 86.49 | 27.85% | 28.68% | +34.09 | Tree Ensemble |
| 12 | HistGradientBoosting | Tree Ensemble | 71.17 | 87.88 | 30.29% | 31.86% | +32.43 | Overfit Baseline |

### Why Did Classical Holt-Winters Outperform ML Tree Models?
1. **Trend Extrapolation Capability**: Decision tree algorithms split feature space orthogonally and predict constant mean values within leaves; they cannot extrapolate multi-year upward linear trends beyond the maximum target values observed in training.
2. **Sample-Size Preservation**: Machine learning models require an explicit `lag_52` feature to capture annual seasonality, which drops the first 52 weeks (33%) of the training dataset. Holt-Winters models the trend and 52-week seasonal cycle directly across the entire 156-week training history without sample loss.

---

## 📈 Forward Production Forecast (2018)

- **Forecast Origin**: `2017-12-25` (End of historical data).
- **Projection Horizon**: 52 weeks (`2018-01-01` to `2018-12-24`).
- **Projected Total 2018 Demand**: **16,266.8 units** (+30.97% YoY vs. 12,420.0 actual units in 2017).
- **Weekly Run Rate**: 312.8 units/week.
- **Forecast Peak**: Week of `2018-11-26` (528.8 units).
- **Forecast Trough**: Week of `2018-01-01` (169.0 units).
- **Planning Horizons Supported**:
  - **1 Week**: 169.0 units ($-27.46\%$ vs. preceding week).
  - **4 Weeks**: 747.7 units ($-44.69\%$ vs. preceding 4 weeks — seasonal Q1 post-holiday dip).
  - **8 Weeks**: 1,659.1 units ($-40.23\%$ vs. preceding 8 weeks).
  - **12 Weeks (Q1)**: 2,504.2 units ($-42.25\%$ vs. preceding 12 weeks).
  - **Full 52 Weeks**: 16,266.8 units ($+30.97\%$ YoY vs. 2017 actuals).

---

## 🖥️ Interactive Web Dashboard Experience

The frontend is a single-page analytics application served directly by FastAPI:

1. **Executive Performance Overview**: High-level KPIs (Total Demand, Gross Revenue, Net Profit, Commercial Margin, AOV) and deterministic automated business insight cards.
2. **Forward Demand Forecast**: Interactive line chart with 95% analytical prediction intervals and dynamic planning horizon switcher (4W, 8W, 12W, 52W).
3. **Commercial & Regional Performance**:
   - **Interactive Multi-Dimensional Filters**: Slices historical transactions across **Year** (`All`, `2014`, `2015`, `2016`, `2017`), **Category** (`All`, `Furniture`, `Office Supplies`, `Technology`), and **Region** (`All`, `Central`, `East`, `South`, `West`).
   - Dynamic filtered KPI strip, Category revenue vs. volume bar chart/table, and Regional demand distribution doughnut chart/table.
4. **Forecasting Model Benchmarks**: Displays the fixed 12-model chronological evaluation matrix and Holt-Winters champion showcase. *(Note: Model Benchmarks remain fixed during historical filtering because they represent the rigorous, predefined 2017 company-wide holdout experiment).*
5. **Grounded AI Business Analyst**: Natural-language Q&A interface with suggested prompt chips, latency tracking, analytical caveat badges, and a secure two-pass Markdown renderer.

---

## 🧠 Grounded Gemini AI Analyst Architecture

```
USER NATURAL LANGUAGE QUESTION
              ↓
DETERMINISTIC ANALYTICS ENGINE (Python computes verified facts)
              ↓
STRUCTURED CONTEXT INJECTION (Strict JSON payload isolation)
              ↓
SYSTEM GROUNDING PROMPT (Rules 1–9: Executive conciseness, no invented numbers)
              ↓
GOOGLE GEMINI API (gemini-3.6-flash, temperature=0.2)
              ↓
TWO-PASS XSS-NEUTRALIZED MARKDOWN PARSER (HTML Entity Escaping + Tag Whitelist)
              ↓
EXECUTIVE BUSINESS ANSWER (With analytical caveats & latency metric)
```

### Safety & Guardrail Design:
- **Verified Analytics Context**: The LLM never computes formulas or aggregates rows independently. All numbers originate from verified Python calculations.
- **Two-Pass Client Sanitization**: Input text is converted to safe HTML entities prior to Markdown formatting, completely mitigating XSS vulnerabilities.
- **Graceful Fallback**: If the Gemini API key is unconfigured or throttled, the backend returns a `503 Service Unavailable` status and the UI alerts the user while keeping all forecasting charts and performance analytics 100% operational.

---

## 📁 Repository Structure

```
customer-demand-forecasting/
├── data/
│   ├── raw/                    # Raw untouched Superstore dataset
│   └── processed/              # Aggregated continuous weekly time-series
├── reports/                    # Generated charts, evaluation JSONs, and audit reports
├── src/
│   ├── analytics/              # Deterministic Analytics Engine & Pydantic schemas
│   │   ├── engine.py           # Master analytical calculations & filtered slicing
│   │   └── schemas.py          # Data contract models
│   ├── api/                    # FastAPI backend layer
│   │   ├── app.py              # Application factory & CORS configuration
│   │   └── routes/             # Health, Analytics, Forecast, Models, Analyst routes
│   ├── features/               # Lag, rolling, and calendar feature engineering
│   ├── llm/                    # Grounded LLM Business Analyst layer
│   │   ├── prompts.py          # Grounding system instructions & prompt builder
│   │   ├── provider.py         # Isolated Gemini API client (google-genai SDK)
│   │   ├── schemas.py          # LLM Request & Response Pydantic models
│   │   ├── service.py          # Grounded business question answering service
│   │   └── smoke_test.py       # Live API smoke test script
│   ├── models/                 # Holt-Winters, SARIMA, and ML forecasters
│   ├── static/                 # Frontend dashboard assets
│   │   ├── app.js              # Client application, API fetchers, safe Markdown parser
│   │   ├── index.html          # Semantic dashboard layout & accessible UI
│   │   └── style.css           # Modern SaaS design system & typography
│   ├── baselines.py            # Naive and Seasonal Naive baselines
│   ├── data_processing.py      # Monday-aligned weekly aggregation pipeline
│   ├── evaluation.py           # Forecast accuracy metrics (MAE, RMSE, MAPE, sMAPE, Bias)
│   └── visualization.py        # High-resolution plotting routines
├── tests/                      # Automated unit and integration test suite (110 tests)
├── .env.example                # Environment variable template (placeholders only)
├── .gitignore                  # Git exclusion rules (protects .env and credentials)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start & Local Execution

### 1. Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd customer-demand-forecasting

# Create and activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows PowerShell
# source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and provide your Google Gemini API key:
```bash
cp .env.example .env
```
In `.env`:
```ini
LLM_API_KEY=your_actual_gemini_api_key_here
LLM_MODEL=gemini-3.6-flash
```
*(Note: `.env` is excluded in `.gitignore` and is never committed to Git).*

### 3. Launch the Application
Start the FastAPI server with auto-reload:
```bash
uvicorn src.api.app:app --reload
```
Open your browser and navigate to:
- **Web Dashboard**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧪 Automated Testing

The repository contains **110 automated unit and integration tests** covering data processing, feature engineering, model training, holdout evaluation, Pydantic schemas, API endpoints, error handling, security headers, and mocked LLM services.

Run the full test suite:
```bash
pytest -v
```

```text
================================ 110 passed in 93.27s ================================
```

---

## 📜 License & Portfolio Citation

This project was engineered as a professional portfolio demonstration of applied time-series data science, machine learning benchmarking, deterministic business analytics, and grounded generative AI architecture. Available under the MIT License.

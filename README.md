# AI-Powered Customer Demand Forecasting & Business Intelligence Platform

A production-grade Data Science, Time-Series Forecasting, and Business Intelligence project that analyzes historical business/sales data, evaluates statistical and ML forecasting models, generates rigorous predictions with uncertainty intervals, and provides grounded business intelligence insights via an LLM layer.

---

## 🎯 Project Objectives

- **Historical & Statistical Analysis**: Uncover demand trends, seasonality, stationarity, volatility, and category-level dynamics.
- **Robust Time-Series Forecasting**: Benchmark baselines against statistical models (Holt-Winters, SARIMA) and ML approaches (Ridge, Tree Ensembles) with chronological cross-validation.
- **Model Evaluation & Error Analysis**: Evaluate MAE, RMSE, sMAPE/MAPE without temporal leakage, inspecting under/over-prediction and seasonal dynamics.
- **Forecast Uncertainty**: Provide analytical prediction intervals alongside point forecasts.
- **Deterministic Business Analytics Engine**: Compute verified KPIs, growth dynamics, category/regional breakdowns, and rolling Z-score anomalies.
- **Grounded AI Business Analyst**: Translate validated numerical analytics into structured business insights without hallucinating metrics or fabricating trends.

---

## 📊 Dataset Information

- **Dataset**: Sample Superstore Dataset
- **Exact Source**: Tableau Sample Data (mirrored publicly at `https://raw.githubusercontent.com/suyog2001/Tableau_Superstore_Sales_Dashboard/main/Sample%20-%20Superstore.csv`)
- **Raw File Stored**: `data/raw/Sample_Superstore.csv` (2,287,806 bytes)
- **Source Notes**: Standard public demonstration dataset provided by Tableau for business intelligence, time-series, and analytics education.
- **Granularity & Dimensions**: 9,994 transaction rows, 21 columns, covering 2014-01-03 to 2017-12-30.

---

## 📁 Repository Structure

```
customer-demand-forecasting/
├── data/
│   ├── raw/                    # Untouched original datasets
│   └── processed/              # Aggregated continuous weekly time-series
├── reports/                    # Generated charts, evaluation JSONs, and audit reports
├── src/
│   ├── analytics/              # Deterministic Analytics Engine & Pydantic schemas
│   ├── features/               # Temporal, lag, and business feature engineering
│   ├── llm/                    # Grounded LLM Business Analyst layer
│   │   ├── provider.py         # Isolated Gemini API client
│   │   ├── prompts.py          # Grounding system prompt & template builder
│   │   ├── schemas.py          # LLM Request & Response Pydantic models
│   │   ├── service.py          # Grounded business question answering service
│   │   └── smoke_test.py       # Live API smoke test script
│   ├── models/                 # Holt-Winters, SARIMA, and ML forecasters
│   ├── baselines.py            # Naive and Seasonal Naive baselines
│   ├── evaluation.py           # Metric calculation (MAE, RMSE, MAPE, sMAPE, Bias)
│   └── visualization.py        # Publication-ready plotting routines
├── tests/                      # Automated unit and integration test suite (82 tests)
├── .env.example                # Environment variable template (placeholders only)
├── .gitignore                  # Git exclusion rules (protects .env and secrets)
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Setup & Environment

1. Clone or open the repository.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1   # Windows PowerShell
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables in `.env` based on `.env.example`:
   ```bash
   LLM_API_KEY=your_actual_gemini_api_key_here
   LLM_MODEL=gemini-3.6-flash
   ```

---

## 🧠 Grounded LLM Business Analyst (Phase 8)

The platform incorporates a **Grounded LLM Business Analyst** that sits strictly on top of the deterministic Business Analytics Engine.

```
USER QUESTION
      ↓
DETERMINISTIC ANALYTICS CONTEXT (Python verified facts)
      ↓
GROUNDED PROMPT (System rules + Context JSON + Question)
      ↓
GEMINI API (gemini-3.6-flash)
      ↓
NATURAL-LANGUAGE BUSINESS ANSWER
```

### Core Architecture Principles:
- **Python as Source of Truth**: All metrics (totals, growth rates, profit margins, anomalies, and forward forecasts) are calculated deterministically by Python. The LLM is strictly an **interpretation layer**.
- **No Hallucinated Numbers**: The system prompt enforces strict adherence to numbers present in the analytics context.
- **Concept Separation**:
  - **Historical Observations**: 2014–2017 actual business metrics (37,873 units, \$2,297,200.86 sales, 12.47% margin).
  - **Historical Model Evaluation**: Out-of-sample holdout test performance on the 2017 test set (MAE = 39.02, RMSE = 52.40, MAPE = 19.24%, Bias = +3.76).
  - **Forward Business Forecast**: Genuinely future 52-week projections for 2018 (from Forecast Origin `2017-12-25`, starting `2018-01-01` through `2018-12-24`: 16,266.8 units, +30.97% vs 2017 actuals).
- **Prompt-Injection Defense**: Dataset text is treated as passive data, preventing prompt override attacks.
- **Secret Protection**: API credentials are loaded solely from the environment and are never logged, echoed, or included in repository files.

### Supported Question Categories:
- **Historical Performance**: *"What are our key business KPIs?"*, *"Which categories generate the most sales and profit?"*
- **Forward Forecasts**: *"What is the demand forecast for next year and how does it compare to 2017?"*, *"When is expected demand to peak?"*
- **Statistical Anomalies**: *"Were there unusual demand weeks and what were the largest deviations?"*
- **Model Evaluation**: *"How accurate is the forecasting model based on historical evaluation?"*

---

## 🧪 Automated Testing

Run the complete test suite (all LLM tests use mocked providers and do not require active network calls):

```bash
pytest -v
```

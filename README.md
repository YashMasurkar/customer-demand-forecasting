# AI-Powered Customer Demand Forecasting & Business Intelligence Platform

A production-grade Data Science, Time-Series Forecasting, and Business Intelligence project that analyzes historical business/sales data, evaluates statistical and ML forecasting models, generates rigorous predictions with uncertainty intervals, and provides grounded business intelligence insights via an LLM layer.

---

## 🎯 Project Objectives

- **Historical & Statistical Analysis**: Uncover demand trends, seasonality, stationarity, volatility, and category-level dynamics.
- **Robust Time-Series Forecasting**: Benchmark baselines against statistical models (e.g., Holt-Winters, SARIMA) and ML approaches with chronological cross-validation.
- **Model Evaluation & Error Analysis**: Evaluate MAE, RMSE, sMAPE/MAPE without temporal leakage, inspecting under/over-prediction and seasonal failures.
- **Forecast Uncertainty**: Provide calibrated prediction intervals alongside point forecasts.
- **Grounded AI Business Analyst**: Translate validated numerical analytics into structured business insights without hallucinating metrics or fabricating trends.

---

## 📊 Dataset Information

- **Dataset**: Sample Superstore Dataset
- **Exact Source**: Tableau Sample Data (mirrored publicly at `https://raw.githubusercontent.com/suyog2001/Tableau_Superstore_Sales_Dashboard/main/Sample%20-%20Superstore.csv`)
- **Raw File Stored**: [`data/raw/Sample_Superstore.csv`](file:///e:/projects/customer-demand-forecasting/data/raw/Sample_Superstore.csv) (2,287,806 bytes)
- **Source Notes**: Standard public demonstration dataset provided by Tableau for business intelligence, time-series, and analytics education.
- **Granularity & Dimensions**: 9,994 transaction rows, 21 columns, covering 2014-01-03 to 2017-12-30.

---

## 📁 Repository Structure

```
customer-demand-forecasting/
├── data/
│   ├── raw/            # Untouched original datasets
│   ├── cleaned/        # Validated and cleaned records
│   └── processed/      # Aggregated time-series ready for modeling
├── notebooks/          # Exploratory Data Analysis & experiments
├── src/                # Core modular source code
├── models/             # Model artifacts and serialization
├── tests/              # Automated unit and integration tests
├── .env.example        # Environment variable template
├── .gitignore          # Git exclusion rules
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🚀 Setup & Environment

1. Clone or open the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables in `.env` (using `.env.example` as a baseline).

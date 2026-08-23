/**
 * DemandIQ — Interactive Client Application
 * Consumes FastAPI backend endpoints strictly without frontend recalculations.
 */

document.addEventListener("DOMContentLoaded", () => {
    // Chart instances
    let forecastChart = null;
    let categoryChart = null;
    let regionalChart = null;

    let cachedDashboard = null;
    let currentHorizon = 52;

    // DOM Elements
    const apiBadge = document.getElementById("api-status-badge");
    const statusText = apiBadge.querySelector(".status-text");
    const insightsContainer = document.getElementById("insights-container");
    const categoryTableBody = document.querySelector("#category-table tbody");
    const regionalTableBody = document.querySelector("#regional-table tbody");
    const modelsTableBody = document.querySelector("#models-table tbody");
    const segButtons = document.querySelectorAll(".seg-btn");

    // Analyst Elements
    const analystForm = document.getElementById("analyst-form");
    const queryInput = document.getElementById("analyst-query-input");
    const submitBtn = document.getElementById("btn-submit-query");
    const btnText = submitBtn.querySelector(".btn-text");
    const btnSpinner = submitBtn.querySelector(".btn-spinner");
    const responseCard = document.getElementById("analyst-response-card");
    const responseContent = document.getElementById("response-content");
    const responseModelTag = document.getElementById("response-model-tag");
    const responseLatencyTag = document.getElementById("response-latency-tag");
    const responseCaveatBox = document.getElementById("response-caveat-box");
    const responseCaveatText = document.getElementById("response-caveat-text");

    // Initialize Dashboard
    initApp();

    async function initApp() {
        try {
            // 1. Check System Health
            await checkHealth();

            // 2. Load Dashboard Executive Metrics & Insights
            await loadDashboardData();

            // 3. Load Forward Forecast & Render Main Chart
            await loadForecast(52);

            // 4. Load Models Benchmark Matrix
            await loadModelsBenchmark();

            // 5. Setup Listeners
            setupEventListeners();

        } catch (err) {
            console.error("Dashboard initialization error:", err);
            setApiStatus("error", "API Offline");
        }
    }

    function setApiStatus(type, label) {
        apiBadge.className = `status-badge status-${type}`;
        statusText.textContent = label;
    }

    async function checkHealth() {
        try {
            const res = await fetch("/api/health");
            if (res.ok) {
                setApiStatus("online", "● System Operational");
            } else {
                setApiStatus("error", "● Service Degraded");
            }
        } catch (e) {
            setApiStatus("error", "● Connection Error");
        }
    }

    async function loadDashboardData() {
        const res = await fetch("/api/dashboard");
        if (!res.ok) throw new Error("Failed to load dashboard data");
        cachedDashboard = await res.json();

        // 1. Populate Executive KPI Cards
        const k = cachedDashboard.historical_kpis;
        document.getElementById("kpi-quantity").textContent = Number(k.total_quantity).toLocaleString() + " u";
        document.getElementById("kpi-sales").textContent = "$" + (Number(k.total_sales) / 1000000).toFixed(2) + "M";
        document.getElementById("kpi-profit").textContent = "$" + (Number(k.total_profit) / 1000).toFixed(1) + "K";
        document.getElementById("kpi-margin").textContent = k.profit_margin_pct.toFixed(1) + "%";
        document.getElementById("kpi-aov").textContent = "$" + k.average_order_value.toFixed(2);

        // 2. Render Dynamic Business Insights
        const insights = cachedDashboard.business_insights || [];
        insightsContainer.innerHTML = insights.map(i => `
            <div class="insight-card">
                <span class="insight-tag">${escapeHtml(i.category)}</span>
                <div class="insight-title">${escapeHtml(i.title)}</div>
                <p class="insight-desc">${escapeHtml(i.description)}</p>
            </div>
        `).join("");

        // 3. Render Category Breakdown (Chart & Table)
        renderCategoryPerformance(cachedDashboard.category_summary);

        // 4. Render Regional Distribution (Chart & Table)
        renderRegionalPerformance(cachedDashboard.regional_summary);
    }

    function renderCategoryPerformance(categories) {
        if (!categories || !categories.length) return;

        // Table
        categoryTableBody.innerHTML = categories.map(c => `
            <tr>
                <td><strong>${escapeHtml(c.dimension_value)}</strong></td>
                <td>${Number(c.quantity).toLocaleString()} u</td>
                <td>$${Number(c.sales).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                <td><span class="badge-tag ${c.profit_margin_pct >= 10 ? 'highlight' : ''}">${c.profit_margin_pct.toFixed(1)}%</span></td>
                <td>${c.quantity_share_pct.toFixed(1)}%</td>
            </tr>
        `).join("");

        // Horizontal Bar Chart
        const ctx = document.getElementById("categoryChart").getContext("2d");
        const labels = categories.map(c => c.dimension_value);
        const salesData = categories.map(c => c.sales);
        const qtyData = categories.map(c => c.quantity);

        if (categoryChart) categoryChart.destroy();

        categoryChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Total Sales ($)",
                        data: salesData,
                        backgroundColor: "#6366f1",
                        borderRadius: 6
                    },
                    {
                        label: "Units Sold",
                        data: qtyData,
                        backgroundColor: "#cbd5e1",
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: "top", labels: { font: { size: 11, weight: 600 } } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => ` ${ctx.dataset.label}: ${ctx.datasetIndex === 0 ? '$' + Number(ctx.raw).toLocaleString() : Number(ctx.raw).toLocaleString() + ' u'}`
                        }
                    }
                },
                scales: {
                    x: { grid: { display: false } },
                    y: { grid: { color: "#f1f5f9" } }
                }
            }
        });
    }

    function renderRegionalPerformance(regions) {
        if (!regions || !regions.length) return;

        // Table
        regionalTableBody.innerHTML = regions.map(r => `
            <tr>
                <td><strong>${escapeHtml(r.dimension_value)}</strong></td>
                <td>${Number(r.quantity).toLocaleString()} u</td>
                <td>$${Number(r.sales).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                <td>$${Number(r.profit).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                <td>${r.quantity_share_pct.toFixed(1)}%</td>
            </tr>
        `).join("");

        // Regional Chart
        const ctx = document.getElementById("regionalChart").getContext("2d");
        const labels = regions.map(r => r.dimension_value);
        const qtyShares = regions.map(r => r.quantity_share_pct);

        if (regionalChart) regionalChart.destroy();

        regionalChart = new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: labels,
                datasets: [{
                    data: qtyShares,
                    backgroundColor: ["#6366f1", "#8b5cf6", "#38bdf8", "#94a3b8"],
                    borderWidth: 2,
                    borderColor: "#ffffff"
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: "right", labels: { font: { size: 11, weight: 600 } } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => ` ${ctx.label}: ${ctx.raw.toFixed(1)}% Volume Share`
                        }
                    }
                },
                cutout: "68%"
            }
        });
    }

    async function loadForecast(horizon = 52) {
        currentHorizon = horizon;
        const res = await fetch(`/api/forecast?horizon=${horizon}`);
        if (!res.ok) throw new Error("Failed to load forecast");
        const fData = await res.json();

        // Update Quick Stat Cards with exact authoritative API values and comparison baselines
        if (cachedDashboard && cachedDashboard.forward_forecast_summary) {
            const h = cachedDashboard.forward_forecast_summary.horizons;
            if (h && h.next_1_week) {
                document.getElementById("fstat-1w").textContent = h.next_1_week.total_forecast_quantity.toFixed(1) + " u";
                const p1 = h.next_1_week.pct_change_vs_prior_period;
                document.getElementById("fstat-1w-sub").textContent = `${p1 >= 0 ? '+' : ''}${p1.toFixed(2)}% vs preceding week`;
            }
            if (h && h.next_4_weeks) {
                document.getElementById("fstat-4w").textContent = h.next_4_weeks.total_forecast_quantity.toFixed(1) + " u";
                const p4 = h.next_4_weeks.pct_change_vs_prior_period;
                document.getElementById("fstat-4w-sub").textContent = `${p4 >= 0 ? '+' : ''}${p4.toFixed(2)}% vs preceding 4w`;
            }
            if (h && h.next_12_weeks) {
                document.getElementById("fstat-12w").textContent = Number(h.next_12_weeks.total_forecast_quantity).toLocaleString(undefined, { maximumFractionDigits: 1 }) + " u";
                const p12 = h.next_12_weeks.pct_change_vs_prior_period;
                document.getElementById("fstat-12w-sub").textContent = `${p12 >= 0 ? '+' : ''}${p12.toFixed(2)}% vs preceding 12w`;
            }
            if (h && h.full_52_weeks) {
                document.getElementById("fstat-52w").textContent = Number(h.full_52_weeks.total_forecast_quantity).toLocaleString(undefined, { maximumFractionDigits: 1 }) + " u";
                const p52 = h.full_52_weeks.pct_change_vs_prior_period;
                document.getElementById("fstat-52w-sub").textContent = `${p52 >= 0 ? '+' : ''}${p52.toFixed(2)}% YoY vs 2017 actuals`;
            }
        }

        renderForecastChart(fData);
    }

    function renderForecastChart(fData) {
        const ctx = document.getElementById("forecastChart").getContext("2d");
        const dates = fData.forecast_records.map(r => r.week_start);
        const forecastValues = fData.forecast_records.map(r => r.forecast_quantity);
        const piLower = fData.forecast_records.map(r => r.pi_lower_95);
        const piUpper = fData.forecast_records.map(r => r.pi_upper_95);

        if (forecastChart) forecastChart.destroy();

        forecastChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: dates,
                datasets: [
                    {
                        label: `Forward Production Forecast (${fData.horizon_weeks} Weeks)`,
                        data: forecastValues,
                        borderColor: "#6366f1",
                        backgroundColor: "rgba(99, 102, 241, 0.08)",
                        borderWidth: 2.8,
                        fill: false,
                        tension: 0.25,
                        pointRadius: fData.horizon_weeks <= 12 ? 4 : 2.5,
                        pointBackgroundColor: "#6366f1",
                        pointHoverRadius: 6
                    },
                    {
                        label: "95% Prediction Upper Bound",
                        data: piUpper,
                        borderColor: "rgba(99, 102, 241, 0.3)",
                        borderDash: [4, 4],
                        borderWidth: 1.5,
                        fill: false,
                        pointRadius: 0
                    },
                    {
                        label: "95% Prediction Lower Bound",
                        data: piLower,
                        borderColor: "rgba(99, 102, 241, 0.3)",
                        borderDash: [4, 4],
                        borderWidth: 1.5,
                        fill: "-1",
                        backgroundColor: "rgba(99, 102, 241, 0.08)",
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: "index" },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: "rgba(15, 23, 42, 0.95)",
                        titleFont: { size: 12, weight: 700 },
                        bodyFont: { size: 12 },
                        padding: 12,
                        cornerRadius: 8,
                        callbacks: {
                            label: function (c) {
                                return ` ${c.dataset.label}: ${Number(c.raw).toFixed(1)} units`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { maxTicksLimit: 14, font: { size: 11, weight: 500 } }
                    },
                    y: {
                        beginAtZero: false,
                        title: { display: true, text: "Weekly Demand Quantity (Units)", font: { weight: 600 } },
                        grid: { color: "#f1f5f9" }
                    }
                }
            }
        });
    }

    async function loadModelsBenchmark() {
        const res = await fetch("/api/models");
        if (!res.ok) throw new Error("Failed to load models");
        const data = await res.json();

        modelsTableBody.innerHTML = data.models.map(m => `
            <tr class="${m.is_champion ? 'champion-row' : ''}">
                <td><span class="rank-pill ${m.rank === 1 ? 'rank-pill-1' : ''}">${m.rank}</span></td>
                <td><strong>${escapeHtml(m.model_name)}</strong></td>
                <td>${escapeHtml(m.model_family)}</td>
                <td><strong>${m.mae.toFixed(2)}</strong></td>
                <td>${m.rmse.toFixed(2)}</td>
                <td>${m.mape.toFixed(2)}%</td>
                <td>${m.smape.toFixed(2)}%</td>
                <td>${(m.bias >= 0 ? '+' : '') + m.bias.toFixed(2)}</td>
                <td>${m.is_champion ? '<span class="badge-champion-tag">★ Production Champion</span>' : '<span class="badge-tag">Benchmark</span>'}</td>
            </tr>
        `).join("");
    }

    function setupEventListeners() {
        // Horizon Switcher
        segButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                segButtons.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                const h = parseInt(btn.getAttribute("data-horizon"));
                loadForecast(h);
            });
        });

        // Prompt Chips
        document.querySelectorAll(".chip").forEach(chip => {
            chip.addEventListener("click", () => {
                const query = chip.getAttribute("data-query");
                queryInput.value = query;
                queryInput.focus();
                handleAnalystQuery(query);
            });
        });

        // Form Submit
        analystForm.addEventListener("submit", (e) => {
            e.preventDefault();
            const q = queryInput.value.trim();
            if (q) handleAnalystQuery(q);
        });
    }

    async function handleAnalystQuery(question) {
        submitBtn.disabled = true;
        btnText.style.display = "none";
        btnSpinner.style.display = "inline-flex";
        responseCard.style.display = "block";
        responseContent.textContent = "Connecting to Gemini Analyst layer and synthesizing deterministic Python context...";
        responseModelTag.textContent = "gemini-3.6-flash";
        responseLatencyTag.textContent = "Analyzing...";
        responseCaveatBox.style.display = "none";

        try {
            const res = await fetch("/api/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question })
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `Server error (${res.status})`);
            }

            const answer = await res.json();
            responseContent.textContent = answer.answer;
            responseModelTag.textContent = answer.model;
            responseLatencyTag.textContent = `Latency: ${answer.execution_time_seconds ? answer.execution_time_seconds.toFixed(2) + 's' : '0.4s'}`;

            if (answer.limitations) {
                responseCaveatBox.style.display = "flex";
                responseCaveatText.textContent = answer.limitations;
            }

        } catch (err) {
            responseContent.textContent = `Error: ${err.message || "Failed to process query."}`;
            responseLatencyTag.textContent = "Error";
        } finally {
            submitBtn.disabled = false;
            btnText.style.display = "inline-flex";
            btnSpinner.style.display = "none";
        }
    }

    function escapeHtml(str) {
        if (!str) return "";
        return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }
});

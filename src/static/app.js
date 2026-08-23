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

    // Performance Filter DOM Elements
    const filterYear = document.getElementById("perf-filter-year");
    const filterCategory = document.getElementById("perf-filter-category");
    const filterRegion = document.getElementById("perf-filter-region");
    const btnResetFilters = document.getElementById("btn-reset-perf-filters");
    const filterStatus = document.getElementById("perf-filter-status");

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

        // 1. Populate Executive KPI Cards (Historical 2014-2017 baseline)
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

        // 3. Populate Initial Filtered KPI Strip (Unfiltered baseline)
        updatePerformanceKPIStrip(k, k.total_transactions);

        // 4. Render Initial Category Breakdown (Chart & Table)
        renderCategoryPerformance(cachedDashboard.category_summary);

        // 5. Render Initial Regional Distribution (Chart & Table)
        renderRegionalPerformance(cachedDashboard.regional_summary);
    }

    function updatePerformanceKPIStrip(k, totalRows) {
        if (!k) return;
        const elQty = document.getElementById("pkpi-quantity");
        const elQtySub = document.getElementById("pkpi-quantity-sub");
        const elSales = document.getElementById("pkpi-sales");
        const elSalesSub = document.getElementById("pkpi-sales-sub");
        const elProfit = document.getElementById("pkpi-profit");
        const elProfitSub = document.getElementById("pkpi-profit-sub");
        const elMargin = document.getElementById("pkpi-margin");
        const elMarginSub = document.getElementById("pkpi-margin-sub");
        const elOrders = document.getElementById("pkpi-orders");
        const elAovSub = document.getElementById("pkpi-aov-sub");

        if (elQty) elQty.textContent = formatCompact(k.total_quantity) + " u";
        if (elQtySub) elQtySub.textContent = `${Number(k.total_quantity).toLocaleString()} total units`;
        if (elSales) elSales.textContent = formatCurrency(k.total_sales);
        if (elSalesSub) elSalesSub.textContent = `$${Number(k.total_sales).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        if (elProfit) elProfit.textContent = formatCurrency(k.total_profit);
        if (elProfitSub) elProfitSub.textContent = `$${Number(k.total_profit).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        if (elMargin) elMargin.textContent = `${k.profit_margin_pct.toFixed(1)}%`;
        if (elMarginSub) elMarginSub.textContent = `Efficiency ratio`;
        if (elOrders) elOrders.textContent = Number(k.total_orders).toLocaleString();
        if (elAovSub) elAovSub.textContent = `AOV: $${k.average_order_value.toFixed(2)}`;
    }

    async function applyPerformanceFilters() {
        const yr = filterYear ? filterYear.value : "All";
        const cat = filterCategory ? filterCategory.value : "All";
        const reg = filterRegion ? filterRegion.value : "All";

        try {
            const res = await fetch(`/api/performance?year=${encodeURIComponent(yr)}&category=${encodeURIComponent(cat)}&region=${encodeURIComponent(reg)}`);
            if (!res.ok) throw new Error("Failed to load filtered performance data");
            const perfData = await res.json();

            // 1. Update Filter Status Badge
            if (filterStatus) {
                const parts = [];
                if (yr !== "All") parts.push(`Year: ${yr}`);
                if (cat !== "All") parts.push(`Category: ${cat}`);
                if (reg !== "All") parts.push(`Region: ${reg}`);
                const desc = parts.length ? parts.join(" • ") : "All Historical Data";
                filterStatus.innerHTML = `Filter: <strong>${escapeHtml(desc)} (${Number(perfData.record_count).toLocaleString()} records)</strong>`;
            }

            // 2. Update Filtered KPI Strip
            updatePerformanceKPIStrip(perfData.filtered_kpis, perfData.record_count);

            // 3. Render Filtered Category Performance Chart & Table
            renderCategoryPerformance(perfData.category_summary);

            // 4. Render Filtered Regional Performance Chart & Table
            renderRegionalPerformance(perfData.regional_summary);

        } catch (err) {
            console.error("Filter update error:", err);
        }
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

        // 1. Update Selected Horizon Date Coverage
        const datesEl = document.getElementById("selected-horizon-dates");
        if (datesEl) {
            datesEl.textContent = `${fData.forecast_start_date} → ${fData.forecast_end_date} (${horizon} Weeks)`;
        }

        // 2. Find Horizon Peak and Trough within the active sliced records
        let maxRec = fData.forecast_records[0];
        let minRec = fData.forecast_records[0];
        fData.forecast_records.forEach(r => {
            if (r.forecast_quantity > maxRec.forecast_quantity) maxRec = r;
            if (r.forecast_quantity < minRec.forecast_quantity) minRec = r;
        });

        // 3. Update Primary Selected-Horizon Demand KPI Card
        document.getElementById("primary-horizon-label").textContent = `Selected Horizon Demand (${horizon}W)`;
        document.getElementById("primary-horizon-value").textContent = `${Number(fData.total_forecast_quantity).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} u`;

        const compBadge = document.getElementById("primary-horizon-comp");
        if (compBadge) {
            if (horizon === 52) {
                compBadge.textContent = `+${fData.projected_growth_pct.toFixed(2)}% YoY vs 2017 actuals (${Number(fData.comparison_historical_total_quantity).toLocaleString()} u baseline)`;
            } else {
                const sign = fData.projected_growth_pct >= 0 ? '+' : '';
                compBadge.textContent = `${sign}${fData.projected_growth_pct.toFixed(2)}% vs preceding ${horizon}w (${Number(fData.comparison_historical_total_quantity).toLocaleString()} u baseline)`;
            }
        }

        // 4. Update Secondary Horizon Cards (Weekly Run Rate, Peak Week, Trough Week)
        document.getElementById("primary-horizon-mean").textContent = `${fData.mean_weekly_forecast.toFixed(1)} u/wk`;
        document.getElementById("primary-horizon-mean-sub").textContent = `Average weekly demand across ${horizon} weeks`;

        document.getElementById("primary-horizon-peak").textContent = `${maxRec.forecast_quantity.toFixed(1)} u`;
        document.getElementById("primary-horizon-peak-sub").textContent = `Week of ${maxRec.week_start}`;

        document.getElementById("primary-horizon-trough").textContent = `${minRec.forecast_quantity.toFixed(1)} u`;
        document.getElementById("primary-horizon-trough-sub").textContent = `Week of ${minRec.week_start}`;

        // 5. Update Secondary Reference Horizons Strip from Authoritative Context
        if (cachedDashboard && cachedDashboard.forward_forecast_summary) {
            const h = cachedDashboard.forward_forecast_summary.horizons;
            if (h) {
                if (h.next_1_week) {
                    const p1 = h.next_1_week.pct_change_vs_prior_period;
                    document.getElementById("ref-pill-1w").innerHTML = `<strong>1W:</strong> ${h.next_1_week.total_forecast_quantity.toFixed(1)} u <span class="ref-sub">(${p1 >= 0 ? '+' : ''}${p1.toFixed(2)}% vs preceding wk)</span>`;
                }
                if (h.next_4_weeks) {
                    const p4 = h.next_4_weeks.pct_change_vs_prior_period;
                    document.getElementById("ref-pill-4w").innerHTML = `<strong>4W:</strong> ${h.next_4_weeks.total_forecast_quantity.toFixed(1)} u <span class="ref-sub">(${p4 >= 0 ? '+' : ''}${p4.toFixed(2)}% vs preceding 4w)</span>`;
                }
                if (h.next_12_weeks) {
                    const p12 = h.next_12_weeks.pct_change_vs_prior_period;
                    document.getElementById("ref-pill-12w").innerHTML = `<strong>12W:</strong> ${Number(h.next_12_weeks.total_forecast_quantity).toLocaleString(undefined, { maximumFractionDigits: 1 })} u <span class="ref-sub">(${p12 >= 0 ? '+' : ''}${p12.toFixed(2)}% vs preceding 12w)</span>`;
                }
                if (h.full_52_weeks) {
                    const p52 = h.full_52_weeks.pct_change_vs_prior_period;
                    document.getElementById("ref-pill-52w").innerHTML = `<strong>52W:</strong> ${Number(h.full_52_weeks.total_forecast_quantity).toLocaleString(undefined, { maximumFractionDigits: 1 })} u <span class="ref-sub">(+${p52.toFixed(2)}% YoY vs 2017)</span>`;
                }
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

        // Performance Filter Dropdowns & Reset
        if (filterYear) filterYear.addEventListener("change", applyPerformanceFilters);
        if (filterCategory) filterCategory.addEventListener("change", applyPerformanceFilters);
        if (filterRegion) filterRegion.addEventListener("change", applyPerformanceFilters);
        if (btnResetFilters) {
            btnResetFilters.addEventListener("click", () => {
                if (filterYear) filterYear.value = "All";
                if (filterCategory) filterCategory.value = "All";
                if (filterRegion) filterRegion.value = "All";
                applyPerformanceFilters();
            });
        }

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
        responseContent.innerHTML = "<p>Connecting to Gemini Analyst layer and synthesizing deterministic Python context...</p>";
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
            responseContent.innerHTML = renderMarkdownToSafeHtml(answer.answer);
            responseModelTag.textContent = answer.model;
            responseLatencyTag.textContent = `Latency: ${answer.execution_time_seconds ? answer.execution_time_seconds.toFixed(2) + 's' : '0.4s'}`;

            if (answer.limitations) {
                responseCaveatBox.style.display = "flex";
                responseCaveatText.textContent = answer.limitations;
            }

        } catch (err) {
            responseContent.innerHTML = `<p style="color: #ef4444;"><strong>Error:</strong> ${escapeHtml(err.message || "Failed to process query.")}</p>`;
            responseLatencyTag.textContent = "Error";
        } finally {
            submitBtn.disabled = false;
            btnText.style.display = "inline-flex";
            btnSpinner.style.display = "none";
        }
    }

    /**
     * Parse and render Markdown to safe, sanitized HTML.
     * Escapes all raw HTML first to prevent XSS, then converts Markdown syntax to safe tags.
     */
    function renderMarkdownToSafeHtml(markdown) {
        if (!markdown) return "";
        try {
            // 1. Sanitize raw text by escaping all HTML special characters first
            let text = String(markdown)
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#39;");

            // Normalize line breaks
            text = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");

            const lines = text.split("\n");
            const out = [];
            let inList = false;
            let listType = "ul"; // 'ul' or 'ol'
            let currentParagraph = [];

            function flushParagraph() {
                if (currentParagraph.length > 0) {
                    const pContent = currentParagraph.join("<br>");
                    if (pContent.trim()) {
                        out.push(`<p>${formatInline(pContent)}</p>`);
                    }
                    currentParagraph = [];
                }
            }

            function closeList() {
                if (inList) {
                    out.push(`</${listType}>`);
                    inList = false;
                }
            }

            function formatInline(str) {
                // Inline code: `code`
                str = str.replace(/`([^`]+)`/g, "<code>$1</code>");
                // Bold + Italic: ***text*** or ___text___
                str = str.replace(/\*\*\*([^*]+)\*\*\*/g, "<strong><em>$1</em></strong>");
                str = str.replace(/___([^_]+)___/g, "<strong><em>$1</em></strong>");
                // Bold: **text** or __text__
                str = str.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
                str = str.replace(/__([^_]+)__/g, "<strong>$1</strong>");
                // Italic: *text* or _text_
                str = str.replace(/\*([^*]+)\*/g, "<em>$1</em>");
                str = str.replace(/_([^_]+)_/g, "<em>$1</em>");
                return str;
            }

            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                const trimmed = line.trim();

                // Blank line
                if (!trimmed) {
                    flushParagraph();
                    closeList();
                    continue;
                }

                // Horizontal Rule: --- or *** or ___
                if (/^(?:---|\*\*\*|___)\s*$/.test(trimmed)) {
                    flushParagraph();
                    closeList();
                    out.push("<hr>");
                    continue;
                }

                // Headings: #, ##, ###, ####
                const hMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
                if (hMatch) {
                    flushParagraph();
                    closeList();
                    const level = hMatch[1].length;
                    const hContent = formatInline(hMatch[2]);
                    out.push(`<h${level}>${hContent}</h${level}>`);
                    continue;
                }

                // Bullet List: * item or - item
                const ulMatch = trimmed.match(/^[\*\-]\s+(.+)$/);
                if (ulMatch) {
                    flushParagraph();
                    if (!inList || listType !== "ul") {
                        closeList();
                        out.push("<ul>");
                        inList = true;
                        listType = "ul";
                    }
                    out.push(`<li>${formatInline(ulMatch[1])}</li>`);
                    continue;
                }

                // Numbered List: 1. item
                const olMatch = trimmed.match(/^\d+\.\s+(.+)$/);
                if (olMatch) {
                    flushParagraph();
                    if (!inList || listType !== "ol") {
                        closeList();
                        out.push("<ol>");
                        inList = true;
                        listType = "ol";
                    }
                    out.push(`<li>${formatInline(olMatch[1])}</li>`);
                    continue;
                }

                // Regular paragraph text
                closeList();
                currentParagraph.push(trimmed);
            }

            flushParagraph();
            closeList();

            return out.join("");
        } catch (err) {
            console.error("Markdown rendering fallback error:", err);
            return `<p>${escapeHtml(markdown)}</p>`;
        }
    }

    function formatCurrency(val) {
        const num = Number(val);
        if (Math.abs(num) >= 1000000) {
            return "$" + (num / 1000000).toFixed(2) + "M";
        }
        if (Math.abs(num) >= 1000) {
            return "$" + (num / 1000).toFixed(1) + "K";
        }
        return "$" + num.toFixed(2);
    }

    function formatCompact(val) {
        const num = Number(val);
        if (Math.abs(num) >= 1000000) {
            return (num / 1000000).toFixed(2) + "M";
        }
        if (Math.abs(num) >= 1000) {
            return (num / 1000).toFixed(1) + "K";
        }
        return num.toLocaleString();
    }

    function escapeHtml(str) {
        if (!str) return "";
        return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }
});

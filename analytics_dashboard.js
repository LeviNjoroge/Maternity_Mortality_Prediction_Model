const DEFAULT_API_BASE = "http://127.0.0.1:5000/api";
const LOCAL_DATASET_PATH = "./Maternal_Mortality.csv";

const state = {
  apiBase: DEFAULT_API_BASE,
  dataMode: "fallback",
  backendConnected: false,
  source: null,
  health: null,
  metrics: {},
  countries: [],
  localDatasetRows: [],
  localCountrySeries: new Map(),
  seriesCache: new Map(),
  predictionCache: new Map(),
  featureImportanceCache: new Map(),
  countryFilterText: "",
  countrySortMode: "az",
  autoRefreshMs: 10000,
  autoRefreshTimer: null,
  autoRefreshRunning: false,
  refreshInFlight: false,
  yearMin: null,
  yearMax: null,
  charts: {
    modelBenchmark: null,
    countryTrend: null,
    feature: null,
    rfHorizon: null,
  },
};

const dom = {};

const ALGORITHM_KEYS = {
  linearRegression: "linear_regression",
  randomForest: "random_forest",
  xgboost: "xgboost",
  arima: "arima",
};

window.addEventListener("DOMContentLoaded", initializeDashboard);

async function initializeDashboard() {
  bindDom();
  bindEvents();

  await bootstrapDashboard();
}

function bindDom() {
  dom.reloadDataBtn = document.getElementById("reloadDataBtn");
  dom.predictBtn = document.getElementById("predictBtn");
  dom.liveRefreshBtn = document.getElementById("liveRefreshBtn");
  dom.startAutoRefreshBtn = document.getElementById("startAutoRefreshBtn");
  dom.stopAutoRefreshBtn = document.getElementById("stopAutoRefreshBtn");

  dom.countryFilterInput = document.getElementById("countryFilterInput");
  dom.countrySortSelect = document.getElementById("countrySortSelect");
  dom.countrySelect = document.getElementById("countrySelect");
  dom.forecastYearSelect = document.getElementById("forecastYearSelect");
  dom.importanceModelSelect = document.getElementById("importanceModelSelect");
  dom.refreshIntervalSelect = document.getElementById("refreshIntervalSelect");

  dom.statusText = document.getElementById("statusText");
  dom.dataModeBadge = document.getElementById("dataModeBadge");

  dom.kpiCountries = document.getElementById("kpiCountries");
  dom.kpiYears = document.getElementById("kpiYears");
  dom.kpiBestModel = document.getElementById("kpiBestModel");
  dom.kpiForecastRows = document.getElementById("kpiForecastRows");

  dom.ensembleValue = document.getElementById("ensembleValue");
  dom.riskTag = document.getElementById("riskTag");
  dom.confidenceText = document.getElementById("confidenceText");
  dom.algorithmBreakdown = document.getElementById("algorithmBreakdown");
  dom.lastUpdateText = document.getElementById("lastUpdateText");

  dom.cardLinearText = document.getElementById("cardLinearText");
  dom.cardRandomForestText = document.getElementById("cardRandomForestText");
  dom.cardXGBoostText = document.getElementById("cardXGBoostText");
  dom.cardArimaText = document.getElementById("cardArimaText");

  dom.modelBenchmarkChart = document.getElementById("modelBenchmarkChart");
  dom.countryTrendChart = document.getElementById("countryTrendChart");
  dom.featureChart = document.getElementById("featureChart");
  dom.rfHorizonChart = document.getElementById("rfHorizonChart");
}

function bindEvents() {
  dom.reloadDataBtn.addEventListener("click", async () => {
    clearTransientCaches();
    await bootstrapDashboard();
  });

  dom.liveRefreshBtn.addEventListener("click", async () => {
    await runLiveRefreshCycle({ forceReconnect: true, showStatus: true });
  });

  dom.startAutoRefreshBtn.addEventListener("click", () => {
    startAutoRefresh();
  });

  dom.stopAutoRefreshBtn.addEventListener("click", () => {
    stopAutoRefresh();
  });

  dom.countryFilterInput.addEventListener("input", () => {
    state.countryFilterText = (dom.countryFilterInput.value || "").trim().toLowerCase();
    populateCountrySelect();
    populateYearSelect();
    renderPredictionAndTrend();
  });

  dom.countrySortSelect.addEventListener("change", () => {
    state.countrySortMode = dom.countrySortSelect.value || "az";
    populateCountrySelect();
    populateYearSelect();
    renderPredictionAndTrend();
  });

  dom.refreshIntervalSelect.addEventListener("change", () => {
    const value = toNumber(dom.refreshIntervalSelect.value);
    if (Number.isFinite(value) && value >= 1000) {
      state.autoRefreshMs = value;
    }

    if (state.autoRefreshRunning) {
      startAutoRefresh();
    }
  });

  dom.predictBtn.addEventListener("click", async () => {
    await renderPredictionAndTrend();
  });

  dom.countrySelect.addEventListener("change", async () => {
    clearPredictionCacheForCountry(dom.countrySelect.value);
    await ensureCountrySeries(dom.countrySelect.value);
    populateYearSelect();
    await renderPredictionAndTrend();
  });

  dom.forecastYearSelect.addEventListener("change", async () => {
    await renderPredictionAndTrend();
  });

  dom.importanceModelSelect.addEventListener("change", async () => {
    await renderFeatureChart();
  });
}

function clearTransientCaches() {
  state.predictionCache = new Map();
  state.featureImportanceCache = new Map();
}

function clearPredictionCacheForCountry(country) {
  const nextMap = new Map();
  for (const [key, value] of state.predictionCache.entries()) {
    if (!key.startsWith(`${country}::`)) {
      nextMap.set(key, value);
    }
  }
  state.predictionCache = nextMap;
}

async function bootstrapDashboard() {
  setStatus("Connecting to backend API and loading analytics context...");

  await loadLocalDatasetContext();
  await connectBackend();
  await populateControls();
  await renderDashboard();
  setRealtimeButtonsState();
  if (!dom.lastUpdateText.textContent || dom.lastUpdateText.textContent === "Last update: -") {
    updateLastUpdateStamp(null);
  }
}

async function loadLocalDatasetContext() {
  const rows = await fetchCsvFile(LOCAL_DATASET_PATH);
  state.localDatasetRows = rows;

  state.localCountrySeries = new Map();

  if (!rows.length) {
    state.yearMin = null;
    state.yearMax = null;
    return;
  }

  const firstRow = rows[0];
  const yearPairs = Object.keys(firstRow)
    .map((column) => {
      const match = String(column).match(/\((\d{4})\)\s*$/);
      if (!match) {
        return null;
      }
      return { column, year: Number(match[1]) };
    })
    .filter(Boolean)
    .sort((a, b) => a.year - b.year);

  if (!yearPairs.length) {
    state.yearMin = null;
    state.yearMax = null;
    return;
  }

  state.yearMin = yearPairs[0].year;
  state.yearMax = yearPairs[yearPairs.length - 1].year;

  for (const row of rows) {
    const country = String(row.Country || "").trim();
    if (!country) {
      continue;
    }

    const series = [];
    for (const item of yearPairs) {
      const value = toNumber(row[item.column]);
      if (Number.isFinite(value)) {
        series.push({ year: item.year, value });
      }
    }

    if (series.length) {
      state.localCountrySeries.set(country, series);
    }
  }
}

async function connectBackend() {
  const health = await apiGet("/health", false);
  if (!health || health.status !== "ok") {
    state.backendConnected = false;
    state.dataMode = "fallback";
    state.health = null;
    state.source = null;
    state.metrics = {};

    state.countries = Array.from(state.localCountrySeries.keys()).sort((a, b) => a.localeCompare(b));

    updateDataModeBadge();
    setStatus(
      "Backend unavailable. Running in fallback simulation mode. Start run_backend_service.bat for live API predictions."
    );
    return;
  }

  state.backendConnected = true;
  state.dataMode = "live-api";
  state.health = health;

  const [source, metricsPayload, countriesPayload] = await Promise.all([
    apiGet("/source", false),
    apiGet("/metrics", false),
    apiGet("/countries", false),
  ]);

  state.source = source || {};
  state.metrics = metricsPayload?.metrics || {};
  state.countries = Array.isArray(countriesPayload?.countries)
    ? countriesPayload.countries.slice().sort((a, b) => a.localeCompare(b))
    : [];

  updateDataModeBadge();

  const countryCount = state.countries.length;
  const xgbFlag = health.xgboost_available ? "XGBoost enabled" : "XGBoost unavailable";
  setStatus(
    `Live API connected at ${state.apiBase}. Countries loaded: ${countryCount}. ${xgbFlag}.`
  );
}

function updateDataModeBadge() {
  dom.dataModeBadge.classList.remove("pipeline", "fallback", "live-api");

  if (state.dataMode === "live-api") {
    dom.dataModeBadge.classList.add("live-api");
    dom.dataModeBadge.textContent = "Live API Mode";
    return;
  }

  dom.dataModeBadge.classList.add("fallback");
  dom.dataModeBadge.textContent = "Simulation Fallback";
}

async function populateControls() {
  if (dom.countrySortSelect) {
    dom.countrySortSelect.value = state.countrySortMode;
  }
  if (dom.refreshIntervalSelect) {
    dom.refreshIntervalSelect.value = String(state.autoRefreshMs);
  }

  populateCountrySelect();

  if (dom.countrySelect.value) {
    await ensureCountrySeries(dom.countrySelect.value);
  }

  populateYearSelect();
}

function populateCountrySelect() {
  const previousCountry = dom.countrySelect.value;
  dom.countrySelect.innerHTML = "";

  const candidates = buildCountryDisplayList();

  if (!candidates.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No countries match filter";
    dom.countrySelect.appendChild(option);
    return;
  }

  for (const country of candidates) {
    const option = document.createElement("option");
    option.value = country;
    option.textContent = country;
    dom.countrySelect.appendChild(option);
  }

  const configuredDefault = state.source?.default_country;
  const preferredCandidates = [
    previousCountry,
    configuredDefault,
    "Kenya",
    candidates[0],
  ].filter(Boolean);

  const preferred = preferredCandidates.find((country) => candidates.includes(country)) || candidates[0];

  dom.countrySelect.value = preferred;
}

function buildCountryDisplayList() {
  const base = state.countries.length
    ? [...state.countries]
    : Array.from(state.localCountrySeries.keys());

  const filtered = state.countryFilterText
    ? base.filter((country) => country.toLowerCase().includes(state.countryFilterText))
    : base;

  const sorted = [...filtered].sort((a, b) => {
    if (state.countrySortMode === "za") {
      return b.localeCompare(a);
    }

    if (state.countrySortMode === "risk_high") {
      const aValue = latestMmrValueForCountry(a);
      const bValue = latestMmrValueForCountry(b);
      const aValid = Number.isFinite(aValue);
      const bValid = Number.isFinite(bValue);

      if (!aValid && !bValid) {
        return a.localeCompare(b);
      }
      if (!aValid) {
        return 1;
      }
      if (!bValid) {
        return -1;
      }
      return bValue - aValue;
    }

    if (state.countrySortMode === "risk_low") {
      const aValue = latestMmrValueForCountry(a);
      const bValue = latestMmrValueForCountry(b);
      const aValid = Number.isFinite(aValue);
      const bValid = Number.isFinite(bValue);

      if (!aValid && !bValid) {
        return a.localeCompare(b);
      }
      if (!aValid) {
        return 1;
      }
      if (!bValid) {
        return -1;
      }
      return aValue - bValue;
    }

    return a.localeCompare(b);
  });

  return sorted;
}

function latestMmrValueForCountry(country) {
  const series = state.seriesCache.get(country) || state.localCountrySeries.get(country) || [];
  if (!series.length) {
    return Number.NaN;
  }
  return toNumber(series[series.length - 1]?.value);
}

function populateYearSelect() {
  dom.forecastYearSelect.innerHTML = "";

  const selectedCountry = dom.countrySelect.value;
  const series = state.seriesCache.get(selectedCountry) || state.localCountrySeries.get(selectedCountry) || [];

  const latestYear = series.length
    ? series[series.length - 1].year
    : Number.isFinite(state.yearMax)
      ? state.yearMax
      : new Date().getFullYear() - 1;

  const forecastSteps = toNumber(state.source?.forecast_steps);
  const horizon = Number.isFinite(forecastSteps) ? Math.max(5, forecastSteps + 4) : 10;

  const years = [];
  for (let year = latestYear + 1; year <= latestYear + horizon; year += 1) {
    years.push(year);
  }

  for (const year of years) {
    const option = document.createElement("option");
    option.value = String(year);
    option.textContent = String(year);
    dom.forecastYearSelect.appendChild(option);
  }

  if (years.length) {
    dom.forecastYearSelect.value = String(latestYear + 1);
  }
}

function setRealtimeButtonsState() {
  if (!dom.startAutoRefreshBtn) {
    return;
  }

  dom.startAutoRefreshBtn.disabled = state.autoRefreshRunning || state.refreshInFlight;
  dom.stopAutoRefreshBtn.disabled = !state.autoRefreshRunning;
  dom.liveRefreshBtn.disabled = state.refreshInFlight;
}

function updateLastUpdateStamp(timestamp) {
  if (!dom.lastUpdateText) {
    return;
  }

  if (!timestamp) {
    dom.lastUpdateText.textContent = "Last update: waiting";
    return;
  }

  const formatted = timestamp.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  dom.lastUpdateText.textContent = `Last update: ${formatted}`;
}

function stopAutoRefresh() {
  if (state.autoRefreshTimer) {
    window.clearInterval(state.autoRefreshTimer);
    state.autoRefreshTimer = null;
  }

  if (state.autoRefreshRunning) {
    state.autoRefreshRunning = false;
    setStatus("Auto refresh stopped.");
  }

  setRealtimeButtonsState();
}

function startAutoRefresh() {
  stopAutoRefresh();
  state.autoRefreshRunning = true;

  void runLiveRefreshCycle({ forceReconnect: true, showStatus: true });
  state.autoRefreshTimer = window.setInterval(() => {
    void runLiveRefreshCycle({ forceReconnect: true, showStatus: false });
  }, state.autoRefreshMs);

  setStatus(`Auto refresh started (${Math.round(state.autoRefreshMs / 1000)}s interval).`);
  setRealtimeButtonsState();
}

async function runLiveRefreshCycle({ forceReconnect = false, showStatus = false } = {}) {
  if (state.refreshInFlight) {
    return;
  }

  state.refreshInFlight = true;
  setRealtimeButtonsState();

  try {
    if (forceReconnect || state.backendConnected) {
      await connectBackend();
    }

    clearTransientCaches();

    const selectedCountry = dom.countrySelect.value;
    const selectedYear = dom.forecastYearSelect.value;

    await populateControls();

    if (selectedCountry && Array.from(dom.countrySelect.options).some((opt) => opt.value === selectedCountry)) {
      dom.countrySelect.value = selectedCountry;
      await ensureCountrySeries(selectedCountry);
      populateYearSelect();
      if (selectedYear && Array.from(dom.forecastYearSelect.options).some((opt) => opt.value === selectedYear)) {
        dom.forecastYearSelect.value = selectedYear;
      }
    }

    await renderDashboard();
    updateLastUpdateStamp(new Date());

    if (showStatus) {
      const mode = state.backendConnected ? "live backend API" : "fallback simulation";
      setStatus(`Analytics refreshed from ${mode}.`);
    }
  } finally {
    state.refreshInFlight = false;
    setRealtimeButtonsState();
  }
}

async function ensureCountrySeries(country) {
  if (!country) {
    return [];
  }

  if (state.seriesCache.has(country)) {
    return state.seriesCache.get(country) || [];
  }

  if (state.backendConnected) {
    const payload = await apiGet(`/country-series?country=${encodeURIComponent(country)}`, false);
    const series = Array.isArray(payload?.series)
      ? payload.series
          .map((p) => ({ year: toNumber(p.year), value: toNumber(p.mmr) }))
          .filter((p) => Number.isFinite(p.year) && Number.isFinite(p.value))
          .sort((a, b) => a.year - b.year)
      : [];

    if (series.length) {
      state.seriesCache.set(country, series);
      refreshGlobalYearRange(series);
      return series;
    }
  }

  const local = state.localCountrySeries.get(country) || [];
  state.seriesCache.set(country, local);
  if (local.length) {
    refreshGlobalYearRange(local);
  }

  return local;
}

function refreshGlobalYearRange(series) {
  if (!series.length) {
    return;
  }

  const minYear = series[0].year;
  const maxYear = series[series.length - 1].year;

  if (!Number.isFinite(state.yearMin) || minYear < state.yearMin) {
    state.yearMin = minYear;
  }
  if (!Number.isFinite(state.yearMax) || maxYear > state.yearMax) {
    state.yearMax = maxYear;
  }
}

async function renderDashboard() {
  renderKpis();
  renderAlgorithmCards();
  renderModelBenchmarkChart();
  renderErrorProfileChart();
  await renderFeatureChart();
  await renderPredictionAndTrend();
}

function renderKpis() {
  const countryCount = state.countries.length || state.localCountrySeries.size;
  dom.kpiCountries.textContent = formatInt(countryCount);

  if (Number.isFinite(state.yearMin) && Number.isFinite(state.yearMax)) {
    dom.kpiYears.textContent = `${state.yearMin}-${state.yearMax}`;
  } else {
    dom.kpiYears.textContent = "-";
  }

  const best = computeBestModel();
  dom.kpiBestModel.textContent = best
    ? `${best.label} (${formatNumber(best.rmse, 2)})`
    : "Insufficient metrics";

  const forecastSteps = toNumber(state.source?.forecast_steps);
  const projectedRows = Number.isFinite(forecastSteps)
    ? countryCount * Math.max(1, Math.floor(forecastSteps))
    : countryCount * 5;
  dom.kpiForecastRows.textContent = formatInt(projectedRows);
}

function renderAlgorithmCards() {
  const lr = getModelMetrics("linearRegression");
  const rf = getModelMetrics("randomForest");
  const xgb = getModelMetrics("xgboost");
  const arima = getModelMetrics("arima");

  dom.cardLinearText.textContent = lr
    ? `RMSE ${formatNumber(lr.rmse, 2)} | R2 ${formatNumber(lr.r2, 3)}`
    : "Metrics unavailable.";

  dom.cardRandomForestText.textContent = rf
    ? `RMSE ${formatNumber(rf.rmse, 2)} | MAE ${formatNumber(rf.mae, 2)}`
    : "Metrics unavailable.";

  const xgbEnabled = Boolean(state.health?.xgboost_available);
  dom.cardXGBoostText.textContent = xgb
    ? `RMSE ${formatNumber(xgb.rmse, 2)} | R2 ${formatNumber(xgb.r2, 3)}`
    : xgbEnabled
      ? "Metrics unavailable."
      : "XGBoost not available in backend environment.";

  dom.cardArimaText.textContent = arima
    ? `RMSE ${formatNumber(arima.rmse, 2)} | ADF p ${formatNumber(arima.adf_pvalue, 4)}`
    : "Metrics unavailable.";
}

async function renderPredictionAndTrend() {
  const country = dom.countrySelect.value;
  const targetYear = toNumber(dom.forecastYearSelect.value);

  if (!country || !Number.isFinite(targetYear)) {
    resetPredictionPanel();
    await renderCountryTrendChart(country, targetYear, null);
    return;
  }

  await ensureCountrySeries(country);

  const prediction = await getPrediction(country, targetYear);
  if (!prediction) {
    resetPredictionPanel();
    await renderCountryTrendChart(country, targetYear, null);
    return;
  }

  const ensembleValue = toNumber(prediction.ensemble?.value);
  const lowerBound = toNumber(prediction.ensemble?.lower_bound);
  const upperBound = toNumber(prediction.ensemble?.upper_bound);
  const riskBand = String(prediction.risk_band || "Low Risk");

  dom.ensembleValue.textContent = Number.isFinite(ensembleValue)
    ? `${formatNumber(ensembleValue, 2)} MMR`
    : "N/A";

  dom.confidenceText.textContent =
    Number.isFinite(lowerBound) && Number.isFinite(upperBound)
      ? `Confidence interval: ${formatNumber(lowerBound, 2)} to ${formatNumber(upperBound, 2)}`
      : "Confidence interval: unavailable";

  dom.riskTag.classList.remove("mid", "high");
  if (riskBand === "Mid Risk") {
    dom.riskTag.classList.add("mid");
  } else if (riskBand === "High Risk") {
    dom.riskTag.classList.add("high");
  }
  dom.riskTag.textContent = `Risk: ${riskBand}`;

  renderAlgorithmBreakdown(prediction);
  await renderCountryTrendChart(country, targetYear, prediction);
}

function resetPredictionPanel() {
  dom.ensembleValue.textContent = "-";
  dom.riskTag.classList.remove("mid", "high");
  dom.riskTag.textContent = "Risk: -";
  dom.confidenceText.textContent = "Confidence: unavailable";
  dom.algorithmBreakdown.innerHTML = "";
}

function renderAlgorithmBreakdown(prediction) {
  const entries = [
    { key: ALGORITHM_KEYS.linearRegression, label: "Linear Regression" },
    { key: ALGORITHM_KEYS.randomForest, label: "Random Forest" },
    { key: ALGORITHM_KEYS.xgboost, label: "XGBoost" },
    { key: ALGORITHM_KEYS.arima, label: "ARIMA" },
  ];

  dom.algorithmBreakdown.innerHTML = "";

  for (const item of entries) {
    const value = toNumber(prediction.prediction?.[item.key]);

    const row = document.createElement("div");
    row.className = "breakdown-item";

    const label = document.createElement("strong");
    label.textContent = item.label;

    const metric = document.createElement("span");
    metric.textContent = Number.isFinite(value) ? formatNumber(value, 2) : "N/A";

    row.appendChild(label);
    row.appendChild(metric);
    dom.algorithmBreakdown.appendChild(row);
  }
}

async function getPrediction(country, year) {
  const cacheKey = `${country}::${year}`;
  if (state.predictionCache.has(cacheKey)) {
    return state.predictionCache.get(cacheKey);
  }

  if (state.backendConnected) {
    const response = await apiGet(
      `/predict?country=${encodeURIComponent(country)}&year=${encodeURIComponent(String(year))}`,
      false
    );

    if (response && !response.error) {
      state.predictionCache.set(cacheKey, response);
      return response;
    }
  }

  const simulated = simulatePrediction(country, year);
  if (simulated) {
    state.predictionCache.set(cacheKey, simulated);
  }
  return simulated;
}

function simulatePrediction(country, targetYear) {
  const series = state.seriesCache.get(country) || state.localCountrySeries.get(country) || [];
  if (!series.length) {
    return null;
  }

  const latest = series[series.length - 1];
  const latestYear = latest.year;
  const latestValue = latest.value;

  const slope = estimateSlope(series.slice(-10));
  const step = Math.max(1, targetYear - latestYear);

  const predictions = {
    linear_regression: Math.max(0, latestValue + slope * step * 0.95),
    random_forest: Math.max(0, latestValue + slope * step * 0.8),
    xgboost: Math.max(0, latestValue + slope * step * 0.88),
    arima: Math.max(0, latestValue + slope * step * 0.84),
  };

  const values = Object.values(predictions).filter(Number.isFinite);
  if (!values.length) {
    return null;
  }

  const ensemble = average(values);
  const lower = Math.max(0, ensemble - Math.max(3, Math.abs(slope) * step));
  const upper = ensemble + Math.max(3, Math.abs(slope) * step);

  const allLastValues = Array.from(state.localCountrySeries.values())
    .map((items) => items[items.length - 1]?.value)
    .filter((v) => Number.isFinite(v));

  const q1 = quantile(allLastValues, 0.33);
  const q2 = quantile(allLastValues, 0.66);

  let risk = "Low Risk";
  if (Number.isFinite(q2) && ensemble > q2) {
    risk = "High Risk";
  } else if (Number.isFinite(q1) && ensemble > q1) {
    risk = "Mid Risk";
  }

  return {
    country,
    target_year: targetYear,
    prediction: predictions,
    ensemble: {
      value: ensemble,
      lower_bound: lower,
      upper_bound: upper,
    },
    risk_band: risk,
  };
}

async function renderModelBenchmarkChart() {
  const metricRows = [
    { label: "Linear Regression", metric: getModelMetrics("linearRegression") },
    { label: "Random Forest", metric: getModelMetrics("randomForest") },
    { label: "XGBoost", metric: getModelMetrics("xgboost") },
    { label: "ARIMA", metric: getModelMetrics("arima") },
  ];

  const labels = metricRows.map((m) => m.label);
  const rmseValues = metricRows.map((m) => finiteOrNull(m.metric?.rmse));
  const r2Values = metricRows.map((m) => finiteOrNull(m.metric?.r2));

  destroyChart("modelBenchmark");

  state.charts.modelBenchmark = new Chart(dom.modelBenchmarkChart, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          type: "bar",
          label: "RMSE",
          data: rmseValues,
          backgroundColor: ["#4a7ea8", "#0fb9a8", "#f76844", "#ffb627"],
          borderRadius: 8,
          yAxisID: "y",
        },
        {
          type: "line",
          label: "R2",
          data: r2Values,
          borderColor: "#15283b",
          borderWidth: 2.2,
          pointBackgroundColor: "#15283b",
          pointRadius: 4,
          tension: 0.25,
          yAxisID: "y1",
        },
      ],
    },
    options: {
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          labels: {
            color: "#20364d",
            font: { family: "Sora", weight: "600" },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#4f5f73", font: { family: "Sora" } },
          grid: { color: "rgba(31,56,81,0.08)" },
        },
        y: {
          position: "left",
          title: { display: true, text: "RMSE", color: "#4f5f73" },
          ticks: { color: "#4f5f73" },
          grid: { color: "rgba(31,56,81,0.08)" },
        },
        y1: {
          position: "right",
          min: -1,
          max: 1,
          title: { display: true, text: "R2", color: "#4f5f73" },
          ticks: { color: "#4f5f73" },
          grid: { drawOnChartArea: false },
        },
      },
    },
  });
}

async function renderCountryTrendChart(country, selectedYear, selectedPrediction) {
  destroyChart("countryTrend");

  const series = await ensureCountrySeries(country);
  if (!series.length) {
    state.charts.countryTrend = new Chart(dom.countryTrendChart, {
      type: "line",
      data: { labels: [], datasets: [] },
    });
    return;
  }

  const latestYear = series[series.length - 1].year;
  const defaultHorizon = toNumber(state.source?.forecast_steps);
  const horizon = Number.isFinite(defaultHorizon) ? Math.max(2, defaultHorizon) : 5;

  const endYear = Math.max(
    Number.isFinite(selectedYear) ? selectedYear : latestYear + 1,
    latestYear + horizon
  );

  const futureYears = [];
  for (let y = latestYear + 1; y <= endYear; y += 1) {
    futureYears.push(y);
  }

  const predictionResults = await Promise.all(
    futureYears.map(async (year) => {
      const p = await getPrediction(country, year);
      return { year, payload: p };
    })
  );

  const predictionMap = new Map(predictionResults.map((row) => [row.year, row.payload]));

  const labels = [];
  for (let y = series[0].year; y <= endYear; y += 1) {
    labels.push(y);
  }

  const historicalMap = new Map(series.map((p) => [p.year, p.value]));

  const historicalData = labels.map((year) => (historicalMap.has(year) ? historicalMap.get(year) : null));
  const lrData = labels.map((year) => toPredictionValue(predictionMap.get(year), ALGORITHM_KEYS.linearRegression));
  const rfData = labels.map((year) => toPredictionValue(predictionMap.get(year), ALGORITHM_KEYS.randomForest));
  const xgbData = labels.map((year) => toPredictionValue(predictionMap.get(year), ALGORITHM_KEYS.xgboost));
  const arimaData = labels.map((year) => toPredictionValue(predictionMap.get(year), ALGORITHM_KEYS.arima));
  const ensembleData = labels.map((year) => {
    const payload = predictionMap.get(year);
    return payload ? finiteOrNull(payload.ensemble?.value) : null;
  });

  state.charts.countryTrend = new Chart(dom.countryTrendChart, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Historical",
          data: historicalData,
          borderColor: "#15283b",
          backgroundColor: "rgba(21,40,59,0.1)",
          pointRadius: 2,
          borderWidth: 2.3,
          tension: 0.18,
        },
        {
          label: "Linear Regression",
          data: lrData,
          borderColor: "#4a7ea8",
          borderDash: [7, 4],
          pointRadius: 2,
          tension: 0.2,
        },
        {
          label: "Random Forest",
          data: rfData,
          borderColor: "#0fb9a8",
          borderDash: [8, 4],
          pointRadius: 2,
          tension: 0.2,
        },
        {
          label: "XGBoost",
          data: xgbData,
          borderColor: "#f76844",
          borderDash: [8, 4],
          pointRadius: 2,
          tension: 0.2,
        },
        {
          label: "ARIMA",
          data: arimaData,
          borderColor: "#ffb627",
          borderDash: [8, 4],
          pointRadius: 2,
          tension: 0.2,
        },
        {
          label: "Ensemble",
          data: ensembleData,
          borderColor: "#0b1d2f",
          pointBackgroundColor: "#0b1d2f",
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.15,
        },
      ],
    },
    options: {
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: "#20364d", boxWidth: 14 } },
        tooltip: {
          callbacks: {
            title: (items) => `Year ${items[0]?.label || ""}`,
            label: (item) => `${item.dataset.label}: ${Number.isFinite(item.parsed.y) ? formatNumber(item.parsed.y, 2) : "N/A"}`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#4f5f73" },
          grid: { color: "rgba(31,56,81,0.08)" },
        },
        y: {
          title: { display: true, text: "Maternal Mortality Ratio", color: "#4f5f73" },
          ticks: { color: "#4f5f73" },
          grid: { color: "rgba(31,56,81,0.08)" },
        },
      },
    },
  });

  if (selectedPrediction) {
    const modeText = state.backendConnected ? "live backend API" : "fallback simulation";
    setStatus(
      `Prediction generated for ${country} (${selectedYear}) using ${modeText}.`
    );
  }
}

function toPredictionValue(payload, key) {
  if (!payload) {
    return null;
  }
  const value = toNumber(payload.prediction?.[key]);
  return Number.isFinite(value) ? value : null;
}

async function renderFeatureChart() {
  destroyChart("feature");

  const selectedModel = dom.importanceModelSelect.value;
  const featureRows = await getFeatureImportanceRows(selectedModel);

  const labels = featureRows.slice(0, 14).map((item) => item.feature);
  const values = featureRows.slice(0, 14).map((item) => item.value);

  state.charts.feature = new Chart(dom.featureChart, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Importance / Absolute Coefficient",
          data: values,
          backgroundColor: "rgba(15, 185, 168, 0.78)",
          borderColor: "#0a9689",
          borderWidth: 1.2,
          borderRadius: 8,
        },
      ],
    },
    options: {
      indexAxis: "y",
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#20364d" },
        },
      },
      scales: {
        x: {
          ticks: { color: "#4f5f73" },
          grid: { color: "rgba(31,56,81,0.08)" },
        },
        y: {
          ticks: { color: "#4f5f73" },
          grid: { color: "rgba(31,56,81,0.06)" },
        },
      },
    },
  });
}

async function getFeatureImportanceRows(modelKey) {
  if (state.featureImportanceCache.has(modelKey)) {
    return state.featureImportanceCache.get(modelKey) || [];
  }

  if (state.backendConnected) {
    const response = await apiGet(
      `/feature-importance?model=${encodeURIComponent(modelKey)}&top_n=20`,
      false
    );

    const rows = Array.isArray(response?.feature_importance)
      ? response.feature_importance
          .map((item) => ({
            feature: shortenFeatureName(item.feature),
            value: Math.abs(toNumber(item.value)),
          }))
          .filter((item) => Number.isFinite(item.value))
          .sort((a, b) => b.value - a.value)
      : [];

    if (rows.length) {
      state.featureImportanceCache.set(modelKey, rows);
      return rows;
    }
  }

  const fallback = buildFallbackFeatureRows();
  state.featureImportanceCache.set(modelKey, fallback);
  return fallback;
}

function buildFallbackFeatureRows() {
  if (!state.localDatasetRows.length) {
    return [];
  }

  const firstRow = state.localDatasetRows[0];
  const yearColumns = Object.keys(firstRow)
    .filter((column) => /\((\d{4})\)\s*$/.test(String(column)))
    .slice(-12);

  const rows = yearColumns
    .map((column) => {
      const values = state.localDatasetRows
        .map((row) => toNumber(row[column]))
        .filter((value) => Number.isFinite(value));

      return {
        feature: shortenFeatureName(column),
        value: variance(values),
      };
    })
    .sort((a, b) => b.value - a.value);

  return rows;
}

function renderErrorProfileChart() {
  destroyChart("rfHorizon");

  const metricRows = [
    { label: "Linear Regression", metric: getModelMetrics("linearRegression") },
    { label: "Random Forest", metric: getModelMetrics("randomForest") },
    { label: "XGBoost", metric: getModelMetrics("xgboost") },
    { label: "ARIMA", metric: getModelMetrics("arima") },
  ];

  const labels = metricRows.map((m) => m.label);
  const maeValues = metricRows.map((m) => finiteOrNull(m.metric?.mae));
  const rmseValues = metricRows.map((m) => finiteOrNull(m.metric?.rmse));
  const r2Percent = metricRows.map((m) => {
    const r2 = finiteOrNull(m.metric?.r2);
    return Number.isFinite(r2) ? r2 * 100 : null;
  });

  state.charts.rfHorizon = new Chart(dom.rfHorizonChart, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "MAE",
          data: maeValues,
          backgroundColor: "rgba(255, 182, 39, 0.75)",
          borderRadius: 7,
          yAxisID: "y",
        },
        {
          label: "RMSE",
          data: rmseValues,
          backgroundColor: "rgba(247, 104, 68, 0.75)",
          borderRadius: 7,
          yAxisID: "y",
        },
        {
          type: "line",
          label: "R2 %",
          data: r2Percent,
          borderColor: "#0fb9a8",
          backgroundColor: "#0fb9a8",
          pointRadius: 3,
          borderWidth: 2,
          tension: 0.25,
          yAxisID: "y1",
        },
      ],
    },
    options: {
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          labels: { color: "#20364d" },
        },
      },
      scales: {
        y: {
          position: "left",
          title: { display: true, text: "Error", color: "#4f5f73" },
          ticks: { color: "#4f5f73" },
          grid: { color: "rgba(31,56,81,0.08)" },
        },
        y1: {
          position: "right",
          min: -100,
          max: 100,
          title: { display: true, text: "R2 %", color: "#4f5f73" },
          ticks: { color: "#4f5f73" },
          grid: { drawOnChartArea: false },
        },
        x: {
          ticks: { color: "#4f5f73" },
          grid: { color: "rgba(31,56,81,0.08)" },
        },
      },
    },
  });
}

function getModelMetrics(modelKey) {
  if (modelKey === "linearRegression") {
    return sanitizeMetricObject(
      state.metrics.linear_regression_multi || state.metrics.linear_regression_simple
    );
  }

  if (modelKey === "randomForest") {
    return sanitizeMetricObject(state.metrics.random_forest);
  }

  if (modelKey === "xgboost") {
    return sanitizeMetricObject(state.metrics.xgboost);
  }

  if (modelKey === "arima") {
    return sanitizeMetricObject(state.metrics.arima);
  }

  return null;
}

function sanitizeMetricObject(metricObj) {
  if (!metricObj || typeof metricObj !== "object") {
    return null;
  }

  const mae = toNumber(metricObj.mae);
  const rmse = toNumber(metricObj.rmse);
  const mse = toNumber(metricObj.mse);
  const r2 = toNumber(metricObj.r2);
  const adf = toNumber(metricObj.adf_pvalue);

  if (![mae, rmse, mse, r2, adf].some((v) => Number.isFinite(v))) {
    return null;
  }

  return {
    mae,
    rmse,
    mse,
    r2,
    adf_pvalue: adf,
  };
}

function computeBestModel() {
  const candidates = [
    { label: "Linear Regression", metric: getModelMetrics("linearRegression") },
    { label: "Random Forest", metric: getModelMetrics("randomForest") },
    { label: "XGBoost", metric: getModelMetrics("xgboost") },
    { label: "ARIMA", metric: getModelMetrics("arima") },
  ]
    .map((row) => ({
      label: row.label,
      rmse: finiteOrNull(row.metric?.rmse),
    }))
    .filter((row) => Number.isFinite(row.rmse));

  if (!candidates.length) {
    return null;
  }

  candidates.sort((a, b) => a.rmse - b.rmse);
  return candidates[0];
}

async function apiGet(endpoint, throwOnError = false) {
  const url = endpoint.startsWith("http") ? endpoint : `${state.apiBase}${endpoint}`;

  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      if (throwOnError) {
        throw new Error(`Request failed (${response.status}) for ${url}`);
      }
      return null;
    }
    return await response.json();
  } catch (error) {
    if (throwOnError) {
      throw error;
    }
    return null;
  }
}

async function fetchCsvFile(path) {
  try {
    const response = await fetch(path, { cache: "no-store" });
    if (!response.ok) {
      return [];
    }

    const text = await response.text();
    const parsed = Papa.parse(text, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: false,
    });

    return Array.isArray(parsed.data) ? parsed.data : [];
  } catch (_error) {
    return [];
  }
}

function setStatus(message) {
  dom.statusText.textContent = message;
}

function destroyChart(key) {
  if (state.charts[key]) {
    state.charts[key].destroy();
    state.charts[key] = null;
  }
}

function finiteOrNull(value) {
  const numeric = toNumber(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function toNumber(value) {
  if (value === null || value === undefined || value === "") {
    return Number.NaN;
  }

  if (typeof value === "number") {
    return Number.isFinite(value) ? value : Number.NaN;
  }

  const parsed = Number(String(value).replace(/,/g, ""));
  return Number.isFinite(parsed) ? parsed : Number.NaN;
}

function average(values) {
  if (!values.length) {
    return 0;
  }
  return values.reduce((acc, value) => acc + value, 0) / values.length;
}

function variance(values) {
  if (!values.length) {
    return 0;
  }
  const mean = average(values);
  return average(values.map((value) => (value - mean) ** 2));
}

function quantile(values, q) {
  if (!values.length) {
    return Number.NaN;
  }

  const sorted = [...values].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const low = Math.floor(pos);
  const high = Math.ceil(pos);

  if (low === high) {
    return sorted[low];
  }

  return sorted[low] + (sorted[high] - sorted[low]) * (pos - low);
}

function estimateSlope(points) {
  if (!Array.isArray(points) || points.length < 2) {
    return 0;
  }

  const xs = points.map((point) => point.year);
  const ys = points.map((point) => point.value);

  const xMean = average(xs);
  const yMean = average(ys);

  let numerator = 0;
  let denominator = 0;

  for (let i = 0; i < xs.length; i += 1) {
    numerator += (xs[i] - xMean) * (ys[i] - yMean);
    denominator += (xs[i] - xMean) ** 2;
  }

  if (!denominator) {
    return 0;
  }

  return numerator / denominator;
}

function formatNumber(value, decimals = 2) {
  if (!Number.isFinite(value)) {
    return "N/A";
  }

  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
}

function formatInt(value) {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
  }).format(value || 0);
}

function shortenFeatureName(name) {
  return String(name)
    .replace(/^num__/, "")
    .replace(/^cat__/, "")
    .replace(/^model__/, "")
    .replace("Maternal Mortality Ratio (deaths per 100,000 live births)", "MMR")
    .replace("UNDP Developing Regions", "UNDP Region")
    .replace("Human Development Groups", "HD Group");
}

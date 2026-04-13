import React, { useState } from "react";

const MODEL_COLORS = {
  "Random Forest": "#7F77DD",
  XGBoost: "#1D9E75",
  SVM: "#D85A30",
  "Logistic Regression": "#378ADD",
  "Linear Regression": "#7F77DD",
  ARIMA: "#378ADD"
};

const RISK_COLORS = {
  "Low Risk": { border: "#639922", bg: "#E9F2DB", text: "#3F5F12" },
  "Mid Risk": { border: "#EF9F27", bg: "#FDF0D6", text: "#7A4E00" },
  "High Risk": { border: "#D85A30", bg: "#FCE4DB", text: "#7C2A12" }
};

const TREND_COLORS = {
  improving: { border: "#1D9E75", bg: "#E1F5EE", text: "#085041" },
  stable: { border: "#EF9F27", bg: "#FDF0D6", text: "#7A4E00" },
  worsening: { border: "#D85A30", bg: "#FCE4DB", text: "#7C2A12" }
};

const initialPatient = {
  age: 28,
  sbp: 118,
  dbp: 76,
  bs: 6.5,
  temp: 98.6,
  hr: 80
};

const initialCountry = {
  country: "Kenya",
  continent: "Africa",
  hdiGroup: "Medium",
  hdiRank: 143,
  baseline: 342
};

const baseCard = {
  background: "var(--color-background-secondary)",
  border: "1px solid var(--color-border-secondary)",
  borderRadius: "var(--border-radius-lg)",
  padding: "16px"
};

const sectionTitle = {
  fontSize: "14px",
  textTransform: "uppercase",
  letterSpacing: "0.08em",
  color: "var(--color-text-tertiary)",
  marginBottom: "12px"
};

const labelStyle = {
  fontSize: "14px",
  fontWeight: 600,
  color: "var(--color-text-primary)"
};

const hintStyle = {
  fontSize: "12px",
  color: "var(--color-text-tertiary)",
  marginTop: "4px"
};

const dividerStyle = {
  height: "1px",
  background: "var(--color-border-tertiary)",
  margin: "16px 0"
};

const pillBase = {
  padding: "4px 10px",
  borderRadius: "999px",
  fontSize: "12px",
  fontWeight: 700
};

const emptyStateStyle = {
  border: "1px dashed var(--color-border-secondary)",
  borderRadius: "var(--border-radius-lg)",
  padding: "24px",
  textAlign: "center",
  color: "var(--color-text-tertiary)",
  fontSize: "14px"
};

function cleanJsonText(text) {
  const stripped = text
    .replace(/```json/gi, "")
    .replace(/```/g, "")
    .trim();
  const start = stripped.indexOf("{");
  const end = stripped.lastIndexOf("}");
  if (start !== -1 && end !== -1) {
    return stripped.slice(start, end + 1);
  }
  return stripped;
}

async function callAnthropic(prompt) {
  // Ensure the user exports their API key in their environment or sets it manually
  // NOTE: Never expose an API key safely on the client dynamically typically, but for standalone run context:
  const API_KEY = import.meta.env.VITE_ANTHROPIC_API_KEY || "";
  
  if(!API_KEY) {
      throw new Error("Missing VITE_ANTHROPIC_API_KEY environment variable. Create a .env file and set VITE_ANTHROPIC_API_KEY.");
  }

  const payload = {
    model: "claude-3-5-sonnet-20240620", 
    max_tokens: 1000,
    messages: [{ role: "user", content: prompt }]
  };

  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": API_KEY,
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true"
    },
    body: JSON.stringify(payload)
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`API error ${res.status}: ${errorText}`);
  }

  const data = await res.json();
  const text =
    Array.isArray(data.content) && data.content.length
      ? data.content.map((c) => c.text || "").join("")
      : "";

  const cleaned = cleanJsonText(text);
  return JSON.parse(cleaned);
}

function SliderField({
  label,
  value,
  min,
  max,
  step,
  hint,
  unit,
  accent,
  accentFill,
  onChange,
  decimals
}) {
  const display =
    typeof value === "number"
      ? decimals != null
        ? value.toFixed(decimals)
        : Math.round(value).toString()
      : value;

  return (
    <div style={{ marginBottom: "16px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={labelStyle}>{label}</div>
        <div
          style={{
            ...pillBase,
            background: accentFill,
            color: accent,
            border: `1px solid ${accent}`
          }}
        >
          {display}
          {unit ? ` ${unit}` : ""}
        </div>
      </div>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: "100%", marginTop: "8px" }}
      />
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "var(--color-text-tertiary)" }}>
        <span>{min}</span>
        <span>{max}</span>
      </div>
      {hint ? <div style={hintStyle}>{hint}</div> : null}
    </div>
  );
}

function RiskBadge({ risk }) {
  const colors = RISK_COLORS[risk] || {
    border: "var(--color-border-secondary)",
    bg: "var(--color-background-tertiary)",
    text: "var(--color-text-secondary)"
  };
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "8px",
        padding: "6px 12px",
        borderRadius: "999px",
        background: colors.bg,
        color: colors.text,
        border: `1px solid ${colors.border}`,
        fontSize: "12px",
        fontWeight: 700
      }}
    >
      <span
        style={{
          width: "8px",
          height: "8px",
          borderRadius: "50%",
          background: colors.border
        }}
      />
      {risk}
    </span>
  );
}

function TrendPill({ trend }) {
  const t = trend || "stable";
  const colors = TREND_COLORS[t] || TREND_COLORS.stable;
  return (
    <span
      style={{
        ...pillBase,
        background: colors.bg,
        color: colors.text,
        border: `1px solid ${colors.border}`
      }}
    >
      {t}
    </span>
  );
}

function LoadingState() {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "12px", color: "var(--color-text-secondary)" }}>
      <div className="spinner" />
      <div>Running model inference...</div>
    </div>
  );
}

function EmptyState({ text }) {
  return (
    <div style={emptyStateStyle}>
      <div style={{ fontSize: "20px", marginBottom: "8px" }}>+</div>
      <div>{text}</div>
    </div>
  );
}

export default function MaternalMortalityApp() {
  const [activeTab, setActiveTab] = useState("patient");

  const [patientInput, setPatientInput] = useState(initialPatient);
  const [patientResult, setPatientResult] = useState(null);
  const [patientLoading, setPatientLoading] = useState(false);
  const [patientError, setPatientError] = useState("");

  const [countryInput, setCountryInput] = useState(initialCountry);
  const [countryResult, setCountryResult] = useState(null);
  const [countryLoading, setCountryLoading] = useState(false);
  const [countryError, setCountryError] = useState("");

  const patientAccent = "#7F77DD";
  const patientAccentFill = "#EEEDFE";
  const countryAccent = "#1D9E75";
  const countryAccentFill = "#E1F5EE";

  const buildPatientPrompt = (data) => `
You are simulating four ML classifiers for maternal risk based on these clinical inputs:
- Age: ${data.age} years
- Systolic BP: ${data.sbp} mmHg
- Diastolic BP: ${data.dbp} mmHg
- Blood Sugar: ${data.bs} mmol/L
- Body Temperature: ${data.temp} F
- Heart Rate: ${data.hr} bpm

Return ONLY this exact JSON (no markdown, no backticks):

{
  "consensus": "Low Risk | Mid Risk | High Risk",
  "summary": "2-sentence clinical interpretation",
  "top_feature": "most critical feature name",
  "models": [
    { "name": "Random Forest", "risk": "Low Risk|Mid Risk|High Risk", "confidence": 0-100 },
    { "name": "XGBoost", "risk": "Low Risk|Mid Risk|High Risk", "confidence": 0-100 },
    { "name": "SVM", "risk": "Low Risk|Mid Risk|High Risk", "confidence": 0-100 },
    { "name": "Logistic Regression", "risk": "Low Risk|Mid Risk|High Risk", "confidence": 0-100 }
  ],
  "feature_importance": [
    { "feature": "Age", "score": 0-100 },
    { "feature": "Systolic BP", "score": 0-100 },
    { "feature": "Diastolic BP", "score": 0-100 },
    { "feature": "Blood Sugar", "score": 0-100 },
    { "feature": "Body Temp", "score": 0-100 },
    { "feature": "Heart Rate", "score": 0-100 }
  ],
  "recommendation": "one actionable clinical recommendation"
}
`;

  const buildCountryPrompt = (data) => `
You are simulating four forecasting algorithms for maternal mortality ratios based on these inputs:
- Country: ${data.country}
- Continent: ${data.continent}
- HDI Group: ${data.hdiGroup}
- HDI Rank (2021): ${data.hdiRank}
- Baseline MMR (2021): ${data.baseline}

Return ONLY this exact JSON (no markdown, no backticks):

{
  "summary": "2-sentence regional trend analysis",
  "top_driver": "most impactful feature driving the forecast",
  "models": [
    {
      "name": "Linear Regression",
      "forecasts": { "1yr": 0, "3yr": 0, "5yr": 0, "10yr": 0 },
      "r2": 0.0,
      "trend": "improving | stable | worsening"
    },
    {
      "name": "Random Forest",
      "forecasts": { "1yr": 0, "3yr": 0, "5yr": 0, "10yr": 0 },
      "confidence_interval": { "lower": 0, "upper": 0 },
      "trend": "improving | stable | worsening"
    },
    {
      "name": "XGBoost",
      "forecasts": { "1yr": 0, "3yr": 0, "5yr": 0, "10yr": 0 },
      "mae": 0,
      "trend": "improving | stable | worsening"
    },
    {
      "name": "ARIMA",
      "forecasts": { "1yr": 0, "3yr": 0, "5yr": 0, "10yr": 0 },
      "params": "ARIMA(p,d,q)",
      "trend": "improving | stable | worsening"
    }
  ],
  "ensemble_forecast": { "1yr": 0, "3yr": 0, "5yr": 0, "10yr": 0 }
}
`;

  const runPatientPrediction = async () => {
    setPatientError("");
    setPatientLoading(true);
    try {
      const prompt = buildPatientPrompt(patientInput);
      const data = await callAnthropic(prompt);
      setPatientResult(data);
    } catch (err) {
      setPatientError(err.message || "Unable to run prediction");
    } finally {
      setPatientLoading(false);
    }
  };

  const runCountryForecast = async () => {
    setCountryError("");
    setCountryLoading(true);
    try {
      const prompt = buildCountryPrompt(countryInput);
      const data = await callAnthropic(prompt);
      setCountryResult(data);
    } catch (err) {
      setCountryError(err.message || "Unable to run forecast");
    } finally {
      setCountryLoading(false);
    }
  };

  const renderPatientResults = () => {
    if (patientLoading) return <LoadingState />;
    if (patientError) return <div style={{ color: "#D85A30" }}>{patientError}</div>;
    if (!patientResult) return <EmptyState text="Run a prediction to see patient risk results." />;

    const models = Array.isArray(patientResult.models) ? patientResult.models : [];
    const featureImp = Array.isArray(patientResult.feature_importance)
      ? [...patientResult.feature_importance].sort((a, b) => (b.score || 0) - (a.score || 0))
      : [];

    return (
      <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px", justifyContent: "space-between" }}>
          <div>
            <div style={sectionTitle}>Consensus Verdict</div>
            <RiskBadge risk={patientResult.consensus || "Mid Risk"} />
          </div>
          <div style={{ textAlign: "right" }}>
            <div style={sectionTitle}>Top Feature</div>
            <div style={{ fontWeight: 700 }}>{patientResult.top_feature}</div>
          </div>
        </div>

        <div style={{ color: "var(--color-text-secondary)", lineHeight: 1.5 }}>
          {patientResult.summary}
        </div>

        <div style={dividerStyle} />

        <div>
          <div style={sectionTitle}>Model Breakdown</div>
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            {models.map((m, idx) => {
              const conf = Math.round(m.confidence || 0);
              const color = MODEL_COLORS[m.name] || "#7F77DD";
              return (
                <div
                  key={`${m.name}-${idx}`}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "180px 160px 1fr",
                    gap: "12px",
                    alignItems: "center",
                    background: "var(--color-background-tertiary)",
                    padding: "10px 12px",
                    borderRadius: "var(--border-radius-md)",
                    border: "1px solid var(--color-border-tertiary)"
                  }}
                >
                  <div style={{ fontWeight: 700 }}>{m.name}</div>
                  <RiskBadge risk={m.risk || "Mid Risk"} />
                  <div>
                    <div style={{ height: "8px", background: "var(--color-border-tertiary)", borderRadius: "8px", overflow: "hidden" }}>
                      <div style={{ width: `${conf}%`, background: color, height: "100%" }} />
                    </div>
                    <div style={{ fontSize: "11px", color: "var(--color-text-tertiary)", marginTop: "4px" }}>
                      Confidence: {conf}%
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div style={dividerStyle} />

        <div>
          <div style={sectionTitle}>Feature Importance</div>
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {featureImp.map((f, idx) => {
              const score = Math.round(f.score || 0);
              return (
                <div key={`${f.feature}-${idx}`} style={{ display: "grid", gridTemplateColumns: "140px 1fr 50px", gap: "12px", alignItems: "center" }}>
                  <div style={{ fontSize: "13px", color: "var(--color-text-secondary)" }}>{f.feature}</div>
                  <div style={{ height: "8px", background: "var(--color-border-tertiary)", borderRadius: "8px", overflow: "hidden" }}>
                    <div style={{ width: `${score}%`, background: patientAccent, height: "100%" }} />
                  </div>
                  <div style={{ fontSize: "12px", textAlign: "right" }}>{score}</div>
                </div>
              );
            })}
          </div>
        </div>

        <div style={{ borderLeft: `4px solid ${patientAccent}`, paddingLeft: "12px", background: "var(--color-background-tertiary)", borderRadius: "var(--border-radius-md)" }}>
          <div style={{ fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.1em", color: "var(--color-text-tertiary)" }}>
            Recommendation
          </div>
          <div style={{ fontWeight: 600, color: "var(--color-text-primary)" }}>
            {patientResult.recommendation}
          </div>
        </div>
      </div>
    );
  };

  const renderCountryResults = () => {
    if (countryLoading) return <LoadingState />;
    if (countryError) return <div style={{ color: "#D85A30" }}>{countryError}</div>;
    if (!countryResult) return <EmptyState text="Run a forecast to see country-level MMR projections." />;

    const models = Array.isArray(countryResult.models) ? countryResult.models : [];
    const ensemble = countryResult.ensemble_forecast || {};
    const baseline = Number(countryInput.baseline) || 0;
    const values = [
      { label: "Baseline", value: baseline, key: "baseline" },
      { label: "+1yr", value: Number(ensemble["1yr"] || 0), key: "1yr" },
      { label: "+3yr", value: Number(ensemble["3yr"] || 0), key: "3yr" },
      { label: "+5yr", value: Number(ensemble["5yr"] || 0), key: "5yr" },
      { label: "+10yr", value: Number(ensemble["10yr"] || 0), key: "10yr" }
    ];
    const maxVal = Math.max(...values.map((v) => v.value), 1);
    const forecast10 = Math.round(Number(ensemble["10yr"] || 0));
    const improving = forecast10 < baseline;
    const trendColor = improving ? TREND_COLORS.improving : TREND_COLORS.worsening;

    return (
      <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
          <div style={{ ...baseCard, padding: "12px" }}>
            <div style={sectionTitle}>Baseline MMR</div>
            <div style={{ fontSize: "28px", fontWeight: 700 }}>{Math.round(baseline)}</div>
            <div style={hintStyle}>per 100k live births</div>
          </div>
          <div style={{ ...baseCard, padding: "12px", border: `1px solid ${trendColor.border}`, background: trendColor.bg }}>
            <div style={{ ...sectionTitle, color: trendColor.text }}>10-yr Forecast</div>
            <div style={{ fontSize: "28px", fontWeight: 700, color: trendColor.text }}>{forecast10}</div>
            <div style={{ ...hintStyle, color: trendColor.text }}>{improving ? "Improving trend" : "Worsening trend"}</div>
          </div>
        </div>

        <div style={{ color: "var(--color-text-secondary)", lineHeight: 1.5 }}>
          {countryResult.summary}
        </div>

        <div style={dividerStyle} />

        <div>
          <div style={sectionTitle}>Ensemble Forecast Trajectory</div>
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {values.map((v) => {
              const valueRounded = Math.round(v.value);
              const change = baseline ? Math.round(((v.value - baseline) / baseline) * 100) : 0;
              const barColor = v.key === "baseline"
                ? "#7F77DD"
                : v.value < baseline
                ? "#1D9E75"
                : v.value > baseline
                ? "#D85A30"
                : "#EF9F27";
              return (
                <div key={v.key} style={{ display: "grid", gridTemplateColumns: "80px 1fr 120px", gap: "12px", alignItems: "center" }}>
                  <div style={{ fontSize: "12px", color: "var(--color-text-secondary)" }}>{v.label}</div>
                  <div style={{ height: "8px", background: "var(--color-border-tertiary)", borderRadius: "8px", overflow: "hidden" }}>
                    <div style={{ width: `${Math.round((v.value / maxVal) * 100)}%`, background: barColor, height: "100%" }} />
                  </div>
                  <div style={{ textAlign: "right", fontSize: "12px" }}>
                    {valueRounded} ({change}%)
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div style={dividerStyle} />

        <div>
          <div style={sectionTitle}>Model Comparison</div>
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {models.map((m, idx) => {
              const forecast10yr = Math.round((m.forecasts && m.forecasts["10yr"]) || 0);
              let metric = "";
              if (m.name === "Linear Regression") {
                metric = `R2: ${Number(m.r2 || 0).toFixed(2)}`;
              } else if (m.name === "Random Forest") {
                const lower = Math.round((m.confidence_interval && m.confidence_interval.lower) || 0);
                const upper = Math.round((m.confidence_interval && m.confidence_interval.upper) || 0);
                metric = `CI: ${lower}-${upper}`;
              } else if (m.name === "XGBoost") {
                metric = `MAE: ${Math.round(m.mae || 0)}`;
              } else if (m.name === "ARIMA") {
                metric = m.params || "ARIMA(p,d,q)";
              }
              return (
                <div
                  key={`${m.name}-${idx}`}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1.2fr 120px 140px 120px",
                    gap: "12px",
                    alignItems: "center",
                    padding: "10px 12px",
                    borderRadius: "var(--border-radius-md)",
                    border: "1px solid var(--color-border-tertiary)",
                    background: "var(--color-background-tertiary)"
                  }}
                >
                  <div style={{ fontWeight: 700 }}>{m.name}</div>
                  <div style={{ fontSize: "12px", color: "var(--color-text-secondary)" }}>
                    10-yr: {forecast10yr}
                  </div>
                  <TrendPill trend={(m.trend || "stable").trim()} />
                  <div style={{ fontSize: "12px", textAlign: "right", color: "var(--color-text-tertiary)" }}>{metric}</div>
                </div>
              );
            })}
          </div>
        </div>

        <div style={{ borderLeft: `4px solid ${countryAccent}`, paddingLeft: "12px", background: "var(--color-background-tertiary)", borderRadius: "var(--border-radius-md)" }}>
          <div style={{ fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.1em", color: "var(--color-text-tertiary)" }}>
            Key Driver
          </div>
          <div style={{ fontWeight: 600, color: "var(--color-text-primary)" }}>
            {countryResult.top_driver}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div style={{ fontFamily: "var(--font-sans)", background: "var(--color-background-primary)", color: "var(--color-text-primary)", minHeight: "100vh", padding: "24px" }}>
      <style>{`
        .spinner {
          width: 18px;
          height: 18px;
          border: 3px solid var(--color-border-tertiary);
          border-top: 3px solid #7F77DD;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @media (max-width: 900px) {
          .grid-two {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>

      <div style={{ maxWidth: "1200px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "8px" }}>
          <div>
            <div style={{ fontSize: "28px", fontWeight: 800 }}>Maternal Mortality Prediction UI</div>
            <div style={{ fontSize: "15px", color: "var(--color-text-tertiary)" }}>
              AI-powered risk assessment and MMR forecasting combining 4 analytical models.
            </div>
          </div>
        </div>

        <div style={{ display: "flex", gap: "16px", borderBottom: "1px solid var(--color-border-tertiary)" }}>
          <button
            onClick={() => setActiveTab("patient")}
            style={{
              background: "transparent",
              border: "none",
              padding: "8px 0",
              color: activeTab === "patient" ? patientAccent : "var(--color-text-tertiary)",
              borderBottom: activeTab === "patient" ? `3px solid ${patientAccent}` : "3px solid transparent",
              fontWeight: 700,
              fontSize: "15px",
              cursor: "pointer"
            }}
          >
            Patient Risk Assessment
          </button>
          <button
            onClick={() => setActiveTab("country")}
            style={{
              background: "transparent",
              border: "none",
              padding: "8px 0",
              color: activeTab === "country" ? countryAccent : "var(--color-text-tertiary)",
              borderBottom: activeTab === "country" ? `3px solid ${countryAccent}` : "3px solid transparent",
              fontWeight: 700,
              fontSize: "15px",
              cursor: "pointer"
            }}
          >
            Country MMR Forecasting
          </button>
        </div>

        {activeTab === "patient" ? (
          <div className="grid-two" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}>
            <div style={baseCard}>
              <div style={sectionTitle}>Patient Vitals</div>
              <SliderField
                label="Age"
                value={patientInput.age}
                min={15}
                max={55}
                step={1}
                hint=""
                unit="yrs"
                accent={patientAccent}
                accentFill={patientAccentFill}
                onChange={(v) => setPatientInput({ ...patientInput, age: v })}
              />
              <SliderField
                label="Systolic BP"
                value={patientInput.sbp}
                min={70}
                max={200}
                step={1}
                hint="Normal: 90-120 mmHg"
                unit="mmHg"
                accent={patientAccent}
                accentFill={patientAccentFill}
                onChange={(v) => setPatientInput({ ...patientInput, sbp: v })}
              />
              <SliderField
                label="Diastolic BP"
                value={patientInput.dbp}
                min={40}
                max={130}
                step={1}
                hint="Normal: 60-80 mmHg"
                unit="mmHg"
                accent={patientAccent}
                accentFill={patientAccentFill}
                onChange={(v) => setPatientInput({ ...patientInput, dbp: v })}
              />
              <SliderField
                label="Blood Sugar"
                value={patientInput.bs}
                min={3.0}
                max={20.0}
                step={0.1}
                hint="Gestational diabetes indicator"
                unit="mmol/L"
                accent={patientAccent}
                accentFill={patientAccentFill}
                onChange={(v) => setPatientInput({ ...patientInput, bs: v })}
                decimals={1}
              />
              <SliderField
                label="Body Temperature"
                value={patientInput.temp}
                min={96.0}
                max={104.0}
                step={0.1}
                hint="Normal: 97-99 F"
                unit="F"
                accent={patientAccent}
                accentFill={patientAccentFill}
                onChange={(v) => setPatientInput({ ...patientInput, temp: v })}
                decimals={1}
              />
              <SliderField
                label="Heart Rate"
                value={patientInput.hr}
                min={40}
                max={160}
                step={1}
                hint="Normal: 60-100 bpm"
                unit="bpm"
                accent={patientAccent}
                accentFill={patientAccentFill}
                onChange={(v) => setPatientInput({ ...patientInput, hr: v })}
              />
              <button
                onClick={runPatientPrediction}
                disabled={patientLoading}
                style={{
                  width: "100%",
                  padding: "12px",
                  borderRadius: "var(--border-radius-md)",
                  background: patientAccent,
                  color: "white",
                  border: "none",
                  fontWeight: 700,
                  cursor: patientLoading ? "default" : "pointer"
                }}
              >
                Run Prediction via Anthropic AI
              </button>
            </div>

            <div style={baseCard}>
              <div style={sectionTitle}>Results</div>
              {renderPatientResults()}
            </div>
          </div>
        ) : (
          <div className="grid-two" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}>
            <div style={baseCard}>
              <div style={sectionTitle}>Country Inputs</div>
              <div style={{ marginBottom: "14px" }}>
                <div style={labelStyle}>Country</div>
                <input
                  type="text"
                  value={countryInput.country}
                  onChange={(e) => setCountryInput({ ...countryInput, country: e.target.value })}
                  style={{ width: "100%", marginTop: "6px", padding: "8px", borderRadius: "var(--border-radius-md)", border: "1px solid var(--color-border-secondary)", background: "var(--color-background-tertiary)", color: "var(--color-text-primary)" }}
                />
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginBottom: "12px" }}>
                <div>
                  <div style={labelStyle}>Continent</div>
                  <select
                    value={countryInput.continent}
                    onChange={(e) => setCountryInput({ ...countryInput, continent: e.target.value })}
                    style={{ width: "100%", marginTop: "6px", padding: "8px", borderRadius: "var(--border-radius-md)", border: "1px solid var(--color-border-secondary)", background: "var(--color-background-tertiary)", color: "var(--color-text-primary)" }}
                  >
                    {["Africa", "Asia", "Europe", "North America", "South America", "Oceania"].map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <div style={labelStyle}>HDI Group</div>
                  <select
                    value={countryInput.hdiGroup}
                    onChange={(e) => setCountryInput({ ...countryInput, hdiGroup: e.target.value })}
                    style={{ width: "100%", marginTop: "6px", padding: "8px", borderRadius: "var(--border-radius-md)", border: "1px solid var(--color-border-secondary)", background: "var(--color-background-tertiary)", color: "var(--color-text-primary)" }}
                  >
                    {["Low", "Medium", "High", "Very High"].map((g) => (
                      <option key={g} value={g}>
                        {g}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <SliderField
                label="HDI Rank 2021"
                value={countryInput.hdiRank}
                min={1}
                max={191}
                step={1}
                hint="1 = highest development"
                unit=""
                accent={countryAccent}
                accentFill={countryAccentFill}
                onChange={(v) => setCountryInput({ ...countryInput, hdiRank: v })}
              />
              <SliderField
                label="Baseline MMR (2021)"
                value={countryInput.baseline}
                min={2}
                max={1200}
                step={1}
                hint="Deaths per 100,000 live births"
                unit="/ 100k"
                accent={countryAccent}
                accentFill={countryAccentFill}
                onChange={(v) => setCountryInput({ ...countryInput, baseline: v })}
              />

              <button
                onClick={runCountryForecast}
                disabled={countryLoading}
                style={{
                  width: "100%",
                  padding: "12px",
                  borderRadius: "var(--border-radius-md)",
                  background: countryAccent,
                  color: "white",
                  border: "none",
                  fontWeight: 700,
                  cursor: countryLoading ? "default" : "pointer",
                  marginTop: "12px"
                }}
              >
                Run Forecast via Anthropic AI
              </button>
            </div>

            <div style={baseCard}>
              <div style={sectionTitle}>Results</div>
              {renderCountryResults()}
            </div>
          </div>
        )}

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", color: "var(--color-text-tertiary)", fontSize: "12px", marginTop: "24px" }}>
          <div style={{ display: "flex", gap: "16px", alignItems: "center" }}>
            {Object.entries(MODEL_COLORS).slice(0, 4).map(([name, color]) => (
              <div key={name} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: color }} />
                <span>{name}</span>
              </div>
            ))}
          </div>
          <div>AI-simulated predictions - for research use only</div>
        </div>
      </div>
    </div>
  );
}
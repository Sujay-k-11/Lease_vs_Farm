"""
AgriConnect+ | Farmer Income Decision Dashboard — Objective 1
Crop Income vs Lease Income — Full Analysis & Downloadable Report
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json, io, csv
from datetime import datetime
from ml_engine import (
    compute_income,
    load_artifacts,
    predict_decision,
    predict_price as ml_predict_price,
    predict_yield as ml_predict_yield,
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & THEME
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriConnect+ | Income Report",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f1f5f9;
}

/* ── Header ── */
.ag-hero {
    background: linear-gradient(135deg, #1e293b 0%, #334155 60%, #475569 100%);
    border-radius: 16px;
    padding: 36px 48px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.18);
}
.ag-hero h1 {
    color: #f8fafc;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}
.ag-hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin: 0;
}
.ag-badge {
    display: inline-block;
    background: #6366f1;
    color: white;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
}

/* ── Form Card ── */
.ag-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 28px 32px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.07);
    margin-bottom: 24px;
}
.ag-section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #1e293b;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding-bottom: 10px;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 20px;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
[data-testid="stMetricLabel"]  { color: #64748b; font-weight: 600; font-size: 0.82rem; }
[data-testid="stMetricValue"]  { color: #1e293b; font-size: 1.7rem !important; font-weight: 800; }
[data-testid="stMetricDelta"]  { color: #6366f1; font-size: 0.85rem; }

/* ── Recommendation boxes ── */
.rec-win {
    background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
    border-left: 5px solid #6366f1;
    border-radius: 12px;
    padding: 20px 24px;
    color: #1e1b4b;
    font-size: 0.97rem;
    line-height: 1.6;
}
.rec-lease {
    background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 60%);
    border-left: 5px solid #f97316;
    border-radius: 12px;
    padding: 20px 24px;
    color: #431407;
    font-size: 0.97rem;
    line-height: 1.6;
}

/* ── Risk badges ── */
.risk-low    { background:#dbeafe; color:#1e40af; border-radius:20px; padding:4px 16px; font-weight:700; font-size:0.88rem; }
.risk-medium { background:#fef9c3; color:#854d0e; border-radius:20px; padding:4px 16px; font-weight:700; font-size:0.88rem; }
.risk-high   { background:#fee2e2; color:#991b1b; border-radius:20px; padding:4px 16px; font-weight:700; font-size:0.88rem; }

/* ── Section header strip ── */
.strip {
    background: linear-gradient(90deg, #1e293b, #334155);
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 0.95rem;
    font-weight: 700;
    margin: 28px 0 16px 0;
    letter-spacing: 0.3px;
}

/* ── Buttons ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #4f46e5, #6366f1);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1rem;
    padding: 12px 28px;
    box-shadow: 0 4px 14px rgba(99,102,241,0.4);
    transition: all 0.2s;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(90deg, #4338ca, #4f46e5);
    box-shadow: 0 6px 20px rgba(99,102,241,0.5);
}

/* ── What-if card ── */
.whatif-card {
    background: linear-gradient(135deg, #f8fafc, #eef2ff);
    border: 1px solid #c7d2fe;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
}

/* ── Download buttons ── */
div[data-testid="stDownloadButton"] > button {
    background: #1e293b;
    color: white;
    border-radius: 10px;
    font-weight: 600;
    border: none;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: #334155;
}

/* ── Divider ── */
hr { border-color: #e2e8f0; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# STATIC DATA
# ─────────────────────────────────────────────────────────────
CROP_CYCLES = {
    'Amaranthus':3,'Banana':1,'Beans':2,'Beetroot':2,
    'Bitter Gourd':2,'Bottle Gourd':2,'Brinjal':2,
    'Cabbage':2,'Capsicum':2,'Carrot':2,'Cauliflower':2,
    'Coconut':1,'Cotton':1,'Garlic':1,'Grapes':1,
    'Green Chilli':2,'Groundnut':2,'Guava':1,'Lemon':1,
    'Maize':2,'Mango':1,'Mustard':1,'Onion':2,
    'Orange':1,'Papaya':1,'Pomegranate':1,'Potato':2,
    'Pumpkin':2,'Rice':2,'Soyabean':1,'Spinach':3,
    'Sugarcane':1,'Sunflower':2,'Tomato':3,
    'Turmeric':1,'Wheat':1,
}
FARMING_COSTS = {
    'Amaranthus':4000,'Banana':35000,'Beans':6000,
    'Beetroot':5000,'Bitter Gourd':6000,'Bottle Gourd':5000,
    'Brinjal':6000,'Cabbage':6000,'Capsicum':9000,
    'Carrot':7500,'Cauliflower':7000,'Coconut':20000,
    'Cotton':18000,'Garlic':25000,'Grapes':45000,
    'Green Chilli':7500,'Groundnut':9000,'Guava':20000,
    'Lemon':18000,'Maize':6000,'Mango':20000,
    'Mustard':10000,'Onion':9000,'Orange':20000,
    'Papaya':22000,'Pomegranate':35000,'Potato':10000,
    'Pumpkin':5000,'Rice':11000,'Soyabean':10000,
    'Spinach':2700,'Sugarcane':25000,'Sunflower':5000,
    'Tomato':6000,'Turmeric':22000,'Wheat':12000,
}
COST_SPLIT = {'seeds':0.25,'fertilizer':0.30,'labor':0.35,'misc':0.10}
DEFAULT_LEASE = {"min":15000,"avg":25000,"max":40000}

# ─────────────────────────────────────────────────────────────
# DATA LOAD
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    prices = pd.read_csv("crop_prices.csv")
    yields = pd.read_csv("crop_yield.csv")
    with open("lease_rates.json") as f:  lr = json.load(f)
    with open("district_lease_rates.json") as f:  dr = json.load(f)
    return prices, yields, lr, dr

price_df, yield_df, STATE_LEASE_RATES, DISTRICT_LEASE_RATES = load_data()

@st.cache_resource
def load_model_artifacts():
    return load_artifacts()

ML_ARTIFACTS = load_model_artifacts()

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def get_districts(state):
    return sorted(price_df[price_df["state"].str.lower()==state.lower()]["district"].dropna().unique())

def get_crops(state, district):
    mask = (price_df["state"].str.lower()==state.lower()) & (price_df["district"].str.lower()==district.lower())
    return sorted(price_df[mask]["crop"].dropna().unique())

def get_price(state, district, crop):
    row = price_df[
        (price_df["state"].str.lower()==state.lower()) &
        (price_df["district"].str.lower()==district.lower()) &
        (price_df["crop"].str.lower()==crop.lower())
    ]
    return float(row["avg_price_quintal"].values[0]) if not row.empty else None

def get_yield(state, crop, irrigation):
    row = yield_df[(yield_df["state"].str.lower()==state.lower()) & (yield_df["crop"].str.lower()==crop.lower())]
    base = float(row["avg_yield_kg_ha"].values[0]) if not row.empty else (
        float(yield_df[yield_df["crop"].str.lower()==crop.lower()]["avg_yield_kg_ha"].mean()) if not yield_df[yield_df["crop"].str.lower()==crop.lower()].empty else 4000
    )
    return base * 0.8 if irrigation == "Rainfed" else base * 1.2

def get_lease(state, district):
    return DISTRICT_LEASE_RATES.get(district, STATE_LEASE_RATES.get(state, DEFAULT_LEASE))

def risk_level(margin_pct, cycles, irrigation, scenario):
    s = 0
    if margin_pct < 20: s += 2
    elif margin_pct < 50: s += 1
    if irrigation == "Rainfed": s += 1
    if "Low" in scenario: s += 2
    elif "High" in scenario: s -= 1
    if cycles >= 3: s -= 1
    return "Low" if s <= 1 else ("Medium" if s <= 3 else "High")

def price_trend(base, years):
    np.random.seed(42)
    n = max(int(years * 2), 4)
    trend = base * (1 + np.cumsum(np.random.uniform(-0.08, 0.10, n)))
    trend = np.clip(trend, base * 0.55, base * 2.0)
    labels = [f"H{i}" if years <= 3 else f"Y{i}" for i in range(1, n+1)]
    return labels, trend

def fmt(n): return f"₹{n:,.0f}"

# ─────────────────────────────────────────────────────────────
# CHART STYLE
# ─────────────────────────────────────────────────────────────
PALETTE = {
    "crop":  "#6366f1",   # indigo
    "lease": "#f97316",   # orange
    "bg":    "#f8fafc",
    "grid":  "#e2e8f0",
}

def style_ax(ax, title=""):
    ax.set_facecolor(PALETTE["bg"])
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["bottom","left"]].set_color(PALETTE["grid"])
    ax.tick_params(colors="#64748b", labelsize=8)
    ax.yaxis.label.set_color("#64748b")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color="#1e293b", pad=10)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8, linestyle="--")

# ─────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────
if "report" not in st.session_state:
    st.session_state.report = None

# ─────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="ag-hero">
    <span class="ag-badge">OBJECTIVE 1</span>
    <h1>🌾 AgriConnect+ Income Dashboard</h1>
    <p>Compare Crop Farming vs Land Leasing — Decide smarter, earn better.</p>
</div>
""", unsafe_allow_html=True)

if ML_ARTIFACTS.get("available"):
    metrics = ML_ARTIFACTS.get("metrics", {})
    trained_at = metrics.get("generated_at", "available")
    st.success(f"ML models loaded. Training metadata: {trained_at}")
else:
    st.info("ML models are not trained yet. Run `python train_models.py`; until then the dashboard uses transparent dataset lookup and formulas.")

# ─────────────────────────────────────────────────────────────
# ① INPUT FORM (full-width, vertical, on main page)
# ─────────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="ag-card">', unsafe_allow_html=True)
    st.markdown('<div class="ag-section-title">📋 Step 1 — Enter Your Farm Details</div>', unsafe_allow_html=True)

    available_states = sorted(price_df["state"].dropna().unique())

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        state = st.selectbox("📍 State", available_states, key="inp_state")
    with r1c2:
        districts = get_districts(state)
        district  = st.selectbox("🏘️ District", districts, key="inp_district")
    with r1c3:
        crops = get_crops(state, district)
        crop  = st.selectbox("🌱 Crop", crops if crops else ["No crops found"], key="inp_crop")
    with r1c4:
        irrigation = st.selectbox("💧 Irrigation Type", ["Irrigated","Rainfed"], key="inp_irr")

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        year_opts = [0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,12.0,15.0]
        years = st.selectbox("📅 Duration", year_opts, index=1,
                              format_func=lambda x: f"{x} {'year' if x==1 else 'years'}", key="inp_years")
    with r2c2:
        acres = st.number_input("🏡 Land Size (Acres)", 0.5, 1000.0, 2.0, step=0.5, key="inp_acres")
    with r2c3:
        scenario = st.selectbox("📊 Market Scenario",
                                 ["Low (Bad Season)","Average","High (Good Season)"],
                                 index=1, key="inp_scen")
    with r2c4:
        lease_data   = get_lease(state, district)
        lease_choice = st.selectbox("🏠 Lease Rate",
                                     ["Minimum","Average","Maximum","Manual"],
                                     index=1, key="inp_lease")

    if lease_choice == "Manual":
        lease_per_acre = st.number_input("Enter Lease Rate (₹/acre/year)",
                                          1000, 500000, lease_data["avg"], step=1000,
                                          key="inp_lease_manual")
    else:
        lease_per_acre = {"Minimum":lease_data["min"],
                          "Average":lease_data["avg"],
                          "Maximum":lease_data["max"]}[lease_choice]
        st.caption(f"📍 {district}, {state} lease range: {fmt(lease_data['min'])} – {fmt(lease_data['max'])}/acre/year")

    st.markdown("</div>", unsafe_allow_html=True)

    # Generate button (centered)
    _, btn_col, _ = st.columns([2.5, 1, 2.5])
    with btn_col:
        generate = st.button("⚡ Generate Report", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────
# When Generate is clicked — compute & store in session_state
# ─────────────────────────────────────────────────────────────
if generate:
    price_result = ml_predict_price(ML_ARTIFACTS, price_df, state, district, crop, scenario)
    yield_result = ml_predict_yield(ML_ARTIFACTS, yield_df, state, crop, irrigation)
    price_base   = price_result["base_price"]
    price_used   = price_result["price"]
    yield_kg     = yield_result["yield_kg_ha"]
    cycles       = CROP_CYCLES.get(crop, 1)
    base_cost_pa = FARMING_COSTS.get(crop, 15000)
    location     = f"{district}, {state}"

    income = compute_income(
        price_per_quintal=price_used,
        yield_kg_ha=yield_kg,
        cost_per_acre_cycle=base_cost_pa,
        cycles_per_year=cycles,
        acres=acres,
        years=years,
        lease_per_acre_year=lease_per_acre,
    )
    net_cycle = income["net_cycle"]
    net_year  = income["net_year"]
    total_crop  = income["total_crop"]
    total_lease = income["total_lease"]
    total_diff  = income["total_diff"]
    margin_pct  = income["margin_pct"]

    scenario_label = "Low" if "Low" in scenario else ("High" if "High" in scenario else "Average")
    deterministic_better = "Crop Farming" if total_diff > 0 else "Leasing"
    decision_features = {
        "state": state,
        "district": district,
        "crop": crop,
        "irrigation": irrigation,
        "market_scenario": scenario_label,
        "price_per_quintal": price_used,
        "yield_kg_ha": yield_kg,
        "cost_per_acre_cycle": base_cost_pa,
        "cycles_per_year": cycles,
        "acres": acres,
        "years": years,
        "lease_per_acre_year": lease_per_acre,
    }
    decision_result = predict_decision(ML_ARTIFACTS, decision_features, deterministic_better)

    step    = 0.5 if years <= 3 else 1.0
    periods = np.arange(step, years + step/2, step)
    ylabels, crop_list, lease_list = [], [], []
    for p in periods:
        ylabels.append(f"Y{p:.1f}" if step==0.5 else f"Y{int(p)}")
        crop_list.append(net_year * acres * step)
        lease_list.append(lease_per_acre * acres * step)

    st.session_state.report = dict(
        state=state, district=district, crop=crop, irrigation=irrigation,
        years=years, acres=acres, scenario=scenario,
        lease_per_acre=lease_per_acre, lease_data=lease_data,
        location=location,
        price_base=price_base, price_used=price_used, yield_kg=yield_kg, cycles=cycles,
        base_cost_pa=base_cost_pa, net_cycle=net_cycle, net_year=net_year,
        total_crop=total_crop, total_lease=total_lease,
        total_diff=total_diff, margin_pct=margin_pct,
        ylabels=ylabels, crop_list=crop_list, lease_list=lease_list,
        price_source=price_result["source"],
        yield_source=yield_result["source"],
        decision_source=decision_result["source"],
        decision_confidence=decision_result["confidence"],
        ml_decision=decision_result["decision"],
        ml_metrics=ML_ARTIFACTS.get("metrics", {}),
    )

# ─────────────────────────────────────────────────────────────
# ② REPORT SECTION (only if we have data in session_state)
# ─────────────────────────────────────────────────────────────
R = st.session_state.report
if R is None:
    st.markdown("""
    <div style="text-align:center; padding:60px 0; color:#94a3b8;">
        <div style="font-size:3rem; margin-bottom:12px;">📊</div>
        <div style="font-size:1.15rem; font-weight:600; color:#475569;">
            Fill in your farm details above and click <strong>Generate Report</strong>
        </div>
        <div style="font-size:0.9rem; margin-top:8px;">
            You'll get charts, insights, what-if analysis & a downloadable report.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Unpack ──
state        = R["state"];      district   = R["district"]
crop         = R["crop"];       irrigation = R["irrigation"]
years        = R["years"];      acres      = R["acres"]
scenario     = R["scenario"];   location   = R["location"]
price_used   = R["price_used"]; yield_kg   = R["yield_kg"]
cycles       = R["cycles"];     base_cost_pa = R["base_cost_pa"]
net_year     = R["net_year"];   net_cycle  = R["net_cycle"]
total_crop   = R["total_crop"]; total_lease = R["total_lease"]
total_diff   = R["total_diff"]; margin_pct = R["margin_pct"]
ylabels      = R["ylabels"];    crop_list  = R["crop_list"]
lease_list   = R["lease_list"]; lease_per_acre = R["lease_per_acre"]
lease_data   = R["lease_data"]
cum_crop     = np.cumsum(crop_list)
cum_lease    = np.cumsum(lease_list)
rlevel       = risk_level(margin_pct, cycles, irrigation, scenario)
ml_decision  = R.get("ml_decision", "Crop Farming" if total_diff > 0 else "Leasing")
decision_confidence = R.get("decision_confidence")
better       = "🌾 Crop Farming" if ml_decision == "Crop Farming" else "🏠 Leasing"
seeds_c  = base_cost_pa * COST_SPLIT["seeds"]
fert_c   = base_cost_pa * COST_SPLIT["fertilizer"]
labor_c  = base_cost_pa * COST_SPLIT["labor"]
misc_c   = base_cost_pa * COST_SPLIT["misc"]

risk_factors = []
if irrigation == "Rainfed":        risk_factors.append("🌧️ Rainfed crops depend on monsoon — income may vary.")
if "Low" in scenario:              risk_factors.append("📉 Bad season mode — good year income could be higher.")
if rlevel == "High":               risk_factors.append("🔴 Tight margins — one bad season could cause a loss.")
if cycles >= 3:                    risk_factors.append("🔄 3 cycles/year needs strong, consistent management.")
risk_factors.append("📊 Mandi prices fluctuate — consider contract farming or FPO linkage.")
risk_factors.append("💧 Upgrade to Irrigated farming to increase yield by ~50%.")

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("""
<div style="font-size:1.4rem; font-weight:800; color:#1e293b; margin-bottom:8px;">
    📑 Step 2 — Your Income Report
</div>""", unsafe_allow_html=True)
st.caption(f"Generated for **{crop}** · {acres} acres · {years} yr(s) · {location} · {scenario}")
st.caption(
    f"Model/data sources: price = {R.get('price_source', 'unknown')}; "
    f"yield = {R.get('yield_source', 'unknown')}; "
    f"decision = {R.get('decision_source', 'unknown')}"
)

# ─────────────────────────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="strip">📊 Top Metrics</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("🌾 Total Crop Income",  fmt(total_crop),  f"{fmt(net_year*acres)}/yr")
m2.metric("🏠 Total Lease Income", fmt(total_lease), f"{fmt(lease_per_acre*acres)}/yr")
m3.metric("💰 Profit Difference",  fmt(abs(total_diff)), f"{margin_pct:.1f}% advantage")
m4.metric("🏆 Best Option",        ml_decision,
          f"{decision_confidence:.1%} model confidence" if decision_confidence is not None else f"Over {years} yr(s)")

risk_css = {"Low":"risk-low","Medium":"risk-medium","High":"risk-high"}[rlevel]
risk_ico = {"Low":"🔵","Medium":"🟡","High":"🔴"}[rlevel]
st.markdown(f"""
<div style="margin:12px 0 4px 0;">
  <span class="{risk_css}">{risk_ico} Risk: {rlevel}</span>
  &nbsp;<small style="color:#94a3b8;">Crop farming risk for {crop} in {location}</small>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# RECOMMENDATION
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="strip">🤖 Recommendation</div>', unsafe_allow_html=True)
if ml_decision == "Crop Farming":
    msg = (f"Growing <b>{crop}</b> on your <b>{acres}-acre</b> land in <b>{location}</b> earns "
           f"<b>{fmt(abs(total_diff))} more</b> than leasing over {years} year(s). "
           f"At {fmt(price_used)}/quintal with {cycles} cycle(s)/year, crop farming is the smarter choice "
           f"under <b>{scenario}</b> market conditions.")
    st.markdown(f'<div class="rec-win">✅ <b>Recommendation: Crop Farming is BETTER</b><br><br>{msg}</div>',
                unsafe_allow_html=True)
else:
    msg = (f"Leasing your <b>{acres}-acre</b> land at {fmt(lease_per_acre)}/acre/year in <b>{location}</b> "
           f"earns <b>{fmt(abs(total_diff))} more</b> than growing {crop} over {years} year(s). "
           f"Consider a different crop, improving irrigation, or renegotiating lease terms.")
    st.markdown(f'<div class="rec-lease">⚠️ <b>Recommendation: Leasing is BETTER right now</b><br><br>{msg}</div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CHARTS — Bar + Pie
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="strip">📈 Income Comparison</div>', unsafe_allow_html=True)
ch1, ch2 = st.columns([2.1, 1])

with ch1:
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#f8fafc")
    x, w = np.arange(len(ylabels)), 0.36
    ax.bar(x-w/2, crop_list,  w, label="Crop Income",  color=PALETTE["crop"],  alpha=0.88, edgecolor="white", linewidth=0.5)
    ax.bar(x+w/2, lease_list, w, label="Lease Income", color=PALETTE["lease"], alpha=0.88, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(ylabels, rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"₹{v/1000:.0f}K"))
    ax.legend(fontsize=9, framealpha=0.7)
    style_ax(ax, f"{crop} — {location} ({acres} acres)")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close(fig)

with ch2:
    if total_crop + total_lease > 0:
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        fig2.patch.set_facecolor("#f8fafc")
        sizes  = [max(total_crop,0), max(total_lease,0)]
        colors = [PALETTE["crop"], PALETTE["lease"]]
        wedges, texts, autotexts = ax2.pie(
            sizes, labels=["Crop","Lease"], colors=colors, explode=(0.05,0),
            autopct="%1.1f%%", startangle=140, textprops={"fontsize":9})
        for at in autotexts: at.set_color("white"); at.set_fontweight("bold")
        style_ax(ax2, "Profit Distribution (%)")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True); plt.close(fig2)

# ─────────────────────────────────────────────────────────────
# LINE CHARTS — Cumulative + Price Trend
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="strip">📉 Trends Over Time</div>', unsafe_allow_html=True)
lc1, lc2 = st.columns(2)

with lc1:
    fig3, ax3 = plt.subplots(figsize=(6, 3.5))
    fig3.patch.set_facecolor("#f8fafc")
    ax3.plot(ylabels, cum_crop,  "o-", color=PALETTE["crop"],  lw=2.5, ms=5, label="Cumulative Crop")
    ax3.plot(ylabels, cum_lease, "s-", color=PALETTE["lease"], lw=2.5, ms=5, label="Cumulative Lease")
    ax3.fill_between(range(len(ylabels)), cum_crop, cum_lease, alpha=0.1,
                     color=PALETTE["crop"] if total_diff>0 else PALETTE["lease"])
    ax3.set_xticks(range(len(ylabels)))
    ax3.set_xticklabels(ylabels, rotation=45, ha="right", fontsize=8)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"₹{v/1000:.0f}K"))
    ax3.legend(fontsize=9)
    style_ax(ax3, "Cumulative Income Comparison")
    fig3.tight_layout(); st.pyplot(fig3, use_container_width=True); plt.close(fig3)

with lc2:
    tlabels, tprices = price_trend(price_used, years)
    fig4, ax4 = plt.subplots(figsize=(6, 3.5))
    fig4.patch.set_facecolor("#f8fafc")
    ax4.plot(tlabels, tprices, "o-", color="#a855f7", lw=2.5, ms=5, label=f"{crop} Price Trend")
    ax4.fill_between(range(len(tlabels)), tprices, alpha=0.12, color="#a855f7")
    ax4.axhline(price_used, color=PALETTE["lease"], ls="--", lw=1.5,
                label=f"Base: {fmt(price_used)}/q")
    ax4.set_xticks(range(len(tlabels))); ax4.set_xticklabels(tlabels, rotation=45, ha="right", fontsize=8)
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"₹{v:,.0f}"))
    ax4.legend(fontsize=9)
    style_ax(ax4, "Crop Price Trend (Simulated)")
    fig4.tight_layout(); st.pyplot(fig4, use_container_width=True); plt.close(fig4)

st.caption("⚠️ Price trend is simulated for reference only. Actual mandi prices reflect real data.")

# ─────────────────────────────────────────────────────────────
# COST BREAKDOWN — Stacked Bar
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="strip">💸 Farming Cost Breakdown (₹/acre/cycle)</div>', unsafe_allow_html=True)
cb1, cb2 = st.columns([1.2, 1])

with cb1:
    fig5, ax5 = plt.subplots(figsize=(5, 3))
    fig5.patch.set_facecolor("#f8fafc")
    cat = [crop[:18]]
    bottoms = [0, seeds_c, seeds_c+fert_c, seeds_c+fert_c+labor_c]
    vals    = [seeds_c, fert_c, labor_c, misc_c]
    colors5 = ["#818cf8","#6366f1","#4f46e5","#3730a3"]
    lbls5   = ["Seeds","Fertilizer","Labor","Misc"]
    for bot, val, col, lbl in zip(bottoms, vals, colors5, lbls5):
        b = ax5.bar(cat, [val], bottom=[bot], color=col, label=lbl, edgecolor="white")
        ax5.text(0, bot+val/2, f" ₹{val:,.0f}", va="center", ha="left",
                 color="white", fontsize=8, fontweight="bold")
    ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"₹{v:,.0f}"))
    ax5.legend(loc="upper right", fontsize=8, framealpha=0.7)
    style_ax(ax5, "Cost per Acre per Cycle")
    fig5.tight_layout(); st.pyplot(fig5, use_container_width=True); plt.close(fig5)

with cb2:
    cost_df = pd.DataFrame({
        "Component": ["🌱 Seeds","🧪 Fertilizer","👷 Labor","🔧 Misc","🔢 Total"],
        "₹/Acre/Cycle": [fmt(seeds_c), fmt(fert_c), fmt(labor_c), fmt(misc_c), fmt(base_cost_pa)],
        "Share": ["25%","30%","35%","10%","100%"],
    })
    st.dataframe(cost_df, use_container_width=True, hide_index=True)
    annual_cost = base_cost_pa * cycles * acres
    st.info(f"**Annual Cost** ({acres}a × {cycles} cycles)\n\n{fmt(annual_cost)}")

# ─────────────────────────────────────────────────────────────
# INSIGHTS
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="strip">💡 Farmer Insights</div>', unsafe_allow_html=True)
ic1, ic2 = st.columns(2)
with ic1:
    st.markdown("**🌾 About Crop Farming**")
    if total_diff > 0:
        st.success(f"{crop} earns **{fmt(net_year*acres)}/year** net on {acres} acres. "
                   f"That's **{fmt(net_cycle)}/cycle/acre** after all costs. "
                   f"With {cycles} cycle(s)/year, your farm stays productive.")
    else:
        st.warning(f"{crop} earns **{fmt(net_year*acres)}/year** — currently less than leasing. "
                   f"Try a higher-value crop or Irrigated mode to boost income.")
with ic2:
    st.markdown("**🏠 About Leasing Land**")
    st.info(f"Leasing at {fmt(lease_per_acre)}/acre/year gives **{fmt(lease_per_acre*acres)}/year** "
            f"— guaranteed with zero farming effort & zero weather risk. "
            f"Good if you have other income or are temporarily away.")

st.markdown("**⚠️ Key Risk Factors**")
cols_rf = st.columns(2)
for i, rf in enumerate(risk_factors):
    cols_rf[i % 2].markdown(f"- {rf}")

# ─────────────────────────────────────────────────────────────
# WHAT-IF ANALYSIS (uses local calculation only — does NOT
# touch session_state, so the main report NEVER resets)
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="strip">🔄 What-If Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="whatif-card">', unsafe_allow_html=True)
st.markdown("Adjust sliders to explore different scenarios — the main report above stays unchanged.")

wa1, wa2, wa3 = st.columns(3)
with wa1:
    wi_price = st.slider("💲 Price (₹/quintal)", 500, 20000, int(price_used), step=100, key="wi_price")
with wa2:
    wi_cost  = st.slider("💸 Cost/Cycle/Acre (₹)", 1000, 80000, int(base_cost_pa), step=500, key="wi_cost")
with wa3:
    wi_yield = st.slider("🌾 Yield (kg/ha)", 500, 20000, int(yield_kg), step=100, key="wi_yield")

wi_net   = ((wi_yield/100)/2.47 * wi_price - wi_cost) * cycles
wi_total = wi_net * acres * years
wi_diff  = wi_total - total_lease
wi_marg  = abs(wi_diff/total_lease)*100 if total_lease > 0 else 0
wi_risk  = risk_level(wi_marg, cycles, irrigation, scenario)

wm1, wm2, wm3, wm4 = st.columns(4)
wm1.metric("🌾 Adjusted Crop Income", fmt(wi_total),
           f"{'▲' if wi_total>total_crop else '▼'} {abs(wi_total-total_crop)/max(total_crop,1)*100:.1f}% vs base")
wm2.metric("💰 vs Lease Income", fmt(abs(wi_diff)),
           "🌾 Farming wins" if wi_diff>0 else "🏠 Lease wins")
wm3.metric("📊 New Advantage", f"{wi_marg:.1f}%",
           f"{'↑ Better' if wi_marg>margin_pct else '↓ Lower'}")
wi_rcls = {"Low":"risk-low","Medium":"risk-medium","High":"risk-high"}[wi_risk]
wi_rico = {"Low":"🔵","Medium":"🟡","High":"🔴"}[wi_risk]
with wm4:
    st.markdown(f"**Adjusted Risk**")
    st.markdown(f'<span class="{wi_rcls}">{wi_rico} {wi_risk}</span>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# YEAR-BY-YEAR TABLE
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="strip">📋 Year-by-Year Breakdown</div>', unsafe_allow_html=True)
tbl_df = pd.DataFrame({
    "Period":               ylabels,
    "Crop Income":          [fmt(v) for v in crop_list],
    "Lease Income":         [fmt(v) for v in lease_list],
    "Cumulative Crop":      [fmt(v) for v in cum_crop],
    "Cumulative Lease":     [fmt(v) for v in cum_lease],
    "Better Option":        ["🌾 Farm" if c>l else "🏠 Lease" for c,l in zip(crop_list,lease_list)],
})
st.dataframe(tbl_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="strip">📥 Download Report</div>', unsafe_allow_html=True)
dl1, dl2 = st.columns(2)

# CSV
with dl1:
    rows = [
        ["AgriConnect+ — Farmer Income Analysis Report"],
        ["Generated", datetime.now().strftime("%d %B %Y, %I:%M %p")],
        [],
        ["PARAMETERS",""],
        ["State", state],["District", district],["Crop", crop],
        ["Irrigation", irrigation],["Years", years],["Acres", acres],
        ["Market Scenario", scenario],["Price (₹/q)", f"{price_used:,.0f}"],
        ["Yield (kg/ha)", f"{yield_kg:,.0f}"],["Cycles/Year", cycles],
        ["Cost/Cycle/Acre", f"{base_cost_pa:,}"],
        [],
        ["RESULTS",""],
        ["Total Crop Income", f"{total_crop:,.0f}"],
        ["Total Lease Income", f"{total_lease:,.0f}"],
        ["Difference", f"{abs(total_diff):,.0f}"],
        ["Advantage (%)", f"{margin_pct:.1f}%"],
        ["Risk Level", rlevel],["Best Option", better],
        [],
        ["COST BREAKDOWN",""],
        ["Seeds", f"{seeds_c:,.0f}"],["Fertilizer", f"{fert_c:,.0f}"],
        ["Labor", f"{labor_c:,.0f}"],["Misc", f"{misc_c:,.0f}"],
        [],
        ["YEAR-BY-YEAR",""],
        ["Period","Crop Income","Lease Income","Cumulative Crop","Cumulative Lease","Better"],
    ] + [
        [lbl, f"{ci:,.0f}", f"{li:,.0f}", f"{cc:,.0f}", f"{cl:,.0f}",
         "Farm" if ci>li else "Lease"]
        for lbl,ci,li,cc,cl in zip(ylabels,crop_list,lease_list,cum_crop,cum_lease)
    ]
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    st.download_button(
        "⬇️ Download CSV Report",
        buf.getvalue().encode("utf-8-sig"),
        f"AgriConnect_{crop}_{state}_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv", use_container_width=True
    )
    st.caption("Opens in Excel or Google Sheets")

# Text report
with dl2:
    sep = "=" * 58
    txt_lines = [
        sep,
        "    AGRICONNECT+  —  FARMER INCOME ANALYSIS REPORT",
        sep,
        f"  Date     : {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
        f"  Location : {location}",
        f"  Crop     : {crop}  |  {acres} acres  |  {years} year(s)",
        f"  Mode     : {irrigation}  |  {scenario}",
        "",
        "  SUMMARY","-"*58,
        f"  Crop Income  : Rs {total_crop:,.0f}",
        f"  Lease Income : Rs {total_lease:,.0f}",
        f"  Difference   : Rs {abs(total_diff):,.0f}  ({margin_pct:.1f}% advantage)",
        f"  Risk Level   : {rlevel}",
        f"  Best Option  : {better}",
        "",
        "  COST BREAKDOWN (Rs/acre/cycle)","-"*58,
        f"  Seeds      : Rs {seeds_c:,.0f}",
        f"  Fertilizer : Rs {fert_c:,.0f}",
        f"  Labor      : Rs {labor_c:,.0f}",
        f"  Misc       : Rs {misc_c:,.0f}",
        f"  TOTAL      : Rs {base_cost_pa:,.0f}",
        "",
        "  YEAR-BY-YEAR","-"*58,
        f"  {'Period':<8}  {'Crop':>14}  {'Lease':>14}  {'Better':>8}",
    ]
    for lbl,ci,li in zip(ylabels,crop_list,lease_list):
        txt_lines.append(f"  {lbl:<8}  Rs{ci:>12,.0f}  Rs{li:>12,.0f}  {'Farm' if ci>li else 'Lease':>8}")
    txt_lines += ["", "  RISK FACTORS","-"*58]
    for rf in risk_factors:
        txt_lines.append(f"  • {rf}")
    txt_lines += ["", sep, "  Powered by AgriConnect+  |  Objective 1", sep]
    st.download_button(
        "⬇️ Download Text Report",
        "\n".join(txt_lines).encode("utf-8"),
        f"AgriConnect_Report_{crop}_{datetime.now().strftime('%Y%m%d')}.txt",
        "text/plain", use_container_width=True
    )
    st.caption("Printable plain-text report")

st.markdown("---")
st.caption("🌾 AgriConnect+ · Objective 1 · Data: NITI Aayog, Mandi Price Database, Crop Yield Statistics")

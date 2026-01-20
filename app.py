import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import glob
import re
from sklearn.linear_model import LinearRegression

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Aadhaar Lifecycle Analysis",
    page_icon="🆔",
    layout="wide"
)

# ==================================================
# CUSTOM UI STYLING
# ==================================================
st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #1f3bb3;
}
.sub-title {
    font-size: 18px;
    color: #555;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# STATE NAME NORMALIZATION
# ==================================================
STATE_CANONICAL_MAP = {
    "westbengal": "West Bengal",
    "westbangal": "West Bengal",
    "westbengli": "West Bengal",
    "uttaranchal": "Uttarakhand",
    "uttarakhand": "Uttarakhand",
    "orissa": "Odisha",
    "odisha": "Odisha",
    "pondicherry": "Puducherry",
    "puducherry": "Puducherry",
    "tamilnadu": "Tamil Nadu",
    "andhrapradesh": "Andhra Pradesh",
    "madhyapradesh": "Madhya Pradesh",
    "uttarpradesh": "Uttar Pradesh",
    "jammuandkashmir": "Jammu and Kashmir"
}

def normalize_state_name(state):
    if pd.isna(state):
        return None
    s = str(state).lower()
    s = re.sub(r"[^a-z]", "", s)
    return STATE_CANONICAL_MAP.get(s, state.title())

# ==================================================
# DATA LOADER
# ==================================================
@st.cache_data
def load_folder_csvs(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ==================================================
# LOAD DATA
# ==================================================
enrol_df = load_folder_csvs("data/api_data_aadhar_enrolment")
demo_df  = load_folder_csvs("data/api_data_aadhar_demographic")
bio_df   = load_folder_csvs("data/api_data_aadhar_biometric")

if "state" in bio_df.columns:
    bio_df["state_clean"] = bio_df["state"].apply(normalize_state_name)

# ==================================================
# ML PREDICTION MODELS
# ==================================================
def linear_predict(values, steps):
    X = np.arange(len(values)).reshape(-1, 1)
    y = np.array(values)
    model = LinearRegression()
    model.fit(X, y)
    future_X = np.arange(len(values), len(values) + steps).reshape(-1, 1)
    return model.predict(future_X).astype(int)

def moving_average_predict(values, steps, window=3):
    values = list(values)
    preds = []
    for _ in range(steps):
        avg = int(np.mean(values[-window:]))
        preds.append(avg)
        values.append(avg)
    return preds

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("🆔 Aadhaar Dashboard")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Age-wise Enrolment",
        "Demographic Updates",
        "Biometric Lifecycle",
        "Regional Insights",
        "Smart Update Framework"
    ]
)

st.sidebar.markdown("### 🔮 Prediction Controls")

model_type = st.sidebar.selectbox(
    "Prediction Model",
    ["Linear Regression", "Moving Average"]
)

future_steps = st.sidebar.slider(
    "Future Cycles",
    1, 12, 3
)

# ==================================================
# OVERVIEW
# ==================================================
if menu == "Overview":
    st.markdown('<div class="main-title">Lifecycle-Based Aadhaar Update Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Aggregated • Anonymised • ML-driven</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Enrolment Records", f"{len(enrol_df):,}")
    c2.metric("Demographic Updates", f"{len(demo_df):,}")
    c3.metric("Biometric Updates", f"{len(bio_df):,}")

# ==================================================
# AGE-WISE ENROLMENT
# ==================================================
elif menu == "Age-wise Enrolment":
    st.header("👶🧑 Age-wise Aadhaar Enrolment")

    age_cols = ["age_0_5", "age_5_17", "age_18_greater"]
    data = enrol_df[age_cols].sum()

    preds = (
        linear_predict(data.values, future_steps)
        if model_type == "Linear Regression"
        else moving_average_predict(data.values, future_steps)
    )

    plot_df = pd.DataFrame({
        "Cycle": list(data.index) + [f"Future {i+1}" for i in range(future_steps)],
        "Enrolments": list(data.values) + list(preds)
    })

    fig = px.line(plot_df, x="Cycle", y="Enrolments", markers=True,
                  title=f"Enrolment Forecast ({model_type})")
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# DEMOGRAPHIC UPDATES
# ==================================================
elif menu == "Demographic Updates":
    st.header("🧾 Demographic Update Forecast")

    demo_cols = ["demo_age_5_17", "demo_age_17_"]
    data = demo_df[demo_cols].sum()

    preds = (
        linear_predict(data.values, future_steps)
        if model_type == "Linear Regression"
        else moving_average_predict(data.values, future_steps)
    )

    plot_df = pd.DataFrame({
        "Cycle": list(data.index) + [f"Future {i+1}" for i in range(future_steps)],
        "Updates": list(data.values) + list(preds)
    })

    fig = px.bar(plot_df, x="Cycle", y="Updates",
                 title=f"Demographic Update Forecast ({model_type})")
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# BIOMETRIC LIFECYCLE
# ==================================================
elif menu == "Biometric Lifecycle":
    st.header("🧬 Biometric Lifecycle Forecast")

    age_cols = [c for c in bio_df.columns if "age" in c.lower()]
    data = bio_df[age_cols].sum()

    preds = (
        linear_predict(data.values, future_steps)
        if model_type == "Linear Regression"
        else moving_average_predict(data.values, future_steps)
    )

    plot_df = pd.DataFrame({
        "Cycle": list(data.index) + [f"Future {i+1}" for i in range(future_steps)],
        "Updates": list(data.values) + list(preds)
    })

    fig = px.line(plot_df, x="Cycle", y="Updates", markers=True,
                  title=f"Biometric Update Forecast ({model_type})")
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# REGIONAL INSIGHTS (CLEAN STATES)
# ==================================================
elif menu == "Regional Insights":
    st.header("🗺️ Regional Insights (Cleaned State Names)")

    states = sorted(bio_df["state_clean"].dropna().unique())
    selected = st.selectbox("Select State", states)

    df = bio_df[bio_df["state_clean"] == selected]
    district_summary = df.groupby("district").size().reset_index(name="Records")

    fig = px.bar(
        district_summary,
        x="district",
        y="Records",
        title=f"Biometric Updates – {selected}"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# SMART UPDATE FRAMEWORK
# ==================================================
elif menu == "Smart Update Framework":
    st.header("🔔 Smart Aadhaar Update Framework")

    st.markdown("""
    **ML-powered lifecycle system** that:
    - Predicts update demand
    - Detects regional risk
    - Enables proactive notifications
    - Preserves citizen privacy
    """)

    st.success("✔ Policy-ready • ✔ Scalable • ✔ Privacy-by-design")

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.caption("🆔 Aadhaar Lifecycle Dashboard | Hackathon Project | Aggregated UIDAI Data")

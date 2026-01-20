import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob

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
.card {
    background-color: #f5f7ff;
    padding: 18px;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# DATA LOADER (SINGLE, SAFE)
# ==================================================
@st.cache_data
def load_folder_csvs(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            st.warning(f"Failed to load {f}: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ==================================================
# LOAD DATA
# ==================================================
enrol_df = load_folder_csvs("data/api_data_aadhar_enrolment")
demo_df  = load_folder_csvs("data/api_data_aadhar_demographic")
bio_df   = load_folder_csvs("data/api_data_aadhar_biometric")

# ==================================================
# SIDEBAR NAVIGATION
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

# ==================================================
# OVERVIEW
# ==================================================
if menu == "Overview":
    st.markdown('<div class="main-title">Lifecycle-Based Aadhaar Update Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Aggregated | Anonymised | Lifecycle-driven</div>', unsafe_allow_html=True)

    st.markdown("### 🎯 Project Objectives")
    st.markdown("""
    - Analyse age-wise Aadhaar enrolment patterns  
    - Identify lifecycle-based update peaks  
    - Detect regional update disparities  
    - Propose a smart, proactive update framework  
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Enrolment Records", f"{len(enrol_df):,}")
    c2.metric("Demographic Update Records", f"{len(demo_df):,}")
    c3.metric("Biometric Update Records", f"{len(bio_df):,}")

# ==================================================
# AGE-WISE ENROLMENT
# ==================================================
elif menu == "Age-wise Enrolment":
    st.header("👶🧑 Age-wise Aadhaar Enrolment")

    age_cols = ["age_0_5", "age_5_17", "age_18_greater"]

    if enrol_df.empty or not all(c in enrol_df.columns for c in age_cols):
        st.error("Required age columns not found in enrolment data.")
    else:
        age_summary = enrol_df[age_cols].sum().reset_index()
        age_summary.columns = ["Age Group", "Total Enrolments"]

        fig = px.bar(
            age_summary,
            x="Age Group",
            y="Total Enrolments",
            text_auto=True,
            title="Age-wise Aadhaar Enrolment Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Higher enrolment volumes are observed during early childhood "
            "and adulthood, reflecting key lifecycle registration phases."
        )

# ==================================================
# DEMOGRAPHIC UPDATES
# ==================================================
elif menu == "Demographic Updates":
    st.header("🧾 Demographic Update Analysis")

    demo_cols = ["demo_age_5_17", "demo_age_17_"]

    if demo_df.empty or not all(c in demo_df.columns for c in demo_cols):
        st.error("Required demographic age columns not found.")
    else:
        demo_summary = demo_df[demo_cols].sum().reset_index()
        demo_summary.columns = ["Age Group", "Demographic Updates"]

        fig = px.bar(
            demo_summary,
            x="Age Group",
            y="Demographic Updates",
            text_auto=True,
            title="Age-wise Demographic Updates"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.success(
            "A sharp rise in demographic updates after age 17 highlights "
            "identity detail changes during adulthood."
        )

# ==================================================
# BIOMETRIC LIFECYCLE
# ==================================================
elif menu == "Biometric Lifecycle":
    st.header("🧬 Biometric Update Lifecycle")

    bio_age_cols = [c for c in bio_df.columns if "age" in c.lower()]

    if bio_df.empty or not bio_age_cols:
        st.error("No biometric age columns found.")
    else:
        bio_summary = bio_df[bio_age_cols].sum().reset_index()
        bio_summary.columns = ["Age Group", "Biometric Updates"]

        fig = px.line(
            bio_summary,
            x="Age Group",
            y="Biometric Updates",
            markers=True,
            title="Biometric Update Peaks Across Lifecycle"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Biometric updates peak during adolescence and early adulthood, "
            "indicating mandatory compliance transition phases."
        )

# ==================================================
# REGIONAL INSIGHTS
# ==================================================
elif menu == "Regional Insights":
    st.header("🗺️ Regional Update Insights")

    if bio_df.empty or "state" not in bio_df.columns or "district" not in bio_df.columns:
        st.error("State or district columns missing in biometric data.")
    else:
        states = sorted(bio_df["state"].dropna().unique())
        selected_state = st.selectbox("Select State", states)

        state_df = bio_df[bio_df["state"] == selected_state]
        district_summary = (
            state_df.groupby("district")
            .size()
            .reset_index(name="Update Records")
        )

        fig = px.bar(
            district_summary,
            x="district",
            y="Update Records",
            title=f"Biometric Updates by District – {selected_state}"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.warning(
            "Districts with lower update activity may indicate "
            "service access gaps or awareness issues."
        )

# ==================================================
# SMART UPDATE FRAMEWORK
# ==================================================
elif menu == "Smart Update Framework":
    st.header("🔔 Smart Aadhaar Update Framework")

    st.markdown("""
    ### 💡 Concept
    A **lifecycle-aware, data-driven reminder system** that ensures timely Aadhaar updates
    **without using personal data**.
    """)

    st.markdown("### 🧠 Core Components")
    st.markdown("""
    1. Lifecycle Trigger Engine  
    2. Regional Risk Scoring  
    3. Update Demand Forecasting  
    4. Citizen Notification Layer  
    """)

    st.success("""
    ✅ Reduced update backlog  
    ✅ Improved Aadhaar accuracy  
    ✅ Better citizen experience  
    ✅ Optimised service planning  
    """)

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.caption(
    "🆔 Aadhaar Lifecycle Analysis | Hackathon Project | "
    "All insights derived from anonymised, aggregated UIDAI data"
)

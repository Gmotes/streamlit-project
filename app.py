import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="FinGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom styling ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #4f8bf9;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        color: #4f8bf9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CHURN_PATH = "data/Churn_Modelling.csv"
PAYSIM_PATH = "data/PaySim.csv"
PAYSIM_SAMPLE = 100_000


# ─── Data loaders (cached) ───────────────────────────────────────────────────────
@st.cache_data
def load_churn():
    df = pd.read_csv(CHURN_PATH)
    df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
    return df


@st.cache_data
def load_paysim(n=PAYSIM_SAMPLE):
    df = pd.read_csv(PAYSIM_PATH, nrows=n)
    return df


# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/shield.png",
        width=60,
    )
    st.title("FinGuard AI")
    st.markdown("Financial Intelligence Dashboard")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏦 Churn Analysis", "🚨 Fraud Detection"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Data sources")
    st.caption("• Churn Modelling – 10,000 customers")
    st.caption(f"• PaySim – {PAYSIM_SAMPLE:,} transactions (sampled)")


# ═══════════════════════════════════════════════════════════════════════════════════
# PAGE 1 – CHURN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════════
if page == "🏦 Churn Analysis":
    st.title("🏦 Customer Churn Analysis")
    st.markdown("Exploring churn patterns across the bank's customer base.")

    df = load_churn()

    # ── Sidebar filters ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Filters")
        geo_options = ["All"] + sorted(df["Geography"].unique().tolist())
        selected_geo = st.selectbox("Geography", geo_options)
        gender_options = ["All"] + sorted(df["Gender"].unique().tolist())
        selected_gender = st.selectbox("Gender", gender_options)
        age_range = st.slider(
            "Age range",
            int(df["Age"].min()),
            int(df["Age"].max()),
            (int(df["Age"].min()), int(df["Age"].max())),
        )

    filtered = df.copy()
    if selected_geo != "All":
        filtered = filtered[filtered["Geography"] == selected_geo]
    if selected_gender != "All":
        filtered = filtered[filtered["Gender"] == selected_gender]
    filtered = filtered[filtered["Age"].between(*age_range)]

    # ── KPI row ──────────────────────────────────────────────────────────────────
    total = len(filtered)
    churned = filtered["Exited"].sum()
    churn_rate = churned / total * 100 if total else 0
    avg_balance = filtered["Balance"].mean()
    avg_credit = filtered["CreditScore"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", f"{total:,}")
    k2.metric("Churned", f"{int(churned):,}", f"{churn_rate:.1f}%")
    k3.metric("Avg Balance", f"${avg_balance:,.0f}")
    k4.metric("Avg Credit Score", f"{avg_credit:.0f}")

    st.divider()

    # ── Row 1: Geography & Gender ────────────────────────────────────────────────
    st.markdown('<div class="section-title">Geographic & Demographic Breakdown</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        geo_churn = (
            filtered.groupby("Geography")["Exited"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "Churned", "count": "Total"})
        )
        geo_churn["Churn Rate"] = geo_churn["Churned"] / geo_churn["Total"] * 100
        fig = px.bar(
            geo_churn,
            x="Geography",
            y="Churn Rate",
            color="Churn Rate",
            color_continuous_scale="RdYlGn_r",
            title="Churn Rate by Geography",
            text=geo_churn["Churn Rate"].apply(lambda x: f"{x:.1f}%"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        gender_churn = filtered.groupby(["Gender", "Exited"]).size().reset_index(name="Count")
        gender_churn["Status"] = gender_churn["Exited"].map({0: "Retained", 1: "Churned"})
        fig = px.bar(
            gender_churn,
            x="Gender",
            y="Count",
            color="Status",
            barmode="group",
            title="Churn by Gender",
            color_discrete_map={"Retained": "#2ecc71", "Churned": "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        active_churn = filtered.groupby(["IsActiveMember", "Exited"]).size().reset_index(name="Count")
        active_churn["Member"] = active_churn["IsActiveMember"].map({0: "Inactive", 1: "Active"})
        active_churn["Status"] = active_churn["Exited"].map({0: "Retained", 1: "Churned"})
        fig = px.bar(
            active_churn,
            x="Member",
            y="Count",
            color="Status",
            barmode="group",
            title="Active vs Inactive Member Churn",
            color_discrete_map={"Retained": "#2ecc71", "Churned": "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Age & Balance distributions ───────────────────────────────────────
    st.markdown('<div class="section-title">Age & Balance Distributions</div>', unsafe_allow_html=True)
    c4, c5 = st.columns(2)

    with c4:
        fig = px.histogram(
            filtered,
            x="Age",
            color=filtered["Exited"].map({0: "Retained", 1: "Churned"}),
            nbins=30,
            barmode="overlay",
            opacity=0.7,
            title="Age Distribution by Churn Status",
            labels={"color": "Status"},
            color_discrete_map={"Retained": "#2ecc71", "Churned": "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with c5:
        fig = px.box(
            filtered,
            x=filtered["Exited"].map({0: "Retained", 1: "Churned"}),
            y="Balance",
            color=filtered["Exited"].map({0: "Retained", 1: "Churned"}),
            title="Balance Distribution by Churn Status",
            labels={"x": "Status", "color": "Status"},
            color_discrete_map={"Retained": "#2ecc71", "Churned": "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Products & Credit Score ───────────────────────────────────────────
    st.markdown('<div class="section-title">Products & Credit</div>', unsafe_allow_html=True)
    c6, c7 = st.columns(2)

    with c6:
        prod_churn = filtered.groupby(["NumOfProducts", "Exited"]).size().reset_index(name="Count")
        prod_churn["Status"] = prod_churn["Exited"].map({0: "Retained", 1: "Churned"})
        fig = px.bar(
            prod_churn,
            x="NumOfProducts",
            y="Count",
            color="Status",
            barmode="group",
            title="Churn by Number of Products",
            color_discrete_map={"Retained": "#2ecc71", "Churned": "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with c7:
        fig = px.histogram(
            filtered,
            x="CreditScore",
            color=filtered["Exited"].map({0: "Retained", 1: "Churned"}),
            nbins=30,
            barmode="overlay",
            opacity=0.7,
            title="Credit Score Distribution by Churn Status",
            labels={"color": "Status"},
            color_discrete_map={"Retained": "#2ecc71", "Churned": "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Correlation heatmap ────────────────────────────────────────────────
    st.markdown('<div class="section-title">Correlation with Churn</div>', unsafe_allow_html=True)
    num_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"]
    corr = filtered[num_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix",
        aspect="auto",
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # ── Raw data toggle ────────────────────────────────────────────────────────────
    with st.expander("View raw data"):
        st.dataframe(filtered.reset_index(drop=True), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════════
# PAGE 2 – FRAUD DETECTION
# ═══════════════════════════════════════════════════════════════════════════════════
else:
    st.title("🚨 Fraud Detection Analysis")
    st.markdown(
        f"Exploring transaction fraud patterns. Showing first **{PAYSIM_SAMPLE:,}** rows of PaySim dataset."
    )

    df = load_paysim()

    # ── Sidebar filters ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Filters")
        tx_types = ["All"] + sorted(df["type"].unique().tolist())
        selected_type = st.selectbox("Transaction Type", tx_types)
        show_fraud_only = st.checkbox("Show fraud only", value=False)

    filtered = df.copy()
    if selected_type != "All":
        filtered = filtered[filtered["type"] == selected_type]
    if show_fraud_only:
        filtered = filtered[filtered["isFraud"] == 1]

    # ── KPIs ──────────────────────────────────────────────────────────────────────
    total_tx = len(filtered)
    fraud_count = filtered["isFraud"].sum()
    fraud_rate = fraud_count / total_tx * 100 if total_tx else 0
    total_fraud_amt = filtered[filtered["isFraud"] == 1]["amount"].sum()
    avg_tx_amt = filtered["amount"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Transactions", f"{total_tx:,}")
    k2.metric("Fraudulent", f"{int(fraud_count):,}", f"{fraud_rate:.2f}%")
    k3.metric("Total Fraud Amount", f"${total_fraud_amt:,.0f}")
    k4.metric("Avg Transaction Amount", f"${avg_tx_amt:,.0f}")

    st.divider()

    # ── Row 1: Transaction type breakdown ────────────────────────────────────────
    st.markdown('<div class="section-title">Transaction Type Breakdown</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        type_counts = filtered["type"].value_counts().reset_index()
        type_counts.columns = ["type", "count"]
        fig = px.pie(
            type_counts,
            names="type",
            values="count",
            title="Transaction Distribution by Type",
            color_discrete_sequence=px.colors.qualitative.Safe,
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        type_fraud = (
            filtered.groupby("type")["isFraud"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "Fraud", "count": "Total"})
        )
        type_fraud["Fraud Rate (%)"] = type_fraud["Fraud"] / type_fraud["Total"] * 100
        fig = px.bar(
            type_fraud,
            x="type",
            y="Fraud Rate (%)",
            color="Fraud Rate (%)",
            color_continuous_scale="Reds",
            title="Fraud Rate by Transaction Type",
            text=type_fraud["Fraud Rate (%)"].apply(lambda x: f"{x:.2f}%"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Amount distributions ──────────────────────────────────────────────
    st.markdown('<div class="section-title">Transaction Amount Patterns</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        fig = px.histogram(
            filtered,
            x="amount",
            color=filtered["isFraud"].map({0: "Legitimate", 1: "Fraud"}),
            nbins=50,
            barmode="overlay",
            opacity=0.7,
            log_y=True,
            title="Transaction Amount Distribution (log scale)",
            labels={"color": "Status"},
            color_discrete_map={"Legitimate": "#3498db", "Fraud": "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.box(
            filtered,
            x=filtered["isFraud"].map({0: "Legitimate", 1: "Fraud"}),
            y="amount",
            color=filtered["isFraud"].map({0: "Legitimate", 1: "Fraud"}),
            log_y=True,
            title="Amount Box Plot by Fraud Status (log scale)",
            labels={"x": "Status", "color": "Status"},
            color_discrete_map={"Legitimate": "#3498db", "Fraud": "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Balance change analysis ───────────────────────────────────────────
    st.markdown('<div class="section-title">Balance Change Analysis</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)

    with c5:
        sample = filtered.sample(min(3000, len(filtered)), random_state=42)
        fig = px.scatter(
            sample,
            x="oldbalanceOrg",
            y="newbalanceOrig",
            color=sample["isFraud"].map({0: "Legitimate", 1: "Fraud"}),
            opacity=0.4,
            title="Originator: Old vs New Balance",
            labels={"color": "Status"},
            color_discrete_map={"Legitimate": "#3498db", "Fraud": "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        type_fraud_amt = (
            filtered[filtered["isFraud"] == 1]
            .groupby("type")["amount"]
            .sum()
            .reset_index()
            .rename(columns={"amount": "Total Fraud Amount"})
            .sort_values("Total Fraud Amount", ascending=True)
        )
        fig = px.bar(
            type_fraud_amt,
            x="Total Fraud Amount",
            y="type",
            orientation="h",
            title="Total Fraud Amount by Transaction Type",
            color="Total Fraud Amount",
            color_continuous_scale="Reds",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Flagged vs actual fraud ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">Flagging Effectiveness</div>', unsafe_allow_html=True)
    flag_matrix = pd.crosstab(
        filtered["isFraud"],
        filtered["isFlaggedFraud"],
        rownames=["Actual Fraud"],
        colnames=["Flagged Fraud"],
    )
    fig = px.imshow(
        flag_matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix: isFraud vs isFlaggedFraud",
        labels={"x": "Flagged as Fraud", "y": "Actual Fraud"},
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # ── Raw data toggle ────────────────────────────────────────────────────────────
    with st.expander("View raw data"):
        st.dataframe(filtered.head(500).reset_index(drop=True), use_container_width=True)

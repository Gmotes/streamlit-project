import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(
    page_title="Financial & Customer Analytics Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------------------------------------------------------
# 1. CACHED DATA LOADING & PROCESSING FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_churn_data():
    """Loads and caches the customer churn dataset."""
    return pd.read_csv("data/Churn_Modelling.csv")


@st.cache_data(ttl=3600)
def load_paysim_data():
    """Loads and caches the PaySim fraud detection dataset."""
    return pd.read_csv("data/PaySim.csv")


@st.cache_data(ttl=3600)
def process_churn_analytics(df):
    """Trains a quick predictive model to generate feature importances and personalized risk scores."""
    features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                'EstimatedSalary']
    X = df[features]
    y = df['Exited']

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)

    df_scored = df.copy()
    df_scored['Churn_Probability'] = rf.predict_proba(X)[:, 1]
    df_scored['Churn_Score'] = (df_scored['Churn_Probability'] * 100).round(2)

    def get_risk_tier(prob):
        if prob < 0.30:
            return 'Low Risk'
        elif prob < 0.70:
            return 'Medium Risk'
        else:
            return 'High Risk'

    df_scored['Risk_Tier'] = df_scored['Churn_Probability'].apply(get_risk_tier)

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=True)

    return df_scored, importance_df


@st.cache_data(ttl=3600)
def process_fraud_segmentation(df):
    """Processes fraud segments based on transaction sizes."""
    df_fraud = df.copy()

    # Calculate quantiles for fraud transactions to form risk segment thresholds dynamically
    fraud_amounts = df_fraud[df_fraud['isFraud'] == 1]['amount']
    if not fraud_amounts.empty:
        q1 = fraud_amounts.quantile(0.33)
        q2 = fraud_amounts.quantile(0.66)
    else:
        q1, q2 = 10000, 100000  # Fallbacks

    def segment_amount_risk(amt):
        if amt <= q1:
            return 'Tier 1: Low-Value Risk'
        elif amt <= q2:
            return 'Tier 2: Mid-Value Risk'
        else:
            return 'Tier 3: High-Value Risk'

    df_fraud['Fraud_Risk_Segment'] = df_fraud['amount'].apply(segment_amount_risk)
    return df_fraud


# Safe initialization of cached datasets
try:
    churn_raw = load_churn_data()
    paysim_raw = load_paysim_data()
    churn_df, feature_importance_df = process_churn_analytics(churn_raw)
    paysim_df = process_fraud_segmentation(paysim_raw)
except Exception as e:
    st.error(
        f"Error loading datasets. Please ensure 'Churn_Modelling.csv' and 'PaySim.csv' are in the working directory. Details: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# 2. DEFINITIONS FOR THE 5 PAGES
# -----------------------------------------------------------------------------

def show_overview_page():
    st.title("📊 Executive KPI Overview Dashboard")
    st.markdown(
        "Welcome to the central command dashboard. Below are the key performance metrics computed dynamically from your cached datasets.")

    total_customers = int(churn_df['CustomerId'].nunique())
    churn_rate = float(churn_df['Exited'].mean() * 100)
    active_rate = float(churn_df['IsActiveMember'].mean() * 100)
    fraud_rate = float(paysim_df['isFraud'].mean() * 100)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Customers", value=f"{total_customers:,}")
    with col2:
        st.metric(label="Customer Churn Rate", value=f"{churn_rate:.2f}%", delta=f"-{churn_rate:.1f}%",
                  delta_color="inverse")
    with col3:
        st.metric(label="Active Customer Rate", value=f"{active_rate:.2f}%", delta=f"{active_rate:.1f}%")
    with col4:
        st.metric(label="Transaction Fraud Rate", value=f"{fraud_rate:.2f}%", delta=f"-{fraud_rate:.2f}%",
                  delta_color="inverse")

    st.divider()
    st.subheader("📋 Dataset Previews (Loaded from Cache)")
    tab1, tab2 = st.tabs(["Churn Dataset Preview", "PaySim Transaction Dataset Preview"])
    with tab1:
        st.dataframe(churn_df.head(10), use_container_width=True)
    with tab2:
        st.dataframe(paysim_df.head(10), use_container_width=True)


def show_churn_analysis_page():
    st.title("📉 Page 2: Customer Churn Deep-Dive")
    st.markdown(
        "This module evaluates organizational churn liabilities using predictive distributions, group segmentations, feature importances, and discrete customer score lookups.")

    st.header("1. Churn Risk Distribution")
    dist_col1, dist_col2 = st.columns([1, 2])
    with dist_col1:
        tier_counts = churn_df['Risk_Tier'].value_counts()
        for tier in ['Low Risk', 'Medium Risk', 'High Risk']:
            count = tier_counts.get(tier, 0)
            percentage = (count / len(churn_df)) * 100
            st.metric(label=tier, value=f"{count:,}", delta=f"{percentage:.1f}% of total")
    with dist_col2:
        counts, bins = np.histogram(churn_df['Churn_Score'], bins=10, range=(0, 100))
        hist_df = pd.DataFrame({'Risk Range': [f"{int(bins[i])}-{int(bins[i + 1])}%" for i in range(len(bins) - 1)],
                                'Customer Count': counts}).set_index('Risk Range')
        st.bar_chart(hist_df, y='Customer Count', color="#FF4B4B")

    st.divider()
    st.header("2. Churn Segmentations")
    seg_tab1, seg_tab2 = st.tabs(["Geography Segmentation", "Product Portfolio Volatility"])
    with seg_tab1:
        geo_df = pd.DataFrame({'Churn Rate (%)': (churn_df.groupby('Geography')['Exited'].mean() * 100).round(2)})
        st.dataframe(geo_df, use_container_width=True)
        st.bar_chart(geo_df)
    with seg_tab2:
        prod_df = pd.DataFrame({'Churn Rate (%)': (churn_df.groupby('NumOfProducts')['Exited'].mean() * 100).round(2)})
        st.dataframe(prod_df, use_container_width=True)
        st.bar_chart(prod_df)

    st.divider()
    st.header("3. Churn Feature Importance")
    st.bar_chart(feature_importance_df.set_index('Feature'), y='Importance', color="#29B5E8")

    st.divider()
    st.header("4. Customer Based Churn Score Lookup")
    search_id = st.number_input("Enter Unique Customer ID:", min_value=int(churn_df['CustomerId'].min()),
                                max_value=int(churn_df['CustomerId'].max()), value=15634602)
    customer_record = churn_df[churn_df['CustomerId'] == search_id]
    if not customer_record.empty:
        st.dataframe(customer_record[
                         ['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Churn_Score',
                          'Risk_Tier']], use_container_width=True)
    else:
        st.warning("No record matches the provided Customer ID.")


def show_fraud_insights_page():
    st.title("🔒 Page 3: Transaction Fraud Analysis")
    st.markdown(
        "This updated module tracks financial transactional vectors from cache to systematically isolate fraudulent activity trends.")

    # -------------------------------------------------------------------------
    # PARAMETER 1: Fraud Rate
    # -------------------------------------------------------------------------
    st.header("1. Global System Fraud Rate")

    total_tx = len(paysim_df)
    fraud_tx_count = int(paysim_df['isFraud'].sum())
    global_fraud_rate = (fraud_tx_count / total_tx) * 100
    total_fraud_volume = paysim_df[paysim_df['isFraud'] == 1]['amount'].sum()

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        st.metric(label="Total Logged Transactions", value=f"{total_tx:,}")
    with f_col2:
        st.metric(label="Identified Fraud Incident Records", value=f"{fraud_tx_count:,}",
                  delta=f"{global_fraud_rate:.3f}% Fraud Rate", delta_color="inverse")
    with f_col3:
        st.metric(label="Total Capital Impact At Risk", value=f"${total_fraud_volume:,.2f}")

    st.divider()

    # -------------------------------------------------------------------------
    # PARAMETER 2: Fraud by Transaction Type
    # -------------------------------------------------------------------------
    st.header("2. Fraud Incidence by Transaction Type")

    # Calculate incidence count and mean rate per category type
    type_metrics = paysim_df.groupby('type').agg(
        Total_Transactions=('isFraud', 'count'),
        Fraud_Incidents=('isFraud', 'sum'),
        Fraud_Rate_Percentage=('isFraud', lambda x: (x.mean() * 100).round(4))
    ).sort_values(by='Fraud_Incidents', ascending=False)

    t_col1, t_col2 = st.columns([1, 1])
    with t_col1:
        st.markdown("**Transaction Type Matrix Table**")
        st.dataframe(type_metrics, use_container_width=True)
    with t_col2:
        st.markdown("**Fraud Count Distribution by Type**")
        st.bar_chart(type_metrics, y='Fraud_Incidents', color="#FF9F1C")

    st.divider()

    # -------------------------------------------------------------------------
    # PARAMETER 3: Fraud Risk Segmentation
    # -------------------------------------------------------------------------
    st.header("3. Fraud Risk Amount Segmentation")
    st.markdown(
        "Transactions grouped into value brackets based on transaction sizing distribution to pinpoint focus fields for risk audit teams.")

    # Group by the generated
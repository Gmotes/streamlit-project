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

    # Train Random Forest Classifier from cache
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)

    # Calculate custom metrics
    df_scored = df.copy()
    df_scored['Churn_Probability'] = rf.predict_proba(X)[:, 1]
    df_scored['Churn_Score'] = (df_scored['Churn_Probability'] * 100).round(2)

    # Bin into Risk Tiers
    def get_risk_tier(prob):
        if prob < 0.30:
            return 'Low Risk'
        elif prob < 0.70:
            return 'Medium Risk'
        else:
            return 'High Risk'

    df_scored['Risk_Tier'] = df_scored['Churn_Probability'].apply(get_risk_tier)

    # Feature Importances Dataframe
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=True)  # Ascending for nice horizontal bar charts

    return df_scored, importance_df


# Safe initialization of cached datasets
try:
    churn_raw = load_churn_data()
    paysim_df = load_paysim_data()
    churn_df, feature_importance_df = process_churn_analytics(churn_raw)
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

    # Calculate Metrics
    total_customers = int(churn_df['CustomerId'].nunique())
    churn_rate = float(churn_df['Exited'].mean() * 100)
    active_rate = float(churn_df['IsActiveMember'].mean() * 100)
    fraud_rate = float(paysim_df['isFraud'].mean() * 100)

    # Display Metrics in Columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Customers", value=f"{total_customers:,}",
                  help="Total unique customer identification numbers.")
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
        "This enhanced page evaluates organizational churn liabilities using predictive distributions, group segmentations, feature importances, and discrete customer score lookups.")

    # -------------------------------------------------------------------------
    # PARAMETER 1: Churn Risk Distribution
    # -------------------------------------------------------------------------
    st.header("1. Churn Risk Distribution")
    dist_col1, dist_col2 = st.columns([1, 2])

    with dist_col1:
        st.markdown("**Risk Tier Composition**")
        tier_counts = churn_df['Risk_Tier'].value_counts()
        for tier in ['Low Risk', 'Medium Risk', 'High Risk']:
            count = tier_counts.get(tier, 0)
            percentage = (count / len(churn_df)) * 100
            st.metric(label=tier, value=f"{count:,}", delta=f"{percentage:.1f}% of total")

    with dist_col2:
        st.markdown("**Risk Score Histogram (0% - 100%)**")
        # Calculate histogram bins using numpy
        counts, bins = np.histogram(churn_df['Churn_Score'], bins=10, range=(0, 100))
        hist_df = pd.DataFrame({
            'Risk Range': [f"{int(bins[i])}-{int(bins[i + 1])}%" for i in range(len(bins) - 1)],
            'Customer Count': counts
        }).set_index('Risk Range')
        st.bar_chart(hist_df, y='Customer Count', color="#FF4B4B")

    st.divider()

    # -------------------------------------------------------------------------
    # PARAMETER 2: Churn Segmentations
    # -------------------------------------------------------------------------
    st.header("2. Churn Segmentations")
    st.markdown("Analyze how the realized churn rate varies across diverse categorical attributes.")

    seg_tab1, seg_tab2, seg_tab3 = st.tabs(
        ["Geography Segmentation", "Product Portfolio Volatility", "Age Demographics"])

    with seg_tab1:
        geo_seg = churn_df.groupby('Geography')['Exited'].mean() * 100
        geo_df = pd.DataFrame({'Churn Rate (%)': geo_seg.round(2)})
        st.dataframe(geo_df, use_container_width=True)
        st.bar_chart(geo_df)

    with seg_tab2:
        prod_seg = churn_df.groupby('NumOfProducts')['Exited'].mean() * 100
        prod_df = pd.DataFrame({'Churn Rate (%)': prod_seg.round(2)})
        st.dataframe(prod_df, use_container_width=True)
        st.bar_chart(prod_df)

    with seg_tab3:
        # Create dynamic age groupings
        churn_df['Age_Group'] = pd.cut(churn_df['Age'], bins=[0, 30, 40, 50, 60, 100],
                                       labels=['<30', '30-40', '40-50', '50-60', '60+'])
        age_seg = churn_df.groupby('Age_Group', observed=False)['Exited'].mean() * 100
        age_df = pd.DataFrame({'Churn Rate (%)': age_seg.round(2)})
        st.dataframe(age_df, use_container_width=True)
        st.bar_chart(age_df)

    st.divider()

    # -------------------------------------------------------------------------
    # PARAMETER 3: Churn Feature Importance
    # -------------------------------------------------------------------------
    st.header("3. Churn Feature Importance")
    st.markdown(
        "The machine learning model evaluated the following structural feature coefficients to isolate the primary triggers of customer churn.")

    chart_data = feature_importance_df.set_index('Feature')
    st.bar_chart(chart_data, y='Importance', color="#29B5E8")

    st.divider()

    # -------------------------------------------------------------------------
    # PARAMETER 4: Customer Based Churn Score Lookup
    # -------------------------------------------------------------------------
    st.header("4. Customer Based Churn Score Lookup")
    st.markdown("Search for an individual customer record below to audit their real-time calculated risk profile.")

    search_col1, search_col2 = st.columns([1, 2])

    with search_col1:
        search_id = st.number_input("Enter Unique Customer ID:", min_value=int(churn_df['CustomerId'].min()),
                                    max_value=int(churn_df['CustomerId'].max()), value=15634602)

    customer_record = churn_df[churn_df['CustomerId'] == search_id]

    if not customer_record.empty:
        with search_col2:
            score = float(customer_record['Churn_Score'].iloc[0])
            tier = str(customer_record['Risk_Tier'].iloc[0])
            surname = str(customer_record['Surname'].iloc[0])

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Customer Name", surname)
            sc2.metric("Calculated Churn Score", f"{score}%")
            sc3.metric("Assigned Risk Bucket", tier)

        st.markdown("**Complete Analytical Profile Vector For Customer:**")
        st.dataframe(customer_record[
                         ['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                          'NumOfProducts', 'IsActiveMember', 'Churn_Score', 'Risk_Tier']], use_container_width=True)
    else:
        st.warning("No record matches the provided Customer ID. Please review the dataset identifier values.")


def show_fraud_insights_page():
    st.title("🔒 Page 3: Transaction Fraud Analysis")
    st.markdown("Granular insights into suspicious transfers, transaction volumes, and flagged system alerts.")
    type_filter = st.selectbox("Select Transaction Type", options=paysim_df['type'].unique())
    filtered_paysim = paysim_df[paysim_df['type'] == type_filter]
    st.dataframe(filtered_paysim.head(50), use_container_width=True)


def show_demographics_page():
    st.title("👥 Page 4: Customer Demographics & Behavior")
    st.markdown("Exploration of gender distributions, age buckets, tenure cycles, and asset portfolios.")
    age_slider = st.slider("Filter Customer Age Range", int(churn_df['Age'].min()), int(churn_df['Age'].max()),
                           (25, 50))
    filtered_demo = churn_df[(churn_df['Age'] >= age_slider[0]) & (churn_df['Age'] <= age_slider[1])]
    st.metric(label="Customers in Age Bracket", value=len(filtered_demo))
    st.dataframe(filtered_demo.head(20), use_container_width=True)


def show_predictive_modeling_page():
    st.title("🤖 Page 5: Predictive Modeling & Simulation")
    st.markdown("What-if scenario simulators or machine learning inference templates for predictive risk mitigation.")
    st.success(
        "This placeholder page can be used to load pickled ML models to run real-time churn predictions or fraud risk scoring profiles based on custom inputs.")
    st.number_input("Input Sample Transaction Amount ($)", min_value=0.0, max_value=1000000.0, value=500.0)
    st.button("Run Simulation Risk Assessment")


# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION CONTROLLER
# -----------------------------------------------------------------------------
st.sidebar.title("Navigation Menu")
st.sidebar.markdown("Navigate across the 5 analytical modules below:")

page_selection = st.sidebar.radio(
    "Select a Page:",
    [
        "1. Executive KPI Overview",
        "2. Customer Churn Deep-Dive",
        "3. Transaction Fraud Analysis",
        "4. Customer Demographics",
        "5. Predictive Risk Modeling"
    ]
)

st.sidebar.divider()
st.sidebar.caption("⚡ *Data status: Fully cached into application memory via Streamlit decorators.*")

# Route to the appropriate page function
if page_selection == "1. Executive KPI Overview":
    show_overview_page()
elif page_selection == "2. Customer Churn Deep-Dive":
    show_churn_analysis_page()
elif page_selection == "3. Transaction Fraud Analysis":
    show_fraud_insights_page()
elif page_selection == "4. Customer Demographics":
    show_demographics_page()
elif page_selection == "5. Predictive Risk Modeling":
    show_predictive_modeling_page()
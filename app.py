import streamlit as st
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Financial & Customer Analytics Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------------------------------------------------------
# 1. CACHED DATA LOADING FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_churn_data():
    """Loads and caches the customer churn dataset."""
    return pd.read_csv("data/Churn_Modelling.csv")


@st.cache_data(ttl=3600)
def load_paysim_data():
    """Loads and caches the PaySim fraud detection dataset."""
    return pd.read_csv("data/PaySim.csv")


# Load data into session state or cache initially
try:
    churn_df = load_churn_data()
    paysim_df = load_paysim_data()
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
        st.metric(
            label="Total Customers",
            value=f"{total_customers:,}",
            help="Total unique customer identification numbers recorded in the customer database."
        )

    with col2:
        st.metric(
            label="Customer Churn Rate",
            value=f"{churn_rate:.2f}%",
            delta=f"-{churn_rate:.1f}%",
            delta_color="inverse",
            help="Percentage of total customers who have closed their accounts or stopped using the service."
        )

    with col3:
        st.metric(
            label="Active Customer Rate",
            value=f"{active_rate:.2f}%",
            delta=f"{active_rate:.1f}%",
            help="Percentage of total customers flagged as actively engaging with the bank platform."
        )

    with col4:
        st.metric(
            label="Transaction Fraud Rate",
            value=f"{fraud_rate:.2f}%",
            delta=f"-{fraud_rate:.2f}%",
            delta_color="inverse",
            help="Percentage of processed financial transactions flagged as fraudulent behavior."
        )

    st.divider()

    # Additional Context Tables for Page 1
    st.subheader("📋 Dataset Previews (Loaded from Cache)")
    tab1, tab2 = st.tabs(["Churn Dataset Preview", "PaySim Transaction Dataset Preview"])

    with tab1:
        st.dataframe(churn_df.head(10), use_container_width=True)
    with tab2:
        st.dataframe(paysim_df.head(10), use_container_width=True)


def show_churn_analysis_page():
    st.title("📉 Page 2: Customer Churn Deep-Dive")
    st.markdown("Detailed breakdown of demographic and behavioral impacts on customer retention.")

    # Example interactivity
    geo_filter = st.multiselect("Filter by Geography", options=churn_df['Geography'].unique(),
                                default=churn_df['Geography'].unique())
    filtered_churn = churn_df[churn_df['Geography'].isin(geo_filter)]

    st.dataframe(filtered_churn.describe(), use_container_width=True)
    st.info(
        "Tip: Use this page to integrate charts tracking credit score thresholds or account balance vs. churn status.")


def show_fraud_insights_page():
    st.title("🔒 Page 3: Transaction Fraud Analysis")
    st.markdown("Granular insights into suspicious transfers, transaction volumes, and flagged system alerts.")

    type_filter = st.selectbox("Select Transaction Type", options=paysim_df['type'].unique())
    filtered_paysim = paysim_df[paysim_df['type'] == type_filter]

    st.write(f"Showing summary for type: **{type_filter}**")
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
        "This placeholder page can be used to load pickled ML models (`scikit-learn` or `XGBoost`) to run real-time churn predictions or fraud risk scoring profiles based on custom inputs.")

    # Sample Mock Input
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
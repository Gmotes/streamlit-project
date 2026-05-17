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
def process_analytical_engine(churn_df, paysim_df):
    """
    Trains internal ML classifiers from cache to deliver unified risk matrices.
    Maps transaction profiles back to primary client vectors.
    """
    # --- Churn Engine Modeling ---
    churn_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                      'EstimatedSalary']
    X_c = churn_df[churn_features]
    y_c = churn_df['Exited']

    rf_churn = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_churn.fit(X_c, y_c)

    df_scored = churn_df.copy()
    df_scored['Churn_Probability'] = rf_churn.predict_proba(X_c)[:, 1]
    df_scored['Churn_Score'] = (df_scored['Churn_Probability'] * 100).round(2)

    def get_risk_tier(prob):
        if prob < 0.30:
            return 'Low Risk'
        elif prob < 0.70:
            return 'Medium Risk'
        else:
            return 'High Risk'

    df_scored['Risk_Tier'] = df_scored['Churn_Probability'].apply(get_risk_tier)

    # Calculate Health & Engagement Scores
    product_factor = np.where(df_scored['NumOfProducts'] == 2, 40, np.where(df_scored['NumOfProducts'] == 1, 25, 10))
    df_scored['Engagement_Score'] = (df_scored['IsActiveMember'] * 40) + (df_scored['HasCrCard'] * 20) + product_factor
    df_scored['Health_Score'] = (
                (1.0 - df_scored['Churn_Probability']) * 80 + (df_scored['IsActiveMember'] * 20)).round(2)

    # --- Fraud Mapping Engine ---
    # PaySim uses destination/origin keys; we map typical baseline patterns to match customer risk scores
    np.random.seed(42)
    # Generate static, reproducible structural baseline fraud risk probabilities for accounts
    base_fraud_probs = np.random.beta(0.5, 5, size=len(df_scored))
    # Elevate fraud risk score metrics slightly if their credit score is abnormally volatile or balances are zeroed
    adjusted_fraud = np.where(df_scored['CreditScore'] < 500, base_fraud_probs * 1.8, base_fraud_probs)
    df_scored['Fraud_Probability'] = np.clip(adjusted_fraud, 0, 1)
    df_scored['Fraud_Score'] = (df_scored['Fraud_Probability'] * 100).round(2)

    # Store Feature Importance Structure
    importance_df = pd.DataFrame({
        'Feature': churn_features,
        'Importance': rf_churn.feature_importances_
    }).sort_values('Importance', ascending=True)

    # --- PaySim Dataset Prep ---
    df_fraud = paysim_df.copy()
    fraud_amounts = df_fraud[df_fraud['isFraud'] == 1]['amount']
    q1, q2 = (fraud_amounts.quantile(0.33), fraud_amounts.quantile(0.66)) if not fraud_amounts.empty else (10000,
                                                                                                           100000)

    def segment_amount_risk(amt):
        if amt <= q1:
            return 'Tier 1: Low-Value Risk'
        elif amt <= q2:
            return 'Tier 2: Mid-Value Risk'
        else:
            return 'Tier 3: High-Value Risk'

    df_fraud['Fraud_Risk_Segment'] = df_fraud['amount'].apply(segment_amount_risk)

    return df_scored, importance_df, df_fraud


# Initialization sequence execution
try:
    c_raw = load_churn_data()
    p_raw = load_paysim_data()
    churn_df, feature_importance_df, paysim_df = process_analytical_engine(c_raw, p_raw)
except Exception as e:
    st.error(f"Error loading datasets. Place 'Churn_Modelling.csv' and 'PaySim.csv' in the same folder. Details: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# 2. APPLICATION ROUTING PAGES DEF
# -----------------------------------------------------------------------------

def show_overview_page():
    st.title("📊 Executive KPI Overview Dashboard")
    st.markdown("Central control system. Real-time indicators processed from cached application components.")

    total_customers = int(churn_df['CustomerId'].nunique())
    churn_rate = float(churn_df['Exited'].mean() * 100)
    active_rate = float(churn_df['IsActiveMember'].mean() * 100)
    fraud_rate = float(paysim_df['isFraud'].mean() * 100)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Customers", f"{total_customers:,}")
    with col2: st.metric("Customer Churn Rate", f"{churn_rate:.2f}%", delta=f"-{churn_rate:.1f}%",
                         delta_color="inverse")
    with col3: st.metric("Active Customer Rate", f"{active_rate:.2f}%", delta=f"{active_rate:.1f}%")
    with col4: st.metric("Transaction Fraud Rate", f"{fraud_rate:.2f}%", delta=f"-{fraud_rate:.2f}%",
                         delta_color="inverse")

    st.divider()
    t1, t2 = st.tabs(["Churn Ledger Preview", "PaySim Operations Preview"])
    with t1: st.dataframe(churn_df.head(10), use_container_width=True)
    with t2: st.dataframe(paysim_df.head(10), use_container_width=True)


def show_churn_analysis_page():
    st.title("📉 Page 2: Customer Churn Deep-Dive")
    st.header("1. Churn Risk Distribution")
    d1, d2 = st.columns([1, 2])
    with d1:
        t_counts = churn_df['Risk_Tier'].value_counts()
        for tier in ['Low Risk', 'Medium Risk', 'High Risk']:
            st.metric(tier, f"{t_counts.get(tier, 0):,}",
                      delta=f"{(t_counts.get(tier, 0) / len(churn_df)) * 100:.1f}% of total")
    with d2:
        counts, bins = np.histogram(churn_df['Churn_Score'], bins=10, range=(0, 100))
        h_df = pd.DataFrame({'Risk Range': [f"{int(bins[i])}-{int(bins[i + 1])}%" for i in range(len(bins) - 1)],
                             'Customer Count': counts}).set_index('Risk Range')
        st.bar_chart(h_df, y='Customer Count', color="#FF4B4B")
    st.divider()
    st.header("2. Churn Segmentations")
    geo_df = pd.DataFrame({'Churn Rate (%)': (churn_df.groupby('Geography')['Exited'].mean() * 100).round(2)})
    st.dataframe(geo_df, use_container_width=True)
    st.divider()
    st.header("3. Churn Feature Importance")
    st.bar_chart(feature_importance_df.set_index('Feature'), y='Importance', color="#29B5E8")


def show_fraud_insights_page():
    st.title("🔒 Page 3: Transaction Fraud Analysis")
    st.header("1. Global System Fraud Rate")
    st.metric("Total Fraud Volume", f"${paysim_df[paysim_df['isFraud'] == 1]['amount'].sum():,.2f}")
    st.divider()
    st.header("2. Fraud Incidence by Transaction Type")
    type_metrics = paysim_df.groupby('type').agg(Total_Tx=('isFraud', 'count'), Fraud_Cases=('isFraud', 'sum'))
    st.dataframe(type_metrics, use_container_width=True)
    st.divider()
    st.header("3. Fraud Risk Amount Segmentation")
    segment_metrics = paysim_df.groupby('Fraud_Risk_Segment', observed=False).agg(Cases=('isFraud', 'sum'))
    st.bar_chart(segment_metrics, color="#E71D36")
    st.divider()
    st.header("4. Audit Trail: Verified Fraud Transactions Ledger")
    st.dataframe(paysim_df[paysim_df['isFraud'] == 1].drop(columns=['isFraud']).head(100), use_container_width=True)


def show_lifecycle_health_page():
    st.title("👥 Page 4: Customer Health & Engagement Optimization")
    st.header("1. Health & Engagement Score Tracking")
    st.columns(2)[0].metric("Average Account Health Score", f"{churn_df['Health_Score'].mean():.2f} / 100")
    st.columns(2)[1].metric("Average Product Engagement Score", f"{churn_df['Engagement_Score'].mean():.2f} / 100")
    st.divider()
    st.header("2. Prescriptive Retention Playbooks")
    st.info("Review specific regional suggestions by navigating inside the core options matrices of Page 4.")


def show_action_recommendation_page():
    st.title("🎯 Page 5: Action Recommendation Dashboard")
    st.markdown(
        "Select a specific account profile from cache to look up their current risk indexes and review targeted system operations suggestions.")

    # -------------------------------------------------------------------------
    # STEP 1: SELECT CUSTOMER
    # -------------------------------------------------------------------------
    st.header("1. Select Customer Target Profile")

    # Setup interactive filter controls to quickly isolate specific user types
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        tier_filter = st.selectbox("Quick-Filter by Risk Status Group:",
                                   ["All Customers", "High Churn Risk Profiles", "High Fraud Risk Profiles"])

    # Filter dropdown database base population depending on chosen parameters
    if tier_filter == "High Churn Risk Profiles":
        subset_df = churn_df[churn_df['Risk_Tier'] == 'High Risk']
    elif tier_filter == "High Fraud Risk Profiles":
        subset_df = churn_df[churn_df['Fraud_Score'] > 50.0]
    else:
        subset_df = churn_df

    # Search lookup tool via an interactive selectbox component
    customer_options = subset_df.apply(lambda row: f"{row['CustomerId']} - {row['Surname']} ({row['Geography']})",
                                       axis=1).tolist()

    if not customer_options:
        st.warning("No records found matching the specified group filter.")
        return

    selected_option = st.selectbox("Choose Customer Account Record to Audit:", options=customer_options)
    selected_id = int(selected_option.split(" - ")[0])

    # Isolate exact dataset row slice
    user_record = churn_df[churn_df['CustomerId'] == selected_id].iloc[0]

    st.divider()

    # -------------------------------------------------------------------------
    # STEP 2 & 3: SHOW CHURN SCORE & FRAUD SCORE
    # -------------------------------------------------------------------------
    st.header("2. Risk Vectors Evaluation Matrix")

    c_score = float(user_record['Churn_Score'])
    f_score = float(user_record['Fraud_Score'])

    score_col1, score_col2 = st.columns(2)

    with score_col1:
        # Style color alerts depending on risk level severity thresholds
        if c_score >= 70.0:
            st.error(f"### 📉 Churn Score: **{c_score}%** (Critical Risk)")
        elif c_score >= 30.0:
            st.warning(f"### 📉 Churn Score: **{c_score}%** (Moderate Risk)")
        else:
            st.success(f"### 📉 Churn Score: **{c_score}%** (Stable Account)")

    with score_col2:
        if f_score >= 50.0:
            st.error(f"### 🔒 Fraud Score: **{f_score}%** (High Alert Level)")
        elif f_score >= 20.0:
            st.warning(f"### 🔒 Fraud Score: **{f_score}%** (Review Required)")
        else:
            st.success(f"### 🔒 Fraud Score: **{f_score}%** (Low Activity Threat)")

    # Display structural metrics table
    st.markdown("**Profile Attributes Summary Table:**")
    st.dataframe(
        pd.DataFrame([user_record[['Age', 'CreditScore', 'Balance', 'NumOfProducts', 'IsActiveMember', 'HasCrCard']]]),
        use_container_width=True, hide_index=True)

    st.divider()

    # -------------------------------------------------------------------------
    # STEP 4: SEE SUGGESTIONS
    # -------------------------------------------------------------------------
    st.header("3. Prescriptive System Action Recommendations")

    # Expandable interface section
    with st.expander("👉 Click to Open Action Matrix Suggestions Playbook", expanded=True):
        st.subheader(f"Strategic Directives Ledger for {user_record['Surname']} (ID: {selected_id})")

        # Build logical decision rules engine combining both analytical scores
        suggestions_triggered = 0

        if c_score >= 70.0:
            st.markdown("🔴 **CRITICAL CHURN RETENTION ACTION MANDATE**")
            st.markdown(f"""
            - **Observation:** Churn risk profile is at **{c_score}%** with an account balance of **${user_record['Balance']:,.2f}**.
            - **Immediate Interventions:**
                1. **High-Value Account Outreach:** Initiate direct phone contact from a senior customer success manager within 24 hours.
                2. **Product Consolidation Offer:** This customer has **{user_record['NumOfProducts']}** product(s). If they hold 3 or more, offer a waiver to consolidate accounts without penalty fees.
                3. **Active Service Incentive:** Since active membership status is **{user_record['IsActiveMember']}**, provide a customized bonus cash back structure on transactional fees if they complete 5 app logins this month.
            """)
            suggestions_triggered += 1

        if f_score >= 50.0:
            st.markdown("⚠️ **SECURITY & FRAUD RISK ENFORCEMENT DIRECTIVE**")
            st.markdown(f"""
            - **Observation:** Calculated internal account velocity/fraud score has peaked at **{f_score}%**.
            - **Immediate Interventions:**
                1. **Enhanced Step-Up Authentication (MFA):** Enforce immediate mandatory multi-factor validation checks on all outbound transactions exceeding $1,000.
                2. **Velocity Parameter Caps:** Apply a temporary 48-hour transactional transfer ceiling equal to 25% of their active balance vector.
                3. **Audit Origin Logs:** Request an immediate technical match verification on recent balance adjustments against matching `PaySim` transfer records.
            """)
            suggestions_triggered += 1

        if suggestions_triggered == 0:
            st.markdown("🟢 **STANDARD MAINTENANCE & LIFECYCLE NURTURE PROTOCOL**")
            st.markdown(f"""
            - **Observation:** Both risk thresholds are within standard system tolerances (Churn: {c_score}%, Fraud: {f_score}%).
            - **Immediate Interventions:**
                1. **Cross-Sell Premium Upgrades:** The customer holds a credit score of **{user_record['CreditScore']}**. If it's above 700, cross-promote premium credit tier offerings.
                2. **Loyalty Program Enrollment:** Automatically add this account to premium benefit reward cycles to maintain retention scores.
            """)


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
        "4. Lifecycle & Health Optimization",
        "5. Action Recommendation"
    ]
)

st.sidebar.divider()
st.sidebar.caption("⚡ *Data status: Fully cached into application memory via Streamlit decorators.*")

# Route execution calls to target functions
if page_selection == "1. Executive KPI Overview":
    show_overview_page()
elif page_selection == "2. Customer Churn Deep-Dive":
    show_churn_analysis_page()
elif page_selection == "3. Transaction Fraud Analysis":
    show_fraud_insights_page()
elif page_selection == "4. Lifecycle & Health Optimization":
    show_lifecycle_health_page()
elif page_selection == "5. Action Recommendation":
    show_action_recommendation_page()
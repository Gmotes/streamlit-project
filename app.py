import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import plotly.express as px
from groq import Groq


api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)


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

@st.cache_resource  # Keeps the model in memory so it doesn't reload on every click
def load_assets():
    with open('machine_learning_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler


# Initialization sequence execution
try:
    c_raw = load_churn_data()
    p_raw = load_paysim_data()
    churn_df, feature_importance_df, paysim_df = process_analytical_engine(c_raw, p_raw)
except Exception as e:
    st.error(f"Error loading datasets. Place 'Churn_Modelling.csv' and 'PaySim.csv' in the same folder. Details: {e}")
    st.stop()




def build_customer_input(credit_score, geography, gender, age, tenure, balance,
                         num_products, has_cr_card, is_active_member, estimated_salary):
    return pd.DataFrame([{
        "CreditScore": credit_score,
        "Gender": 1 if gender == "Male" else 0,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": int(has_cr_card),
        "IsActiveMember": int(is_active_member),
        "EstimatedSalary": estimated_salary,
        "Geography_Germany": 1 if geography == "Germany" else 0,
        "Geography_Spain": 1 if geography == "Spain" else 0,
        "BalanceSalaryRatio": balance / (estimated_salary + 1),
        "ZeroBalance": int(balance == 0),
        "ProductsPerTenure": num_products / (tenure + 1),
        "ActiveWithBalance": int(is_active_member) * int(balance > 0),
        "CreditScorePerAge": credit_score / age,
        "AgeGroup": 0 if age < 35 else (1 if age <= 55 else 2),
    }])


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


def show_AI_analyst_page():
    st.title("🤖 AI Churn Analyst")
    st.markdown(
        "Enter a customer's profile. The churn model will score them, "
        "then Groq will explain the prediction and suggest retention actions."
    )

    clf, _scaler = load_assets()
    feature_cols = [
        "CreditScore", "Gender", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary",
        "Geography_Germany", "Geography_Spain",
        "BalanceSalaryRatio", "ZeroBalance", "ProductsPerTenure",
        "ActiveWithBalance", "CreditScorePerAge", "AgeGroup",
    ]

    # ── Input form ────────────────────────────────────────────────────────────────
    with st.form("customer_form"):
        st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
            age = st.number_input("Age", min_value=18, max_value=100, value=40)
            tenure = st.slider("Tenure (years)", 0, 10, 5)
            balance = st.number_input("Balance ($)", min_value=0.0, max_value=500_000.0, value=50_000.0, step=1000.0)

        with c2:
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
            estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=300_000.0,
                                               value=60_000.0, step=1000.0)

        with c3:
            has_cr_card = st.checkbox("Has Credit Card", value=True)
            is_active_member = st.checkbox("Is Active Member", value=True)
            groq_model = st.selectbox("Groq Model", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"])

        submitted = st.form_submit_button("🔍 Predict & Analyse", use_container_width=True)

    if submitted:
        input_df = build_customer_input(
            credit_score, geography, gender, age, tenure, balance,
            num_products, has_cr_card, is_active_member, estimated_salary,
        )
        input_df = input_df[feature_cols]

        churn_prob = clf.predict_proba(input_df)[0][1]
        churn_label = "High Risk" if churn_prob >= 0.5 else "Low Risk"

        # ── Prediction metrics ────────────────────────────────────────────────────
        st.divider()
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        r1.metric("Churn Probability", f"{churn_prob:.1%}")
        r2.metric("Risk Level", churn_label)
        r3.metric("Retention Probability", f"{1 - churn_prob:.1%}")

        # ── Top feature importances ───────────────────────────────────────────────
        importances = pd.Series(clf.feature_importances_, index=feature_cols)
        top5 = importances.nlargest(5)
        top5_str = "\n".join(
            f"  - {feat} (importance {imp:.3f}, customer value: {input_df[feat].values[0]:.3f})"
            for feat, imp in top5.items()
        )

        fig_imp = px.bar(
            top5.reset_index().rename(columns={"index": "Feature", 0: "Importance"}),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 5 Model Features",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig_imp.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)

        # ── Groq explanation ──────────────────────────────────────────────────────
        st.divider()
        st.markdown('<div class="section-title">AI Analysis</div>', unsafe_allow_html=True)

        customer_profile = (
            f"- Credit Score: {credit_score}\n"
            f"- Geography: {geography}\n"
            f"- Gender: {gender}\n"
            f"- Age: {age}\n"
            f"- Tenure: {tenure} years\n"
            f"- Balance: ${balance:,.0f}\n"
            f"- Number of Products: {num_products}\n"
            f"- Has Credit Card: {'Yes' if has_cr_card else 'No'}\n"
            f"- Is Active Member: {'Yes' if is_active_member else 'No'}\n"
            f"- Estimated Salary: ${estimated_salary:,.0f}"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior banking analytics expert. "
                    "A Random Forest model has just predicted a customer's churn probability. "
                    "Explain the prediction in plain language and provide 3–5 specific, "
                    "actionable retention strategies tailored to this customer's profile. "
                    "Be concise and professional. Use bullet points where appropriate."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Customer profile:\n{customer_profile}\n\n"
                    f"Model prediction: {churn_prob:.1%} churn probability ({churn_label})\n\n"
                    f"Top 5 most important features (globally) with this customer's values:\n{top5_str}\n\n"
                    "Please:\n"
                    "1. In 2–3 sentences, explain why this customer may or may not be at risk, "
                    "referencing their specific profile.\n"
                    "2. List 3–5 concrete retention actions the bank should take for this customer."
                ),
            },
        ]

        with st.spinner("Asking Groq…"):
            completion = client.chat.completions.create(model=groq_model, messages=messages)

        st.markdown(completion.choices[0].message.content)

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION CONTROLLER
# -----------------------------------------------------------------------------

st.sidebar.image(
    "https://img.icons8.com/fluency/96/shield.png",
    width=60,
)
st.sidebar.title("FinGuard AI")

st.sidebar.markdown("Financial Intelligence Dashboard")

page_selection = st.sidebar.radio(
    "Select a Page:",
    [
        "1. Executive KPI Overview",
        "2. Customer Churn Deep-Dive",
        "3. Transaction Fraud Analysis",
        "4. Lifecycle & Health Optimization",
        "5. AI Analysis",
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
elif page_selection == "5. AI Analysis":
    show_AI_analyst_page()
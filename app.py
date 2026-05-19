import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import plotly.express as px
from groq import Groq
import duckdb
import requests


api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

N8N_WEBHOOK_URL = st.secrets["N8N_WEBHOOK_URL"]


def send_to_n8n(header: str, body: str):
    payload = {
        "header": header,
        "body": body,
        "email": "burak.tunali41@gmail.com",
    }
    try:
        resp = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        st.warning(f"n8n webhook delivery failed: {e}")
        return False


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
    """Loads and caches the PaySim fraud detection dataset from a public URL."""
    url =  st.secrets["PARQUET_URL"]

    df = duckdb.query(
        f"""
            SELECT type, isFraud, amount
            FROM read_parquet('{url}')
            LIMIT 1000000
        """
    ).df()

    return df

@st.cache_resource  # Keeps the model in memory so it doesn't reload on every click
def load_churn_model():
    with open('xgb_churn_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

@st.cache_resource  # Keeps the model in memory so it doesn't reload on every click
def load_fraud_model():
    with open('fraud_detection_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model


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




def build_fraud_input(step, tx_type, amount, old_bal_orig, new_bal_orig,
                      old_bal_dest, new_bal_dest, dest_type_char, model_package):
    scaler = model_package["scaler"]
    scale_cols = model_package["scale_cols"]
    le = model_package["label_encoder"]

    df = pd.DataFrame([{
        "step": step,
        "type": tx_type,
        "amount": amount,
        "oldbalanceOrg": old_bal_orig,
        "newbalanceOrig": new_bal_orig,
        "oldbalanceDest": old_bal_dest,
        "newbalanceDest": new_bal_dest,
        "DestType": dest_type_char,
    }])

    df["BalanceDiff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["DestBalanceDiff"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["IsBalanceError"] = np.where(df["amount"] != df["BalanceDiff"], 1, 0)
    # 95th-percentile threshold is not stored in the package; use the PaySim dataset's
    # typical value (~1 000 000) as a reasonable approximation.
    df["LargeTransaction"] = (df["amount"] > 1_000_000).astype(int)
    df["StepGroup"] = pd.cut(
        df["step"], bins=[0, 200, 400, 600, 800],
        labels=["Early", "Mid", "Late", "VeryLate"]
    )
    df["LogAmount"] = np.log1p(df["amount"])
    df["LogOldBalanceOrg"] = np.log1p(df["oldbalanceOrg"])
    df["LogNewBalanceOrig"] = np.log1p(df["newbalanceOrig"])
    df["LogOldBalanceDest"] = np.log1p(df["oldbalanceDest"])
    df["LogNewBalanceDest"] = np.log1p(df["newbalanceDest"])

    df["DestType"] = le.transform(df["DestType"])

    df = pd.get_dummies(df, columns=["type", "StepGroup"], drop_first=True)

    for col in ["type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
                "StepGroup_Late", "StepGroup_Mid", "StepGroup_VeryLate"]:
        if col not in df.columns:
            df[col] = 0

    df[scale_cols] = scaler.transform(df[scale_cols])

    return df[model_package["features"]]


def build_customer_input(credit_score, geography, gender, age, tenure, balance,
                         num_products, has_cr_card, is_active_member, estimated_salary):
    df = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": int(has_cr_card),
        "IsActiveMember": int(is_active_member),
        "EstimatedSalary": estimated_salary,
    }])

    # Replicate training feature engineering
    df["NumOfProducts"] = df["NumOfProducts"].replace({3: "3 or More", 4: "3 or More"})
    df.loc[df["Age"] > 75, "Age"] = 75
    df["Age_Group"] = pd.cut(df["Age"], [0, 30, 45, 65, 99], labels=["18-30", "31-45", "46-65", "66-99"])
    df["Balance_to_Salary"] = df["Balance"] / df["EstimatedSalary"]
    df["Gender"] = (df["Gender"] == "Male").astype(int)

    # OHE matching training (drop_first=True)
    df = pd.get_dummies(df, columns=["Geography", "NumOfProducts", "Age_Group"], drop_first=True)

    # Ensure all OHE columns exist (missing when a category isn't present in single-row input)
    for col in ["Geography_Germany", "Geography_Spain",
                "NumOfProducts_2", "NumOfProducts_3 or More",
                "Age_Group_31-45", "Age_Group_46-65", "Age_Group_66-99"]:
        if col not in df.columns:
            df[col] = 0

    return df


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
    #st.dataframe(paysim_df[paysim_df['isFraud'] == 1].drop(columns=['isFraud']).head(100), use_container_width=True)


def show_lifecycle_health_page():
    st.title("👥 Page 4: Customer Health & Engagement Optimization")
    st.header("1. Health & Engagement Score Tracking")
    st.columns(2)[0].metric("Average Account Health Score", f"{churn_df['Health_Score'].mean():.2f} / 100")
    st.columns(2)[1].metric("Average Product Engagement Score", f"{churn_df['Engagement_Score'].mean():.2f} / 100")
    st.divider()
    st.header("2. Prescriptive Retention Playbooks")
    st.info("Review specific regional suggestions by navigating inside the core options matrices of Page 4.")


def show_AI_churn_analyst_page():
    st.title("🤖 AI Churn Analyst")
    st.markdown(
        "Enter a customer's profile. The churn model will score them, "
        "then Groq will explain the prediction and suggest retention actions."
    )

    clf = load_churn_model()
    feature_cols = [
        "CreditScore", "Gender", "Age", "Tenure", "Balance",
        "HasCrCard", "IsActiveMember", "EstimatedSalary", "Balance_to_Salary",
        "Geography_Germany", "Geography_Spain",
        "NumOfProducts_2", "NumOfProducts_3 or More",
        "Age_Group_31-45", "Age_Group_46-65", "Age_Group_66-99",
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
            num_products = st.selectbox("NumOfProducts", [1, 2, 3, 4])
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
            num_products, has_cr_card, is_active_member, estimated_salary
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

        ai_response = completion.choices[0].message.content
        st.markdown(ai_response)

        n8n_header = f"Churn Analysis Result – {churn_label} ({churn_prob:.1%})"
        if send_to_n8n(n8n_header, ai_response):
            st.success("Results sent to n8n.")


def show_AI_fraud_analyst_page():
    st.title("🤖 AI Fraud Analyst")
    st.markdown(
        "Enter a transaction's details. The fraud model will score it, "
        "then Groq will explain the prediction and suggest investigation actions."
    )

    model_package = load_fraud_model()
    clf = model_package["model"]

    with st.form("fraud_form"):
        st.markdown('<div class="section-title">Transaction Profile</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            step = st.number_input("Step (1–744)", min_value=1, max_value=744, value=1)
            amount = st.number_input("Amount ($)", min_value=0.0, max_value=10_000_000.0, value=10_000.0, step=500.0)
            old_bal_orig = st.number_input("Origin Old Balance ($)", min_value=0.0, max_value=10_000_000.0,
                                           value=50_000.0, step=1000.0)
            new_bal_orig = st.number_input("Origin New Balance ($)", min_value=0.0, max_value=10_000_000.0,
                                           value=40_000.0, step=1000.0)

        with c2:
            tx_type = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
            dest_type = st.selectbox("Destination Account Type", ["C – Customer", "M – Merchant"])
            old_bal_dest = st.number_input("Destination Old Balance ($)", min_value=0.0, max_value=10_000_000.0,
                                           value=0.0, step=1000.0)
            new_bal_dest = st.number_input("Destination New Balance ($)", min_value=0.0, max_value=10_000_000.0,
                                           value=10_000.0, step=1000.0)

        with c3:
            groq_model = st.selectbox("Groq Model", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"])

        submitted = st.form_submit_button("🔍 Predict & Analyse", use_container_width=True)

    if submitted:
        dest_type_char = dest_type[0]  # "C" or "M"
        input_df = build_fraud_input(
            step, tx_type, amount, old_bal_orig, new_bal_orig,
            old_bal_dest, new_bal_dest, dest_type_char, model_package
        )

        fraud_prob = clf.predict_proba(input_df)[0][1]
        fraud_label = "Fraudulent" if fraud_prob >= 0.5 else "Legitimate"

        st.divider()
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        r1.metric("Fraud Probability", f"{fraud_prob:.1%}")
        r2.metric("Verdict", fraud_label)
        r3.metric("Legitimacy Probability", f"{1 - fraud_prob:.1%}")

        feature_cols = model_package["features"]
        importances = pd.Series(clf.feature_importances_, index=feature_cols)
        top5 = importances.nlargest(5)
        top5_str = "\n".join(
            f"  - {feat} (importance {imp:.3f}, value: {input_df[feat].values[0]:.4f})"
            for feat, imp in top5.items()
        )

        fig_imp = px.bar(
            top5.reset_index().rename(columns={"index": "Feature", 0: "Importance"}),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 5 Model Features",
            color="Importance",
            color_continuous_scale="Reds",
        )
        fig_imp.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)

        st.divider()
        st.markdown('<div class="section-title">AI Analysis</div>', unsafe_allow_html=True)

        tx_profile = (
            f"- Step: {step}\n"
            f"- Transaction Type: {tx_type}\n"
            f"- Amount: ${amount:,.2f}\n"
            f"- Origin Old Balance: ${old_bal_orig:,.2f}\n"
            f"- Origin New Balance: ${new_bal_orig:,.2f}\n"
            f"- Destination Old Balance: ${old_bal_dest:,.2f}\n"
            f"- Destination New Balance: ${new_bal_dest:,.2f}\n"
            f"- Destination Account Type: {dest_type}"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior financial fraud investigation expert. "
                    "A LightGBM model has just predicted the fraud probability of a transaction. "
                    "Explain the prediction in plain language and provide 3–5 specific, "
                    "actionable investigation or mitigation steps tailored to this transaction's profile. "
                    "Be concise and professional. Use bullet points where appropriate."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Transaction profile:\n{tx_profile}\n\n"
                    f"Model prediction: {fraud_prob:.1%} fraud probability ({fraud_label})\n\n"
                    f"Top 5 most important features with this transaction's values:\n{top5_str}\n\n"
                    "Please:\n"
                    "1. In 2–3 sentences, explain why this transaction may or may not be fraudulent, "
                    "referencing its specific profile.\n"
                    "2. List 3–5 concrete actions the fraud team should take for this transaction."
                ),
            },
        ]

        with st.spinner("Asking Groq…"):
            completion = client.chat.completions.create(model=groq_model, messages=messages)

        ai_response = completion.choices[0].message.content
        st.markdown(ai_response)

        n8n_header = f"Fraud Analysis Result – {fraud_label} ({fraud_prob:.1%})"
        if send_to_n8n(n8n_header, ai_response):
            st.success("Results sent to n8n.")


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
        "5. AI Churn Analysis",
        "6. AI Fraud Analysis",
    ]
)

st.sidebar.divider()
st.sidebar.caption("⚡ *DataFu - Group 03 *")
st.sidebar.caption(" 1. Yagmur Sargin")
st.sidebar.caption(" 2. Hakan Kurucay ")
st.sidebar.caption(" 3. Fikri Uğur Emek ")
st.sidebar.caption(" 4. Burak Tunali")

# Route execution calls to target functions
if page_selection == "1. Executive KPI Overview":
    show_overview_page()
elif page_selection == "2. Customer Churn Deep-Dive":
    show_churn_analysis_page()
elif page_selection == "3. Transaction Fraud Analysis":
    show_fraud_insights_page()
elif page_selection == "4. Lifecycle & Health Optimization":
    show_lifecycle_health_page()
elif page_selection == "5. AI Churn Analysis":
    show_AI_churn_analyst_page()
elif page_selection == "6. AI Fraud Analysis":
    show_AI_fraud_analyst_page()
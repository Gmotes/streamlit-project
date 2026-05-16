import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq
import pickle

api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

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


@st.cache_resource  # Keeps the model in memory so it doesn't reload on every click
def load_assets():
    with open('machine_learning_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error("Model or Scaler file not found. Please check your file paths.")


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
        ["🏦 Churn Analysis", "🚨 Fraud Detection", "🤖 AI Analyst"],
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

    st.markdown('<div class="section-title">Retention Action Engine </div>', unsafe_allow_html=True)

    [c1] = st.columns(1)

    with c1:
        model = st.sidebar.selectbox("Choose Model", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"])
        # 2. Prepare the context
        # We convert the first few rows (or a summary) to a string so the LLM understands the structure
        csv_sample = df.head(400).to_csv(index=False)
        column_names = ", ".join(df.columns)

        if prompt := st.chat_input("Ex: What is the average credit score of users who exited?"):
            st.chat_message("user").markdown(prompt)

            # Build the payload for Groq
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a data expert analyzing a Churn dataset. 
                    The dataset contains these columns: {column_names}.
                    Here is a sample of the data to understand the formatting:
                    {csv_sample}

                    Always provide insights based on this specific data structure."""
                },
                {"role": "user", "content": prompt}
            ]

            # Call Groq
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
            )

            response = completion.choices[0].message.content
            with st.chat_message("assistant"):
                st.markdown(response)

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
elif page == "🚨 Fraud Detection":
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


# ═══════════════════════════════════════════════════════════════════════════════════
# PAGE 3 – AI ANALYST
# ═══════════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Analyst":
    st.title("🤖 AI Churn Analyst")
    st.markdown(
        "Enter a customer's profile. The churn model will score them, "
        "then Groq will explain the prediction and suggest retention actions."
    )

    clf, feature_cols = load_assets()

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
            estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=300_000.0, value=60_000.0, step=1000.0)

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

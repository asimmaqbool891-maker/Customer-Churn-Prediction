"""
==============================================================================
 Customer Churn Prediction System
 Author  : Senior ML Engineering Team
 Purpose : Production-grade Streamlit dashboard that consumes a fully-fitted
           scikit-learn Pipeline (preprocessing + Logistic Regression) to
           predict telecom customer churn.

 IMPORTANT
 ---------
 The pipeline stored in `Telco_Churn_LR(1).pkl` already contains ALL
 preprocessing steps (PowerTransformer, StandardScaler, OrdinalEncoder,
 OneHotEncoder). This application NEVER manually encodes or scales data.
 It only builds a raw pandas DataFrame with the exact original column
 names and feeds it straight into `model.predict()` / `model.predict_proba()`.
==============================================================================
"""

import os
import io
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction | ML Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------
MODEL_PATH = "Telco_Churn_LR.pkl"

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

RISK_COLORS = {
    "Low": "#22c55e",
    "Medium": "#f59e0b",
    "High": "#ef4444",
}


# ------------------------------------------------------------------------
# UTILITY / HELPER FUNCTIONS
# ------------------------------------------------------------------------
def load_css(file_path: str) -> None:
    """Inject a local CSS file into the Streamlit app, failing gracefully."""
    try:
        with open(file_path, "r", encoding="utf-8") as css_file:
            st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(
            "⚠️ Custom stylesheet 'style.css' was not found. "
            "The app will run with default Streamlit styling."
        )


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """
    Load the trained sklearn Pipeline from disk using joblib.

    Returns
    -------
    object or None
        The loaded pipeline, or None if loading failed.
    """
    if not os.path.exists(model_path):
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as exc:  # noqa: BLE001
        st.session_state["model_load_error"] = str(exc)
        return None


def build_customer_dataframe(inputs: dict) -> pd.DataFrame:
    """
    Build a single-row pandas DataFrame using the EXACT original feature
    names expected by the pipeline. No manual encoding/scaling is applied.
    """
    data = {col: [inputs[col]] for col in FEATURE_COLUMNS}
    df = pd.DataFrame(data)
    return df[FEATURE_COLUMNS]  # enforce exact column order


def get_risk_level(churn_probability: float) -> str:
    """Classify churn probability into a Low / Medium / High risk bucket."""
    if churn_probability < 0.35:
        return "Low"
    elif churn_probability < 0.65:
        return "Medium"
    return "High"


def make_gauge_chart(churn_probability: float, risk_level: str) -> go.Figure:
    """Create a Plotly gauge chart visualizing churn probability."""
    color = RISK_COLORS[risk_level]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(churn_probability * 100, 2),
            number={"suffix": "%", "font": {"size": 40, "color": "#f8fafc"}},
            title={"text": "Churn Probability", "font": {"size": 18, "color": "#cbd5e1"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8", "tickfont": {"color": "#94a3b8"}},
                "bar": {"color": color},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "rgba(34,197,94,0.25)"},
                    {"range": [35, 65], "color": "rgba(245,158,11,0.25)"},
                    {"range": [65, 100], "color": "rgba(239,68,68,0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#f8fafc", "width": 3},
                    "thickness": 0.8,
                    "value": round(churn_probability * 100, 2),
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
        margin=dict(l=30, r=30, t=60, b=20),
        height=320,
    )
    return fig


def make_probability_pie(stay_probability: float, churn_probability: float) -> go.Figure:
    """Create a Plotly donut chart comparing stay vs churn probability."""
    fig = go.Figure(
        go.Pie(
            labels=["Stay", "Churn"],
            values=[stay_probability * 100, churn_probability * 100],
            hole=0.62,
            marker=dict(colors=["#22c55e", "#ef4444"], line=dict(color="#0f172a", width=3)),
            textinfo="label+percent",
            textfont={"color": "#f8fafc", "size": 13},
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        annotations=[
            dict(
                text=f"{churn_probability*100:.1f}%<br><span style='font-size:11px'>Churn</span>",
                x=0.5,
                y=0.5,
                font_size=22,
                font_color="#f8fafc",
                showarrow=False,
            )
        ],
    )
    return fig


def make_risk_bar(risk_level: str) -> go.Figure:
    """Create a horizontal risk-indicator bar chart."""
    order = ["Low", "Medium", "High"]
    values = [1 if lvl == risk_level else 0.15 for lvl in order]
    colors = [RISK_COLORS[lvl] for lvl in order]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=order,
            orientation="h",
            marker=dict(color=colors),
            text=order,
            textposition="inside",
            insidetextfont={"color": "#0f172a", "size": 13},
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
        xaxis=dict(visible=False),
        yaxis=dict(visible=True, showgrid=False),
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        showlegend=False,
    )
    return fig


def risk_badge_html(risk_level: str) -> str:
    """Return an HTML badge span for the given risk level."""
    color = RISK_COLORS[risk_level]
    return (
        f"<span style='background-color:{color}22; color:{color}; "
        f"border:1px solid {color}; padding:4px 14px; border-radius:999px; "
        f"font-weight:600; font-size:0.85rem;'>{risk_level} Risk</span>"
    )


# ------------------------------------------------------------------------
# SIDEBAR — CUSTOMER INPUT FORM
# ------------------------------------------------------------------------
def render_sidebar() -> dict:
    """Render all customer input widgets in the sidebar and return a dict."""
    st.sidebar.markdown(
        "<h2 style='text-align:center; margin-bottom:0;'>🧾 Customer Profile</h2>"
        "<p style='text-align:center; color:#94a3b8; font-size:0.85rem;'>"
        "Fill in the details below</p>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    st.sidebar.markdown("#### 👤 Demographics")
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"], index=0)
    senior_citizen_label = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"], index=0)
    senior_citizen = 1 if senior_citizen_label == "Yes" else 0
    partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"], index=1)
    dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"], index=1)

    st.sidebar.markdown("#### 📞 Services")
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"], index=0)
    multiple_lines = st.sidebar.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"], index=0
    )
    internet_service = st.sidebar.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"], index=1
    )
    online_security = st.sidebar.selectbox(
        "Online Security", ["No", "Yes", "No internet service"], index=0
    )
    online_backup = st.sidebar.selectbox(
        "Online Backup", ["No", "Yes", "No internet service"], index=0
    )
    device_protection = st.sidebar.selectbox(
        "Device Protection", ["No", "Yes", "No internet service"], index=0
    )
    tech_support = st.sidebar.selectbox(
        "Tech Support", ["No", "Yes", "No internet service"], index=0
    )
    streaming_tv = st.sidebar.selectbox(
        "Streaming TV", ["No", "Yes", "No internet service"], index=0
    )
    streaming_movies = st.sidebar.selectbox(
        "Streaming Movies", ["No", "Yes", "No internet service"], index=0
    )

    st.sidebar.markdown("#### 📃 Account & Billing")
    contract = st.sidebar.selectbox(
        "Contract Type", ["Month-to-month", "One year", "Two year"], index=0
    )
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"], index=0)
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        index=0,
    )

    st.sidebar.markdown("#### 💰 Charges & Tenure")
    tenure = st.sidebar.slider("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
    monthly_charges = st.sidebar.number_input(
        "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5
    )
    total_charges = st.sidebar.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=float(round(monthly_charges * max(tenure, 1), 2)),
        step=1.0,
    )

    st.sidebar.markdown("---")
    predict_clicked = st.sidebar.button("🔮 Predict Churn", use_container_width=True, type="primary")

    inputs = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    return inputs, predict_clicked


# ------------------------------------------------------------------------
# MAIN PAGE SECTIONS
# ------------------------------------------------------------------------
def render_header() -> None:
    """Render the gradient hero header and project description."""
    st.markdown(
        """
        <div class="hero-header">
            <h1>📡 Customer Churn Prediction System</h1>
            <p class="hero-subtitle">
                AI-powered dashboard that predicts telecom customer churn risk
                using a production-grade Logistic Regression pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown(
            """
            <div class="glass-card">
                <h4>📋 About this project</h4>
                <p style="color:#cbd5e1; line-height:1.6;">
                This dashboard consumes a fully-trained scikit-learn
                <b>Pipeline</b> — combining feature preprocessing
                (Power Transform, Scaling, Ordinal &amp; One-Hot Encoding) with a
                <b>Logistic Regression</b> classifier — to estimate the probability
                that a customer will churn. Fill in the customer profile from the
                sidebar and click <b>Predict Churn</b> to see the results.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_prediction_results(model, customer_df: pd.DataFrame) -> None:
    """Run prediction and render all result sections."""
    try:
        prediction = model.predict(customer_df)[0]
        probabilities = model.predict_proba(customer_df)[0]
    except Exception as exc:  # noqa: BLE001
        st.error(
            "❌ **Prediction Error**\n\n"
            "Something went wrong while generating the prediction. "
            "This can happen if the input data contains a category the "
            "model has never seen, or if the model file doesn't match "
            "the expected feature schema.\n\n"
            f"**Technical details:** `{exc}`"
        )
        return

    # class ordering assumption: [No Churn, Churn] -> works for typical
    # binary sklearn classifiers where classes_ = [0, 1] or ['No','Yes']
    classes = list(getattr(model, "classes_", [0, 1]))
    try:
        churn_index = classes.index(1) if 1 in classes else classes.index("Yes")
    except ValueError:
        churn_index = 1  # fallback assumption

    churn_probability = float(probabilities[churn_index])
    stay_probability = float(1 - churn_probability)
    confidence_score = float(max(churn_probability, stay_probability))
    risk_level = get_risk_level(churn_probability)
    will_churn = churn_probability >= 0.5

    st.markdown("## 🎯 Prediction Result")

    # --- Big result card ---
    if will_churn:
        st.markdown(
            f"""
            <div class="result-card result-card-churn">
                <h2>⚠️ Customer Likely to CHURN</h2>
                <p>{risk_badge_html(risk_level)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card result-card-stay">
                <h2>✅ Customer Likely to STAY</h2>
                <p>{risk_badge_html(risk_level)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.progress(churn_probability, text=f"Churn Probability: {churn_probability*100:.1f}%")

    # --- Metrics row ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customer Status", "Churn" if will_churn else "Stay")
    m2.metric("Stay Probability", f"{stay_probability*100:.1f}%")
    m3.metric("Churn Probability", f"{churn_probability*100:.1f}%")
    m4.metric("Confidence Score", f"{confidence_score*100:.1f}%")

    # --- Charts ---
    st.markdown("### 📊 Visual Analysis")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(make_gauge_chart(churn_probability, risk_level), use_container_width=True)
    with c2:
        st.plotly_chart(make_probability_pie(stay_probability, churn_probability), use_container_width=True)
    with c3:
        st.markdown("<p style='text-align:center; color:#cbd5e1; margin-bottom:0;'>Risk Indicator</p>", unsafe_allow_html=True)
        st.plotly_chart(make_risk_bar(risk_level), use_container_width=True)

    # --- Customer summary ---
    st.markdown("### 🧾 Customer Summary")
    st.dataframe(customer_df.T.rename(columns={0: "Value"}), use_container_width=True)

    # --- Download ---
    result_df = customer_df.copy()
    result_df["Prediction"] = "Churn" if will_churn else "Stay"
    result_df["Churn_Probability_%"] = round(churn_probability * 100, 2)
    result_df["Stay_Probability_%"] = round(stay_probability * 100, 2)
    result_df["Risk_Level"] = risk_level
    result_df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="⬇️ Download Prediction Report (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"churn_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ------------------------------------------------------------------------
# APPLICATION ENTRY POINT
# ------------------------------------------------------------------------
def main() -> None:
    load_css("style.css")
    render_header()

    inputs, predict_clicked = render_sidebar()

    model = load_model(MODEL_PATH)

    if model is None:
        st.error(
            f"❌ **Model Not Found**\n\n"
            f"Could not locate or load `{MODEL_PATH}`. Please make sure the "
            f"trained pipeline file is placed in the same folder as `app.py` "
            f"before running the app."
        )
        if "model_load_error" in st.session_state:
            st.caption(f"Technical details: {st.session_state['model_load_error']}")
        st.stop()

    if predict_clicked:
        customer_df = build_customer_dataframe(inputs)
        render_prediction_results(model, customer_df)
    else:
        st.info(
            "👈 Configure the customer profile in the sidebar, then click "
            "**Predict Churn** to generate a prediction."
        )

    st.markdown(
        """
        <div class="footer">
            Built with ❤️ using Streamlit, scikit-learn &amp; Plotly —
            Customer Churn Prediction System
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

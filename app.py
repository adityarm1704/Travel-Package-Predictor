from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ----------------------------- Page setup -----------------------------
st.set_page_config(
    page_title="TripWise | Travel Package Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Custom styling -----------------------------
# Use Streamlit's theme variables so text and cards remain readable in light/dark mode.
st.markdown(
    """
    <style>
      :root {
        --tw-text: var(--text-color, inherit);
        --tw-bg: var(--background-color, transparent);
        --tw-panel: var(--secondary-background-color, transparent);
        --tw-border: rgba(127, 127, 127, 0.28);
      }

      /* Main content: inherit the active Streamlit theme instead of forcing black text. */
      [data-testid="stMain"],
      [data-testid="stMain"] p,
      [data-testid="stMain"] label,
      [data-testid="stMain"] li,
      [data-testid="stMain"] [data-testid="stMarkdownContainer"],
      [data-testid="stMain"] [data-testid="stWidgetLabel"],
      [data-testid="stMain"] [data-testid="stCaptionContainer"],
      [data-testid="stMain"] [data-testid="stMetricLabel"],
      [data-testid="stMain"] [data-testid="stMetricValue"],
      [data-testid="stMain"] [data-testid="stMetricValue"] *,
      [data-testid="stMain"] [data-testid="stExpander"] {
        color: var(--tw-text);
      }

      /* Headings follow the selected theme as well. */
      [data-testid="stMain"] h1,
      [data-testid="stMain"] h2,
      [data-testid="stMain"] h3,
      [data-testid="stMain"] h4 {
        color: var(--tw-text);
      }

      /* Hero banner uses its own dark background, so keep its text white. */
      [data-testid="stMain"] .hero {
        padding: 1.7rem 1.9rem;
        margin: 0 0 1.25rem 0;
        border-radius: 18px;
        background: linear-gradient(135deg, #172554, #2563eb);
        color: #ffffff;
      }
      [data-testid="stMain"] .hero h1,
      [data-testid="stMain"] .hero p,
      [data-testid="stMain"] .hero .eyebrow {
        color: #ffffff !important;
      }
      [data-testid="stMain"] .hero h1 {
        margin: 0.45rem 0;
      }
      [data-testid="stMain"] .hero p {
        margin-bottom: 0;
        opacity: 0.94;
      }
      [data-testid="stMain"] .eyebrow {
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.12em;
      }

      /* Feature cards adapt to light and dark themes. */
      [data-testid="stMain"] .mini-card {
        padding: 1rem 1.1rem;
        border: 1px solid var(--tw-border);
        border-radius: 14px;
        background: var(--tw-panel);
        min-height: 92px;
      }
      [data-testid="stMain"] .mini-label,
      [data-testid="stMain"] .footnote {
        color: var(--text-color, inherit);
        opacity: 0.72;
        font-size: 0.82rem;
      }
      [data-testid="stMain"] .mini-value {
        color: var(--tw-text);
        font-size: 1.12rem;
        font-weight: 700;
        margin-top: 0.3rem;
      }

      [data-testid="stMain"] .section-title {
        color: var(--tw-text);
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 0.4rem;
      }
      [data-testid="stMain"] .section-sub {
        color: var(--text-color, inherit);
        opacity: 0.75;
        margin-bottom: 1rem;
      }

      /* Do not override input/select text; Streamlit/BaseWeb handles theme contrast. */
      [data-testid="stMain"] input,
      [data-testid="stMain"] textarea {
        color: var(--tw-text);
      }

      [data-testid="stMain"] .stButton button,
      [data-testid="stMain"] [data-testid="stFormSubmitButton"] button {
        font-weight: 650;
      }

      [data-testid="stMain"] .result-good {
        color: #168567;
        font-weight: 700;
        font-size: 1.05rem;
      }
      [data-testid="stMain"] .result-warn {
        color: #c27a12;
        font-weight: 700;
        font-size: 1.05rem;
      }
      [data-testid="stMain"] .footnote {
        margin-top: 2rem;
        padding-top: 0.8rem;
        border-top: 1px solid var(--tw-border);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown('<div class="brand">✈️ TripWise</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Travel package insights")
    st.markdown("Use customer and trip details to estimate the likelihood of a travel package purchase.")
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("1. Enter customer details.\n2. Select **Predict**.\n3. Review the model result.")
    st.markdown("---")
    st.caption("Machine-learning demo · Predictions are estimates")

# ----------------------------- Hero -----------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">CUSTOMER ANALYTICS · ML PREDICTION</div>
      <h1>Travel Package Predictor</h1>
      <p>Understand whether a customer may purchase a travel package. Enter the customer profile below to generate a prediction from your trained model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick feature cards
k1, k2, k3 = st.columns(3)
for col, label, value in [
    (k1, "INPUT", "Customer profile"),
    (k2, "ENGINE", "Trained ML model"),
    (k3, "OUTPUT", "Purchase likelihood"),
]:
    with col:
        st.markdown(
            f'<div class="mini-card"><div class="mini-label">{label}</div><div class="mini-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")
st.markdown('<div class="section-title">Customer details</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Complete the fields. Values start with sensible examples and can be adjusted.</div>', unsafe_allow_html=True)

# ----------------------------- Load model assets -----------------------------
@st.cache_resource
def load_assets():
    model_path = Path("model.joblib")
    prep_path = Path("preprocessor.joblib")
    if not model_path.exists() or not prep_path.exists():
        missing = [p.name for p in (model_path, prep_path) if not p.exists()]
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))
    saved_models = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    if isinstance(saved_models, dict):
        model = saved_models.get("Xg boost", next(iter(saved_models.values())))
    else:
        model = saved_models
    return model, preprocessor

# ----------------------------- Input form -----------------------------
with st.form("customer_input_form", clear_on_submit=False):
    st.markdown("#### 01 · Customer profile")
    a, b, c = st.columns(3)
    with a:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital status", ["Married", "Unmarried", "Divorced"])
    with b:
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly income", min_value=0, max_value=1000000, value=25000, step=1000)
    with c:
        city_tier = st.selectbox("City tier", [1, 2, 3])
        contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
        passport = st.selectbox("Has passport?", ["No", "Yes"])

    st.markdown("---")
    st.markdown("#### 02 · Travel & sales interaction")
    d, e, f = st.columns(3)
    with d:
        product = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        property_star = st.selectbox("Preferred property star", [3, 4, 5])
        trips = st.number_input("Number of trips", min_value=0, max_value=100, value=2, step=1)
    with e:
        duration = st.number_input("Duration of pitch (minutes)", min_value=0, max_value=300, value=15, step=1)
        followups = st.number_input("Number of follow-ups", min_value=0, max_value=50, value=3, step=1)
        pitch_satisfaction = st.selectbox("Pitch satisfaction score", [1, 2, 3, 4, 5])
    with f:
        own_car = st.selectbox("Owns a car?", ["No", "Yes"])
        persons_visiting = st.number_input("Adults / persons visiting", min_value=1, max_value=20, value=2, step=1)
        children_visiting = st.number_input("Children visiting", min_value=0, max_value=20, value=0, step=1)

    st.write("")
    submitted = st.form_submit_button("✦  Predict purchase likelihood", use_container_width=True)

# ----------------------------- Prediction -----------------------------
if submitted:
    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfFollowups": followups,
        "ProductPitched": product,
        "PreferredPropertyStar": property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_satisfaction,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
        "TotalVisiting": persons_visiting + children_visiting,
    }])

    try:
        model, preprocessor = load_assets()
        transformed_input = preprocessor.transform(input_data)
        prediction = int(model.predict(transformed_input)[0])
        st.markdown("---")
        st.markdown('<div class="section-title">Prediction result</div>', unsafe_allow_html=True)
        left, right = st.columns([1.3, 1])
        with left:
            if prediction == 1:
                st.markdown('<div class="result-good">✓  Higher purchase intent predicted</div>', unsafe_allow_html=True)
                st.success("The model predicts that this customer may purchase the travel package.")
            else:
                st.markdown('<div class="result-warn">↗  Lower purchase intent predicted</div>', unsafe_allow_html=True)
                st.info("The model predicts that this customer may not purchase the travel package.")
        with right:
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(transformed_input)[0]
                classes = list(getattr(model, "classes_", range(len(probabilities))))
                positive_index = classes.index(1) if 1 in classes else len(probabilities) - 1
                probability = float(probabilities[positive_index]) * 100
                st.metric("Estimated purchase probability", f"{probability:.1f}%")
                st.progress(max(0.0, min(1.0, probability / 100)))
            else:
                st.caption("This model does not provide probability estimates.")
        with st.expander("View submitted feature values"):
            st.dataframe(input_data, use_container_width=True, hide_index=True)
    except FileNotFoundError as exc:
        st.error(str(exc) + ". Place both files in the same folder as main.py.")
    except Exception as exc:
        st.error("Prediction could not be completed. Check that model.joblib and preprocessor.joblib match this notebook and its feature columns.")
        st.exception(exc)

st.markdown('<div class="footnote">TripWise · Travel Package Purchase Prediction · For demonstration purposes only</div>', unsafe_allow_html=True)

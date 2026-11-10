import joblib
import pandas as pd
import streamlit as st
from features import BASE_COLS          # same file used during training

st.set_page_config(page_title="Bankruptcy Predictor", page_icon="🏦", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


model = load_model()
LEVEL_LABEL = {0.0: "Low (0)", 0.5: "Medium (0.5)", 1.0: "High (1)"}

st.title("🏦 Bankruptcy Prediction")
st.write("Rate a company on six qualitative factors and the model estimates its **chance of bankruptcy**.")

tab_single, tab_batch, tab_about = st.tabs(["🔮 Single company", "📂 Many companies (CSV)", "ℹ️ About"])

# ---------------------------------------------------------------- single prediction
with tab_single:
    with st.form("company_form"):
        st.subheader("Risk factors  (High = worse)")
        c1, c2, c3 = st.columns(3)
        industrial = c1.selectbox("Industrial risk", [0.0, 0.5, 1.0], index=1, format_func=LEVEL_LABEL.get)
        management = c2.selectbox("Management risk", [0.0, 0.5, 1.0], index=1, format_func=LEVEL_LABEL.get)
        operating = c3.selectbox("Operating risk", [0.0, 0.5, 1.0], index=1, format_func=LEVEL_LABEL.get)

        st.subheader("Strength factors  (High = better)")
        c4, c5, c6 = st.columns(3)
        flexibility = c4.selectbox("Financial flexibility", [0.0, 0.5, 1.0], index=1, format_func=LEVEL_LABEL.get)
        credibility = c5.selectbox("Credibility", [0.0, 0.5, 1.0], index=1, format_func=LEVEL_LABEL.get)
        competitiveness = c6.selectbox("Competitiveness", [0.0, 0.5, 1.0], index=1, format_func=LEVEL_LABEL.get)

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        values = {
            "industrial_risk": industrial, "management_risk": management, "operating_risk": operating,
            "financial_flexibility": flexibility, "credibility": credibility, "competitiveness": competitiveness,
        }
        row = pd.DataFrame([values])[BASE_COLS]
        prob = float(model.predict_proba(row)[0, 1])        # probability of class 1 = bankruptcy

        st.divider()
        st.metric("Probability of bankruptcy", f"{prob:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))

        if prob >= 0.5:
            st.error("⚠️ HIGH RISK – the model predicts **BANKRUPTCY**.")
        elif prob >= 0.2:
            st.warning("🟠 MEDIUM RISK – keep a close watch on this company.")
        else:
            st.success("✅ LOW RISK – the model predicts **NO bankruptcy**.")

# ---------------------------------------------------------------- batch prediction
with tab_batch:
    st.write("Upload a CSV with these columns: " + ", ".join(f"`{c}`" for c in BASE_COLS))
    file = st.file_uploader("CSV file", type=["csv"])

    if file is not None:
        data = pd.read_csv(file, sep=None, engine="python")      # works for ; or ,
        data.columns = data.columns.str.strip()
        missing = [c for c in BASE_COLS if c not in data.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            probs = model.predict_proba(data[BASE_COLS])[:, 1]
            data["bankruptcy_probability"] = probs.round(4)
            data["prediction"] = ["bankruptcy" if p >= 0.5 else "non-bankruptcy" for p in probs]
            st.dataframe(data)
            st.download_button("Download results", data.to_csv(index=False), "predictions.csv", "text/csv")

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown(
        """
**Model:** trained on 250 companies; best model picked by cross-validation (identical rows kept together so scores are honest).

**Inputs** – each is 0 (low), 0.5 (medium) or 1 (high):
- *Risk factors* (industrial, management, operating): higher means more risk.
- *Strength factors* (financial flexibility, credibility, competitiveness): higher means a stronger company.

**Limits:** the data set is small and the factors are qualitative judgements, so treat the result as decision support, not as a final verdict.
        """
    )

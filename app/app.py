import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "credit_scoring_model.joblib"

st.set_page_config(page_title="Credit Scoring", layout="centered")
st.title("Credit Scoring — Previsão de Risco de Inadimplência")
st.write("Aplicação demonstrativa para estimar a probabilidade de inadimplência de uma solicitação de crédito.")

model = joblib.load(MODEL_PATH)

age = st.number_input("Idade", 18, 100, 30)
income = st.number_input("Renda anual", 0, 1_000_000, 60000)
home = st.selectbox("Moradia", ["RENT", "OWN", "MORTGAGE", "OTHER"])
emp = st.number_input("Tempo de emprego", 0.0, 80.0, 5.0)
intent = st.selectbox("Finalidade do empréstimo", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
grade = st.selectbox("Grade do empréstimo", list("ABCDEFG"))
amount = st.number_input("Valor do empréstimo", 500, 50000, 10000)
rate = st.number_input("Taxa de juros (%)", 0.0, 40.0, 12.0)
pct_income = amount / income if income else 1.0
default_file = st.selectbox("Histórico prévio de default", ["N", "Y"])
cred_hist = st.number_input("Tempo de histórico de crédito", 0, 50, 5)

data = pd.DataFrame([{
    "person_age": age,
    "person_income": income,
    "person_home_ownership": home,
    "person_emp_length": emp,
    "loan_intent": intent,
    "loan_grade": grade,
    "loan_amnt": amount,
    "loan_int_rate": rate,
    "loan_percent_income": pct_income,
    "cb_person_default_on_file": default_file,
    "cb_person_cred_hist_length": cred_hist,
}])

if st.button("Calcular risco"):
    prob = model.predict_proba(data)[0, 1]
    st.metric("Probabilidade estimada de inadimplência", f"{prob:.1%}")
    if prob >= 0.5:
        st.error("Alto risco")
    elif prob >= 0.25:
        st.warning("Risco moderado")
    else:
        st.success("Baixo risco")

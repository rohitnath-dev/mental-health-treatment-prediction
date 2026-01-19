import streamlit as st
import joblib
import pandas as pd

# =========================
# Load model & preprocessors
# =========================

model = joblib.load("/content/drive/MyDrive/Mental_Health_Project/mental_health_model.pkl")
ordinal_encoder = joblib.load("/content/drive/MyDrive/Mental_Health_Project/ordinal_encoder.pkl")
label_encoders = joblib.load("/content/drive/MyDrive/Mental_Health_Project/label_encoders.pkl")
feature_columns = joblib.load("/content/drive/MyDrive/Mental_Health_Project/feature_columns.pkl")

# =========================
# UI → Backend mappings
# =========================

GENDER_MAP = {
    "Male": 0,
    "Female": 1,
    "Other": 2
}

DAYS_INDOORS_MAP = {
    "Go out every day": 0,
    "1–14 days": 1,
    "15–30 days": 2,
    "31–60 days": 3,
    "More than 60 days": 4
}

BINARY_MAP = {
    "Yes": 1,
    "No": 0,
    "Maybe": -1,
    "Not sure": -1,
    "Unknown": -1
}

# =========================
# App UI
# =========================

st.title("Mental Health Treatment Predictor")
st.markdown("Predicts likelihood of seeking mental health treatment based on survey inputs.")
st.divider()

# =========================
# User Inputs (TEXT ONLY)
# =========================

gender_text = st.selectbox("Gender", list(GENDER_MAP.keys()))
occupation_text = st.selectbox(
    "Occupation",
    list(label_encoders["Occupation"].classes_)
)

self_employed = st.selectbox("Self Employed?", ["Yes", "No", "Unknown"])
family_history = st.selectbox("Family History of Mental Illness?", ["Yes", "No"])

days_text = st.selectbox("Days Indoors", list(DAYS_INDOORS_MAP.keys()))

growing_stress = st.selectbox("Growing Stress Level", ["Low", "Medium", "High"])
mood_swings = st.selectbox("Mood Swings", ["Low", "Medium", "High"])

mental_health_interview = st.selectbox(
    "Mental Health Interview at Workplace?", ["Yes", "No", "Maybe"]
)
care_options = st.selectbox(
    "Care Options Available?", ["Yes", "No", "Not sure"]
)
mental_health_history = st.selectbox(
    "Mental Health History?", ["Yes", "No", "Maybe"]
)

changes_habits = st.selectbox(
    "Changes in Habits?",
    list(label_encoders["Changes_Habits"].classes_)
)
coping_struggles = st.selectbox(
    "Coping Struggles?",
    list(label_encoders["Coping_Struggles"].classes_)
)
work_interest = st.selectbox(
    "Work Interest Affected?",
    list(label_encoders["Work_Interest"].classes_)
)
social_weakness = st.selectbox(
    "Social Weakness?",
    list(label_encoders["Social_Weakness"].classes_)
)

# =========================
# Prediction
# =========================

if st.button("Predict"):

    df = pd.DataFrame([{
        "Gender": GENDER_MAP[gender_text],
        "Occupation": occupation_text,
        "self_employed": BINARY_MAP[self_employed],
        "family_history": BINARY_MAP[family_history],
        "Days_Indoors": DAYS_INDOORS_MAP[days_text],
        "Growing_Stress": growing_stress,
        "Mood_Swings": mood_swings,
        "mental_health_interview": BINARY_MAP[mental_health_interview],
        "care_options": BINARY_MAP[care_options],
        "Mental_Health_History": BINARY_MAP[mental_health_history],
        "Changes_Habits": changes_habits,
        "Coping_Struggles": coping_struggles,
        "Work_Interest": work_interest,
        "Social_Weakness": social_weakness,
    }])

    # Ordinal encoding
    df[["Growing_Stress", "Mood_Swings"]] = ordinal_encoder.transform(
        df[["Growing_Stress", "Mood_Swings"]]
    )

    # Label encoding (ONLY text columns)
    SKIP_COLS = ["Gender", "Days_Indoors"]
    for col, le in label_encoders.items():
        if col not in SKIP_COLS:
            df[col] = le.transform(df[col].astype(str))

    # Reorder columns
    df = df[feature_columns]

    # Predict
    prediction = model.predict(df)[0]

    st.divider()
    if prediction == 1:
        st.error("Likely to seek mental health treatment")
    else:
        st.success("Unlikely to seek mental health treatment")

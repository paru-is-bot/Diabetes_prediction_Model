import os

import streamlit as st

from backend import DATA_PATH, MODEL_PATH, load_data, load_model, predict, train_model


st.set_page_config(page_title="Diabetes Prediction", layout="centered")


@st.cache(allow_output_mutation=True)
def _get_data():
    return load_data(DATA_PATH)


def get_default_inputs(df):
    # medians provide reasonable defaults
    med = df.median()
    return {
        "Pregnancies": int(med["Pregnancies"]),
        "Glucose": float(med["Glucose"]),
        "BloodPressure": float(med["BloodPressure"]),
        "SkinThickness": float(med["SkinThickness"]),
        "Insulin": float(med["Insulin"]),
        "BMI": float(med["BMI"]),
        "DiabetesPedigreeFunction": float(med["DiabetesPedigreeFunction"]),
        "Age": int(med["Age"]),
    }


def main():
    st.title("Diabetes Prediction")
    st.write("Enter patient details to predict whether they have diabetes.")

    df = _get_data()
    defaults = get_default_inputs(df.drop(columns=["Outcome"]))

    with st.form("input_form"):
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=defaults["Pregnancies"])
        glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=float(defaults["Glucose"]))
        bp = st.number_input("BloodPressure", min_value=0.0, max_value=200.0, value=float(defaults["BloodPressure"]))
        skin = st.number_input("SkinThickness", min_value=0.0, max_value=100.0, value=float(defaults["SkinThickness"]))
        insulin = st.number_input("Insulin", min_value=0.0, max_value=1000.0, value=float(defaults["Insulin"]))
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=float(defaults["BMI"]))
        dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=10.0, value=float(defaults["DiabetesPedigreeFunction"]))
        age = st.number_input("Age", min_value=0, max_value=120, value=defaults["Age"])

        submitted = st.form_submit_button("Predict")

    # model area
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            st.info("Loaded saved model.")
        except Exception:
            model = None

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Train / Retrain model"):
            with st.spinner("Training model... this may take a few seconds"):
                model, acc = train_model(df)
            st.success(f"Model trained — test accuracy: {acc:.3f}")

    with col2:
        if st.button("Load existing model"):
            try:
                model = load_model(MODEL_PATH)
                st.success("Model loaded from disk.")
            except FileNotFoundError:
                st.error("No saved model found. Train a model first.")

    if submitted:
        if model is None:
            st.warning("No model available. Training a model now...")
            with st.spinner("Training model..."):
                model, acc = train_model(df)
            st.success(f"Model trained — test accuracy: {acc:.3f}")

        features = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }

        res = predict(model, features)
        prob = res["probability"]
        label = res["label"]

        st.markdown("---")
        if label == 1:
            st.error(f"Prediction: Patient likely HAS diabetes (probability {prob:.2%})")
        else:
            st.success(f"Prediction: Patient likely DOES NOT have diabetes (probability {prob:.2%})")


if __name__ == "__main__":
    main()
#THis is the frontend file
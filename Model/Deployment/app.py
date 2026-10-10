import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path
from datetime import datetime

# -------------------------
# Page configuration
# -------------------------
st.set_page_config(
    page_title="Medical Diagnostic Network",
    page_icon="🩺",
    layout="centered"
)

# -------------------------
# Interface styling
# -------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 12px 16px;
    }
    div.stButton > button, div.stDownloadButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Test metrics (fill in your actual test results)
# -------------------------
# Replace each None with the value from your test set evaluation.
# Enter values as decimals between 0 and 1, for example 0.95 for 95%.
# Metrics left as None are shown as "Not set" in the app.
TEST_METRICS = {
    "Accuracy": None,
    "Precision": None,
    "Recall": None,
    "F1-score": None,
    "ROC-AUC": None,
}

# Number of records in your test set, for example 104. Leave as None if unknown.
TEST_SET_SIZE = None

# -------------------------
# Load trained model and scaler
# -------------------------
BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "final_model.keras"
SCALER_PATH = BASE_DIR / "diabetes_age_scaler .pkl"

@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

try:
    model, age_scaler = load_model_and_scaler()
except Exception as e:
    st.error(
        "The application could not load the trained model or age scaler. "
        "Please check that both files are present and compatible."
    )
    st.stop()

# -------------------------
# Helper functions
# -------------------------
def yes_no_to_number(value):
    return 1 if value == "Yes" else 0

def gender_to_number(value):
    return 1 if value == "Male" else 0

def format_metric(value):
    if value is None:
        return "Not set"
    return f"{value:.1%}"

def build_report(patient_name, generated_at, inputs, prediction, probability):
    label = "Positive for the diabetes class" if prediction == 1 else "Negative for the diabetes class"

    lines = []
    lines.append("MEDICAL DIAGNOSTIC NETWORK")
    lines.append("AI-Based Diabetes Screening Report")
    lines.append("=" * 50)
    lines.append(f"Patient name: {patient_name}")
    lines.append(f"Date generated: {generated_at}")
    lines.append("")
    lines.append("PATIENT INFORMATION SUMMARY")
    lines.append("-" * 50)
    for key, value in inputs.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("SCREENING RESULT")
    lines.append("-" * 50)
    lines.append(f"Classification: {label}")
    lines.append(f"Model output score for the positive class: {probability:.1%}")
    lines.append("")
    lines.append("MODEL TEST METRICS")
    lines.append("-" * 50)
    for key, value in TEST_METRICS.items():
        lines.append(f"{key}: {format_metric(value)}")
    if TEST_SET_SIZE is not None:
        lines.append(f"Test set size: {TEST_SET_SIZE}")
    lines.append("")
    lines.append("NOTE")
    lines.append("-" * 50)
    lines.append(
        "This score is the model's output, not necessarily a calibrated "
        "estimate of the patient's probability of having diabetes."
    )
    lines.append(
        "This result cannot confirm or rule out diabetes. A qualified "
        "healthcare professional should assess the patient and arrange "
        "appropriate testing."
    )
    return "\n".join(lines)

# -------------------------
# Application interface
# -------------------------
st.title("🩺 Medical Diagnostic Network")
st.subheader("AI-Based Diabetes Screening Support")

st.write(
    "Enter the patient's information and symptoms to obtain a "
    "model-generated classification. This tool is intended for "
    "screening support and does not replace professional medical "
    "assessment or laboratory testing."
)

st.info(
    "Please answer the questions based on the patient's information. "
    "The model's output is a prediction, not a confirmed diagnosis."
)

screening_tab, about_tab = st.tabs(["Screening", "About the Model"])

# =========================
# Screening tab
# =========================
with screening_tab:

    # -------------------------
    # Patient information
    # -------------------------
    st.header("Patient Information")

    patient_name = st.text_input(
        "Patient name (optional)",
        placeholder="Enter the patient's full name",
        help="The name is only used on the screen and in the downloadable "
             "report. It is not given to the model."
    )

    col_age, col_gender = st.columns(2)

    with col_age:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30,
            step=1
        )

    with col_gender:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

    st.header("Symptoms and Other Features")
    st.write("Select Yes or No for each item.")

    col_left, col_right = st.columns(2)

    with col_left:
        polyuria = st.radio(
            "Does the patient experience frequent urination (polyuria)?",
            ["No", "Yes"],
            horizontal=True
        )

        polydipsia = st.radio(
            "Does the patient experience excessive thirst (polydipsia)?",
            ["No", "Yes"],
            horizontal=True
        )

        sudden_weight_loss = st.radio(
            "Has the patient experienced sudden weight loss?",
            ["No", "Yes"],
            horizontal=True
        )

        weakness = st.radio(
            "Does the patient experience weakness?",
            ["No", "Yes"],
            horizontal=True
        )

    with col_right:
        polyphagia = st.radio(
            "Does the patient experience excessive hunger (polyphagia)?",
            ["No", "Yes"],
            horizontal=True
        )

        genital_thrush = st.radio(
            "Does the patient have genital thrush?",
            ["No", "Yes"],
            horizontal=True
        )

        visual_blurring = st.radio(
            "Does the patient experience blurred vision?",
            ["No", "Yes"],
            horizontal=True
        )

        obesity = st.radio(
            "Is obesity recorded for the patient?",
            ["No", "Yes"],
            horizontal=True
        )

    # -------------------------
    # Patient information summary
    # -------------------------
    display_name = patient_name.strip() if patient_name.strip() else "Not provided"

    inputs_summary = {
        "Patient name": display_name,
        "Age": f"{age} years",
        "Gender": gender,
        "Frequent urination (polyuria)": polyuria,
        "Excessive thirst (polydipsia)": polydipsia,
        "Sudden weight loss": sudden_weight_loss,
        "Weakness": weakness,
        "Excessive hunger (polyphagia)": polyphagia,
        "Genital thrush": genital_thrush,
        "Blurred vision": visual_blurring,
        "Obesity": obesity,
    }

    st.header("Patient Information Summary")
    st.caption("Review the information below before generating the result.")

    summary_df = pd.DataFrame(
        {
            "Item": list(inputs_summary.keys()),
            "Entered value": list(inputs_summary.values()),
        }
    )
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    # -------------------------
    # Prepare model input
    # -------------------------
    if st.button("Generate Screening Result", type="primary"):

        # Scale age using the fitted scaler from preprocessing.
        scaled_age = age_scaler.transform(
            pd.DataFrame({"age": [age]})
        )[0][0]

        # Preserve the exact training feature order.
        input_data = pd.DataFrame(
            [[
                scaled_age,
                gender_to_number(gender),
                yes_no_to_number(polyuria),
                yes_no_to_number(polydipsia),
                yes_no_to_number(sudden_weight_loss),
                yes_no_to_number(weakness),
                yes_no_to_number(polyphagia),
                yes_no_to_number(genital_thrush),
                yes_no_to_number(visual_blurring),
                yes_no_to_number(obesity)
            ]],
            columns=[
                "age",
                "gender",
                "polyuria",
                "polydipsia",
                "sudden_weight_loss",
                "weakness",
                "polyphagia",
                "genital_thrush",
                "visual_blurring",
                "obesity"
            ]
        )

        # Generate prediction
        probability = float(
            model.predict(input_data, verbose=0)[0][0]
        )

        prediction = 1 if probability >= 0.5 else 0

        # Keep the result and the inputs it was based on, so the result
        # stays on screen after the download button is clicked.
        st.session_state["result"] = {
            "name": display_name,
            "generated_at": datetime.now().strftime("%d %B %Y, %H:%M"),
            "inputs": dict(inputs_summary),
            "prediction": prediction,
            "probability": probability,
        }

    # -------------------------
    # Display result
    # -------------------------
    if "result" in st.session_state:
        result = st.session_state["result"]

        st.divider()
        st.header("Screening Result")

        if result["name"] != "Not provided":
            st.write(f"**Patient:** {result['name']}")
        st.caption(f"Generated on {result['generated_at']}")

        if result["prediction"] == 1:
            st.warning(
                "The model classified this patient's input as "
                "**Positive for the diabetes class**."
            )
        else:
            st.success(
                "The model classified this patient's input as "
                "**Negative for the diabetes class**."
            )

        st.write(
            f"Model output score for the positive class: "
            f"**{result['probability']:.1%}**"
        )

        st.caption(
            "This score is the model's output, not necessarily a calibrated "
            "estimate of the patient's probability of having diabetes."
        )

        with st.expander("Inputs used for this result"):
            used_df = pd.DataFrame(
                {
                    "Item": list(result["inputs"].keys()),
                    "Entered value": list(result["inputs"].values()),
                }
            )
            st.dataframe(used_df, hide_index=True, use_container_width=True)

        st.warning(
            "This result cannot confirm or rule out diabetes. "
            "A qualified healthcare professional should assess the patient "
            "and arrange appropriate testing."
        )

        report_text = build_report(
            result["name"],
            result["generated_at"],
            result["inputs"],
            result["prediction"],
            result["probability"],
        )

        safe_name = "".join(
            c for c in result["name"] if c.isalnum() or c in (" ", "_", "-")
        ).strip().replace(" ", "_")
        file_name = (
            f"screening_result_{safe_name}.txt"
            if safe_name and result["name"] != "Not provided"
            else "screening_result.txt"
        )

        st.download_button(
            label="Download Results and Input Summary",
            data=report_text,
            file_name=file_name,
            mime="text/plain"
        )

# =========================
# About the Model tab
# =========================
with about_tab:
    st.header("About the Model")

    st.write(
        "This application uses a neural network built with TensorFlow and "
        "Keras to classify whether a patient's input pattern resembles the "
        "diabetes class. The model uses ten features: age, gender, and eight "
        "yes or no symptom and condition items (polyuria, polydipsia, sudden "
        "weight loss, weakness, polyphagia, genital thrush, blurred vision, "
        "and obesity). Age is scaled with the scaler fitted during "
        "preprocessing. A model output score of 50% or higher is classified "
        "as positive."
    )

    st.subheader("Test Performance")

    metric_items = list(TEST_METRICS.items())
    for start in range(0, len(metric_items), 3):
        row = metric_items[start:start + 3]
        cols = st.columns(3)
        for col, (metric_name, metric_value) in zip(cols, row):
            col.metric(metric_name, format_metric(metric_value))

    if TEST_SET_SIZE is not None:
        st.caption(f"Metrics were calculated on a test set of {TEST_SET_SIZE} records.")
    else:
        st.caption("Metrics were calculated on the held-out test set.")

    st.subheader("Limitations")
    st.write(
        "The model was trained on a limited set of records and only on the "
        "features listed above. Patient names are not used by the model. "
        "The output is a screening aid and must be followed by clinical "
        "assessment and laboratory testing."
    )

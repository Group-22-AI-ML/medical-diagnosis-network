import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="Medical Diagnostic Network",
    layout="centered"
)

# page style
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

# test set results from the notebook
TEST_METRICS = {
    "Accuracy": 0.8974,
    "Precision": 0.8600,
    "Recall": 0.9773,
    "F1-score": 0.9149,
}

# load the trained model and the age scaler
BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "final_model.keras"
SCALER_PATH = BASE_DIR / "diabetes_age_scaler.pkl"

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

def yes_no_to_number(value):
    return 1 if value == "Yes" else 0

def gender_to_number(value):
    return 1 if value == "Male" else 0

def build_report(patient_name, generated_at, inputs, prediction):
    label = (
        "Positive for the diabetes class"
        if prediction == 1
        else "Negative for the diabetes class"
    )

    lines = [
        "Medical Diagnostic Network",
        "Diabetes Screening Report",
        "",
        f"Patient name: {patient_name}",
        f"Date: {generated_at}",
        "",
        "Patient Information",
    ]
    for key, value in inputs.items():
        if key != "Patient name":
            lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Screening Result")
    lines.append(f"Classification: {label}")
    return "\n".join(lines)

st.title("Medical Diagnostic Network")
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

with screening_tab:

    st.header("Patient Information")

    patient_name = st.text_input(
        "Patient name (optional)",
        placeholder="Enter the patient's full name",
        help="The name only appears on the screen and in the downloaded "
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

    # summary of what has been entered so far
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
    st.dataframe(summary_df, hide_index=True, width="stretch")

    if st.button("Generate Screening Result", type="primary"):

        # scale age with the scaler fitted during preprocessing
        scaled_age = age_scaler.transform(
            pd.DataFrame({"age": [age]})
        )[0][0]

        # feature order has to match the training data
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

        probability = float(
            model.predict(input_data, verbose=0)[0][0]
        )

        prediction = 1 if probability >= 0.5 else 0

        # keep the result in session state so it stays on screen
        # after the download button is pressed
        st.session_state["result"] = {
            "name": display_name,
            "generated_at": datetime.now().strftime("%d %B %Y, %H:%M"),
            "inputs": dict(inputs_summary),
            "prediction": prediction,
        }

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

        with st.expander("Inputs used for this result"):
            used_df = pd.DataFrame(
                {
                    "Item": list(result["inputs"].keys()),
                    "Entered value": list(result["inputs"].values()),
                }
            )
            st.dataframe(used_df, hide_index=True, width="stretch")

        report_text = build_report(
            result["name"],
            result["generated_at"],
            result["inputs"],
            result["prediction"],
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

with about_tab:
    st.header("About the Model")

    st.write(
        "This application uses a neural network built with TensorFlow and "
        "Keras to classify whether a patient's input pattern resembles the "
        "diabetes class. The model uses ten features: age, gender, and eight "
        "yes or no symptom and condition items (polyuria, polydipsia, sudden "
        "weight loss, weakness, polyphagia, genital thrush, blurred vision "
        "and obesity). Age is scaled with the scaler fitted during "
        "preprocessing, and an output of 50% or higher is classified as "
        "positive. Patient names are not used by the model."
    )

    st.subheader("Test Performance")

    cols = st.columns(len(TEST_METRICS))
    for col, (metric_name, metric_value) in zip(cols, TEST_METRICS.items()):
        col.metric(metric_name, f"{metric_value:.2%}")

    st.caption("Metrics were calculated on the held-out test set.")
ult["name"] != "Not provided":
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

# About the Model tab
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

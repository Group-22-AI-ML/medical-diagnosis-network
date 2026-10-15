# Medical Diagnostic Network
### An AI-Based Decision Support System for Diabetes Screening

## 1. Project Overview

The Medical Diagnostic Network is a machine learning project that
screens for early-stage diabetes using a patient's age and reported
symptoms.

The system uses a feedforward neural network to classify input cases
as Positive or Negative and presents the result through an interactive
web application built with Streamlit.

The project demonstrates the development of a machine learning
solution from data preprocessing and model training to evaluation
and deployment.

## 2. Project Objectives

- Preprocess the Early Stage Diabetes Risk Prediction dataset.
- Develop and evaluate a neural network classifier.
- Compare model configurations using validation F1-score.
- Evaluate the selected model on a held-out test set.
- Deploy the model through a user-friendly Streamlit application.

## 3. Dataset

The project uses the Early Stage Diabetes Risk Prediction dataset,
which contains 520 records collected through questionnaires at the
Sylhet Diabetes Hospital in Bangladesh.

The model uses age and reported symptoms as input features.

The target variable is `class`, representing the positive or
negative class in the dataset.

**Dataset limitation:** The dataset is relatively small and was
collected at a single hospital. The model has not been clinically
validated on other populations.

Dataset source:
https://archive.ics.uci.edu/dataset/529

## 4. Model Architecture

The selected model is a fully connected feedforward neural network
implemented with TensorFlow and Keras.

| Component | Configuration |
|---|---|
| Input layer | 9 features |
| Hidden layer 1 | 32 neurons, ReLU |
| Hidden layer 2 | 16 neurons, ReLU |
| Hidden layer 3 | 8 neurons, ReLU |
| Output layer | 1 neuron, sigmoid |
| Loss function | Binary cross-entropy |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Dropout rate | 0.1 |
| Decision threshold | 0.5 |

The model's input configuration must match the preprocessing pipeline
used during training.

## 5. Model Evaluation

The final model was evaluated on a held-out test set containing
78 records.

| Metric | Test result |
|---|---:|
| Accuracy | 92.31% |
| Precision | 93.18% |
| Recall | 93.18% |
| F1-score | 93.18% |

The test confusion matrix recorded 31 true negatives, 41 true
positives, 3 false positives, and 3 false negatives.

These results describe performance on the available test set and do
not establish clinical effectiveness or performance on other
populations.

## 6. Technologies Used

- Python
- TensorFlow and Keras
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Joblib
- Google Colab
- GitHub
- Streamlit Community Cloud

## 7. Running the Application Locally

### Prerequisites

- Python installed on your computer
- Git
- The project files and trained model artefacts

### Installation

Clone the repository:

```bash
git clone https://github.com/Group-22-AI-ML/medical-diagnosis-network.git
cd medical-diagnosis-network
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

Note: The model and scaler files must be available at the paths
expected by the application.

## 8. Deployment

The application is deployed using Streamlit Community Cloud.

Live application: https://diab-nndss.streamlit.app/
## 9. Limitations

- The model was developed using a dataset of 520 records.
- The dataset was collected at a single hospital in Bangladesh.
- Inputs are based on reported symptoms and age.
- The model may not generalise to other populations.
- False positives and false negatives are possible.
- The system has not undergone clinical validation.

## 10. Future Work

Future improvements could include expanding the dataset, evaluating
additional classification algorithms, testing the model on
independent populations, improving model interpretability, and
conducting appropriate clinical evaluation.

## 11. Disclaimer

This project is intended for educational and research purposes.
Its output is not a confirmed diagnosis. Anyone concerned about
diabetes should seek assessment from a qualified healthcare
professional.

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

# Title and Description
st.set_page_config(page_title="ML Model Comparison", layout="wide")
st.title("Machine Learning Model Comparison App")
st.markdown("""
This app allows you to:
- Explore a dataset
- Choose features and a target for classification
- Select a machine learning model (SVC, Logistic Regression, Random Forest)
- View evaluation metrics (Confusion Matrix, ROC Curve, Precision-Recall Curve)
""")

# File Upload
st.sidebar.header("Dataset Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"], help="Upload a CSV file to get started.")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df, width=800, height=400)

    # Feature Selection
    st.sidebar.header("Feature Selection")
    target_col = st.sidebar.selectbox("Select Target Column", options=df.columns, help="Choose the target variable for classification.")

    # Encode categorical variables
    le = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = le.fit_transform(df[column])

    features = df.drop(columns=[target_col])
    target = df[target_col]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Model Selection
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["SVC", "Logistic Regression", "Random Forest"],
        help="Choose a machine learning model to train and evaluate."
    )

    if model_choice == "SVC":
        model = SVC(probability=True, random_state=42)
    elif model_choice == "Logistic Regression":
        model = LogisticRegression(random_state=42)
    else:
        model = RandomForestClassifier(random_state=42)

    # Train Model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')

    st.write("### Model Performance")
    st.metric(label="Precision", value=f"{precision:.2f}")
    st.metric(label="Recall", value=f"{recall:.2f}")

    # Visualization
    st.write("### Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("#### Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap='Blues')
        st.pyplot(fig)

    with col2:
        st.write("#### ROC Curve")
        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
        st.pyplot(fig)

    with col3:
        st.write("#### Precision-Recall Curve")
        fig, ax = plt.subplots()
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
        st.pyplot(fig)
else:
    st.info("Please upload a CSV file to start using the app.")

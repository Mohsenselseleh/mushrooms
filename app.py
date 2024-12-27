import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("https://github.com/Mohsenselseleh/mushrooms/blob/main/mushrooms.csv")
    data = pd.get_dummies(data, drop_first=True)  # Convert categorical to numeric
    return data

# Train model
@st.cache_data
def train_model(data):
    X = data.drop(columns=['type_poisonous'])  # Features
    y = data['type_poisonous']                # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# Main app
st.title("Mushroom Classification 🍄")
st.write("Predict whether a mushroom is edible or poisonous based on its features.")

# Load and display data
data = load_data()
if st.checkbox("Show raw data"):
    st.write(data)

# Train model
model, accuracy = train_model(data)
st.write(f"Model Accuracy: {accuracy:.2f}")

# User inputs
st.sidebar.header("Input Mushroom Features")
cap_shape = st.sidebar.slider("Cap Shape", 0, 5, 1)
cap_surface = st.sidebar.slider("Cap Surface", 0, 3, 1)
cap_color = st.sidebar.slider("Cap Color", 0, 9, 1)

# Prepare input data
input_features = pd.DataFrame({
    'cap_shape': [cap_shape],
    'cap_surface': [cap_surface],
    'cap_color': [cap_color]
})

# Predict
if st.sidebar.button("Predict"):
    prediction = model.predict(input_features)
    result = "Poisonous ☠️" if prediction[0] else "Edible 🍲"
    st.subheader("Prediction")
    st.write(result)

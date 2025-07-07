import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from model_utils import load_model

data = fetch_california_housing()
feature_names = data.feature_names

st.title("🏠 House Price Predictor App")
st.write("Select an algorithm and input values to predict California house prices.")

model_option = st.sidebar.selectbox(
    "Choose Model",
    ("Linear Regression", "Decision Tree", "Random Forest", "Support Vector Machine")
)

model_map = {
    "Linear Regression": "linear_reg",
    "Random Forest": "random_forest",
    "Decision Tree": "decision_tree",
    "Support Vector Machine": "svm"
}

model, accuracy = load_model(model_map[model_option])

user_inputs = []
st.header("📥 Input Feature Values")
for feature in feature_names:
    value = st.slider(feature, float(np.min(data.data[:, feature_names.index(feature)])),
                      float(np.max(data.data[:, feature_names.index(feature)])),
                      float(np.mean(data.data[:, feature_names.index(feature)])))
    user_inputs.append(value)

input_array = np.array(user_inputs).reshape(1, -1)
prediction = model.predict(input_array)[0]

st.subheader("📊 Prediction Result")
st.success(f"Predicted House Price (in $100,000s): **{prediction:.2f}**")
st.write(f"🔍 Model Used: **{model_option}**")
st.write(f"📈 Model R² Score: **{accuracy:.2f}**")

if st.checkbox("Show Feature Importance (if available)"):
    if hasattr(model, 'feature_importances_'):
        import matplotlib.pyplot as plt
        import seaborn as sns
        st.write("Feature Importance:")
        fig, ax = plt.subplots()
        sns.barplot(x=model.feature_importances_, y=feature_names, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("This model does not support feature importance.")

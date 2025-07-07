import pickle
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# Ensure the directory exists
os.makedirs("models", exist_ok=True)

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "linear_reg": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=100),
    "decision_tree": DecisionTreeRegressor(),
    "svm": SVR()
}

# Train and save each model
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = r2_score(y_test, preds)

    # Save model and accuracy
    with open(f"models/{name}.pkl", "wb") as f:
        pickle.dump((model, acc), f)

print("✅ Models trained and saved to /models folder.")

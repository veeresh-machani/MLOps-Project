# Iris species classification: load data, train Random Forest, evaluate, save model.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# --- Data loading and exploration ---
# Load CSV (features + species label). Update path if your file is data/iris.csv.
data = pd.read_csv(
    r"/Users/0573mpind/Documents/Personal/Agentic AI/MLOps Project/data/iris.csv"
)
print(data.head())
print(data["species"].value_counts())

# Encode species names as integers for sklearn (setosa=0, versicolor=1, virginica=2).
data["species"] = data["species"].map({"setosa": 0, "versicolor": 1, "virginica": 2})

# Features (X) and target (y).
x = data.drop("species", axis=1)
y = data["species"]

# 80% train / 20% test split; random_state keeps splits reproducible.
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# --- Train model ---
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# --- Evaluate on held-out test set ---
y_pred = model.predict(X_test)
print(y_pred)

accuracy = accuracy_score(Y_test, y_pred)
print("Model Accuracy:", accuracy)

# --- Persist model for inference / deployment ---
joblib.dump(model, "model/model.pkl")

# Example: reload saved model (path should match dump above, e.g. model.pkl or model/model.pkl).
loaded_model = joblib.load("model/model.pkl")

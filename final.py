import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")

if "user_id" in df.columns:
    df = df.drop(columns=["user_id"])

X = df.drop("needs_assistance_flag", axis=1)
y = df["needs_assistance_flag"]

X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(
    max_iter=2000,
    solver="liblinear",
    class_weight="balanced"
)

model.fit(X_train, y_train)

joblib.dump(model, "logistic_model.pkl")
joblib.dump(X_encoded.columns.tolist(), "model_features.pkl")

print("Model and features saved successfully.")

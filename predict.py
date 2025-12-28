import pandas as pd
import joblib

model = joblib.load("logistic_model.pkl")
features = joblib.load("model_features.pkl")

print("=== User Assistance Prediction System ===")

error_count = int(input("Enter error count: "))
task_time = float(input("Enter task completion time (seconds): "))
satisfaction = int(input("Enter user satisfaction score (1–5): "))

age_group = input("Enter age group (18-30 / 30-45 / 45-60 / 60+): ")
device = input("Enter device type (Smartphone / Smartwatch / Tablet): ")
input_mode = input("Enter input mode (Touch / Voice): ")
task_type = input("Enter task type (Make Payment / Transfer Money): ")

data = {
    "error_count": error_count,
    "task_completion_time_sec": task_time,
    "user_satisfaction_score": satisfaction,
    "age_group": age_group,
    "device_type": device,
    "input_mode": input_mode,
    "task_type": task_type
}

df_input = pd.DataFrame([data])
df_input_encoded = pd.get_dummies(df_input)

df_input_encoded = df_input_encoded.reindex(
    columns=features,
    fill_value=0
)

prediction = model.predict(df_input_encoded)[0]
probability = model.predict_proba(df_input_encoded)[0][1]

print("\nPrediction Result:")
if prediction == 1:
    print(f"User NEEDS assistance (probability: {probability:.2f})")
else:
    print(f"User does NOT need assistance (probability: {probability:.2f})")

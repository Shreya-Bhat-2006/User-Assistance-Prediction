# User Assistance Prediction

## 1. Project Overview

The goal of this project is to predict whether a user **needs assistance** while performing a task, based on their interaction behavior and device-related features.

This is a **binary classification problem**, where:

- `1` → User needs assistance
- `0` → User does not need assistance

The project follows a **complete machine learning workflow**, from data preparation to model deployment and GitHub publishing.

---

## 2. Dataset Description

The dataset contains user interaction data such as:

- Error count during task execution
- Task completion time
- User satisfaction score
- Age group
- Device type
- Input mode
- Task type

Target variable:

- **`needs_assistance_flag`**

An ID column (`user_id`) was present but removed since it does not contribute to prediction.

---

## 3. Data Preprocessing

The following preprocessing steps were applied:

1. **Dropped irrelevant column**
    - `user_id` was removed as it has no predictive value.
2. **Separated features and target**
    - Features (`X`)
    - Target label (`y`)
3. **Handled categorical variables**
    - One-hot encoding was applied using `pd.get_dummies()`
    - `drop_first=True` was used to avoid multicollinearity
4. **Train–test split**
    - 80% training data
    - 20% testing data
    - Stratified split to preserve class distribution

---

## 4. Model Selection

Initially, multiple models were tested:

- Logistic Regression
- Decision Tree
- Random Forest

### Why Logistic Regression was chosen

Although tree-based models showed very high accuracy, they were found to **over-rely on strong interaction features** and were prone to memorization.

Logistic Regression was chosen because:

- It generalizes better
- It produces stable train and test performance
- It is less sensitive to overfitting
- It provides interpretable coefficients
- It reflects realistic learning instead of memorization

Class imbalance was handled using:

```python
class_weight="balanced"

```

---

## 5. Model Training

The Logistic Regression model was trained using:

- Solver: `liblinear`
- Maximum iterations: `2000`
- Balanced class weights

The model was trained on the training dataset and evaluated on unseen test data.

---

## 6. Model Evaluation

### Performance Summary

- **Train Accuracy:** ~89%
- **Test Accuracy:** ~89%

This indicates:

- No overfitting
- Good generalization

### Confusion Matrix Analysis

- Most users needing assistance were correctly identified
- False predictions were minimal
- The model showed strong precision and recall for both classes

### Classification Metrics

- High precision for class `1` (needs assistance)
- Strong recall for class `0` (does not need assistance)
- Balanced overall performance

---

## 7. Feature Importance Interpretation

Logistic Regression coefficients were analyzed to understand model behavior.

### Strong positive indicators:

- `error_count`
- `task_completion_time_sec`
- Certain input modes

### Strong negative indicators:

- Specific age groups
- Certain device types
- Familiar task types

These relationships align with real-world intuition, confirming meaningful learning.

---

## 8. Model Saving and Reusability

To make the project reusable and interactive:

- The trained model was saved using `joblib`
- Feature column names were saved separately to ensure consistent input format

Saved files:

- `logistic_model.pkl`
- `model_features.pkl`

---

## 9. Interactive Prediction Script

An interactive command-line script (`predict.py`) was created that:

- Loads the trained model
- Accepts user input
- Encodes input features
- Outputs prediction and probability

This allows predictions **without retraining the model**.

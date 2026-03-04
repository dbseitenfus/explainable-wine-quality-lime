"""
Explainable Wine Quality with LIME
===================================
Uses LIME (Local Interpretable Model-agnostic Explanations) to explain
individual predictions from ML models trained on the Red Wine Quality dataset.

Instructor: Dr. E. Kapetanios (PhD, ETH Zurich)
"""

from training import load_and_preprocess, train_models
from explanation import build_explainer, explain_instance


# --- Data Loading & Preprocessing ---
X_train, X_test, y_train, y_test, feature_names = load_and_preprocess("winequality-red.csv")

model1, model2, model3 = train_models(X_train, X_test, y_train, y_test)

explainer = build_explainer(X_train, feature_names)

# --- Explain a Low-Quality Prediction (index 20) ---
explain_instance(explainer, X_test, y_test, model2, index=20, output_path="output_32_0.png")

# --- Explain a High-Quality Prediction (index 2) ---
explain_instance(explainer, X_test, y_test, model2, index=2, output_path="output_36_0.png")

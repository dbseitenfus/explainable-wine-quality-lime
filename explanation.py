"""
Explanation module
==================
Builds a LIME explainer and generates explanation plots for individual predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular


def build_explainer(X_train: np.ndarray, feature_names: list):
    return lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=['goodquality'],
        verbose=True,
        mode='regression',
    )


def explain_instance(explainer, X_test, y_test, model, index: int, output_path: str):
    print(f"\ny_test[{index}:{index+5}]:")
    print(y_test[index:index + 5])
    print(f"Model predictions: {model.predict(X_test[index:index + 5])}")

    exp = explainer.explain_instance(X_test[index], model.predict)
    fig = exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

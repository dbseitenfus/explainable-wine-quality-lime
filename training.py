"""
Training module
===============
Loads and preprocesses the Red Wine Quality dataset, then trains three classifiers.
Returns all artifacts needed for LIME explanation.
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


def load_and_preprocess(csv_path: str = "winequality-red.csv"):
    df = pd.read_csv(csv_path)
    print("Rows, columns:", df.shape)
    print(df.head())

    # Missing values
    print(df.isna().sum())

    # Feature types
    X_feature_names = [col for col in df.columns if df[col].dtype == float]
    print("Continuous features:", X_feature_names)

    Y_feature_names = [col for col in df.columns if df[col].dtype == np.int64]
    print("Target features:", Y_feature_names)

    # Binary classification: good quality if score >= 7
    df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]
    print(df['goodquality'].value_counts())

    X_raw = df.drop(['quality', 'goodquality'], axis=1)
    y = df['goodquality']

    X = StandardScaler().fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0
    )
    print("Train set distribution:\n", y_train.value_counts())
    print("Test set distribution:\n", y_test.value_counts())

    return X_train, X_test, y_train, y_test, X_feature_names


def train_models(X_train, X_test, y_train, y_test):
    model1 = DecisionTreeClassifier(random_state=1)
    model1.fit(X_train, y_train)
    print("\nDecision Tree:")
    print(classification_report(y_test, model1.predict(X_test)))

    model2 = RandomForestClassifier(random_state=1)
    model2.fit(X_train, y_train)
    print("\nRandom Forest:")
    print(classification_report(y_test, model2.predict(X_test)))

    model3 = AdaBoostClassifier(random_state=1)
    model3.fit(X_train, y_train)
    print("\nAdaBoost:")
    print(classification_report(y_test, model3.predict(X_test)))

    return model1, model2, model3

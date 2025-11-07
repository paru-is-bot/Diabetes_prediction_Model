import os
from typing import Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = os.path.join(os.path.dirname(__file__), "diabetes.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the diabetes CSV into a DataFrame."""
    return pd.read_csv(path)


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Simple preprocessing: replace zeroes in some columns with the median.

    The Pima dataset uses 0 to indicate missing values for some features.
    """
    df = df.copy()
    cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for c in cols_with_zeros:
        # replace 0 with NaN then fill with median
        df[c] = df[c].replace(0, np.nan)
        median = df[c].median()
        df[c] = df[c].fillna(median)
    return df


def train_model(df: pd.DataFrame = None, save_path: str = MODEL_PATH) -> Tuple[Pipeline, float]:
    """Train a simple pipeline (Scaler + LogisticRegression) and save it.

    Returns the trained pipeline and test accuracy.
    """
    if df is None:
        df = load_data()
    df = _preprocess(df)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    # persist
    joblib.dump(pipe, save_path)
    return pipe, acc


def load_model(path: str = MODEL_PATH) -> Pipeline:
    """Load a saved model pipeline. Raises if not found."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def predict(model: Pipeline, features: Dict[str, Any]) -> Dict[str, Any]:
    """Predict outcome for a single sample given a features dict.

    features: mapping from column name to value (Pregnancies, Glucose, ... Age)
    Returns dict with 'label' (0/1) and 'probability' (float for positive class).
    """
    # preserve order used in dataset
    cols = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
    x = np.array([[float(features.get(c, 0)) for c in cols]])
    pred = int(model.predict(x)[0])
    proba = float(model.predict_proba(x)[0][1])
    return {"label": pred, "probability": proba}

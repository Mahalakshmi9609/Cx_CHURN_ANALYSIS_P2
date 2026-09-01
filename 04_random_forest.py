"""Step 4 - Train and evaluate a Random Forest churn classifier."""

from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from config import DATA_CLEANED, OUTPUT_DIR, TARGET


# 4.2 Hyperparameters Used
RANDOM_STATE = 42
VALIDATION_SIZE = 0.10
TEST_SIZE = 0.10
N_ESTIMATORS = 100
MAX_DEPTH = 8


def main():
    """Train, evaluate, and save the Random Forest model."""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4.3 Code Walkthrough: load cleaned data and separate X from y.
    df = pd.read_csv(DATA_CLEANED)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE + TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    validation_predictions = model.predict(X_validation)
    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # 4.4 Results & Metrics
    validation_metrics = {
        "Accuracy": accuracy_score(y_validation, validation_predictions),
        "Precision": precision_score(y_validation, validation_predictions, zero_division=0),
        "Recall": recall_score(y_validation, validation_predictions, zero_division=0),
        "F1": f1_score(y_validation, validation_predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_validation, validation_probabilities),
    }
    test_metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }

    print("Random Forest Evaluation (80% train / 10% validation / 10% test)")
    print("=" * 30)
    print("Validation metrics:")
    for name, value in validation_metrics.items():
        print(f"{name}: {value:.4f}")
    print("\nTest metrics:")
    for name, value in test_metrics.items():
        print(f"{name}: {value:.4f}")

    # 4.5 Feature Importance
    feature_importance = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
    )
    print("\nFeature Importance")
    print(feature_importance.to_string())

    importance_path = output_dir / "random_forest_feature_importance.png"
    plt.figure(figsize=(10, 6))
    feature_importance.sort_values().plot(kind="barh")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(importance_path, dpi=150)
    plt.close()

    model_path = output_dir / "random_forest_model.pkl"
    with model_path.open("wb") as model_file:
        pickle.dump(model, model_file)

    print(f"\nSaved feature importance chart to: {importance_path}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()

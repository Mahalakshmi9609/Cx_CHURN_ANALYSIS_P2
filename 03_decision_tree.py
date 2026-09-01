"""Step 3 - Train and evaluate a decision tree churn classifier."""

from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from config import DATA_CLEANED, OUTPUT_DIR, TARGET


RANDOM_STATE = 42
VALIDATION_SIZE = 0.10
TEST_SIZE = 0.10


def main():
    """Train, evaluate, visualize, and save the decision tree model."""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    model = DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    validation_predictions = model.predict(X_validation)
    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

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

    print("Decision Tree Evaluation (80% train / 10% validation / 10% test)")
    print("=" * 30)
    print("Validation metrics:")
    for name, value in validation_metrics.items():
        print(f"{name}: {value:.4f}")
    print("\nTest metrics:")
    for name, value in test_metrics.items():
        print(f"{name}: {value:.4f}")

    feature_importance = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
    )
    print("\nFeature Importance")
    print(feature_importance.to_string())

    tree_path = output_dir / "decision_tree_top_4_levels.png"
    plt.figure(figsize=(24, 12))
    plot_tree(
        model,
        feature_names=X.columns,
        class_names=[str(label) for label in model.classes_],
        max_depth=4,
        filled=True,
        rounded=True,
        proportion=True,
        impurity=False,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(tree_path, dpi=150, bbox_inches="tight")
    plt.close()

    model_path = output_dir / "decision_tree_model.pkl"
    with model_path.open("wb") as model_file:
        pickle.dump(model, model_file)

    print(f"\nSaved tree visualization to: {tree_path}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()

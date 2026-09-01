"""Step 5 - Compare and evaluate the churn classification models."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier

from config import DATA_CLEANED, OUTPUT_DIR, TARGET


RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def build_models():
    """Return the models used in Steps 3 and 4."""
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def main():
    """Compare both models and save evaluation results and visualizations."""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_CLEANED)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    models = build_models()
    validation_metrics = {}
    test_metrics = {}
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        validation_predictions = model.predict(X_validation)
        validation_probabilities = model.predict_proba(X_validation)[:, 1]
        predictions[name] = model.predict(X_test)
        probabilities[name] = model.predict_proba(X_test)[:, 1]
        validation_metrics[name] = {
            "Accuracy": accuracy_score(y_validation, validation_predictions),
            "Precision": precision_score(y_validation, validation_predictions, zero_division=0),
            "Recall": recall_score(y_validation, validation_predictions, zero_division=0),
            "F1": f1_score(y_validation, validation_predictions, zero_division=0),
            "ROC-AUC": roc_auc_score(y_validation, validation_probabilities),
        }
        test_metrics[name] = {
            "Accuracy": accuracy_score(y_test, predictions[name]),
            "Precision": precision_score(y_test, predictions[name], zero_division=0),
            "Recall": recall_score(y_test, predictions[name], zero_division=0),
            "F1": f1_score(y_test, predictions[name], zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probabilities[name]),
        }

    metrics_df = pd.DataFrame(test_metrics).T
    metrics_df.index.name = "Model"
    metrics_df.to_csv(output_dir / "model_comparison_metrics.csv")
    validation_df = pd.DataFrame(validation_metrics).T
    validation_df.index.name = "Model"
    validation_df.to_csv(output_dir / "model_validation_metrics.csv")

    # 5.1 Objective and 5.2 Metrics Compared
    print("Model Comparison and Evaluation")
    print("=" * 38)
    print("Objective: compare Decision Tree and Random Forest churn classifiers.")
    print("\nValidation-set metrics:")
    print(validation_df.round(4).to_string())
    print("\nTest-set metrics:")
    print(metrics_df.round(4).to_string())

    # 5.3 ROC Curve Comparison
    plt.figure(figsize=(9, 7))
    for name, scores in probabilities.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, scores)
        curve_auc = auc(false_positive_rate, true_positive_rate)
        plt.plot(false_positive_rate, true_positive_rate, label=f"{name} (AUC = {curve_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve_comparison.png", dpi=150)
    plt.close()

    # 5.4 Precision-Recall Curve
    plt.figure(figsize=(9, 7))
    positive_rate = y_test.mean()
    for name, scores in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.4f})")
    plt.axhline(positive_rate, color="k", linestyle="--", label="Class prevalence")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve_comparison.png", dpi=150)
    plt.close()

    # 5.5 Confusion Matrices
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, (name, prediction) in zip(axes, predictions.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            prediction,
            display_labels=["No Churn", "Churn"],
            cmap="Blues",
            ax=axis,
            colorbar=False,
        )
        axis.set_title(name)
    figure.suptitle("Confusion Matrices")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrices.png", dpi=150)
    plt.close(figure)

    # 5.6 Cross-Validation Results
    scoring = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1": "f1",
        "ROC-AUC": "roc_auc",
    }
    cross_validator = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    cv_rows = []
    for name, model in build_models().items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cross_validator,
            scoring=scoring,
            n_jobs=1,
        )
        row = {"Model": name}
        for metric_name in scoring:
            key = f"test_{metric_name}"
            row[f"{metric_name} Mean"] = scores[key].mean()
            row[f"{metric_name} Std"] = scores[key].std()
        cv_rows.append(row)

    cv_df = pd.DataFrame(cv_rows).set_index("Model")
    cv_df.to_csv(output_dir / "cross_validation_results.csv")
    print("\n5-fold cross-validation results:")
    print(cv_df.round(4).to_string())

    # 5.7 Which Model is Better?
    winner = cv_df["F1 Mean"].idxmax()
    print(f"\nBetter model by mean cross-validated F1: {winner}")
    print("The winner is selected using F1 because it balances precision and recall.")


if __name__ == "__main__":
    main()

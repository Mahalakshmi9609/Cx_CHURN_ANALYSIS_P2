# Customer Churn Prediction Project Summary

## 1. Project Objective

This project analyzes customer behavior and builds machine learning models to predict customer churn. It combines data exploration, data cleaning, classification models, model evaluation, and SQL-based business analysis.

## 2. Project Workflow

1. Explore the raw customer churn dataset.
2. Clean missing values, duplicates, outliers, and categorical columns.
3. Remove `CustomerID` because it is not a predictive feature.
4. Save the cleaned dataset to `outputs/cleaned_data.csv`.
5. Train and evaluate a Decision Tree classifier.
6. Train and evaluate a Random Forest classifier.
7. Compare the models using classification metrics, ROC curves, precision-recall curves, confusion matrices, and cross-validation.
8. Run SQL queries using an in-memory SQLite database to identify business insights.

## 3. Files

- `01_data_exploration.py`: Exploratory data analysis and distribution charts.
- `02_data_cleaning.py`: Data cleaning, encoding, outlier capping, and export.
- `03_decision_tree.py`: Decision Tree training, evaluation, visualization, and pickle export.
- `04_random_forest.py`: Random Forest training, evaluation, feature importance, and pickle export.
- `05_model_evaluation.py`: Model comparison and evaluation visualizations.
- `06_sql_queries.py`: SQLite setup and 15 churn-analysis queries.
- `config.py`: Shared dataset paths, output path, and target-column configuration.

## 4. Data Split

Both machine learning models use a stratified three-way split:

- Training: 80%
- Validation: 10%
- Testing: 10%

The models are trained on the training set, checked on the validation set, and evaluated finally on the untouched test set. The validation set supports model comparison without using the final test set prematurely.

## 5. Models and Hyperparameters

### Decision Tree

- Algorithm: `DecisionTreeClassifier`
- `max_depth=8`
- `random_state=42`

### Random Forest

- Algorithm: `RandomForestClassifier`
- `n_estimators=100`
- `max_depth=8`
- `random_state=42`
- `n_jobs=-1`

## 6. Evaluation Metrics

- **Accuracy:** Percentage of all predictions that are correct.
- **Precision:** Percentage of predicted churn customers who actually churned.
- **Recall:** Percentage of actual churn customers correctly identified.
- **F1 Score:** Harmonic mean of Precision and Recall.
- **ROC-AUC:** Measures how well the model separates churn from non-churn across classification thresholds.

## 7. Model Results

### Validation Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0.9941 | 0.9934 | 0.9941 | 0.9938 | 0.9999 |
| Random Forest | 0.9938 | 0.9980 | 0.9888 | 0.9934 | 0.9998 |

### Final Test Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0.9957 | 0.9938 | 0.9970 | 0.9954 | 0.9998 |
| Random Forest | 0.9941 | 0.9970 | 0.9905 | 0.9938 | 0.9998 |

### Five-Fold Cross-Validation

| Model | Accuracy Mean | Precision Mean | Recall Mean | F1 Mean | ROC-AUC Mean |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0.9939 | 0.9931 | 0.9941 | 0.9936 | 0.9993 |
| Random Forest | 0.9928 | 0.9973 | 0.9875 | 0.9924 | 0.9999 |

## 8. Which Model Is Better?

The Decision Tree is the better overall model for this project because it has the highest final test accuracy, recall, and F1 score. It also has the highest mean cross-validated F1 score.

The Random Forest has higher precision and slightly higher cross-validated ROC-AUC. It produces fewer false-positive churn predictions, but it misses more actual churn customers. Since identifying actual churn customers is important for retention actions, the Decision Tree is preferred.

## 9. Important Features

The strongest Decision Tree features were:

1. Payment Delay
2. Support Calls
3. Gender
4. Tenure
5. Usage Frequency

The strongest Random Forest features were:

1. Payment Delay
2. Support Calls
3. Tenure
4. Usage Frequency
5. Gender

Payment delay and support calls are the clearest recurring warning signals for churn.

## 10. SQL Business Insights

The SQL analysis found these important patterns:

- Overall churn rate: `47.37%`.
- Customers with payment delays of 30 days or more have a `77.07%` churn rate.
- Customers with payment delays under 15 days have a `10.10%` churn rate.
- Monthly contracts have the highest contract churn rate at `51.61%`.
- Customers with seven support calls have the highest observed support-call churn rate at `61.78%`.
- Customers with total spend under 500 have a higher churn rate than customers spending 500 or more.
- Customers aged 50 or older have a higher churn rate than younger age groups.

Recommended business actions include payment reminders, proactive support follow-up, and retention offers for customers showing multiple risk signals.

## 11. Key Machine Learning Concepts

- **Parameter:** A value learned by a model during training, such as tree split rules.
- **Hyperparameter:** A setting selected before training, such as `max_depth` or `n_estimators`.
- **Overfitting:** The model memorizes training data and performs worse on unseen data.
- **Underfitting:** The model is too simple and performs poorly on both training and unseen data.
- **Validation set:** Data used during development to compare models or tune choices.
- **Test set:** Held-out data used once for the final unbiased evaluation.

## 12. Generated Outputs

The `outputs` folder contains:

- Cleaned CSV data
- Exploratory distribution charts
- Decision Tree visualization
- Decision Tree pickle model
- Random Forest feature-importance chart
- Random Forest pickle model
- ROC curve comparison
- Precision-recall curve comparison
- Confusion matrices
- Test-set model comparison CSV
- Validation-set model comparison CSV
- Cross-validation results CSV

## Final Conclusion

The project successfully prepares the churn data, trains two classification models, evaluates them using a three-way split and cross-validation, and extracts business insights through SQL. The Decision Tree is the recommended model because it provides the strongest balance of accuracy, recall, and F1 score for identifying customers who may churn.

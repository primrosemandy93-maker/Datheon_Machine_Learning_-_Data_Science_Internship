# Datheon Machine Learning & Data Science Internship

Two project notebooks are included. Both contain code, saved tables and plots, and written conclusions.

## Project index

| Task | Notebook | Work included |
|---|---|---|
| 1 — Iris classification | [Task_1_Iris_Classification.ipynb](task_1_iris/Task_1_Iris_Classification.ipynb) | Data checks, pairplot, stratified split, scaled pipelines, Logistic Regression and Decision Tree, classification metrics, confusion matrices, two-feature decision boundaries |
| 2 — Housing prediction | [Task_2_Housing_Price_Prediction.ipynb](task_2_housing/Task_2_Housing_Price_Prediction.ipynb) | Data checks, training-only correlations, feature engineering, Linear Regression, Ridge and Lasso tuning, five-fold validation, test metrics, residual plots and price-band error analysis |

No Task 3 or Task 4 deliverables are included. This index describes the files present; it does not certify compliance with an unavailable full internship rubric.

## Run the notebooks

### Google Colab (recommended for both tasks)

Open a notebook and use its **Open in Colab** badge. Start a fresh Python runtime and run all cells in order.

Task 1 loads Iris using `sklearn.datasets.load_iris(as_frame=True)`; no separate data file is needed.

Task 2 reads `/content/sample_data/california_housing_train.csv`, the 17,000-row sample CSV used for its saved results. Confirm that this file exists in the Colab runtime before running. This CSV is not committed to the repository. If absent, obtain the same Colab sample CSV before proceeding. Do not substitute `fetch_california_housing` silently: it is a different dataset representation and will not reproduce these results.

### Local Jupyter

Use Python 3.10 or later as a practical starting point:

```bash
python -m venv .venv
# Activate .venv using the command for your operating system.
python -m pip install -r requirements.txt
python -m jupyter lab
```

Task 1 runs without a separate dataset. For Task 2, obtain the same CSV and change only its read path to your local copy. The current Task 2 loader is Colab-specific.

The dependency list is an unpinned starting environment. Original package versions were not recorded, so it is not an exact lockfile for the saved outputs.

## Results recorded in the notebooks

These are saved notebook outputs, not a claim that both notebooks have been freshly executed in every environment.

### Task 1

120 training rows and 30 test rows; 80/20 stratified split with `random_state=42`. Precision, recall and F1 are macro averages.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.933 | 0.933 | 0.933 | 0.933 |
| Decision Tree (max depth 3) | 0.967 | 0.970 | 0.967 | 0.967 |

The decision-boundary figures use separate two-feature models. The table uses all four features. A one-prediction advantage on this small test set does not establish general superiority.

### Task 2

13,600 training rows and 3,400 test rows; 80/20 random split with `random_state=42`. Five-fold shuffled cross-validation is performed on training data. Imputation and scaling are fitted inside pipelines.

| Model | Test RMSE ($) | Test MAE ($) | Test R² |
|---|---:|---:|---:|
| Linear Regression — original | 68,078.326 | 49,983.475 | 0.664 |
| Linear Regression — engineered | 67,358.059 | 49,219.454 | 0.671 |
| Ridge — engineered | 67,358.064 | 49,219.444 | 0.671 |
| Lasso — engineered | 67,358.062 | 49,219.453 | 0.671 |

Feature engineering reduced test RMSE by about 1.1%, but **mean cross-validation RMSE increased** from $70,049.11 (original) to $70,805.39 (engineered). Fold standard deviation also increased, from $1,859.51 to $5,178.29. Therefore, the test-set gain is not consistent evidence of a general improvement.

Ridge and Lasso both selected alpha 0.01 from the tested grid and added no meaningful test improvement. The $300,000-and-above group had the largest absolute dollar error. The CSV describes historical census block groups, not current individual property prices; capped targets and a random geographic split limit interpretation.

## Review and submission

See [REPOSITORY_REVIEW.md](REPOSITORY_REVIEW.md) for the audit findings and remaining verification steps. Submit the **repository root URL**, rather than a single notebook link, if Datheon requests another submission:

https://github.com/primrosemandy93-maker/Datheon_Machine_Learning_-_Data_Science_Internship

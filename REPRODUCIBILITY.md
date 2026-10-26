# Reproducibility record

**Validated:** 26 October 2026, Linux, Python 3.12.14.

## What was checked

- Both notebooks' code cells executed top-to-bottom in separate, fresh Python processes using `python scripts/reproduce.py --update-notebooks`.
- Notebook structures were validated with nbformat.
- Iris accuracy and macro F1 matched the historical results; housing RMSE and R² matched within the runner's documented numeric tolerances.
- Housing price-band counts sum to 3,400 test areas for every model.
- Additional training-only robustness cells ran successfully: Iris majority-class baseline plus 50 repeated stratified splits per model; housing median baseline plus five random and five geographic folds per model. Geographic group overlap is checked within the notebook.
- Housing data matched SHA-256 `bb9ca4041a66e0ed691cb825188f16fdab518435c4950a080951be2fdfe30d4d` and shape 17,000 × 9. The exact file was retrieved from the Google source linked in the project README.
- The housing loader was separately executed with an empty temporary project directory; its fresh download and checksum validation passed.
- The exported confusion matrices and housing actual-versus-predicted chart were visually inspected. Project pages link directly to measured CSVs, charts and notebooks.

## Environment

| Package | Validated version |
|---|---|
| Python | 3.12.14 |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| SciPy | 1.17.0 |
| scikit-learn | 1.8.0 |
| Matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| nbformat | 5.10.4 |

`requirements.txt` pins direct dependencies, not a complete transitive lock. Small numerical or rendering differences may occur across operating systems or numerical libraries. Package installation and the first housing-data download require internet access; subsequent executions can use the cached dataset.

## Execution details

The runner executes the actual ordinary-Python notebook cells using a fresh child process for each project. It captures standard output, display tables and Matplotlib plots without requiring a Jupyter server. It uses the invoking Python interpreter, sets numerical thread limits to one and stops on execution errors. Fixed split/fold seeds are 42. Housing GridSearchCV uses one worker. Interactive Colab execution uses its own packages and is not a substitute for the pinned local reference environment.

Generated CSVs and PNGs are committed for readers who do not run code. Full executed notebooks are regenerated in `results/executed.ipynb` and ignored by Git. `--update-notebooks` refreshes source notebook outputs with links to the saved PNGs, avoiding duplicated image payloads. CSV results are regenerated from code, not hand-entered.

## Interpretation boundaries

Reproducing a metric verifies this workflow on these data and splits. It does not prove deployment readiness or future predictive performance. Iris has just 30 test examples. Housing has geographic dependence, capped historical targets, and mixed validation/test findings. No external validation, production deployment or measured business impact is claimed.

The original notebook paths, historical test splits and test-model settings are retained. Changes make execution portable, export evidence, clarify the mixed housing findings, and add explicitly labelled training-only robustness analyses. New validation measurements are reported separately from the historical test results. No new model was selected by retuning against the existing test set.

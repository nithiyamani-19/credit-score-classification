# Credit Score Classification

Machine learning model that predicts a customer's credit score (Poor / Standard / Good) from financial and personal attributes.

## Dataset
Kaggle "Credit Score Classification". `train.csv` has 100,000 labeled rows and `test.csv` has 50,000 rows without labels. Each customer has 8 monthly records.

## Approach
1. Data exploration
2. Cleaning (junk values, impossible values, per-customer imputation)
3. Feature engineering (loan counts, loan-type flags, financial ratios)
4. EDA visualizations (Matplotlib, Seaborn, Plotly)
5. Customer-grouped 80/20 split to avoid leakage
6. Models: Logistic Regression, Random Forest, XGBoost
7. Evaluation: accuracy, classification report, confusion matrix, macro F1
8. Feature importance: built-in, permutation (SHAP optional in the notebook)
9. Predictions on test.csv

## Results (validation set)
| Model               |   Accuracy |   Macro F1 |
|:--------------------|-----------:|-----------:|
| Random Forest       |     0.701  |     0.6783 |
| XGBoost             |     0.6956 |     0.6698 |
| Logistic Regression |     0.6632 |     0.633  |

Best model: **Random Forest**

## Top features (permutation importance)
1. Interest_Rate
2. Credit_Mix
3. Outstanding_Debt
4. Delay_from_due_date
5. Payment_of_Min_Amount_Yes

## Repository contents
- `notebooks/credit_score_classification.ipynb`: full analysis and model code
- `reports/final_report.docx`: final report
- `outputs/`: predictions, evaluation summary, feature importance tables
- `figures/`: all charts

## Note
`test.csv` has no `Credit_Score`, so metrics come from a customer-grouped validation split of `train.csv`. `test.csv` is used only for final predictions.

## How to run
```
pip install -r requirements.txt
```
Open the notebook in Jupyter or Google Colab, place `train.csv` and `test.csv` next to it, and run all cells.

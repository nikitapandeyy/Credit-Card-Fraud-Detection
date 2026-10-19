# Credit-Card-Fraud-Detection
# Credit Card Fraud Detection (Random Forest, XGBoost, Logistic Regression)

Detecting fraudulent card transactions in a highly imbalanced dataset (~0.17% fraud), with a focus on
metrics and decisions that matter in production: PR-AUC, recall at fixed precision, cost-based thresholds,
and explainability.

## Dataset
Kaggle "Credit Card Fraud Detection" (ULB): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Download `creditcard.csv` and place it in the project root (it is not included in this repo).

## Approach
- Stratified 60/20/20 train / validation / test split; scaler fit on train only (no leakage)
- Models: Logistic Regression (baseline), Random Forest, XGBoost
- Imbalance handling: class weights vs SMOTE (applied to the training set only)
- Metrics: PR-AUC, ROC-AUC, precision, recall, F1, recall at 90% precision (accuracy is deliberately not used)
- Cost-based decision threshold chosen on validation data, reported once on the test set
- SHAP for feature-level explanations

## Results (replace with your numbers after running)
| Model | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | | | | |
| Random Forest | | | | |
| XGBoost | | | | |
| Random Forest + SMOTE | | | | |

Key findings: (2-3 lines in your own words)

## Limitations
- Features V1-V28 are anonymised (PCA), so SHAP rankings cannot be mapped to business meaning
- Cost values (FN / FP) are assumptions
- Random split, not time-based; a time-based split would be a stricter test

## Run it
```
pip install -r requirements.txt
jupyter notebook fraud_detection.ipynb
```
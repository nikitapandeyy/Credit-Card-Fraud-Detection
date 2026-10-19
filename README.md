# Credit Card Fraud Detection: Random Forest, XGBoost and Logistic Regression

Detecting fraudulent card transactions in a highly imbalanced dataset (0.17% fraud). The project focuses on the
decisions that matter in a real fraud setting: leakage-safe evaluation, precision-recall metrics instead of accuracy,
a cost-based decision threshold, and model explainability.

**Headline result:** Random Forest + SMOTE reached **0.868 PR-AUC** on a held-out test set. With a cost-tuned threshold it
caught **84 of 95 frauds (88.4% recall)**, versus 76 of 95 (80.0%) at the default threshold, and cut expected loss by about
**24%** under the stated cost assumptions.

## Dataset

Kaggle "Credit Card Fraud Detection" (ULB): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

| Item | Value |
|---|---|
| Transactions | 284,807 (European cardholders, two days in September 2013) |
| Frauds | 492 (0.173%) |
| Duplicates removed | 1,081 |
| Features | `V1`-`V28` (PCA-anonymised), `Time`, `Amount` |
| Engineered | `Hour` (hour of day from `Time`); `Amount` scaled with `RobustScaler` |

The CSV is not stored in this repo. The notebook downloads it automatically in its first cell.

## Approach

1. **Leakage-safe split.** Stratified 60 / 20 / 20 train / validation / test split after removing duplicates
   (train 170,235 rows with 284 frauds; validation 56,745 with 94; test 56,746 with 95). The scaler is fit on the training data only.
2. **Models.**
   - Logistic Regression (baseline, `class_weight="balanced"`)
   - Random Forest (300 trees, `min_samples_leaf=2`, `class_weight="balanced_subsample"`)
   - XGBoost (400 trees, depth 5, learning rate 0.05, `scale_pos_weight`)
   - Random Forest + SMOTE (fraud oversampled to 10% of legit, applied to the training set only)
3. **Metrics.** PR-AUC, ROC-AUC, precision, recall, F1 and recall at 90% precision. Accuracy is deliberately not used,
   because predicting "legit" for everything is 99.8% accurate and catches no fraud.
4. **Model selection on validation only.** The test set is used once, at the end.
5. **Cost-based threshold.** The decision threshold is chosen on the validation set by minimising
   `FN x 100 + FP x 5` (assumed costs), then evaluated once on the test set.
6. **Explainability.** SHAP values on the XGBoost model.

## Results

### Model comparison (validation set: 56,745 transactions, 94 frauds)

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Recall @ 90% precision |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.698 | 0.973 | 0.056 | 0.872 | 0.105 | 0.000 |
| Random Forest | 0.790 | 0.946 | 0.944 | 0.723 | 0.819 | 0.777 |
| XGBoost | 0.806 | 0.969 | 0.890 | 0.777 | 0.830 | 0.777 |
| **Random Forest + SMOTE** | **0.811** | 0.965 | 0.901 | 0.777 | 0.834 | 0.777 |

Precision, recall and F1 above are at the default 0.50 threshold.

### Final model on the held-out test set (56,746 transactions, 95 frauds)

Random Forest + SMOTE, PR-AUC 0.868, ROC-AUC 0.987.

| Threshold | Precision | Recall | F1 | Frauds caught | False alarms | Cost (FN x 100 + FP x 5) |
|---|---|---|---|---|---|---|
| 0.50 (default) | 0.927 | 0.800 | 0.859 | 76 of 95 | 6 | 1,930 |
| 0.15 (cost-optimal on validation) | 0.535 | 0.884 | 0.667 | 84 of 95 | 73 | 1,465 |

Lowering the threshold catches 8 more frauds at the price of 67 more false alarms, which reduces total expected cost by
about 24%. Because a missed fraud is assumed to cost 20 times more than a false alarm, accepting lower precision is the
rational trade-off here.

### Key findings

- **Tree models clearly beat the linear baseline** on PR-AUC (0.79 to 0.81 versus 0.70). Logistic Regression's high ROC-AUC (0.973)
  is misleading: at the default threshold its precision is only 5.6%. This is why PR-AUC is the primary metric.
- **SMOTE and class weights perform similarly.** SMOTE gave Random Forest a small gain (0.790 to 0.811 PR-AUC), but the
  difference between Random Forest + SMOTE and XGBoost (0.811 vs 0.806) is within noise given only 94 validation frauds.
- **Top SHAP drivers (XGBoost):** V14, V4, V12, V10 and V3, followed by V11. `Amount` and `Hour` rank in the middle.

## Limitations

- Features `V1`-`V28` are anonymised, so SHAP rankings cannot be translated into business meaning.
- The FN / FP costs (100 and 5) are assumptions, not real loss figures. The threshold should be re-tuned with actual costs.
- With 95 test frauds, recall has roughly a plus or minus 6-point margin of error, so treat it as "about 88%".
- The split is random, not time-based. A time-based split would be a stricter and more realistic test.
- SHAP was computed on XGBoost, while the final selected model is Random Forest + SMOTE.
- The threshold was tuned on a validation set with only 94 frauds, so it is somewhat noisy.

## Possible extensions

- Time-based train / validation / test split
- Probability calibration and hyperparameter search (Optuna)
- A FastAPI `/score` endpoint with Docker for serving
- An LLM layer that turns SHAP output into a short analyst-style case summary, with a grounding check against the model's evidence

## Reproduce

```bash
pip install -r requirements.txt
jupyter notebook fraud_detection.ipynb
```

Run all cells from the top. The full run takes roughly 10 to 20 minutes.

## Files

- `fraud_detection.ipynb`: full analysis, from data download to SHAP
- `requirements.txt`: Python dependencies

@'
# Lab 3 Model Design — Track A: Churn Prediction

## 1. Objective and data

The objective is to identify customers at elevated risk of churn so that
retention resources can be prioritized. The primary ranking metrics are
precision@10% and recall@10%, supplemented by ROC-AUC.

Training data was retrieved from the SageMaker Feature Store offline store
using Athena. The experiment used 9,999 customer records, with 6,999
training records and 3,000 validation records. The validation positive
rate was 22%.

The target is `churn_label`. The model uses engineered customer features
and does not use `churn_label` as an input. The recency-only baseline uses
`days_since_last_purchase` as its sole feature.

## 2. Model and experiment design

The classifier is XGBoost. Hyperparameters are supplied through command-line
arguments. `scale_pos_weight` is derived from the negative-to-positive
class ratio in the training split when not explicitly supplied.

Three experiments were tracked in the SageMaker MLflow App under the
`northstar-churn` experiment:

| Run | Max depth | Eta | Rounds | ROC-AUC | Precision@10% | Recall@10% |
|---|---:|---:|---:|---:|---:|---:|
| xgb-baseline | 6 | 0.10 | 200 | 0.7779 | 0.7033 | 0.3197 |
| xgb-shallow | 4 | 0.10 | 300 | 0.7842 | 0.7167 | 0.3258 |
| xgb-deeper | 8 | 0.05 | 300 | 0.7796 | 0.7300 | 0.3318 |

The shallow model achieved the highest ROC-AUC. The deeper model achieved
the highest precision@10% and recall@10%, so it is the preferred candidate
when the main operational objective is to find as many churners as possible
within the top 10% of ranked customers. This choice trades a small amount
of overall AUC for better top-decile targeting.

## 3. Evaluation and baseline comparison

The selected `xgb-deeper` run achieved:

- ROC-AUC: 0.7796
- Precision@10%: 0.7300 (evaluation gate: at least 0.50)
- Recall@10%: 0.3318 (evaluation gate: at least 0.25)
- Recency-only baseline ROC-AUC: 0.7229
- AUC lift over baseline: 0.0568
- Bootstrap 95% confidence interval for AUC lift: [0.0361, 0.0759]

The confidence interval excludes zero, supporting an improvement over the
recency-only baseline on this validation split. The two top-decile metrics
also pass their evaluation gates.

At a probability threshold of 0.5, the confusion matrix, with rows as
actual classes [not churn, churn] and columns as predicted classes
[not churn, churn], is:

| | Predicted not churn | Predicted churn |
|---|---:|---:|
| Actual not churn | 1797 | 543 |
| Actual churn | 265 | 395 |

The threshold-based confusion matrix is a different view of performance
from precision@10% and recall@10%, which evaluate the highest-ranked 10%
of customers.

## 4. Loyalty-tier slice evaluation

| Loyalty tier | Validation n | ROC-AUC | Recall@10% |
|---|---:|---:|---:|
| Bronze | 1,071 | 0.7543 | 0.2853 |
| Gold | 483 | 0.7400 | 0.2885 |
| Platinum | 307 | 0.8828 | 0.5714 |
| Silver | 1,139 | 0.7116 | 0.2699 |

Silver is the weakest slice on both ROC-AUC and recall@10%. Its AUC is
below the aggregate model AUC, and its recall@10% is below the aggregate
recall@10% of 0.3318. However, the implemented slice gate did not flag
any tier as more than 10 percentage points below aggregate recall.
All tier sample sizes exceed the script's minimum slice-size threshold
of 30, although the smaller Platinum and Gold samples still warrant
caution when interpreting differences.

Possible follow-up work includes checking calibration and churn prevalence
by tier, collecting more validation data, and testing tier-specific
ranking performance before making separate targeting policies.

## 5. Artifacts, tracking, and limitations

The local outputs include:

- `model/model.xgb` — trained model
- `model/model_metadata.json` — feature names, hyperparameters, and metrics
- `output/evaluation_metrics.json` — aggregate metrics and slice evaluation
- `output/confusion_matrix.png` — confusion-matrix visualization
- `output/feature_importance.png` — feature-importance visualization

The experiments are tracked in the SageMaker MLflow App, not an MLflow
Tracking Server. Model Registry registration is opt-in and, when performed,
uses `PendingManualApproval`.

The evaluation uses a held-out random split rather than a temporal split.
Therefore, results demonstrate performance on this validation sample and
do not by themselves establish performance under future time-based drift.
The selected model should be monitored after deployment, and approval
should remain a human decision.

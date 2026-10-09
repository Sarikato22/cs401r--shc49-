"""
NorthStar Retail — Churn Prediction Training Script
CS 401R Lab 3 Starter Kit

This script is the SageMaker XGBoost training entry point for the NorthStar
churn prediction model. It reads features from the SageMaker Feature Store,
trains an XGBoost classifier, tracks the run in a SageMaker MLflow App, and
registers the model in the Model Registry.

Experiment tracking needs two packages that are NOT in the training container:

    pip install mlflow sagemaker-mlflow

`sagemaker-mlflow` is the SigV4 auth plugin for arn:aws:sagemaker:... tracking
URIs. Install `mlflow` alone and you get a connection error that never mentions
credentials. See track_run() for the cost warning about the OTHER MLflow
product -- read it before you create anything.

Usage (local testing):
    python churn_training_skeleton.py \
        --feature-group-name northstar-dev-customer-features \
        --artifacts-bucket northstar-dev-data-<account> \
        --max-depth 6 --eta 0.1 --num-round 200 \
        --mlflow-arn arn:aws:sagemaker:us-east-1:<account>:mlflow-app/app-XXXX \
        --run-name xgb-depth6-eta0.1

Usage (SageMaker training job — parameters passed via hyperparameter dict):
    Configured via the SageMaker Python SDK Estimator.
"""

import argparse
import json
import os
import tarfile
import tempfile
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd

# SageMaker imports — available in SageMaker training containers.
#
# The two failure modes below are different problems and used to produce the
# same message. sagemaker 3.x REMOVED the feature_store package, so on a 3.x
# install the SDK imports fine and only the second line fails. Reporting that
# as "sagemaker not available" sends you looking for a missing install that is
# not missing. Pin sagemaker<3.0.0 (see requirements.txt, defect 55).
try:
    import sagemaker
except ImportError:
    sagemaker = None
    SAGEMAKER_AVAILABLE = False
    print("WARNING: sagemaker not installed — running in local test mode")
else:
    try:
        from sagemaker.feature_store.feature_group import FeatureGroup
        from sagemaker.session import Session
        SAGEMAKER_AVAILABLE = True
    except ImportError:
        SAGEMAKER_AVAILABLE = False
        print("WARNING: sagemaker is installed but has no feature_store module. "
              "This is sagemaker 3.x, which dropped it. Run "
              "`pip install 'sagemaker>=2.200.0,<3.0.0'` — the Feature Store "
              "path in this script cannot work on 3.x.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    raise ImportError("xgboost is required: pip install xgboost")

from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# ── Argument Parsing ───────────────────────────────────────────────────────────
# SageMaker passes hyperparameters as CLI arguments.
def parse_args():
    parser = argparse.ArgumentParser(
        description="NorthStar churn prediction training"
    )

    # Data configuration
    # Names come from Lab 2's Terraform: `terraform output -raw feature_group_name`
    # and `terraform output -raw s3_bucket_name`. Do not invent new ones.
    parser.add_argument(
        "--feature-group-name",
        type=str,
        default="northstar-dev-customer-features",
        help="SageMaker Feature Store feature group name (Lab 2 output)",
    )
    # These dates bound the INGEST batch, not the feature window. Lab 2 stamps
    # event_time with the wall-clock time its Glue job ran (time.time()), not
    # with the feature cutoff T = 2026-04-01 -- T was already enforced when the
    # features were computed. A window that ends before your Lab 2 run (the old
    # default here was 2026-06-01) matches zero rows. These defaults match the
    # reference implementation and cover any Lab 2 run during the semester.
    parser.add_argument(
        "--training-start-date",
        type=str,
        default="2026-01-01",
        help="Earliest Lab 2 ingest date to include (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--training-end-date",
        type=str,
        default="2027-01-01",
        help="Latest Lab 2 ingest date to include (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--artifacts-bucket",
        type=str,
        required=True,
        help="Your Lab 1 data bucket, northstar-dev-data-<account>. "
             "Athena results and model artifacts go under its artifacts/ "
             "prefix, the only prefix your MLEngineer role can write",
    )
    parser.add_argument(
        "--local-data-path",
        type=str,
        default=None,
        help="Path to local feature CSV for development testing (bypasses Feature Store)",
    )

    # XGBoost hyperparameters
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--num-round", type=int, default=200)
    parser.add_argument("--min-child-weight", type=int, default=5)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument(
        "--scale-pos-weight",
        type=float,
        default=None,
        help=(
            "XGBoost positive-class weight. Leave unset and it is computed "
            "from YOUR training split as negatives/positives, which is what "
            "you want. The reference 10,000-customer dataset is 22.0%% "
            "churners, giving (1 - 0.220) / 0.220 = 3.545. Passing a value "
            "tuned for a different class balance will quietly skew your "
            "precision/recall trade-off, so only override this deliberately."
        ),
    )

    # Experiment tracking (Task 1, 5 points)
    parser.add_argument(
        "--mlflow-arn",
        type=str,
        default=os.environ.get("MLFLOW_APP_ARN"),
        help=(
            "ARN of your SageMaker MLflow APP, e.g. "
            "arn:aws:sagemaker:us-east-1:<account>:mlflow-app/app-XXXX. "
            "Create it once with `aws sagemaker create-mlflow-app`. "
            "This is NOT an MLflow Tracking Server -- see the warning "
            "on track_run() below. Omit to skip tracking."
        ),
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="northstar-churn",
        help="MLflow experiment name; runs group under this",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run. Make it descriptive -- 'run3' tells "
             "you nothing three weeks later.",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "./model"),
    )
    parser.add_argument(
        "--output-data-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"),
    )
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Upload and register the model only if evaluation gates pass.",
    )
    parser.add_argument(
        "--model-package-group",
        type=str,
        default="northstar-churn-models",
    )
    parser.add_argument(
        "--inference-xgboost-version",
        type=str,
        default="1.7-1",
        help="XGBoost version for the SageMaker inference image.",
    )

    return parser.parse_args()


# ── Feature Store Loading ──────────────────────────────────────────────────────

# These are exactly the features your Lab 2 pipeline wrote to the Feature
# Group. If a name here does not match your feature group, the Athena query
# below will fail - check `aws sagemaker describe-feature-group` first.
#
# Note what is NOT in this list: churn_label (the target) and customer_id
# (an identifier, not a signal). Including either as an input is leakage.
FEATURE_COLUMNS = [
    # Recency and tenure
    "days_since_last_purchase",
    "customer_tenure_days",
    # Frequency
    "purchase_frequency_30d",
    "purchase_frequency_90d",
    "purchase_frequency_180d",
    # Monetary
    "avg_order_value",
    "total_spend_90d",
    "total_lifetime_value",
    "avg_basket_size_6m",
    # Behavioural - these are the features that catch churners who still
    # look active on recency alone. Drop them and your model collapses
    # toward the recency-only baseline.
    "category_diversity_score",
    "online_to_store_ratio",
]

# churn_risk_score is available in the feature group but deliberately excluded
# here: it is a pure recency heuristic and doubles as the baseline you must
# beat. If you add it, report your metrics both with and without it.
BASELINE_COLUMN = "churn_risk_score"

# The recency-only baseline model uses just this one feature. Task 1 requires
# you to train it and show your full model beats it by a margin whose 95%
# confidence interval excludes zero. There is no fixed AUC-lift threshold: a
# ">= 0.03 lift" gate lived here until 2026-08-02 and was removed because 0.03
# is smaller than the metric's own run-to-run standard deviation, so it failed
# on 21% of splits regardless of model quality. See EVAL_THRESHOLDS below.
BASELINE_FEATURE = "days_since_last_purchase"

LABEL_COLUMN = "churn_label"

# Not a feature. This is the column Task 1's slice evaluation groups by, so it
# has to come back from the query even though it never enters the model. Keep it
# out of FEATURE_COLUMNS: loyalty_tier correlates with spend, and feeding it in
# both leaks and makes the per-tier fairness check circular.
SLICE_COLUMN = "loyalty_tier"


def load_features_from_feature_store(
    feature_group_name: str,
    start_date: str,
    end_date: str,
    artifacts_bucket: str,
) -> pd.DataFrame:
    """
    Load features from the SageMaker Feature Store offline store using Athena.

    Args:
        feature_group_name: your Lab 2 feature group
        start_date, end_date: ISO dates (YYYY-MM-DD) bounding the event_time range
        artifacts_bucket: S3 bucket that Athena writes its result set to. Athena
            will not run a query without an output location, so this has to be
            threaded in from --artifacts-bucket rather than assumed.
    """
    if not SAGEMAKER_AVAILABLE:
        raise RuntimeError(
            "SageMaker SDK not available - use --local-data-path for local testing"
        )

    start_dt = datetime.strptime(
        start_date, "%Y-%m-%d"
    ).replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(
        end_date, "%Y-%m-%d"
    ).replace(tzinfo=timezone.utc)

    start_epoch = start_dt.timestamp()
    end_epoch = end_dt.timestamp()

    if end_epoch <= start_epoch:
        raise ValueError(
            "--training-end-date must be later than --training-start-date"
        )

    session = Session()
    feature_group = FeatureGroup(
        name=feature_group_name,
        sagemaker_session=session,
    )

    # event_time was written as Fractional (epoch seconds) by the Lab 2 job, so
    # the date bounds must be converted before comparison. Passing ISO strings
    # straight into the query fails with TYPE_MISMATCH rather than returning
    # rows, so at least the failure is loud.
    #
    # Two things in the query below are easy to get wrong, so read them:
    #
    # 1. event_time is Fractional - Unix epoch SECONDS, not an ISO string.
    #    Comparing it to '2026-04-01' fails the query outright with:
    #      TYPE_MISMATCH: Cannot check if double is BETWEEN varchar(10)...
    #    Convert your date bounds to epoch seconds before substituting them.
    #
    # 2. The offline store is append-only. Re-running the Lab 2 feature job
    #    writes a SECOND record for every customer, and a naive SELECT then
    #    returns duplicate customers with stale feature values. Deduplicate
    #    with a window function keyed on customer_id, keeping the most recent
    #    event_time. Filtering on a global MAX(write_time) does NOT work -
    #    it keeps only the customers written in the final microsecond.
    #
    #    The offline store also soft-deletes: rows carry is_deleted, and
    #    deleted records must be excluded.
    #
    # 3. The outer SELECT must ORDER BY customer_id, and this is the one people
    #    skip because it looks cosmetic. It is not. Athena parallelises the scan
    #    across the offline store's Parquet objects and returns rows in whatever
    #    order the splits happen to finish, which varies from run to run.
    #    train_test_split(random_state=42) is deterministic only for a GIVEN row
    #    order -- so without ORDER BY, the identical data produces a different
    #    train/test split, and therefore different metrics, on every single run.
    #
    #    This was measured, not theorised: four runs on byte-identical data
    #    produced AUC between 0.7276 and 0.7431, and a Platinum-slice AUC
    #    between 0.430 and 0.700. An entire "the model is worse than random on
    #    your best customers" finding turned out to be an artefact of row order.
    #
    #    The rn = 1 filter guarantees customer_id is unique here, so ordering on
    #    it is a TOTAL order and the pipeline becomes reproducible.
    #
    #    If your metrics move between runs and your data did not, this is why.
    #
    #    loyalty_tier is selected but is NOT in FEATURE_COLUMNS, and that is
    #    deliberate. It is not an input to the model - it is the column Task 1's
    #    slice evaluation groups by. Keep it out of the feature matrix (see
    #    preprocess_features, which selects FEATURE_COLUMNS explicitly) and
    #    carry it alongside X and y through the split so each row's prediction
    #    can still be attributed to a tier.

    query = feature_group.athena_query()
    query_string = f"""
        WITH ranked AS (
            SELECT
                customer_id,
                {", ".join(FEATURE_COLUMNS)},
                {LABEL_COLUMN},
                {SLICE_COLUMN},
                event_time,
                write_time,
                ROW_NUMBER() OVER (
                    PARTITION BY customer_id
                    ORDER BY event_time DESC, write_time DESC
                ) AS rn
            FROM "{query.table_name}"
            WHERE event_time >= {start_epoch}
              AND event_time < {end_epoch}
              AND is_deleted = false
        )
        SELECT
            customer_id,
            {", ".join(FEATURE_COLUMNS)},
            {LABEL_COLUMN},
            {SLICE_COLUMN}
        FROM ranked
        WHERE rn = 1
        ORDER BY customer_id
    """

    output_location = (
        f"s3://{artifacts_bucket}/artifacts/athena-results/"
    )
    query.run(
        query_string=query_string,
        output_location=output_location,
    )
    query.wait()
    df = query.as_dataframe()

    if df.empty:
        raise ValueError(
            f"Athena returned no rows for feature group '{feature_group_name}' "
            f"between {start_date} and {end_date}. Check the ingest dates and "
            "the Feature Store offline store."
        )

    required = FEATURE_COLUMNS + [LABEL_COLUMN, SLICE_COLUMN]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Athena results missing required columns: {missing}")

    print(
        f"Loaded {len(df):,} customers from the Feature Store offline store."
    )
    return df


def load_features_local(path: str) -> pd.DataFrame:
    """Load features from a local CSV file (development mode)."""
    print(f"Loading features from local file: {path}")
    df = pd.read_csv(path)
    missing = [
        c for c in FEATURE_COLUMNS + [LABEL_COLUMN]
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Local data missing required columns: {missing}")

    # Not fatal - the model trains fine without it - but Task 1's slice
    # evaluation cannot be done at all, so fail the task rather than the run.
    if SLICE_COLUMN not in df.columns:
        raise ValueError(
            f"'{SLICE_COLUMN}' is required for Task 1 slice evaluation."
        )
    return df


# ── Feature Preprocessing ──────────────────────────────────────────────────────

def preprocess_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Clean and prepare features for XGBoost training.
    Returns (X, y, cleaned dataframe) so slice metadata remains aligned.
    """
    print(f"Preprocessing {len(df):,} samples...")

    # Drop rows where label is missing
    df = df.dropna(subset=[LABEL_COLUMN]).copy().reset_index(drop=True)

    if df.empty:
        raise ValueError("No rows with a non-missing churn_label were found.")

    if SLICE_COLUMN not in df.columns:
        raise ValueError(
            f"'{SLICE_COLUMN}' is required for Task 1 slice evaluation."
        )
    if df[SLICE_COLUMN].isna().any():
        raise ValueError("loyalty_tier contains missing values.")

    # XGBoost can handle NaN natively, but document your imputation strategy.
    # For features with business-meaningful nulls (e.g., clickstream for non-web customers),
    # consider median imputation or a sentinel value.
    X_df = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    X = X_df.fillna(-1).to_numpy(dtype=np.float32)  # Sentinel -1 for missing

    labels = pd.to_numeric(df[LABEL_COLUMN], errors="raise")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("churn_label must contain only 0 and 1.")
    y = labels.to_numpy(dtype=np.int32)

    if len(np.unique(y)) != 2:
        raise ValueError(
            "Training data must contain both churn_label classes (0 and 1)."
        )

    print(f"  Positive class rate: {y.mean():.1%}")
    print(f"  Feature matrix shape: {X.shape}")

    return X, y, df


# ── Model Training ─────────────────────────────────────────────────────────────

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args,
) -> xgb.Booster:
    """Train the XGBoost churn model."""

    dtrain = xgb.DMatrix(
        X_train, label=y_train, feature_names=FEATURE_COLUMNS
    )
    dval = xgb.DMatrix(
        X_val, label=y_val, feature_names=FEATURE_COLUMNS
    )

    # Class imbalance, computed from the training split rather than hardcoded.
    # On the reference dataset this lands at 3.545.
    spw = (
        args.scale_pos_weight
        if args.scale_pos_weight is not None
        else float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    )
    if spw <= 0:
        raise ValueError("--scale-pos-weight must be greater than zero.")

    # Record the RESOLVED value so track_run() can log what was actually used.
    # Logging args.scale_pos_weight instead would log None on every run that
    # let it compute itself -- which is most of them -- and your MLflow
    # comparison would be missing the one parameter most likely to explain a
    # precision/recall difference between runs.
    args.resolved_scale_pos_weight = spw

    params = {
        # eval_metric ORDER MATTERS. XGBoost early-stops on the LAST metric in
        # this list, not the first. This used to read ["auc", "logloss"], which
        # silently made logloss the early-stopping criterion: under
        # scale_pos_weight the reweighted logloss keeps improving long after
        # ranking quality peaks, so training ran the full 200 rounds instead of
        # stopping near round 31 and validation AUC came out 0.7603 instead of
        # 0.7822. Measured across 50 splits, that cost the model the promotion
        # gate on 28% of them -- a student doing everything right was told their
        # features did not beat recency. Keep "auc" last (defect 49).
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": args.max_depth,
        "eta": args.eta,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "scale_pos_weight": spw,
        "seed": 42,
        "verbosity": 1,
    }

    print(f"\nTraining XGBoost with params: {json.dumps(params, indent=2)}")

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=20,
        verbose_eval=25,
    )

    return model


# ── Evaluation ─────────────────────────────────────────────────────────────────

EVAL_THRESHOLDS = {
    "precision_top10": 0.50,  # Minimum precision @ top 10% scored customers
    "recall_top10": 0.25,     # Minimum recall @ top 10% scored customers
    # Note the recall ceiling: with ~22% positives, targeting the top 10% of
    # customers caps achievable recall near 0.45. Hitting 0.25 means capturing
    # roughly half of what is reachable within that contact budget.
}

# There is deliberately NO absolute AUC threshold. One used to live here
# (auc_roc >= 0.72) and it was removed on 2026-08-02: measured across 200
# random train/test splits of the same data, the reference model fell below
# 0.72 on 58% of them. A gate the reference clears by luck of the shuffle
# grades your random seed, not your model. Report AUC; do not gate on it.
#
# The baseline gate is an INTERVAL, not a threshold: the 95% CI on
# (your AUC - recency-only baseline AUC) must exclude zero.
# BASELINE_FEATURE is defined once, near FEATURE_COLUMNS above.
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
MIN_SLICE_SIZE = 30


def train_recency_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
) -> np.ndarray:
    """Train the recency-only baseline and score the validation split.

    The baseline is a model over `days_since_last_purchase` ALONE, trained on
    the same training rows and scored on the same validation rows as the full
    model. That is what makes the comparison fair.

    It is deliberately NOT a constant predictor. A constant prediction has an
    AUC of exactly 0.5 by construction, so "beating" it proves only that your
    model is better than a coin flip. The question Lab 3 asks is whether your
    feature engineering beats the rule the business already has for free:
    "contact whoever has not purchased in a while."
    """
    i = FEATURE_COLUMNS.index(BASELINE_FEATURE)
    dtr = xgb.DMatrix(
        X_train[:, [i]],
        label=y_train,
        feature_names=[BASELINE_FEATURE],
    )
    dva = xgb.DMatrix(
        X_val[:, [i]],
        feature_names=[BASELINE_FEATURE],
    )
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 4,
        "eta": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "scale_pos_weight": spw,
        "seed": 42,
    }
    return xgb.train(
        params,
        dtr,
        num_boost_round=200,
    ).predict(dva)


def bootstrap_lift_ci(
    y_true: np.ndarray,
    proba: np.ndarray,
    baseline_proba: np.ndarray,
    n: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for (model AUC - baseline AUC), by resampling the val set.

    PAIRED: each replicate resamples row indices once and scores BOTH models on
    those same rows, so the interval is on the difference and the correlation
    between the two models is preserved. Resampling them independently breaks
    the pairing and inflates the interval, which would make a real improvement
    look inconclusive.

    Replicates whose resample happens to be single-class are skipped, because
    AUC is undefined there.

    The seed is fixed so the gate is reproducible: the same split must always
    produce the same verdict.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    baseline_proba = np.asarray(baseline_proba)

    if not (len(y_true) == len(proba) == len(baseline_proba)):
        raise ValueError("Bootstrap input arrays must have equal lengths.")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    diffs = []

    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        ys = y_true[s]
        if len(np.unique(ys)) < 2:
            continue
        diffs.append(
            roc_auc_score(ys, proba[s])
            - roc_auc_score(ys, baseline_proba[s])
        )

    if len(diffs) < n // 2:
        raise ValueError(
            "Too few usable bootstrap replicates to form a confidence interval. "
            "Your validation set is too small or too imbalanced to support this gate."
        )

    d = np.sort(diffs)
    return (
        float(np.quantile(d, alpha / 2)),
        float(np.quantile(d, 1 - alpha / 2)),
    )


def evaluate_model(
    model: xgb.Booster,
    X_val: np.ndarray,
    y_val: np.ndarray,
    baseline_proba: np.ndarray,
) -> dict:
    """
    Evaluate the trained model against required thresholds.
    Returns a metrics dict and records whether evaluation gates passed.
    """
    dval = xgb.DMatrix(X_val, feature_names=FEATURE_COLUMNS)
    y_pred_proba = model.predict(dval)

    auc_roc = roc_auc_score(y_val, y_pred_proba)

    # Precision and recall @ top 10%
    top10_threshold = np.percentile(y_pred_proba, 90)
    top10_mask = y_pred_proba >= top10_threshold
    precision_top10 = (
        y_val[top10_mask].mean()
        if top10_mask.sum() > 0
        else 0.0
    )
    recall_top10 = (
        y_val[top10_mask].sum() / y_val.sum()
        if y_val.sum() > 0
        else 0.0
    )

    # Baseline: a model trained on days_since_last_purchase ALONE.
    #
    # This used to be `np.full_like(y_pred_proba, y_val.mean())` - a constant
    # prediction, whose AUC is exactly 0.5 by definition. Beating a coin flip
    # is not evidence that your feature engineering did anything, and it is not
    # what Lab 3 asks for. The comparison that matters is against recency,
    # because recency is the rule the business already has for free.
    #
    baseline_auc = roc_auc_score(y_val, baseline_proba)
    auc_vs_baseline = auc_roc - baseline_auc

    lift_ci_low, lift_ci_high = bootstrap_lift_ci(
        y_val,
        y_pred_proba,
        baseline_proba,
    )

    metrics = {
        "auc_roc": round(float(auc_roc), 4),
        "precision_top10": round(float(precision_top10), 4),
        "recall_top10": round(float(recall_top10), 4),
        "auc_vs_baseline": round(float(auc_vs_baseline), 4),
        "baseline_auc": round(float(baseline_auc), 4),
        "lift_ci_low": round(float(lift_ci_low), 4),
        "lift_ci_high": round(float(lift_ci_high), 4),
        "positive_rate_val": round(float(y_val.mean()), 4),
        "n_val_samples": int(len(y_val)),
        "eval_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print("\n── Evaluation Results ──────────────────────────────")
    failures = []

    for metric, threshold in EVAL_THRESHOLDS.items():
        value = metrics[metric]
        status = "✓ PASS" if value >= threshold else "✗ FAIL"
        print(
            f"  {metric:25s}: {value:.4f}  "
            f"(threshold: {threshold:.4f})  {status}"
        )
        if value < threshold:
            failures.append(f"{metric}={value:.4f} < {threshold}")

    # The baseline gate: the interval must exclude zero.
    ci_status = "✓ PASS" if lift_ci_low > 0 else "✗ FAIL"
    print(
        f"  {'auc_lift 95% CI':25s}: "
        f"[{lift_ci_low:.4f}, {lift_ci_high:.4f}]"
        f"  (must exclude 0)  {ci_status}"
    )
    if lift_ci_low <= 0:
        failures.append(
            f"auc_lift 95% CI [{lift_ci_low:.4f}, {lift_ci_high:.4f}] "
            "includes zero - no evidence the model beats the recency-only baseline"
        )

    metrics["evaluation_passed"] = not failures
    metrics["evaluation_failures"] = failures

    if failures:
        print(
            "Evaluation gates not met: " + "; ".join(failures)
        )
    else:
        print("  All thresholds passed ✓")

    # Evaluation failures are recorded instead of raising immediately so
    # metrics and slice results can still be saved for diagnosis.
    return metrics


# ── Slice Evaluation ───────────────────────────────────────────────────────────

def evaluate_slices(
    model: xgb.Booster,
    X_val: np.ndarray,
    y_val: np.ndarray,
    slice_column: pd.Series,
) -> dict:
    """
    Evaluate model performance by slice (e.g., loyalty_tier, age_band).
    Flag slices where recall@10% drops more than 10pp below aggregate.
    """
    dval = xgb.DMatrix(X_val, feature_names=FEATURE_COLUMNS)
    y_pred_proba = model.predict(dval)
    y_true = np.asarray(y_val)
    slice_column = pd.Series(slice_column).reset_index(drop=True)

    if len(slice_column) != len(y_true):
        raise ValueError(
            "slice_column and y_val must have the same number of rows."
        )
    if slice_column.isna().any():
        raise ValueError("Slice column contains missing values.")

    aggregate_threshold = np.percentile(y_pred_proba, 90)
    aggregate_mask = y_pred_proba >= aggregate_threshold
    aggregate_recall = (
        float(y_true[aggregate_mask].sum() / y_true.sum())
        if y_true.sum() > 0
        else 0.0
    )

    slice_results = {}

    for slice_value in sorted(slice_column.unique(), key=str):
        mask = (slice_column == slice_value).to_numpy()
        y_slice = y_true[mask]
        proba_slice = y_pred_proba[mask]
        n_slice = int(mask.sum())

        # Calculate recall@10% within this slice, using that slice's scores.
        slice_threshold = np.percentile(proba_slice, 90)
        top10_mask = proba_slice >= slice_threshold
        positives = int(y_slice.sum())
        recall_top10 = (
            float(y_slice[top10_mask].sum() / positives)
            if positives > 0
            else 0.0
        )

        auc_slice = (
            float(roc_auc_score(y_slice, proba_slice))
            if len(np.unique(y_slice)) == 2
            else None
        )

        below_aggregate = recall_top10 < aggregate_recall - 0.10
        too_small = n_slice < MIN_SLICE_SIZE

        slice_results[str(slice_value)] = {
            "n": n_slice,
            "auc_roc": (
                round(auc_slice, 4)
                if auc_slice is not None
                else None
            ),
            "recall_top10": round(recall_top10, 4),
            "aggregate_recall_top10": round(aggregate_recall, 4),
            "below_aggregate_by_10pp": bool(below_aggregate),
            "too_small_for_reliable_claim": bool(too_small),
            "auc_undefined_single_class": auc_slice is None,
        }

        print(
            f"  {slice_value}: n={n_slice}, "
            f"AUC={slice_results[str(slice_value)]['auc_roc']}, "
            f"recall@10%={recall_top10:.4f}, "
            f"below aggregate by >10pp={below_aggregate}, "
            f"too small={too_small}"
        )

    return slice_results


# ── Experiment Tracking (MLflow App) ───────────────────────────────────────────

def track_run(args, metrics: dict, scale_pos_weight: float) -> None:
    """
    Log this training run to your SageMaker MLflow App.

    Worth 5 points in Task 1: >=3 runs, each with logged params AND metrics,
    retrievable via mlflow.search_runs.

    ###########################################################################
    #  THERE ARE TWO MLflow PRODUCTS ON SAGEMAKER. USE THE APP.               #
    #                                                                         #
    #    CreateMlflowApp            serverless   NO ADDITIONAL CHARGE   <-- ok #
    #    CreateMlflowTrackingServer $0.60/hour   until you delete it    <-- NO #
    #                                                                         #
    #  $0.60/hr breaches the entire $10 course budget in 16.7 hours and costs #
    #  about $43 over a weekend -- more per hour than any endpoint in this    #
    #  course. It is not an endpoint, so "did I delete my endpoints?" will    #
    #  not find it. Most tutorials describe the Tracking Server because it    #
    #  shipped first. If anything asks you to choose a size (Small/Medium),   #
    #  you are on the wrong product.                                          #
    ###########################################################################

    Two things that fail in ways the error does not explain:

      1. `create-mlflow-app` postdates many installed AWS CLIs. An old CLI
         says "Invalid choice: 'create-mlflow-app'", which reads like a typo
         rather than a version problem. Check `aws --version` and upgrade.

      2. You need BOTH `mlflow` and `sagemaker-mlflow` installed. The second
         is the SigV4 auth plugin for arn:aws:sagemaker:... tracking URIs.
         Without it you get a connection error that never mentions credentials.
    """
    if not args.mlflow_arn:
        print("\n(no --mlflow-arn given; skipping experiment tracking)")
        return

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow tracking requires both packages: "
            "pip install mlflow sagemaker-mlflow"
        ) from exc

    mlflow.set_tracking_uri(args.mlflow_arn)
    mlflow.set_experiment(args.mlflow_experiment)

    run_name = args.run_name or (
        f"xgb-depth{args.max_depth}-eta{args.eta}-rounds{args.num_round}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "max_depth": args.max_depth,
            "eta": args.eta,
            "num_round": args.num_round,
            "min_child_weight": args.min_child_weight,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
            "xgboost_version": xgb.__version__,
        })

        numeric_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float, np.integer, np.floating))
            and np.isfinite(value)
        }
        mlflow.log_metrics(numeric_metrics)

        # THREE RUNS IS THE FLOOR, NOT THE GOAL. Three runs with identical
        # hyperparameters and different seeds is the same experiment three
        # times; vary a parameter you can defend and compare the metrics.
    print(f"Logged MLflow run: {run_name}")


# ── Model Saving & Registry ────────────────────────────────────────────────────

def save_and_register_model(
    model: xgb.Booster,
    metrics: dict,
    args,
) -> None:
    """
    Save the model artifact and register it in SageMaker Model Registry.
    Status is set to PendingManualApproval — human approval required before deployment.
    """
    os.makedirs(args.model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(args.model_dir, "model.xgb")
    model.save_model(model_path)
    print(f"\n✓ Model saved to {model_path}")

    # Save feature metadata alongside model
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "label_column": LABEL_COLUMN,
        "hyperparameters": {
            "max_depth": args.max_depth,
            "eta": args.eta,
            "num_round": args.num_round,
            "min_child_weight": args.min_child_weight,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "scale_pos_weight": getattr(
                args, "resolved_scale_pos_weight", None
            ),
        },
        "evaluation_metrics": metrics,
        "training_date": datetime.now(timezone.utc).isoformat(),
    }

    with open(
        os.path.join(args.model_dir, "model_metadata.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2, default=str)

    if not args.register_model:
        print(
            "Model registration skipped. Use --register-model to opt in."
        )
        return

    if not metrics.get("evaluation_passed", False):
        print(
            "Model registration skipped because evaluation gates did not pass."
        )
        return

    if not SAGEMAKER_AVAILABLE:
        raise RuntimeError(
            "SageMaker SDK v2 is required for Model Registry registration."
        )

    region = boto3.Session().region_name
    if not region:
        raise RuntimeError(
            "AWS region is not configured in the current credentials/profile."
        )

    sm_client = boto3.client("sagemaker", region_name=region)
    s3_client = boto3.client("s3", region_name=region)

    # SageMaker XGBoost inference expects the model file named xgboost-model
    # inside model.tar.gz.
    with tempfile.TemporaryDirectory() as temp_dir:
        inference_model_path = os.path.join(
            temp_dir, "xgboost-model"
        )
        model.save_model(inference_model_path)

        archive_path = os.path.join(temp_dir, "model.tar.gz")
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(
                inference_model_path,
                arcname="xgboost-model",
            )

        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        object_key = (
            f"artifacts/model-registry/{args.model_package_group}/"
            f"model-{timestamp}.tar.gz"
        )
        s3_client.upload_file(
            archive_path,
            args.artifacts_bucket,
            object_key,
        )

    model_data_url = (
        f"s3://{args.artifacts_bucket}/{object_key}"
    )

    try:
        sm_client.describe_model_package_group(
            ModelPackageGroupName=args.model_package_group
        )
    except sm_client.exceptions.ResourceNotFound:
        sm_client.create_model_package_group(
            ModelPackageGroupName=args.model_package_group,
            ModelPackageGroupDescription=(
                "NorthStar Retail churn prediction models"
            ),
        )

    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version=args.inference_xgboost_version,
    )

    response = sm_client.create_model_package(
        ModelPackageGroupName=args.model_package_group,
        ModelPackageDescription=(
            f"Churn model trained {datetime.now(timezone.utc).date()}; "
            f"AUC={metrics['auc_roc']}; "
            f"baseline AUC={metrics['baseline_auc']}"
        ),
        InferenceSpecification={
            "Containers": [
                {
                    "Image": image_uri,
                    "ModelDataUrl": model_data_url,
                }
            ],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large"],
            "SupportedTransformInstanceTypes": ["ml.m5.large"],
        },
        ModelApprovalStatus="PendingManualApproval",
        CustomerMetadataProperties={
            key: str(value)
            for key, value in metrics.items()
            if isinstance(value, (str, int, float, bool))
        },
    )

    print(
        "Registered model package as PendingManualApproval: "
        f"{response['ModelPackageArn']}"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)

    print("=" * 60)
    print("NorthStar Churn Prediction — Training Script")
    print("=" * 60)

    # 1. Load features
    if args.local_data_path:
        df = load_features_local(args.local_data_path)
    else:
        df = load_features_from_feature_store(
            args.feature_group_name,
            args.training_start_date,
            args.training_end_date,
            args.artifacts_bucket,
        )

    # 2. Preprocess
    X, y, clean_df = preprocess_features(df)

    # 3. Train/validation split (temporal split is better — split by customer_id here for simplicity)
    # test_size=0.30 matches the reference run and every published figure in
    # Lab 3 (6,999 train / 3,000 test on the 10k dataset). The bootstrap lift CI
    # is calibrated on a test set that size; a 0.20 split leaves 2,000 rows and
    # widens the interval enough to matter.
    #
    # Pass df[SLICE_COLUMN] as a third array to this same call and it is split
    # on the identical indices:
    #   X_train, X_val, y_train, y_val, tier_train, tier_val = train_test_split(
    #       X, y, df[SLICE_COLUMN], test_size=0.30, random_state=42, stratify=y)
    # Re-deriving the tiers afterwards from df will NOT line up.
    X_train, X_val, y_train, y_val, _, tier_val = train_test_split(
        X,
        y,
        clean_df[SLICE_COLUMN].reset_index(drop=True),
        test_size=0.30,
        random_state=42,
        stratify=y,
    )
    print(f"\nTrain: {len(X_train):,} | Validation: {len(X_val):,}")

    # 4. Train
    model = train_model(X_train, y_train, X_val, y_val, args)

    # 5. Evaluate
    baseline_proba = train_recency_baseline(
        X_train, y_train, X_val
    )
    metrics = evaluate_model(
        model, X_val, y_val, baseline_proba
    )
    slices = evaluate_slices(
        model, X_val, y_val, tier_val
    )
    metrics["slice_evaluation"] = slices

    # Save evaluation artifacts, including the confusion matrix and feature
    # importance plot when matplotlib is available.
    dval = xgb.DMatrix(X_val, feature_names=FEATURE_COLUMNS)
    proba = model.predict(dval)
    predictions = (proba >= 0.5).astype(int)
    matrix = confusion_matrix(y_val, predictions, labels=[0, 1])
    metrics["confusion_matrix_threshold"] = 0.5
    metrics["confusion_matrix_labels"] = ["not_churn", "churn"]
    metrics["confusion_matrix"] = matrix.tolist()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        image = ax.imshow(matrix)
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Not churn", "Churn"],
            yticklabels=["Not churn", "Churn"],
            xlabel="Predicted label",
            ylabel="True label",
            title="Confusion matrix (threshold = 0.5)",
        )
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, str(value), ha="center", va="center")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(
            os.path.join(args.output_data_dir, "confusion_matrix.png"),
            dpi=150,
        )
        plt.close(fig)

        importance = model.get_score(importance_type="gain")
        if importance:
            ordered = sorted(
                importance.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            names, values = zip(*ordered)
            fig, ax = plt.subplots(
                figsize=(9, max(4, len(names) * 0.35))
            )
            ax.barh(list(names)[::-1], list(values)[::-1])
            ax.set_xlabel("Gain")
            ax.set_title("XGBoost feature importance")
            fig.tight_layout()
            fig.savefig(
                os.path.join(args.output_data_dir, "feature_importance.png"),
                dpi=150,
            )
            plt.close(fig)
    except ImportError:
        print(
            "WARNING: matplotlib is not installed; plots were not generated."
        )

    # 6. Save and optionally register
    save_and_register_model(model, metrics, args)

    # 7. Track this run in your MLflow App (Task 1, 5 pts)
    track_run(
        args,
        metrics,
        getattr(args, "resolved_scale_pos_weight", None),
    )

    # 8. Write metrics to output. SageMaker also collects these as job output;
    #    that is job lineage, NOT experiment tracking. It records that a job
    #    ran, not what you varied or why. Step 7 is the graded one.
    metrics_path = os.path.join(
        args.output_data_dir,
        "evaluation_metrics.json",
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print(f"Training complete. AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"Evaluation gates passed: {metrics['evaluation_passed']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
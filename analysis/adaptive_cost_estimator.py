
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/processed/processed_userops.csv")
OUTPUT_FILE = Path("data/processed/adaptive_cost_predictions.csv")
SUMMARY_FILE = Path("data/processed/adaptive_cost_summary.csv")

CALIBRATION_FRACTION = 0.20
MIN_HISTORY = 20

# For Paymasters with insufficient history, use:
# Paymaster historical median * global P90/median ratio.
# If a Paymaster has no history, use global calibration P90.


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

print("Loading processed UserOperation dataset...")

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "userop_hash",
    "transaction_hash",
    "log_index",
    "paymaster",
    "block_number",
    "gas_cost_eth",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["block_number"] = pd.to_numeric(
    df["block_number"], errors="raise"
)

df["log_index"] = pd.to_numeric(
    df["log_index"], errors="raise"
)

df["gas_cost_eth"] = pd.to_numeric(
    df["gas_cost_eth"], errors="raise"
)

if df[required_columns].isnull().any().any():
    raise ValueError(
        "Required columns contain missing values. "
        "Check the processed dataset."
    )

if (df["gas_cost_eth"] < 0).any():
    raise ValueError("Gas costs cannot be negative.")

# A deterministic chronological order.
# transaction_hash and log_index break ties within blocks.
df = df.sort_values(
    ["block_number", "transaction_hash", "log_index"]
).reset_index(drop=True)


# ============================================================
# CHRONOLOGICAL CALIBRATION / TEST SPLIT
# ============================================================

print("Creating chronological calibration and test sets...")

calibration_indices = []
test_indices = []

for paymaster, group in df.groupby("paymaster", sort=False):

    group_indices = group.index.tolist()

    split_position = int(
        len(group_indices) * CALIBRATION_FRACTION
    )

    # Ensure that a Paymaster with multiple operations has
    # at least one calibration observation.
    if len(group_indices) > 1:
        split_position = max(1, split_position)
        split_position = min(
            split_position, len(group_indices) - 1
        )

    calibration_indices.extend(
        group_indices[:split_position]
    )

    test_indices.extend(
        group_indices[split_position:]
    )

calibration = df.loc[calibration_indices].copy()

test = df.loc[test_indices].copy()

# Keep test rows in chronological order.
test = test.sort_values(
    ["block_number", "transaction_hash", "log_index"]
)


# ============================================================
# INITIALIZE HISTORICAL COSTS
# ============================================================

print("Initializing historical cost distributions...")

history = {}

for paymaster, group in calibration.groupby("paymaster"):

    history[paymaster] = (
        group["gas_cost_eth"].astype(float).tolist()
    )

global_median = calibration["gas_cost_eth"].median()
global_p90 = calibration["gas_cost_eth"].quantile(0.90)

if global_median <= 0:
    raise ValueError(
        "Global calibration median must be positive."
    )

global_scale = global_p90 / global_median


# ============================================================
# ADAPTIVE COST PREDICTION
# ============================================================

print("Predicting test operation costs...")

predictions = []

for _, row in test.iterrows():

    paymaster = row["paymaster"]
    actual_cost = float(row["gas_cost_eth"])

    previous_costs = history.get(paymaster, [])

    # --------------------------------------------------------
    # Estimate using only information available before
    # the current operation.
    # --------------------------------------------------------

    if len(previous_costs) >= MIN_HISTORY:

        predicted_cost = float(
            np.quantile(previous_costs, 0.90)
        )

        method = "expanding_historical_p90"

    elif len(previous_costs) > 0:

        historical_median = float(
            np.median(previous_costs)
        )

        predicted_cost = (
            historical_median * global_scale
        )

        method = "scaled_historical_median"

    else:

        predicted_cost = float(global_p90)

        method = "global_calibration_p90"

    # Safety check.
    predicted_cost = max(predicted_cost, 0.0)

    # --------------------------------------------------------
    # Save prediction BEFORE adding current actual cost.
    # --------------------------------------------------------

    predictions.append({

        "userop_hash": row["userop_hash"],

        "transaction_hash": row["transaction_hash"],

        "log_index": row["log_index"],

        "paymaster": paymaster,

        "block_number": int(row["block_number"]),

        "actual_cost_eth": actual_cost,

        "predicted_cost_eth": predicted_cost,

        "prediction_method": method,

        "history_size_before_prediction": len(
            previous_costs
        ),

        "prediction_error_eth": (
            actual_cost - predicted_cost
        ),

        "actual_within_estimate": (
            actual_cost <= predicted_cost
        ),
    })

    # --------------------------------------------------------
    # Update historical information AFTER prediction.
    # This is a chronological historical replay.
    # --------------------------------------------------------

    history.setdefault(paymaster, []).append(actual_cost)


# ============================================================
# CREATE PREDICTION RESULTS
# ============================================================

results = pd.DataFrame(predictions)

if results.empty:
    raise ValueError(
        "No test predictions were generated. "
        "Check the dataset and calibration split."
    )

results["absolute_error_eth"] = (
    results["actual_cost_eth"]
    - results["predicted_cost_eth"]
).abs()

results["underestimated"] = (
    results["actual_cost_eth"]
    > results["predicted_cost_eth"]
)


# ============================================================
# OVERALL EVALUATION
# ============================================================

coverage = (
    results["actual_within_estimate"].mean() * 100
)

actual_total = results["actual_cost_eth"].sum()

predicted_total = results["predicted_cost_eth"].sum()

actual_to_predicted_ratio = (
    actual_total / predicted_total
    if predicted_total > 0 else np.nan
)

mae = results["absolute_error_eth"].mean()

underestimation_rate = (
    results["underestimated"].mean() * 100
)

summary = pd.DataFrame([{

    "test_operations": len(results),

    "paymasters": results["paymaster"].nunique(),

    "coverage_percentage": coverage,

    "underestimation_percentage": underestimation_rate,

    "actual_total_cost_eth": actual_total,

    "predicted_total_cost_eth": predicted_total,

    "actual_to_predicted_total_ratio": (
        actual_to_predicted_ratio
    ),

    "mean_absolute_error_eth": mae,

}])


# ============================================================
# EVALUATION BY PREDICTION METHOD
# ============================================================

method_summary = (
    results.groupby("prediction_method")
    .agg(
        operations=("userop_hash", "count"),

        coverage_percentage=(
            "actual_within_estimate",
            lambda values: values.mean() * 100
        ),

        mean_actual_cost_eth=(
            "actual_cost_eth", "mean"
        ),

        mean_predicted_cost_eth=(
            "predicted_cost_eth", "mean"
        ),

        mean_absolute_error_eth=(
            "absolute_error_eth", "mean"
        ),

        actual_total_cost_eth=(
            "actual_cost_eth", "sum"
        ),

        predicted_total_cost_eth=(
            "predicted_cost_eth", "sum"
        ),
    )
    .reset_index()
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

results.to_csv(OUTPUT_FILE, index=False)

summary.to_csv(SUMMARY_FILE, index=False)

method_summary.to_csv(
    Path("data/processed/adaptive_cost_method_summary.csv"),
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("ADAPTIVE COST ESTIMATOR RESULTS")
print("==========================================")

print(f"Calibration operations: {len(calibration)}")
print(f"Test operations: {len(test)}")

print(
    f"Test Paymasters: {results['paymaster'].nunique()}"
)

print(f"Coverage: {coverage:.2f}%")

print(
    f"Underestimation rate: "
    f"{underestimation_rate:.2f}%"
)

print(
    f"Actual total cost: "
    f"{actual_total:.12f} ETH"
)

print(
    f"Predicted total cost: "
    f"{predicted_total:.12f} ETH"
)

print(
    f"Actual / predicted total: "
    f"{actual_to_predicted_ratio:.4f}"
)

print(
    f"Mean absolute error: "
    f"{mae:.12e} ETH"
)

print("\nPrediction methods:")

print(
    results["prediction_method"].value_counts()
)

print("\nMethod-level evaluation:")

print(method_summary.to_string(index=False))

print("\nSaved files:")

print(OUTPUT_FILE)
print(SUMMARY_FILE)
print("data/processed/adaptive_cost_method_summary.csv")

print("\nAdaptive cost estimation completed successfully.")

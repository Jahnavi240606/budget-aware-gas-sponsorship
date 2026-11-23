
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_FILE = Path("data/processed/processed_userops.csv")
OUTPUT_FILE = Path("data/processed/rolling_cost_predictions.csv")
SUMMARY_FILE = Path("data/processed/rolling_cost_summary.csv")

CALIBRATION_FRACTION = 0.20
MIN_HISTORY = 20
WINDOW_SIZES = [50, 100]

# ==========================================
# LOAD DATA
# ==========================================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

required = [
    "userop_hash",
    "transaction_hash",
    "log_index",
    "paymaster",
    "block_number",
    "gas_cost_eth"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["block_number"] = pd.to_numeric(df["block_number"])
df["log_index"] = pd.to_numeric(df["log_index"])
df["gas_cost_eth"] = pd.to_numeric(df["gas_cost_eth"])

if df[required].isnull().any().any():
    raise ValueError("Required columns contain missing values.")

if (df["gas_cost_eth"] < 0).any():
    raise ValueError("Negative gas cost detected.")

df = df.sort_values(
    ["block_number", "transaction_hash", "log_index"]
).reset_index(drop=True)

# ==========================================
# CHRONOLOGICAL SPLIT
# ==========================================

print("Creating calibration and test sets...")

calibration_indices = []
test_indices = []

for paymaster, group in df.groupby("paymaster", sort=False):

    indices = group.index.tolist()

    split = int(len(indices) * CALIBRATION_FRACTION)

    if len(indices) > 1:
        split = max(1, split)
        split = min(split, len(indices) - 1)

    calibration_indices.extend(indices[:split])
    test_indices.extend(indices[split:])

calibration = df.loc[calibration_indices].copy()

test = df.loc[test_indices].copy()

test = test.sort_values(
    ["block_number", "transaction_hash", "log_index"]
)

# ==========================================
# INITIAL HISTORICAL DATA
# ==========================================

history = {}

for paymaster, group in calibration.groupby("paymaster"):

    history[paymaster] = (
        group["gas_cost_eth"].astype(float).tolist()
    )

global_median = calibration["gas_cost_eth"].median()
global_p90 = calibration["gas_cost_eth"].quantile(0.90)

if global_median <= 0:
    raise ValueError("Global calibration median must be positive.")

global_scale = global_p90 / global_median

# ==========================================
# PREDICTION EXPERIMENT
# ==========================================

print("Running rolling-window predictions...")

predictions = []

for _, row in test.iterrows():

    paymaster = row["paymaster"]
    actual_cost = float(row["gas_cost_eth"])

    previous_costs = history.get(paymaster, [])

    # --------------------------------------
    # Expanding P90 baseline
    # --------------------------------------

    if len(previous_costs) >= MIN_HISTORY:
        expanding_prediction = float(
            np.quantile(previous_costs, 0.90)
        )
    elif len(previous_costs) > 0:
        expanding_prediction = (
            float(np.median(previous_costs)) * global_scale
        )
    else:
        expanding_prediction = float(global_p90)

    # --------------------------------------
    # Rolling P90 predictions
    # --------------------------------------

    rolling_predictions = {}

    for window in WINDOW_SIZES:

        if len(previous_costs) >= MIN_HISTORY:

            recent_costs = previous_costs[-window:]

            rolling_prediction = float(
                np.quantile(recent_costs, 0.90)
            )

        elif len(previous_costs) > 0:

            rolling_prediction = (
                float(np.median(previous_costs)) * global_scale
            )

        else:

            rolling_prediction = float(global_p90)

        rolling_predictions[window] = rolling_prediction

    # --------------------------------------
    # Record predictions before history update
    # --------------------------------------

    record = {
        "userop_hash": row["userop_hash"],
        "transaction_hash": row["transaction_hash"],
        "log_index": row["log_index"],
        "paymaster": paymaster,
        "block_number": int(row["block_number"]),
        "actual_cost_eth": actual_cost,
        "expanding_p90": expanding_prediction,
        "history_size": len(previous_costs)
    }

    for window in WINDOW_SIZES:
        record[f"rolling_p90_{window}"] = (
            rolling_predictions[window]
        )

    predictions.append(record)

    # Update history only after all predictions.
    history.setdefault(paymaster, []).append(actual_cost)

# ==========================================
# RESULTS
# ==========================================

results = pd.DataFrame(predictions)

if results.empty:
    raise ValueError("No predictions generated.")

# ==========================================
# EVALUATION
# ==========================================

print("Evaluating prediction methods...")

methods = [
    "expanding_p90",
    "rolling_p90_50",
    "rolling_p90_100"
]

summary_rows = []

for method in methods:

    actual = results["actual_cost_eth"]
    predicted = results[method]

    coverage = (actual <= predicted).mean() * 100

    mae = np.abs(actual - predicted).mean()

    actual_total = actual.sum()
    predicted_total = predicted.sum()

    underestimation = (actual > predicted).mean() * 100

    summary_rows.append({

        "method": method,

        "operations": len(results),

        "coverage_percentage": coverage,

        "underestimation_percentage": underestimation,

        "mean_absolute_error_eth": mae,

        "actual_total_cost_eth": actual_total,

        "predicted_total_cost_eth": predicted_total,

        "actual_to_predicted_ratio": (
            actual_total / predicted_total
            if predicted_total > 0 else np.nan
        )
    })

summary = pd.DataFrame(summary_rows)

# ==========================================
# SAVE RESULTS
# ==========================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

results.to_csv(OUTPUT_FILE, index=False)

summary.to_csv(SUMMARY_FILE, index=False)

# ==========================================
# DISPLAY
# ==========================================

print("\n======================================")
print("ROLLING-WINDOW ESTIMATOR RESULTS")
print("======================================")

print(f"Calibration operations: {len(calibration)}")
print(f"Test operations: {len(test)}")

print("\nMethod comparison:")

print(summary.to_string(index=False))

print("\nSaved files:")
print(OUTPUT_FILE)
print(SUMMARY_FILE)

print("\nExperiment completed successfully.")


import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_FILE = Path("data/processed/processed_userops.csv")
OUTPUT_FILE = Path("data/processed/cost_estimation_analysis.csv")

CALIBRATION_RATIO = 0.20

# ==========================================
# LOAD DATA
# ==========================================

print("GAS COST ESTIMATION ANALYSIS")
print("============================")

df = pd.read_csv(INPUT_FILE)

df = df.sort_values(
    ["block_number", "log_index"]
).reset_index(drop=True)

# ==========================================
# SPLIT CALIBRATION AND TEST DATA
# ==========================================

calibration_parts = []
test_parts = []

for paymaster, group in df.groupby("paymaster", sort=False):

    group = group.copy()

    split_index = max(
        1,
        int(len(group) * CALIBRATION_RATIO)
    )

    if len(group) > 1:
        split_index = min(split_index, len(group) - 1)

    calibration_parts.append(group.iloc[:split_index])
    test_parts.append(group.iloc[split_index:])

calibration = pd.concat(
    calibration_parts,
    ignore_index=True
)

test = pd.concat(
    test_parts,
    ignore_index=True
)

print("Calibration operations:", len(calibration))
print("Test operations:", len(test))

# ==========================================
# CALCULATE HISTORICAL COST ESTIMATES
# ==========================================

estimates = calibration.groupby("paymaster")[
    "gas_cost_eth"
].agg(
    calibration_count="count",
    p50=lambda x: x.quantile(0.50),
    p75=lambda x: x.quantile(0.75),
    p90=lambda x: x.quantile(0.90)
).reset_index()

test = test.merge(
    estimates,
    on="paymaster",
    how="left"
)

if test[["p50", "p75", "p90"]].isna().any().any():
    raise ValueError("Missing Paymaster cost estimates")

# ==========================================
# EVALUATE ESTIMATION ACCURACY
# ==========================================

def evaluate_estimate(actual, estimate):

    coverage = (
        actual <= estimate
    ).mean() * 100

    mean_ratio = (
        actual / estimate.replace(0, float("nan"))
    ).mean()

    total_ratio = (
        actual.sum() / estimate.sum() * 100
        if estimate.sum() > 0 else 0
    )

    return {
        "coverage_percent": coverage,
        "mean_actual_to_estimated_ratio": mean_ratio,
        "total_actual_vs_estimated_percent": total_ratio
    }


summary_rows = []

for column in ["p50", "p75", "p90"]:

    metrics = evaluate_estimate(
        test["gas_cost_eth"],
        test[column]
    )

    summary_rows.append({
        "estimator": column.upper(),
        **metrics
    })

summary = pd.DataFrame(summary_rows)

# ==========================================
# PAYMASTER-LEVEL ANALYSIS
# ==========================================

paymaster_rows = []

for paymaster, group in test.groupby("paymaster"):

    row = {
        "paymaster": paymaster,
        "calibration_operations": int(
            group["calibration_count"].iloc[0]
        ),
        "test_operations": len(group)
    }

    for column in ["p50", "p75", "p90"]:

        row[f"{column}_coverage_percent"] = (
            group["gas_cost_eth"] <= group[column]
        ).mean() * 100

    row["actual_total_cost_eth"] = (
        group["gas_cost_eth"].sum()
    )

    row["p90_estimated_total_cost_eth"] = (
        group["p90"].sum()
    )

    paymaster_rows.append(row)

paymaster_summary = pd.DataFrame(paymaster_rows)

# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\nOVERALL ESTIMATION PERFORMANCE")
print("===============================")

print(
    summary.round(4).to_string(index=False)
)

print("\nPAYMASTER-LEVEL ESTIMATION")
print("=========================")

print(
    paymaster_summary.round(8).to_string(index=False)
)

print("\nPAYMASTERS WITH SMALL CALIBRATION SAMPLES")

print(
    paymaster_summary[
        paymaster_summary["calibration_operations"] < 20
    ][
        ["paymaster", "calibration_operations", "test_operations"]
    ].to_string(index=False)
)

# ==========================================
# SAVE RESULTS
# ==========================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

paymaster_summary.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nAnalysis saved to:", OUTPUT_FILE)
print("COST ESTIMATION ANALYSIS COMPLETED")

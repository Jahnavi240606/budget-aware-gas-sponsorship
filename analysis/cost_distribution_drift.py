
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/processed_userops.csv")

print("GAS COST DISTRIBUTION ANALYSIS")
print("==============================")

df = pd.read_csv(INPUT_FILE)

df = df.sort_values(
    ["block_number", "log_index"]
).reset_index(drop=True)

calibration_parts = []
test_parts = []

# Split each Paymaster chronologically
for paymaster, group in df.groupby("paymaster", sort=False):

    group = group.copy()

    split_index = max(1, int(len(group) * 0.20))

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

# Calculate cost distribution statistics
def calculate_statistics(group):

    costs = group["gas_cost_eth"]

    return {
        "operations": len(group),
        "mean": costs.mean(),
        "median": costs.median(),
        "p90": costs.quantile(0.90),
        "p99": costs.quantile(0.99),
        "maximum": costs.max()
    }

results = []

for paymaster in df["paymaster"].unique():

    cal_group = calibration[
        calibration["paymaster"] == paymaster
    ]

    test_group = test[
        test["paymaster"] == paymaster
    ]

    cal = calculate_statistics(cal_group)
    tst = calculate_statistics(test_group)

    results.append({
        "paymaster": paymaster,
        "calibration_count": cal["operations"],
        "test_count": tst["operations"],
        "calibration_median": cal["median"],
        "test_median": tst["median"],
        "calibration_p90": cal["p90"],
        "test_p90": tst["p90"],
        "calibration_p99": cal["p99"],
        "test_p99": tst["p99"],
        "calibration_max": cal["maximum"],
        "test_max": tst["maximum"]
    })

result_df = pd.DataFrame(results)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

print("\nPAYMASTER COST DISTRIBUTION COMPARISON")
print("======================================")

print(
    result_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.12f}"
    )
)

print("\nOVERALL CALIBRATION STATISTICS")
print("==============================")

print(
    pd.Series(
        calculate_statistics(calibration)
    )
)

print("\nOVERALL TEST STATISTICS")
print("=======================")

print(
    pd.Series(
        calculate_statistics(test)
    )
)

output = Path("data/processed/cost_distribution_drift.csv")

output.parent.mkdir(
    parents=True,
    exist_ok=True
)

result_df.to_csv(output, index=False)

print("\nAnalysis saved to:", output)
print("DISTRIBUTION ANALYSIS COMPLETED")


import pandas as pd
import numpy as np
from pathlib import Path

# ==================================================
# CONFIGURATION
# ==================================================

INPUT_FILE = Path("data/processed/processed_userops.csv")
RESULTS_FILE = Path("data/processed/allocation_results.csv")
SUMMARY_FILE = Path("data/processed/allocation_summary.csv")

CALIBRATION_RATIO = 0.20

# Calibration spending is 20% of the sample.
# Multiplying it by 2.5 creates a simulated budget
# equal to 50% of extrapolated sample spending.
BUDGET_MULTIPLIER = 2.5

# ==================================================
# LOAD DATA
# ==================================================

print("BUDGET-AWARE GAS SPONSORSHIP SIMULATION")
print("=======================================")

if not INPUT_FILE.exists():
    raise FileNotFoundError("Processed dataset not found!")

df = pd.read_csv(INPUT_FILE)

df["block_timestamp"] = pd.to_datetime(
    df["block_timestamp"], utc=True
)

df = df.sort_values(
    ["block_number", "log_index"]
).reset_index(drop=True)

df["success"] = (
    df["success"].astype(str).str.lower()
    .map({"true": True, "false": False, "1": True, "0": False})
)

if df["success"].isna().any():
    raise ValueError("Invalid success values in dataset")

print("Total operations:", len(df))
print("Unique users:", df["sender"].nunique())
print("Unique Paymasters:", df["paymaster"].nunique())

# ==================================================
# SPLIT CALIBRATION AND TEST DATA
# ==================================================

calibration_parts = []
test_parts = []

for paymaster, group in df.groupby("paymaster", sort=False):

    group = group.sort_values(
        ["block_number", "log_index"]
    ).copy()

    calibration_size = max(
        1,
        int(len(group) * CALIBRATION_RATIO)
    )

    # Keep at least one test operation where possible.
    if len(group) > 1:
        calibration_size = min(
            calibration_size,
            len(group) - 1
        )

    calibration_parts.append(
        group.iloc[:calibration_size].copy()
    )

    test_parts.append(
        group.iloc[calibration_size:].copy()
    )

calibration_df = pd.concat(
    calibration_parts,
    ignore_index=True
)

test_df = pd.concat(
    test_parts,
    ignore_index=True
)

print("\nCALIBRATION AND TEST SPLIT")
print("==========================")
print("Calibration operations:", len(calibration_df))
print("Test operations:", len(test_df))

if test_df.empty:
    raise ValueError("No test operations available")

# ==================================================
# HISTORICAL USER INFORMATION
# ==================================================

# Only calibration-period information is used
# to calculate historical user features.

user_history = calibration_df.groupby("sender").agg(
    previous_successful_count=("success", "sum"),
    previous_successful_spending=(
        "successful_gas_cost_eth", "sum"
    )
).to_dict("index")

# ==================================================
# PAYMASTER COST ESTIMATES AND BUDGETS
# ==================================================

paymaster_medians = {}
paymaster_budgets = {}
paymaster_calibration_spending = {}

for paymaster, group in calibration_df.groupby("paymaster"):

    median_cost = group["gas_cost_eth"].median()

    calibration_spending = group["gas_cost_eth"].sum()

    paymaster_medians[paymaster] = median_cost

    paymaster_calibration_spending[paymaster] = (
        calibration_spending
    )

    paymaster_budgets[paymaster] = (
        calibration_spending * BUDGET_MULTIPLIER
    )

# Use only calibration-period Paymaster median cost
# as the estimated cost for each test operation.

test_df["estimated_cost_eth"] = (
    test_df["paymaster"].map(paymaster_medians)
)

if test_df["estimated_cost_eth"].isna().any():
    raise ValueError("Missing historical Paymaster cost estimate")

# ==================================================
# PREPARE USER HISTORY FOR TEST OPERATIONS
# ==================================================

test_df["previous_user_count"] = (
    test_df["sender"].map(
        lambda user: user_history.get(
            user, {}
        ).get("previous_successful_count", 0)
    )
)

test_df["previous_user_spending"] = (
    test_df["sender"].map(
        lambda user: user_history.get(
            user, {}
        ).get("previous_successful_spending", 0.0)
    )
)

# ==================================================
# FAIRNESS METRIC
# ==================================================

def jain_fairness(values):

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return 0.0

    denominator = len(values) * np.sum(values ** 2)

    if denominator == 0:
        return 0.0

    return float(
        (np.sum(values) ** 2) / denominator
    )

# ==================================================
# ALLOCATION FUNCTION
# ==================================================

def run_strategy(strategy):

    decisions = []

    # Each strategy starts with the same budgets.
    remaining_budget = paymaster_budgets.copy()

    for paymaster, group in test_df.groupby(
        "paymaster", sort=False
    ):

        group = group.copy()

        # ------------------------------------------
        # STRATEGY 1: FIRST-COME, FIRST-SERVED
        # ------------------------------------------

        if strategy == "FCFS":

            group = group.sort_values(
                ["block_number", "log_index"]
            )

            group["priority_score"] = 0.0

        # ------------------------------------------
        # STRATEGY 2: FAIRNESS-BASED
        # ------------------------------------------

        elif strategy == "Fairness":

            # Users with fewer previous successful
            # sponsorships receive higher priority.

            group = group.sort_values(
                [
                    "previous_user_count",
                    "block_number",
                    "log_index"
                ],
                ascending=[True, True, True]
            )

            group["priority_score"] = (
                1 / (1 + group["previous_user_count"])
            )

        # ------------------------------------------
        # STRATEGY 3: PROPOSED METHOD
        # ------------------------------------------

        elif strategy == "Proposed":

            calibration_spending = (
                paymaster_calibration_spending[paymaster]
            )

            median_cost = paymaster_medians[paymaster]

            # Fairness component:
            # Fewer previous sponsorships = higher score.

            fairness_score = (
                1 / (1 + group["previous_user_count"])
            )

            # Spending component:
            # Lower historical user spending = higher score.

            spending_score = 1 / (
                1
                + group["previous_user_spending"]
                / max(calibration_spending, 1e-18)
            )

            # Cost component:
            # Lower estimated operation cost = higher score.

            cost_score = 1 / (
                1
                + group["estimated_cost_eth"]
                / max(median_cost, 1e-18)
            )

            # Combined proposed priority score.

            group["priority_score"] = (
                0.50 * fairness_score
                + 0.30 * spending_score
                + 0.20 * cost_score
            )

            group = group.sort_values(
                [
                    "priority_score",
                    "block_number",
                    "log_index"
                ],
                ascending=[False, True, True]
            )

        else:
            raise ValueError("Unknown strategy")

        # ------------------------------------------
        # SPONSORSHIP DECISIONS
        # ------------------------------------------

        for _, row in group.iterrows():

            estimated_cost = float(
                row["estimated_cost_eth"]
            )

            actual_cost = float(
                row["gas_cost_eth"]
            )

            available_budget = remaining_budget[paymaster]

            # Reserve estimated cost, not actual cost.
            # Actual cost is unknown at decision time.

            sponsored = (
                estimated_cost <= available_budget
            )

            if sponsored:

                remaining_budget[paymaster] -= (
                    estimated_cost
                )

            decisions.append({

                "strategy": strategy,

                "user": row["sender"],

                "paymaster": paymaster,

                "block_number": row["block_number"],

                "log_index": row["log_index"],

                "estimated_cost_eth": estimated_cost,

                "actual_cost_eth": actual_cost,

                "priority_score": float(
                    row["priority_score"]
                ),

                "sponsored": sponsored,

                "budget_eth": paymaster_budgets[paymaster],

                "remaining_estimated_budget_eth": (
                    remaining_budget[paymaster]
                )
            })

    return pd.DataFrame(decisions)

# ==================================================
# RUN ALL THREE ALGORITHMS
# ==================================================

all_results = []

for strategy in ["FCFS", "Fairness", "Proposed"]:

    print(f"\nRunning {strategy} strategy...")

    result = run_strategy(strategy)

    all_results.append(result)

all_results = pd.concat(
    all_results,
    ignore_index=True
)

# ==================================================
# PERFORMANCE EVALUATION
# ==================================================

summary_rows = []

total_budget = sum(paymaster_budgets.values())

eligible_users = test_df["sender"].unique()

for strategy in ["FCFS", "Fairness", "Proposed"]:

    subset = all_results[
        all_results["strategy"] == strategy
    ].copy()

    sponsored = subset[
        subset["sponsored"]
    ]

    sponsored_counts = sponsored.groupby(
        "user"
    ).size()

    total_operations = len(subset)

    sponsored_operations = len(sponsored)

    covered_users = sponsored_counts.index.nunique()

    operation_coverage = (
        sponsored_operations / total_operations * 100
        if total_operations else 0
    )

    user_coverage = (
        covered_users / len(eligible_users) * 100
        if len(eligible_users) else 0
    )

    actual_spending = sponsored[
        "actual_cost_eth"
    ].sum()

    estimated_spending = sponsored[
        "estimated_cost_eth"
    ].sum()

    estimated_utilization = (
        estimated_spending / total_budget * 100
        if total_budget else 0
    )

    actual_utilization = (
        actual_spending / total_budget * 100
        if total_budget else 0
    )

    # Calculate actual spending beyond budget
    # separately for each Paymaster.

    actual_by_paymaster = sponsored.groupby(
        "paymaster"
    )["actual_cost_eth"].sum()

    budget_excess = 0.0

    for paymaster, budget in paymaster_budgets.items():

        actual = actual_by_paymaster.get(
            paymaster, 0.0
        )

        budget_excess += max(
            0.0, actual - budget
        )

    fairness_values = (
        sponsored_counts
        .reindex(eligible_users, fill_value=0)
        .values
    )

    fairness_index = jain_fairness(
        fairness_values
    )

    summary_rows.append({

        "strategy": strategy,

        "total_test_operations": total_operations,

        "sponsored_operations": sponsored_operations,

        "operation_coverage_percent": operation_coverage,

        "users_sponsored": covered_users,

        "user_coverage_percent": user_coverage,

        "actual_spending_eth": actual_spending,

        "estimated_spending_eth": estimated_spending,

        "budget_eth": total_budget,

        "estimated_budget_utilization_percent": (
            estimated_utilization
        ),

        "actual_budget_utilization_percent": (
            actual_utilization
        ),

        "actual_budget_excess_eth": budget_excess,

        "jain_fairness_index": fairness_index
    })

summary = pd.DataFrame(summary_rows)

# ==================================================
# SAVE OUTPUT FILES
# ==================================================

RESULTS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

all_results.to_csv(
    RESULTS_FILE,
    index=False
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)

# ==================================================
# DISPLAY RESULTS
# ==================================================

print("\nALLOCATION PERFORMANCE")
print("======================")

print(
    summary.round(6).to_string(index=False)
)

print("\nSIMULATION COMPLETED")
print("====================")

print("Detailed results:", RESULTS_FILE)
print("Summary results:", SUMMARY_FILE)

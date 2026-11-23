
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FILE = Path("data/processed/processed_userops.csv")

PREDICTION_FILE = Path(
    "data/processed/rolling_cost_predictions.csv"
)

RESULT_FILE = Path(
    "data/processed/budget_allocation_results.csv"
)

SUMMARY_FILE = Path(
    "data/processed/budget_allocation_summary.csv"
)

CALIBRATION_FRACTION = 0.20

BUDGET_MULTIPLIER = 2.5

# Proposed algorithm weights
FAIRNESS_WEIGHT = 0.50
SPENDING_WEIGHT = 0.30
COST_WEIGHT = 0.20


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading processed UserOperation data...")

df = pd.read_csv(DATA_FILE)

predictions = pd.read_csv(PREDICTION_FILE)

required_data_columns = [
    "userop_hash",
    "transaction_hash",
    "log_index",
    "paymaster",
    "sender",
    "block_number",
    "gas_cost_eth"
]

required_prediction_columns = [
    "userop_hash",
    "rolling_p90_50"
]

for column in required_data_columns:

    if column not in df.columns:
        raise ValueError(
            f"Missing dataset column: {column}"
        )

for column in required_prediction_columns:

    if column not in predictions.columns:
        raise ValueError(
            f"Missing prediction column: {column}"
        )


# ============================================================
# 3. DATA VALIDATION AND CLEANING
# ============================================================

print("Validating dataset...")

# Normalize hexadecimal identifiers and addresses.
for column in [
    "userop_hash",
    "transaction_hash",
    "paymaster",
    "sender"
]:

    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .str.lower()
    )

predictions["userop_hash"] = (
    predictions["userop_hash"]
    .astype("string")
    .str.strip()
    .str.lower()
)

# Convert numerical columns.
for column in [
    "block_number",
    "log_index",
    "gas_cost_eth"
]:

    df[column] = pd.to_numeric(
        df[column],
        errors="raise"
    )

predictions["rolling_p90_50"] = pd.to_numeric(
    predictions["rolling_p90_50"],
    errors="raise"
)

# Check missing values.
if df[required_data_columns].isnull().any().any():
    raise ValueError(
        "Missing values found in required dataset columns."
    )

if predictions[
    required_prediction_columns
].isnull().any().any():

    raise ValueError(
        "Missing values found in prediction data."
    )

# Check duplicate UserOperations.
if df["userop_hash"].duplicated().any():

    raise ValueError(
        "Duplicate UserOperation hashes in dataset."
    )

if predictions["userop_hash"].duplicated().any():

    raise ValueError(
        "Duplicate UserOperation hashes in predictions."
    )

# Check gas costs.
if (df["gas_cost_eth"] < 0).any():

    raise ValueError(
        "Negative actual gas cost found."
    )

if (predictions["rolling_p90_50"] < 0).any():

    raise ValueError(
        "Negative predicted gas cost found."
    )

# Chronological ordering.
df = df.sort_values(
    [
        "block_number",
        "transaction_hash",
        "log_index"
    ]
).reset_index(drop=True)


# ============================================================
# 4. CHRONOLOGICAL CALIBRATION AND TEST SPLIT
# ============================================================

print("Creating chronological calibration and test sets...")

calibration_indices = []
test_indices = []

for paymaster, group in df.groupby(
    "paymaster",
    sort=False
):

    indices = group.index.tolist()

    split = int(
        len(indices) * CALIBRATION_FRACTION
    )

    # For groups with multiple observations,
    # ensure at least one calibration and one test row.
    if len(indices) > 1:

        split = max(1, split)

        split = min(
            split,
            len(indices) - 1
        )

    calibration_indices.extend(
        indices[:split]
    )

    test_indices.extend(
        indices[split:]
    )

calibration = df.loc[
    calibration_indices
].copy()

test = df.loc[
    test_indices
].copy()

test = test.sort_values(
    [
        "block_number",
        "transaction_hash",
        "log_index"
    ]
).reset_index(drop=True)

print(
    f"Calibration operations: {len(calibration)}"
)

print(
    f"Test operations: {len(test)}"
)


# ============================================================
# 5. JOIN ROLLING P90 PREDICTIONS
# ============================================================

print("Connecting rolling-window cost predictions...")

test = test.merge(
    predictions[
        [
            "userop_hash",
            "rolling_p90_50"
        ]
    ],
    on="userop_hash",
    how="left",
    validate="one_to_one"
)

if test["rolling_p90_50"].isnull().any():

    missing_count = int(
        test["rolling_p90_50"].isnull().sum()
    )

    raise ValueError(
        f"{missing_count} test operations have no "
        "rolling P90 prediction."
    )

test = test.rename(
    columns={
        "rolling_p90_50": "predicted_cost_eth"
    }
)


# ============================================================
# 6. CALCULATE PAYMASTER BUDGETS
# ============================================================

print("Calculating simulated Paymaster budgets...")

budget_data = {}

# Include every Paymaster appearing in the test set.
# This is important because a Paymaster may have no
# calibration observations.

for paymaster in test["paymaster"].unique():

    paymaster_calibration = calibration[
        calibration["paymaster"] == paymaster
    ]

    if len(paymaster_calibration) == 0:

        calibration_spending = 0.0

        simulated_budget = 0.0

        has_calibration_history = False

    else:

        calibration_spending = float(
            paymaster_calibration["gas_cost_eth"].sum()
        )

        simulated_budget = (
            calibration_spending * BUDGET_MULTIPLIER
        )

        has_calibration_history = True

    budget_data[paymaster] = {

        "calibration_spending_eth": calibration_spending,

        "budget_eth": simulated_budget,

        "has_calibration_history": has_calibration_history

    }

print(
    "Paymasters in test set:",
    len(budget_data)
)

print(
    "Paymasters without calibration history:",
    sum(
        not item["has_calibration_history"]
        for item in budget_data.values()
    )
)


# ============================================================
# 7. CALCULATE USER HISTORY FROM CALIBRATION ONLY
# ============================================================

print("Calculating historical user sponsorship statistics...")

user_history = (
    calibration.groupby(
        [
            "paymaster",
            "sender"
        ]
    )
    .agg(
        previous_sponsorship_count=(
            "userop_hash",
            "count"
        ),

        previous_sponsorship_spending=(
            "gas_cost_eth",
            "sum"
        )
    )
    .reset_index()
)

# Remove existing columns from the processed dataset.
# We must use calibration-period statistics only.

columns_to_remove = [
    "previous_sponsorship_count",
    "previous_sponsorship_spending"
]

test = test.drop(
    columns=columns_to_remove,
    errors="ignore"
)

test = test.merge(
    user_history,
    on=[
        "paymaster",
        "sender"
    ],
    how="left",
    validate="many_to_one"
)

# New users receive zero historical counts and spending.

test["previous_sponsorship_count"] = (
    test["previous_sponsorship_count"].fillna(0)
)

test["previous_sponsorship_spending"] = (
    test["previous_sponsorship_spending"].fillna(0.0)
)


# ============================================================
# 8. CALCULATE PRIORITY SCORES
# ============================================================

print("Calculating allocation priority scores...")

test["fairness_score"] = 0.0

test["spending_score"] = 0.0

test["cost_score"] = 0.0

for paymaster, group in test.groupby("paymaster"):

    indices = group.index

    # --------------------------------------------------------
    # Fairness:
    # Users with fewer historical sponsorships get priority.
    # --------------------------------------------------------

    fairness_rank = group[
        "previous_sponsorship_count"
    ].rank(
        method="average",
        pct=True
    )

    test.loc[
        indices,
        "fairness_score"
    ] = 1.0 - fairness_rank

    # --------------------------------------------------------
    # Spending:
    # Users with lower previous spending get priority.
    # --------------------------------------------------------

    spending_rank = group[
        "previous_sponsorship_spending"
    ].rank(
        method="average",
        pct=True
    )

    test.loc[
        indices,
        "spending_score"
    ] = 1.0 - spending_rank

    # --------------------------------------------------------
    # Cost:
    # Operations with lower predicted cost get priority.
    # --------------------------------------------------------

    cost_rank = group[
        "predicted_cost_eth"
    ].rank(
        method="average",
        pct=True
    )

    test.loc[
        indices,
        "cost_score"
    ] = 1.0 - cost_rank

# Combined proposed priority score.

test["proposed_score"] = (

    FAIRNESS_WEIGHT * test["fairness_score"]

    + SPENDING_WEIGHT * test["spending_score"]

    + COST_WEIGHT * test["cost_score"]

)


# ============================================================
# 9. ALLOCATION FUNCTION
# ============================================================

def run_allocation(strategy_name):

    print(
        f"Running {strategy_name} allocation..."
    )

    allocated_records = []

    for paymaster, group in test.groupby("paymaster"):

        # Safe lookup: all test Paymasters have a budget entry.
        budget = budget_data.get(
            paymaster,
            {"budget_eth": 0.0}
        )["budget_eth"]

        remaining_budget = budget

        # ----------------------------------------------------
        # FCFS STRATEGY
        # ----------------------------------------------------

        if strategy_name == "FCFS":

            ordered = group.sort_values(
                [
                    "block_number",
                    "transaction_hash",
                    "log_index"
                ],
                ascending=[
                    True,
                    True,
                    True
                ]
            )

        # ----------------------------------------------------
        # FAIRNESS STRATEGY
        # ----------------------------------------------------

        elif strategy_name == "FAIRNESS":

            ordered = group.sort_values(
                [
                    "fairness_score",
                    "block_number",
                    "transaction_hash",
                    "log_index"
                ],
                ascending=[
                    False,
                    True,
                    True,
                    True
                ]
            )

        # ----------------------------------------------------
        # PROPOSED STRATEGY
        # ----------------------------------------------------

        elif strategy_name == "PROPOSED":

            ordered = group.sort_values(
                [
                    "proposed_score",
                    "block_number",
                    "transaction_hash",
                    "log_index"
                ],
                ascending=[
                    False,
                    True,
                    True,
                    True
                ]
            )

        else:

            raise ValueError(
                f"Unknown allocation strategy: {strategy_name}"
            )

        # ----------------------------------------------------
        # PROCESS OPERATIONS
        # ----------------------------------------------------

        for _, row in ordered.iterrows():

            predicted_cost = float(
                row["predicted_cost_eth"]
            )

            actual_cost = float(
                row["gas_cost_eth"]
            )

            # Reserve predicted cost before accepting.
            accepted = (
                predicted_cost <= remaining_budget
            )

            if accepted:

                remaining_budget -= predicted_cost

            allocated_records.append({

                "strategy": strategy_name,

                "userop_hash": row["userop_hash"],

                "transaction_hash": row["transaction_hash"],

                "log_index": row["log_index"],

                "paymaster": paymaster,

                "sender": row["sender"],

                "block_number": row["block_number"],

                "predicted_cost_eth": predicted_cost,

                "actual_cost_eth": actual_cost,

                "accepted": accepted,

                "fairness_score": row["fairness_score"],

                "spending_score": row["spending_score"],

                "cost_score": row["cost_score"],

                "proposed_score": row["proposed_score"],

                "paymaster_budget_eth": budget,

                "remaining_budget_eth": remaining_budget

            })

    return pd.DataFrame(allocated_records)


# ============================================================
# 10. RUN THREE ALLOCATION STRATEGIES
# ============================================================

print("\nStarting allocation experiments...")

strategies = [
    "FCFS",
    "FAIRNESS",
    "PROPOSED"
]

all_results = []

for strategy in strategies:

    strategy_result = run_allocation(strategy)

    all_results.append(
        strategy_result
    )

results = pd.concat(
    all_results,
    ignore_index=True
)


# ============================================================
# 11. JAIN'S FAIRNESS INDEX
# ============================================================

def calculate_jain_index(values):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) == 0:
        return 0.0

    denominator = (
        len(values) * np.sum(values ** 2)
    )

    if denominator == 0:
        return 0.0

    return (
        np.sum(values) ** 2 / denominator
    )


# ============================================================
# 12. EVALUATE ALLOCATION STRATEGIES
# ============================================================

print("\nEvaluating allocation performance...")

summary_rows = []

total_test_operations = len(test)

# Include every user in the test period, even if not sponsored.
all_test_users = sorted(
    test["sender"].unique()
)

total_test_users = len(all_test_users)

total_simulated_budget = sum(
    item["budget_eth"]
    for item in budget_data.values()
)

for strategy in strategies:

    strategy_results = results[
        results["strategy"] == strategy
    ].copy()

    accepted = strategy_results[
        strategy_results["accepted"]
    ].copy()

    sponsored_operations = len(accepted)

    unique_sponsored_users = accepted[
        "sender"
    ].nunique()

    # --------------------------------------------------------
    # Operation coverage
    # --------------------------------------------------------

    operation_coverage = (
        sponsored_operations / total_test_operations * 100
        if total_test_operations > 0 else 0
    )

    # --------------------------------------------------------
    # User coverage
    # --------------------------------------------------------

    user_coverage = (
        unique_sponsored_users / total_test_users * 100
        if total_test_users > 0 else 0
    )

    # --------------------------------------------------------
    # Actual and predicted spending
    # --------------------------------------------------------

    actual_spending = float(
        accepted["actual_cost_eth"].sum()
    )

    predicted_spending = float(
        accepted["predicted_cost_eth"].sum()
    )

    # --------------------------------------------------------
    # Budget utilisation
    # --------------------------------------------------------

    actual_budget_utilization = (
        actual_spending / total_simulated_budget * 100
        if total_simulated_budget > 0 else 0
    )

    estimated_budget_utilization = (
        predicted_spending / total_simulated_budget * 100
        if total_simulated_budget > 0 else 0
    )

    # --------------------------------------------------------
    # Actual budget excess per Paymaster
    # --------------------------------------------------------

    budget_excess = 0.0

    paymasters_exceeding_budget = 0

    for paymaster, budget_info in budget_data.items():

        paymaster_accepted = accepted[
            accepted["paymaster"] == paymaster
        ]

        paymaster_actual_spending = float(
            paymaster_accepted["actual_cost_eth"].sum()
        )

        excess = max(
            0.0,
            paymaster_actual_spending
            - budget_info["budget_eth"]
        )

        budget_excess += excess

        if excess > 0:
            paymasters_exceeding_budget += 1

    # --------------------------------------------------------
    # Jain fairness index
    # --------------------------------------------------------

    user_counts = (
        accepted.groupby("sender")["userop_hash"]
        .count()
    )

    # Add zero-sponsored users.
    user_counts = user_counts.reindex(
        all_test_users,
        fill_value=0
    )

    jain_index = calculate_jain_index(
        user_counts.values
    )

    # --------------------------------------------------------
    # Store metrics
    # --------------------------------------------------------

    summary_rows.append({

        "strategy": strategy,

        "test_operations": total_test_operations,

        "sponsored_operations": sponsored_operations,

        "operation_coverage_percentage": operation_coverage,

        "total_test_users": total_test_users,

        "unique_sponsored_users": unique_sponsored_users,

        "user_coverage_percentage": user_coverage,

        "actual_spending_eth": actual_spending,

        "predicted_reserved_spending_eth": predicted_spending,

        "total_simulated_budget_eth": total_simulated_budget,

        "actual_budget_utilization_percentage": (
            actual_budget_utilization
        ),

        "estimated_budget_utilization_percentage": (
            estimated_budget_utilization
        ),

        "actual_budget_excess_eth": budget_excess,

        "paymasters_exceeding_budget": (
            paymasters_exceeding_budget
        ),

        "jain_fairness_index": jain_index

    })

summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

print("\nSaving allocation results...")

RESULT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

results.to_csv(
    RESULT_FILE,
    index=False
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# 14. DISPLAY RESULTS
# ============================================================

print("\n==============================================")

print("BUDGET-AWARE ALLOCATION COMPARISON")

print("==============================================")

print(
    summary.to_string(index=False)
)

print("\nSaved files:")

print(RESULT_FILE)

print(SUMMARY_FILE)

print("\nAllocation experiments completed successfully.")

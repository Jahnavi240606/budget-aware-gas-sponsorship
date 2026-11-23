
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FILE = Path("data/processed/processed_userops.csv")

RESULT_FILE = Path(
    "data/processed/dynamic_allocation_results.csv"
)

SUMMARY_FILE = Path(
    "data/processed/dynamic_allocation_summary.csv"
)

CALIBRATION_FRACTION = 0.20

ROLLING_WINDOW = 50

MIN_HISTORY = 20

BUDGET_MULTIPLIERS = [1.5, 2.5, 5.0]

FAIRNESS_WEIGHT = 0.50
SPENDING_WEIGHT = 0.30
COST_WEIGHT = 0.20

STRATEGIES = ["FCFS", "FAIRNESS", "PROPOSED"]


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

print("Loading processed UserOperation dataset...")

df = pd.read_csv(DATA_FILE)

required_columns = [
    "userop_hash",
    "transaction_hash",
    "log_index",
    "paymaster",
    "sender",
    "block_number",
    "gas_cost_eth"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

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

for column in [
    "block_number",
    "log_index",
    "gas_cost_eth"
]:
    df[column] = pd.to_numeric(
        df[column],
        errors="raise"
    )

if df[required_columns].isnull().any().any():
    raise ValueError(
        "Missing values found in required columns."
    )

if df["userop_hash"].duplicated().any():
    raise ValueError(
        "Duplicate UserOperation hashes found."
    )

if (df["gas_cost_eth"] < 0).any():
    raise ValueError(
        "Negative gas cost found."
    )

# Process operations in block and event-log order.
df = df.sort_values(
    ["block_number", "log_index", "transaction_hash"]
).reset_index(drop=True)

print(f"Total operations: {len(df)}")
print(f"Total Paymasters: {df['paymaster'].nunique()}")
print(f"Total users: {df['sender'].nunique()}")


# ============================================================
# 3. CHRONOLOGICAL CALIBRATION / TEST SPLIT
# ============================================================

print("\nCreating calibration and test sets...")

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

    if len(indices) > 1:
        split = max(1, split)
        split = min(split, len(indices) - 1)

    calibration_indices.extend(indices[:split])
    test_indices.extend(indices[split:])

calibration = df.loc[
    calibration_indices
].copy()

test = df.loc[
    test_indices
].copy()

test = test.sort_values(
    ["block_number", "log_index", "transaction_hash"]
).reset_index(drop=True)

print(f"Calibration operations: {len(calibration)}")
print(f"Test operations: {len(test)}")


# ============================================================
# 4. GENERATE CHRONOLOGICAL ROLLING P90 PREDICTIONS
# ============================================================

print("\nGenerating rolling P90 predictions...")

cost_history = {}

for paymaster, group in calibration.groupby("paymaster"):

    cost_history[paymaster] = (
        group["gas_cost_eth"].astype(float).tolist()
    )

global_median = float(
    calibration["gas_cost_eth"].median()
)

global_p90 = float(
    calibration["gas_cost_eth"].quantile(0.90)
)

if global_median <= 0:
    raise ValueError(
        "Global calibration median must be positive."
    )

global_scale = global_p90 / global_median

predicted_costs = []

for _, row in test.iterrows():

    paymaster = row["paymaster"]

    history = cost_history.get(paymaster, [])

    if len(history) >= MIN_HISTORY:

        recent_history = history[-ROLLING_WINDOW:]

        prediction = float(
            np.quantile(recent_history, 0.90)
        )

        method = "rolling_p90"

    elif len(history) > 0:

        prediction = (
            float(np.median(history)) * global_scale
        )

        method = "scaled_historical_median"

    else:

        prediction = global_p90

        method = "global_calibration_p90"

    prediction = max(prediction, 0.0)

    predicted_costs.append({
        "userop_hash": row["userop_hash"],
        "predicted_cost_eth": prediction,
        "prediction_method": method,
        "history_size": len(history)
    })

    # Update the estimator after making the prediction.
    # This represents newly observed on-chain cost data.
    cost_history.setdefault(paymaster, []).append(
        float(row["gas_cost_eth"])
    )

prediction_df = pd.DataFrame(predicted_costs)

test = test.merge(
    prediction_df,
    on="userop_hash",
    how="left",
    validate="one_to_one"
)

if test["predicted_cost_eth"].isnull().any():
    raise ValueError(
        "Some operations do not have cost predictions."
    )

print("Rolling P90 predictions generated successfully.")


# ============================================================
# 5. CALCULATE BASE PAYMASTER BUDGETS
# ============================================================

print("\nCalculating simulated Paymaster budgets...")

base_budgets = {}

paymasters = sorted(
    test["paymaster"].unique()
)

for paymaster in paymasters:

    calibration_group = calibration[
        calibration["paymaster"] == paymaster
    ]

    if len(calibration_group) == 0:

        calibration_spending = 0.0

    else:

        calibration_spending = float(
            calibration_group["gas_cost_eth"].sum()
        )

    base_budgets[paymaster] = calibration_spending

print(
    f"Paymasters in test set: {len(paymasters)}"
)

print(
    "Paymasters without calibration history:",
    sum(
        base_budgets[p] == 0
        for p in paymasters
    )
)


# ============================================================
# 6. FAIRNESS INDEX
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
# 7. DYNAMIC ALLOCATION FUNCTION
# ============================================================

def run_dynamic_allocation(strategy_name, budget_multiplier):

    print(
        f"Running {strategy_name}, "
        f"budget multiplier {budget_multiplier}..."
    )

    # Each strategy starts with its own independent state.
    remaining_budgets = {}

    for paymaster in paymasters:

        remaining_budgets[paymaster] = (
            base_budgets[paymaster] * budget_multiplier
        )

    # Accepted sponsorship history during simulation.
    user_counts = {}

    user_spending = {}

    actual_spending = {
        paymaster: 0.0
        for paymaster in paymasters
    }

    predicted_spending = {
        paymaster: 0.0
        for paymaster in paymasters
    }

    allocated_records = []

    # --------------------------------------------------------
    # Process blocks chronologically.
    # Operations from different blocks are never reordered.
    # --------------------------------------------------------

    for block_number, block_group in test.groupby(
        "block_number",
        sort=True
    ):

        # Each Paymaster has an independent budget.
        for paymaster, group in block_group.groupby(
            "paymaster",
            sort=False
        ):

            # ------------------------------------------------
            # Calculate priorities using history available
            # before processing this Paymaster's block.
            # ------------------------------------------------

            scored_rows = []

            for _, row in group.iterrows():

                sender = row["sender"]

                state_key = (paymaster, sender)

                previous_count = user_counts.get(
                    state_key, 0
                )

                previous_spending = user_spending.get(
                    state_key, 0.0
                )

                # Fewer previous accepted sponsorships
                # means higher fairness priority.
                fairness_score = (
                    1.0 / (1.0 + previous_count)
                )

                # Lower previous accepted spending
                # means higher spending priority.
                typical_cost = max(
                    global_median, 1e-18
                )

                spending_score = (
                    1.0 / (
                        1.0
                        + previous_spending / typical_cost
                    )
                )

                scored_rows.append({

                    "row": row,

                    "fairness_score": fairness_score,

                    "spending_score": spending_score

                })

            # ------------------------------------------------
            # Cost score within the current block.
            # Lower predicted cost receives higher priority.
            # ------------------------------------------------

            predicted_values = [
                item["row"]["predicted_cost_eth"]
                for item in scored_rows
            ]

            cost_ranks = pd.Series(
                predicted_values
            ).rank(
                method="average",
                pct=True
            ).tolist()

            for index, item in enumerate(scored_rows):

                item["cost_score"] = (
                    1.0 - cost_ranks[index]
                )

                item["proposed_score"] = (

                    FAIRNESS_WEIGHT
                    * item["fairness_score"]

                    + SPENDING_WEIGHT
                    * item["spending_score"]

                    + COST_WEIGHT
                    * item["cost_score"]

                )

            # ------------------------------------------------
            # Determine order within this block.
            # ------------------------------------------------

            if strategy_name == "FCFS":

                scored_rows.sort(
                    key=lambda item: (
                        int(item["row"]["log_index"]),
                        str(item["row"]["transaction_hash"])
                    )
                )

            elif strategy_name == "FAIRNESS":

                scored_rows.sort(
                    key=lambda item: (
                        -item["fairness_score"],
                        int(item["row"]["log_index"]),
                        str(item["row"]["transaction_hash"])
                    )
                )

            elif strategy_name == "PROPOSED":

                scored_rows.sort(
                    key=lambda item: (
                        -item["proposed_score"],
                        int(item["row"]["log_index"]),
                        str(item["row"]["transaction_hash"])
                    )
                )

            else:

                raise ValueError(
                    f"Unknown strategy: {strategy_name}"
                )

            # ------------------------------------------------
            # Process operations in the selected order.
            # ------------------------------------------------

            for item in scored_rows:

                row = item["row"]

                sender = row["sender"]

                state_key = (paymaster, sender)

                predicted_cost = float(
                    row["predicted_cost_eth"]
                )

                actual_cost = float(
                    row["gas_cost_eth"]
                )

                remaining = remaining_budgets[paymaster]

                accepted = (
                    predicted_cost <= remaining
                )

                if accepted:

                    # Reserve predicted gas cost.
                    remaining_budgets[paymaster] -= (
                        predicted_cost
                    )

                    predicted_spending[paymaster] += (
                        predicted_cost
                    )

                    # Record actual spending for evaluation.
                    actual_spending[paymaster] += (
                        actual_cost
                    )

                    # Update dynamic user history.
                    user_counts[state_key] = (
                        user_counts.get(state_key, 0) + 1
                    )

                    user_spending[state_key] = (
                        user_spending.get(state_key, 0.0)
                        + actual_cost
                    )

                allocated_records.append({

                    "strategy": strategy_name,

                    "budget_multiplier": budget_multiplier,

                    "userop_hash": row["userop_hash"],

                    "transaction_hash": row["transaction_hash"],

                    "log_index": int(row["log_index"]),

                    "block_number": int(block_number),

                    "paymaster": paymaster,

                    "sender": sender,

                    "predicted_cost_eth": predicted_cost,

                    "actual_cost_eth": actual_cost,

                    "accepted": accepted,

                    "fairness_score": item["fairness_score"],

                    "spending_score": item["spending_score"],

                    "cost_score": item["cost_score"],

                    "proposed_score": item["proposed_score"],

                    "remaining_budget_eth": (
                        remaining_budgets[paymaster]
                    )

                })

    return (
        pd.DataFrame(allocated_records),
        actual_spending,
        predicted_spending,
        remaining_budgets
    )


# ============================================================
# 8. RUN ALL EXPERIMENTS
# ============================================================

print("\nStarting dynamic allocation experiments...")

all_results = []

summary_rows = []

all_test_users = sorted(
    test["sender"].unique()
)

total_test_operations = len(test)

total_test_users = len(all_test_users)

for budget_multiplier in BUDGET_MULTIPLIERS:

    total_budget = sum(
        base_budgets[p] * budget_multiplier
        for p in paymasters
    )

    for strategy in STRATEGIES:

        result, actual_spending, predicted_spending, remaining_budgets = (
            run_dynamic_allocation(
                strategy,
                budget_multiplier
            )
        )

        all_results.append(result)

        accepted = result[
            result["accepted"]
        ]

        sponsored_operations = len(accepted)

        unique_users = accepted[
            "sender"
        ].nunique()

        operation_coverage = (
            sponsored_operations / total_test_operations * 100
            if total_test_operations > 0 else 0
        )

        user_coverage = (
            unique_users / total_test_users * 100
            if total_test_users > 0 else 0
        )

        actual_total = sum(
            actual_spending.values()
        )

        predicted_total = sum(
            predicted_spending.values()
        )

        actual_utilization = (
            actual_total / total_budget * 100
            if total_budget > 0 else 0
        )

        predicted_utilization = (
            predicted_total / total_budget * 100
            if total_budget > 0 else 0
        )

        # Actual budget excess is evaluated per Paymaster.
        budget_excess = 0.0

        exceeding_paymasters = 0

        for paymaster in paymasters:

            budget = (
                base_budgets[paymaster]
                * budget_multiplier
            )

            excess = max(
                0.0,
                actual_spending[paymaster] - budget
            )

            budget_excess += excess

            if excess > 0:
                exceeding_paymasters += 1

        # Include users receiving zero sponsorships.
        user_counts_series = (
            accepted.groupby("sender")["userop_hash"]
            .count()
        )

        user_counts_series = user_counts_series.reindex(
            all_test_users,
            fill_value=0
        )

        fairness_index = calculate_jain_index(
            user_counts_series.values
        )

        summary_rows.append({

            "strategy": strategy,

            "budget_multiplier": budget_multiplier,

            "test_operations": total_test_operations,

            "sponsored_operations": sponsored_operations,

            "operation_coverage_percentage": operation_coverage,

            "total_test_users": total_test_users,

            "unique_sponsored_users": unique_users,

            "user_coverage_percentage": user_coverage,

            "actual_spending_eth": actual_total,

            "predicted_reserved_spending_eth": predicted_total,

            "total_simulated_budget_eth": total_budget,

            "actual_budget_utilization_percentage": (
                actual_utilization
            ),

            "predicted_budget_utilization_percentage": (
                predicted_utilization
            ),

            "actual_budget_excess_eth": budget_excess,

            "paymasters_exceeding_budget": (
                exceeding_paymasters
            ),

            "jain_fairness_index": fairness_index

        })


# ============================================================
# 9. COMBINE RESULTS
# ============================================================

print("\nCombining experiment results...")

final_results = pd.concat(
    all_results,
    ignore_index=True
)

summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# 10. SAVE OUTPUT FILES
# ============================================================

print("Saving experiment results...")

RESULT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

final_results.to_csv(
    RESULT_FILE,
    index=False
)

SUMMARY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n==============================================")

print("DYNAMIC BUDGET-AWARE ALLOCATION RESULTS")

print("==============================================")

print(
    summary.to_string(index=False)
)

print("\nPrediction method counts:")

print(
    test["prediction_method"].value_counts()
)

print("\nSaved files:")

print(RESULT_FILE)

print(SUMMARY_FILE)

print("\nDynamic allocation experiments completed successfully.")

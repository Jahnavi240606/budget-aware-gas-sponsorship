
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FILE = Path("data/processed/processed_userops.csv")

RESULT_FILE = Path(
    "data/processed/batch_dynamic_allocation_results.csv"
)

SUMMARY_FILE = Path(
    "data/processed/batch_dynamic_allocation_summary.csv"
)

CALIBRATION_FRACTION = 0.20

BATCH_SIZE = 100

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

# Normalize identifiers and addresses.
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

# Convert numeric columns.
for column in [
    "block_number",
    "log_index",
    "gas_cost_eth"
]:
    df[column] = pd.to_numeric(
        df[column],
        errors="raise"
    )

# Validate missing values.
if df[required_columns].isnull().any().any():
    raise ValueError(
        "Missing values found in required columns."
    )

# Validate duplicate operations.
if df["userop_hash"].duplicated().any():
    raise ValueError(
        "Duplicate UserOperation hashes found."
    )

# Validate costs.
if (df["gas_cost_eth"] < 0).any():
    raise ValueError(
        "Negative gas costs found."
    )

# Chronological ordering.
df = df.sort_values(
    [
        "block_number",
        "transaction_hash",
        "log_index"
    ]
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

print(f"Calibration operations: {len(calibration)}")
print(f"Test operations: {len(test)}")


# ============================================================
# 4. GENERATE ROLLING P90 COST PREDICTIONS
# ============================================================

print("\nGenerating rolling P90 cost predictions...")

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

prediction_records = []

for _, row in test.iterrows():

    paymaster = row["paymaster"]

    history = cost_history.get(paymaster, [])

    # Estimate using only previously observed costs.
    if len(history) >= MIN_HISTORY:

        recent_history = history[-ROLLING_WINDOW:]

        predicted_cost = float(
            np.quantile(recent_history, 0.90)
        )

        method = "rolling_p90_50"

    elif len(history) > 0:

        predicted_cost = (
            float(np.median(history)) * global_scale
        )

        method = "scaled_historical_median"

    else:

        predicted_cost = global_p90

        method = "global_calibration_p90"

    prediction_records.append({

        "userop_hash": row["userop_hash"],

        "predicted_cost_eth": max(
            predicted_cost, 0.0
        ),

        "prediction_method": method

    })

    # Add the current actual cost only after prediction.
    cost_history.setdefault(
        paymaster, []
    ).append(
        float(row["gas_cost_eth"])
    )

prediction_df = pd.DataFrame(
    prediction_records
)

test = test.merge(
    prediction_df,
    on="userop_hash",
    how="left",
    validate="one_to_one"
)

if test["predicted_cost_eth"].isnull().any():
    raise ValueError(
        "Some operations have no cost predictions."
    )

print("Cost predictions generated successfully.")


# ============================================================
# 5. CALCULATE SIMULATED PAYMASTER BUDGETS
# ============================================================

print("\nCalculating simulated Paymaster budgets...")

paymasters = sorted(
    test["paymaster"].unique()
)

base_budgets = {}

for paymaster in paymasters:

    historical_data = calibration[
        calibration["paymaster"] == paymaster
    ]

    if historical_data.empty:

        base_budgets[paymaster] = 0.0

    else:

        base_budgets[paymaster] = float(
            historical_data["gas_cost_eth"].sum()
        )

print(f"Test Paymasters: {len(paymasters)}")

print(
    "Paymasters without calibration history:",
    sum(
        base_budgets[p] == 0
        for p in paymasters
    )
)


# ============================================================
# 6. INITIAL USER HISTORY
# ============================================================

print("\nPreparing calibration-period user history...")

initial_user_counts = {}

initial_user_spending = {}

for _, row in calibration.iterrows():

    key = (
        row["paymaster"],
        row["sender"]
    )

    initial_user_counts[key] = (
        initial_user_counts.get(key, 0) + 1
    )

    initial_user_spending[key] = (
        initial_user_spending.get(key, 0.0)
        + float(row["gas_cost_eth"])
    )


# ============================================================
# 7. JAIN'S FAIRNESS INDEX
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
# 8. DYNAMIC CANDIDATE SCORING
# ============================================================

def calculate_candidate_scores(
    candidates,
    user_counts,
    user_spending,
    paymaster
):

    candidate_costs = [
        float(row["predicted_cost_eth"])
        for row in candidates
    ]

    min_cost = min(candidate_costs)
    max_cost = max(candidate_costs)

    typical_cost = max(
        global_median,
        1e-18
    )

    scored = []

    for row in candidates:

        sender = row["sender"]

        key = (
            paymaster,
            sender
        )

        previous_count = user_counts.get(
            key, 0
        )

        previous_spending = user_spending.get(
            key, 0.0
        )

        # Fewer previous sponsorships means
        # higher fairness priority.
        fairness_score = (
            1.0 / (1.0 + previous_count)
        )

        # Lower previous spending means
        # higher spending priority.
        spending_score = (
            1.0 / (
                1.0 + previous_spending / typical_cost
            )
        )

        # Lower predicted cost means
        # higher cost-efficiency priority.
        if max_cost > min_cost:

            cost_score = (
                max_cost
                - float(row["predicted_cost_eth"])
            ) / (max_cost - min_cost)

        else:

            cost_score = 0.5

        proposed_score = (

            FAIRNESS_WEIGHT * fairness_score

            + SPENDING_WEIGHT * spending_score

            + COST_WEIGHT * cost_score

        )

        scored.append({

            "row": row,

            "fairness_score": fairness_score,

            "spending_score": spending_score,

            "cost_score": cost_score,

            "proposed_score": proposed_score

        })

    return scored


# ============================================================
# 9. SELECT NEXT CANDIDATE
# ============================================================

def select_candidate(
    candidates,
    strategy,
    user_counts,
    user_spending,
    paymaster
):

    scored = calculate_candidate_scores(
        candidates,
        user_counts,
        user_spending,
        paymaster
    )

    def chronological_key(item):

        row = item["row"]

        return (
            int(row["block_number"]),
            str(row["transaction_hash"]),
            int(row["log_index"]),
            str(row["userop_hash"])
        )

    if strategy == "FCFS":

        selected = min(
            scored,
            key=chronological_key
        )

    elif strategy == "FAIRNESS":

        selected = min(
            scored,
            key=lambda item: (
                -item["fairness_score"],
                *chronological_key(item)
            )
        )

    elif strategy == "PROPOSED":

        selected = min(
            scored,
            key=lambda item: (
                -item["proposed_score"],
                *chronological_key(item)
            )
        )

    else:

        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    return selected


# ============================================================
# 10. RUN ONE BATCH-BASED ALLOCATION EXPERIMENT
# ============================================================

def run_experiment(strategy, budget_multiplier):

    print(
        f"Running {strategy}, "
        f"budget multiplier {budget_multiplier}..."
    )

    # Every experiment starts with independent budgets.
    remaining_budgets = {
        paymaster: (
            base_budgets[paymaster] * budget_multiplier
        )
        for paymaster in paymasters
    }

    # Every strategy starts with the same calibration history.
    user_counts = initial_user_counts.copy()

    user_spending = initial_user_spending.copy()

    actual_spending = {
        paymaster: 0.0
        for paymaster in paymasters
    }

    predicted_spending = {
        paymaster: 0.0
        for paymaster in paymasters
    }

    records = []

    # Create chronological batches.
    experiment_data = test.copy()

    experiment_data["batch_id"] = (
        np.arange(len(experiment_data)) // BATCH_SIZE
    )

    for batch_id, batch in experiment_data.groupby(
        "batch_id",
        sort=True
    ):

        # Process each Paymaster separately.
        for paymaster, paymaster_batch in batch.groupby(
            "paymaster",
            sort=False
        ):

            # IMPORTANT FIX:
            # Convert Pandas Series to ordinary Python dictionaries.
            # This avoids ambiguous Series comparisons during removal.
            candidates = paymaster_batch.to_dict(
                orient="records"
            )

            decision_order = 0

            while candidates:

                selected = select_candidate(
                    candidates,
                    strategy,
                    user_counts,
                    user_spending,
                    paymaster
                )

                row = selected["row"]

                # Remove selected operation using unique hash.
                selected_hash = row["userop_hash"]

                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate["userop_hash"] != selected_hash
                ]

                predicted_cost = float(
                    row["predicted_cost_eth"]
                )

                actual_cost = float(
                    row["gas_cost_eth"]
                )

                remaining = remaining_budgets[paymaster]

                # Accept only if predicted cost fits.
                accepted = (
                    predicted_cost <= remaining
                )

                if accepted:

                    # Reserve predicted cost.
                    remaining_budgets[paymaster] -= (
                        predicted_cost
                    )

                    predicted_spending[paymaster] += (
                        predicted_cost
                    )

                    # Record actual cost for evaluation.
                    actual_spending[paymaster] += (
                        actual_cost
                    )

                    # Update user history after acceptance.
                    key = (
                        paymaster,
                        row["sender"]
                    )

                    user_counts[key] = (
                        user_counts.get(key, 0) + 1
                    )

                    user_spending[key] = (
                        user_spending.get(key, 0.0)
                        + actual_cost
                    )

                decision_order += 1

                records.append({

                    "strategy": strategy,

                    "budget_multiplier": budget_multiplier,

                    "batch_id": int(batch_id),

                    "decision_order": decision_order,

                    "userop_hash": row["userop_hash"],

                    "transaction_hash": row["transaction_hash"],

                    "log_index": int(row["log_index"]),

                    "block_number": int(row["block_number"]),

                    "paymaster": paymaster,

                    "sender": row["sender"],

                    "predicted_cost_eth": predicted_cost,

                    "actual_cost_eth": actual_cost,

                    "prediction_method": row["prediction_method"],

                    "accepted": accepted,

                    "fairness_score": selected["fairness_score"],

                    "spending_score": selected["spending_score"],

                    "cost_score": selected["cost_score"],

                    "proposed_score": selected["proposed_score"],

                    "remaining_budget_eth": (
                        remaining_budgets[paymaster]
                    )

                })

    return (
        pd.DataFrame(records),
        actual_spending,
        predicted_spending
    )


# ============================================================
# 11. EVALUATE ONE EXPERIMENT
# ============================================================

def evaluate_experiment(
    strategy,
    budget_multiplier,
    result,
    actual_spending,
    predicted_spending
):

    accepted = result[
        result["accepted"]
    ].copy()

    total_operations = len(test)

    all_test_users = sorted(
        test["sender"].unique()
    )

    total_users = len(all_test_users)

    sponsored_operations = len(accepted)

    unique_users = accepted[
        "sender"
    ].nunique()

    operation_coverage = (
        sponsored_operations / total_operations * 100
        if total_operations > 0 else 0
    )

    user_coverage = (
        unique_users / total_users * 100
        if total_users > 0 else 0
    )

    actual_total = sum(
        actual_spending.values()
    )

    predicted_total = sum(
        predicted_spending.values()
    )

    total_budget = sum(
        base_budgets[p] * budget_multiplier
        for p in paymasters
    )

    actual_utilization = (
        actual_total / total_budget * 100
        if total_budget > 0 else 0
    )

    predicted_utilization = (
        predicted_total / total_budget * 100
        if total_budget > 0 else 0
    )

    # Calculate actual budget excess per Paymaster.
    budget_excess = 0.0

    exceeding_paymasters = 0

    for paymaster in paymasters:

        paymaster_budget = (
            base_budgets[paymaster] * budget_multiplier
        )

        excess = max(
            0.0,
            actual_spending[paymaster] - paymaster_budget
        )

        budget_excess += excess

        if excess > 0:
            exceeding_paymasters += 1

    # Include users with zero accepted sponsorships.
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

    return {

        "strategy": strategy,

        "budget_multiplier": budget_multiplier,

        "batch_size": BATCH_SIZE,

        "test_operations": total_operations,

        "sponsored_operations": sponsored_operations,

        "operation_coverage_percentage": operation_coverage,

        "total_test_users": total_users,

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

    }


# ============================================================
# 12. RUN ALL STRATEGIES AND BUDGET LEVELS
# ============================================================

print("\nStarting batch-dynamic allocation experiments...")

all_results = []

summary_records = []

for multiplier in BUDGET_MULTIPLIERS:

    for strategy in STRATEGIES:

        result, actual_spending, predicted_spending = (
            run_experiment(
                strategy,
                multiplier
            )
        )

        all_results.append(result)

        summary_records.append(
            evaluate_experiment(
                strategy,
                multiplier,
                result,
                actual_spending,
                predicted_spending
            )
        )


# ============================================================
# 13. COMBINE RESULTS
# ============================================================

print("\nCombining experiment results...")

final_results = pd.concat(
    all_results,
    ignore_index=True
)

summary = pd.DataFrame(
    summary_records
)


# ============================================================
# 14. SAVE OUTPUTS
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

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# 15. DISPLAY RESULTS
# ============================================================

print("\n==============================================")

print("BATCH-DYNAMIC BUDGET ALLOCATION RESULTS")

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

print("\nBatch-dynamic allocation experiments completed successfully.")

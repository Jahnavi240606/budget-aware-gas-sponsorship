
"""
WEIGHT-TUNING EXPERIMENT:
BUDGET-AWARE GAS SPONSORSHIP ALLOCATION

Compares:
1. FCFS baseline
2. FAIRNESS baseline
3. PROPOSED algorithm with five weight configurations

Uses:
- Historical Optimism UserOperation data
- Chronological calibration/test split
- Rolling P90 gas-cost prediction
- Batch size of 100
- Three simulated budget multipliers
- Five safety margins
- Per-Paymaster budget enforcement
- Jain fairness and coverage metrics

IMPORTANT:
This is a historical simulation.
It does not execute transactions or modify Paymaster balances.
Budgets are simulated from calibration-period spending.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    ROOT / "data" / "processed" / "processed_userops.csv"
)

RESULT_FILE = (
    ROOT / "data" / "processed"
    / "weight_tuning_allocation_results.csv"
)

SUMMARY_FILE = (
    ROOT / "data" / "processed"
    / "weight_tuning_allocation_summary.csv"
)

PAYMASTER_FILE = (
    ROOT / "data" / "processed"
    / "weight_tuning_allocation_by_paymaster.csv"
)

CALIBRATION_FRACTION = 0.20
BATCH_SIZE = 100
ROLLING_WINDOW = 50
MIN_HISTORY = 20

BUDGET_MULTIPLIERS = [1.5, 2.5, 5.0]

SAFETY_MULTIPLIERS = [
    1.00, 1.10, 1.25, 1.50, 2.00
]

WEIGHT_CONFIGS = {
    "current_50_30_20": {
        "fairness": 0.50,
        "spending": 0.30,
        "cost": 0.20
    },

    "fairness_heavy_70_20_10": {
        "fairness": 0.70,
        "spending": 0.20,
        "cost": 0.10
    },

    "balanced_40_30_30": {
        "fairness": 0.40,
        "spending": 0.30,
        "cost": 0.30
    },

    "cost_heavy_30_20_50": {
        "fairness": 0.30,
        "spending": 0.20,
        "cost": 0.50
    },

    "spending_heavy_30_50_20": {
        "fairness": 0.30,
        "spending": 0.50,
        "cost": 0.20
    }
}

BASELINE_WEIGHTS = WEIGHT_CONFIGS["current_50_30_20"]


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

def load_dataset():

    print("Loading processed blockchain dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required = [
        "userop_hash",
        "transaction_hash",
        "log_index",
        "paymaster",
        "sender",
        "block_number",
        "gas_cost_eth"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    for col in [
        "userop_hash",
        "transaction_hash",
        "paymaster",
        "sender"
    ]:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    for col in [
        "block_number",
        "log_index",
        "gas_cost_eth"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise"
        )

    if df[required].isnull().any().any():
        raise ValueError(
            "Missing values found in required columns."
        )

    if df["userop_hash"].duplicated().any():
        raise ValueError(
            "Duplicate UserOperation hashes found."
        )

    if (df["gas_cost_eth"] < 0).any():
        raise ValueError(
            "Negative gas costs found."
        )

    df = df.sort_values(
        [
            "block_number",
            "transaction_hash",
            "log_index"
        ]
    ).reset_index(drop=True)

    print("Total operations:", len(df))
    print("Unique users:", df["sender"].nunique())
    print("Unique Paymasters:", df["paymaster"].nunique())

    return df


# ============================================================
# 3. CHRONOLOGICAL CALIBRATION / TEST SPLIT
# ============================================================

def create_split(df):

    print("\nCreating chronological calibration/test split...")

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

        # A Paymaster with one operation has no calibration
        # observations and one test observation.

        calibration_indices.extend(indices[:split])
        test_indices.extend(indices[split:])

    calibration = df.loc[
        calibration_indices
    ].copy()

    test = df.loc[
        test_indices
    ].copy()

    sort_columns = [
        "block_number",
        "transaction_hash",
        "log_index"
    ]

    calibration = calibration.sort_values(
        sort_columns
    ).reset_index(drop=True)

    test = test.sort_values(
        sort_columns
    ).reset_index(drop=True)

    if calibration.empty or test.empty:
        raise ValueError(
            "Calibration or test dataset is empty."
        )

    print("Calibration operations:", len(calibration))
    print("Test operations:", len(test))

    return calibration, test


# ============================================================
# 4. CHRONOLOGICAL ROLLING P90 COST PREDICTIONS
# ============================================================

def generate_predictions(calibration, test):

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

    prediction_records = []

    # Predict first; append the current actual cost afterward.

    for _, row in test.iterrows():

        paymaster = row["paymaster"]

        history = cost_history.get(paymaster, [])

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
            "predicted_cost_eth": max(predicted_cost, 0.0),
            "prediction_method": method,
            "prediction_history_size": len(history)
        })

        # This assumes historical on-chain costs are observable
        # independently of the simulated acceptance decision.

        cost_history.setdefault(paymaster, []).append(
            float(row["gas_cost_eth"])
        )

    predictions = pd.DataFrame(prediction_records)

    test = test.merge(
        predictions,
        on="userop_hash",
        how="left",
        validate="one_to_one"
    )

    if test["predicted_cost_eth"].isna().any():
        raise ValueError(
            "Some operations do not have cost predictions."
        )

    return test


# ============================================================
# 5. INITIALIZE BUDGETS AND USER HISTORY
# ============================================================

def initialize_state(calibration, paymasters):

    # Simulated base budget is calibration-period spending.

    base_budgets = (
        calibration.groupby("paymaster")["gas_cost_eth"]
        .sum()
        .to_dict()
    )

    base_budgets = {
        pm: float(base_budgets.get(pm, 0.0))
        for pm in paymasters
    }

    initial_user_counts = {}
    initial_user_spending = {}

    for (paymaster, sender), group in calibration.groupby(
        ["paymaster", "sender"]
    ):

        key = (paymaster, sender)

        initial_user_counts[key] = len(group)

        initial_user_spending[key] = float(
            group["gas_cost_eth"].sum()
        )

    return (
        base_budgets,
        initial_user_counts,
        initial_user_spending
    )


# ============================================================
# 6. JAIN FAIRNESS INDEX
# ============================================================

def calculate_jain_index(values):

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return 0.0

    denominator = len(values) * np.sum(values ** 2)

    if denominator == 0:
        return 0.0

    return float(
        np.sum(values) ** 2 / denominator
    )


# ============================================================
# 7. DYNAMIC CANDIDATE SCORING
# ============================================================

def calculate_candidate_scores(
    candidates,
    user_counts,
    user_spending,
    paymaster,
    typical_cost,
    weights
):

    candidate_costs = [
        float(row["predicted_cost_eth"])
        for row in candidates
    ]

    min_cost = min(candidate_costs)
    max_cost = max(candidate_costs)

    typical_cost = max(typical_cost, 1e-18)

    scored = []

    for row in candidates:

        sender = row["sender"]
        key = (paymaster, sender)

        previous_count = user_counts.get(key, 0)
        previous_spending = user_spending.get(key, 0.0)

        # Higher score for users with fewer previous sponsorships.
        fairness_score = 1.0 / (1.0 + previous_count)

        # Higher score for users with lower previous spending.
        spending_score = 1.0 / (
            1.0 + previous_spending / typical_cost
        )

        # Higher score for lower predicted gas cost.
        if max_cost > min_cost:

            cost_score = (
                max_cost - float(row["predicted_cost_eth"])
            ) / (max_cost - min_cost)

        else:
            cost_score = 0.5

        proposed_score = (
            weights["fairness"] * fairness_score
            + weights["spending"] * spending_score
            + weights["cost"] * cost_score
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
# 8. SELECT NEXT CANDIDATE
# ============================================================

def chronological_key(item):

    row = item["row"]

    return (
        int(row["block_number"]),
        str(row["transaction_hash"]),
        int(row["log_index"]),
        str(row["userop_hash"])
    )


def select_candidate(
    candidates,
    strategy,
    user_counts,
    user_spending,
    paymaster,
    typical_cost,
    weights
):

    scored = calculate_candidate_scores(
        candidates,
        user_counts,
        user_spending,
        paymaster,
        typical_cost,
        weights
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
# 9. RUN ONE ALLOCATION EXPERIMENT
# ============================================================

def run_experiment(
    test,
    paymasters,
    base_budgets,
    initial_user_counts,
    initial_user_spending,
    strategy,
    budget_multiplier,
    safety_multiplier,
    typical_cost,
    weight_config,
    weights
):

    remaining_budgets = {
        pm: base_budgets[pm] * budget_multiplier
        for pm in paymasters
    }

    user_counts = initial_user_counts.copy()
    user_spending = initial_user_spending.copy()

    actual_spending = {
        pm: 0.0 for pm in paymasters
    }

    reserved_spending = {
        pm: 0.0 for pm in paymasters
    }

    records = []

    experiment_data = test.copy()

    experiment_data["batch_id"] = (
        np.arange(len(experiment_data)) // BATCH_SIZE
    )

    for batch_id, batch in experiment_data.groupby(
        "batch_id",
        sort=True
    ):

        for paymaster, paymaster_batch in batch.groupby(
            "paymaster",
            sort=False
        ):

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
                    paymaster,
                    typical_cost,
                    weights
                )

                row = selected["row"]
                selected_hash = row["userop_hash"]

                # Remove the selected operation from candidates.
                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate["userop_hash"] != selected_hash
                ]

                predicted_cost = float(
                    row["predicted_cost_eth"]
                )

                # Apply safety margin before checking budget.
                adjusted_reserve = (
                    predicted_cost * safety_multiplier
                )

                actual_cost = float(row["gas_cost_eth"])

                remaining_before = remaining_budgets[paymaster]

                accepted = (
                    adjusted_reserve <= remaining_before
                )

                if accepted:

                    remaining_budgets[paymaster] -= adjusted_reserve

                    reserved_spending[paymaster] += adjusted_reserve

                    actual_spending[paymaster] += actual_cost

                    # Update simulated user history only after acceptance.
                    key = (paymaster, row["sender"])

                    user_counts[key] = (
                        user_counts.get(key, 0) + 1
                    )

                    user_spending[key] = (
                        user_spending.get(key, 0.0) + actual_cost
                    )

                decision_order += 1

                records.append({

                    "strategy": strategy,
                    "weight_config": weight_config,

                    "fairness_weight": weights["fairness"],
                    "spending_weight": weights["spending"],
                    "cost_weight": weights["cost"],

                    "budget_multiplier": budget_multiplier,
                    "safety_multiplier": safety_multiplier,

                    "batch_id": int(batch_id),
                    "decision_order": decision_order,

                    "userop_hash": row["userop_hash"],
                    "transaction_hash": row["transaction_hash"],
                    "log_index": int(row["log_index"]),
                    "block_number": int(row["block_number"]),

                    "paymaster": paymaster,
                    "sender": row["sender"],

                    "predicted_cost_eth": predicted_cost,
                    "safety_adjusted_reserve_eth": adjusted_reserve,
                    "actual_cost_eth": actual_cost,

                    "prediction_method": row["prediction_method"],
                    "prediction_history_size": int(
                        row["prediction_history_size"]
                    ),

                    "accepted": accepted,

                    "actual_exceeded_adjusted_prediction": (
                        actual_cost > adjusted_reserve
                    ),

                    "fairness_score": selected["fairness_score"],
                    "spending_score": selected["spending_score"],
                    "cost_score": selected["cost_score"],
                    "proposed_score": selected["proposed_score"],

                    "remaining_budget_before_eth": remaining_before,
                    "remaining_budget_after_eth": (
                        remaining_budgets[paymaster]
                    )
                })

    return (
        pd.DataFrame(records),
        actual_spending,
        reserved_spending
    )


# ============================================================
# 10. EVALUATE ONE EXPERIMENT
# ============================================================

def evaluate_experiment(
    test,
    paymasters,
    base_budgets,
    strategy,
    budget_multiplier,
    safety_multiplier,
    weight_config,
    weights,
    result,
    actual_spending,
    reserved_spending
):

    accepted = result[result["accepted"]].copy()

    total_operations = len(test)
    total_users = test["sender"].nunique()

    sponsored_operations = len(accepted)
    unique_users = accepted["sender"].nunique()

    operation_coverage = (
        sponsored_operations / total_operations * 100
        if total_operations > 0 else 0.0
    )

    user_coverage = (
        unique_users / total_users * 100
        if total_users > 0 else 0.0
    )

    actual_total = sum(actual_spending.values())
    reserved_total = sum(reserved_spending.values())

    total_budget = sum(
        base_budgets[pm] * budget_multiplier
        for pm in paymasters
    )

    actual_utilization = (
        actual_total / total_budget * 100
        if total_budget > 0 else 0.0
    )

    reserved_utilization = (
        reserved_total / total_budget * 100
        if total_budget > 0 else 0.0
    )

    budget_excess = 0.0
    exceeding_paymasters = 0

    for pm in paymasters:

        pm_budget = base_budgets[pm] * budget_multiplier

        excess = max(
            0.0,
            actual_spending[pm] - pm_budget
        )

        budget_excess += excess

        if excess > 0:
            exceeding_paymasters += 1

    accepted_counts = (
        accepted.groupby("sender")["userop_hash"].count()
    )

    accepted_counts = accepted_counts.reindex(
        test["sender"].unique(),
        fill_value=0
    )

    fairness_index = calculate_jain_index(
        accepted_counts.values
    )

    underestimated_count = (
        int(
            accepted["actual_exceeded_adjusted_prediction"].sum()
        )
        if not accepted.empty else 0
    )

    accepted_underestimation_rate = (
        underestimated_count / sponsored_operations * 100
        if sponsored_operations > 0 else 0.0
    )

    return {

        "strategy": strategy,
        "weight_config": weight_config,

        "fairness_weight": weights["fairness"],
        "spending_weight": weights["spending"],
        "cost_weight": weights["cost"],

        "budget_multiplier": budget_multiplier,
        "safety_multiplier": safety_multiplier,

        "batch_size": BATCH_SIZE,

        "test_operations": total_operations,
        "sponsored_operations": sponsored_operations,
        "rejected_operations": (
            total_operations - sponsored_operations
        ),

        "operation_coverage_percentage": operation_coverage,

        "total_test_users": total_users,
        "unique_sponsored_users": unique_users,
        "user_coverage_percentage": user_coverage,

        "actual_spending_eth": actual_total,
        "safety_adjusted_reserved_spending_eth": reserved_total,
        "total_simulated_budget_eth": total_budget,

        "actual_budget_utilization_percentage": actual_utilization,
        "reserved_budget_utilization_percentage": reserved_utilization,

        "actual_budget_excess_eth": budget_excess,
        "paymasters_exceeding_budget": exceeding_paymasters,

        "accepted_operations_exceeding_adjusted_prediction": (
            underestimated_count
        ),

        "accepted_underestimation_rate_percentage": (
            accepted_underestimation_rate
        ),

        "reserve_to_actual_spending_ratio": (
            reserved_total / actual_total
            if actual_total > 0 else np.nan
        ),

        "jain_fairness_index": fairness_index
    }


# ============================================================
# 11. PER-PAYMASTER SUMMARY
# ============================================================

def create_paymaster_summary(results):

    records = []

    grouped = results.groupby(
        [
            "strategy",
            "weight_config",
            "budget_multiplier",
            "safety_multiplier",
            "paymaster"
        ]
    )

    for keys, group in grouped:

        (
            strategy,
            weight_config,
            budget,
            safety,
            paymaster
        ) = keys

        accepted = group[group["accepted"]]

        records.append({

            "strategy": strategy,
            "weight_config": weight_config,

            "budget_multiplier": budget,
            "safety_multiplier": safety,

            "paymaster": paymaster,

            "test_operations": len(group),
            "sponsored_operations": len(accepted),

            "actual_spending_eth": (
                accepted["actual_cost_eth"].sum()
            ),

            "reserved_spending_eth": (
                accepted["safety_adjusted_reserve_eth"].sum()
            ),

            "accepted_underestimated_operations": int(
                accepted[
                    "actual_exceeded_adjusted_prediction"
                ].sum()
            )
        })

    return pd.DataFrame(records)


# ============================================================
# 12. MAIN EXPERIMENT
# ============================================================

def main():

    df = load_dataset()

    calibration, test = create_split(df)

    test = generate_predictions(calibration, test)

    paymasters = sorted(df["paymaster"].unique())

    (
        base_budgets,
        initial_user_counts,
        initial_user_spending
    ) = initialize_state(calibration, paymasters)

    typical_cost = float(
        calibration["gas_cost_eth"].median()
    )

    if typical_cost <= 0:
        typical_cost = 1e-18

    all_results = []
    summary_records = []

    # For each budget/safety pair:
    # - Run FCFS once.
    # - Run FAIRNESS once.
    # - Run PROPOSED for all five weight configurations.

    total_runs = (
        len(BUDGET_MULTIPLIERS)
        * len(SAFETY_MULTIPLIERS)
        * (2 + len(WEIGHT_CONFIGS))
    )

    run_number = 0

    print("\nStarting weight-tuning experiments...")
    print("Weight configurations:", len(WEIGHT_CONFIGS))
    print("Total experiment runs:", total_runs)

    for budget_multiplier in BUDGET_MULTIPLIERS:

        for safety_multiplier in SAFETY_MULTIPLIERS:

            runs = [
                (
                    "FCFS",
                    "baseline_reference",
                    BASELINE_WEIGHTS
                ),
                (
                    "FAIRNESS",
                    "baseline_reference",
                    BASELINE_WEIGHTS
                )
            ]

            runs.extend(
                (
                    "PROPOSED",
                    name,
                    weights
                )
                for name, weights in WEIGHT_CONFIGS.items()
            )

            for strategy, weight_config, weights in runs:

                run_number += 1

                print(
                    f"\nExperiment {run_number}/{total_runs}: "
                    f"{strategy} | {weight_config} | "
                    f"Budget={budget_multiplier}x | "
                    f"Safety={safety_multiplier}x"
                )

                (
                    result,
                    actual_spending,
                    reserved_spending
                ) = run_experiment(
                    test,
                    paymasters,
                    base_budgets,
                    initial_user_counts,
                    initial_user_spending,
                    strategy,
                    budget_multiplier,
                    safety_multiplier,
                    typical_cost,
                    weight_config,
                    weights
                )

                all_results.append(result)

                summary_records.append(
                    evaluate_experiment(
                        test,
                        paymasters,
                        base_budgets,
                        strategy,
                        budget_multiplier,
                        safety_multiplier,
                        weight_config,
                        weights,
                        result,
                        actual_spending,
                        reserved_spending
                    )
                )

    # Combine all experiment results.

    final_results = pd.concat(
        all_results,
        ignore_index=True
    )

    summary = pd.DataFrame(summary_records)

    paymaster_summary = create_paymaster_summary(
        final_results
    )

    # Save results.

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

    paymaster_summary.to_csv(
        PAYMASTER_FILE,
        index=False
    )

    # Display important results.

    print("\n" + "=" * 100)
    print("WEIGHT-TUNING ALLOCATION RESULTS")
    print("=" * 100)

    display_columns = [
        "strategy",
        "weight_config",
        "budget_multiplier",
        "safety_multiplier",
        "sponsored_operations",
        "operation_coverage_percentage",
        "unique_sponsored_users",
        "actual_budget_excess_eth",
        "paymasters_exceeding_budget",
        "jain_fairness_index"
    ]

    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.8e}"
        )
    )

    print("\nPrediction methods:")

    print(
        test["prediction_method"].value_counts()
    )

    print("\nSaved output files:")

    print(RESULT_FILE)
    print(SUMMARY_FILE)
    print(PAYMASTER_FILE)

    print("\nAll weight-tuning experiments completed successfully.")


if __name__ == "__main__":
    main()

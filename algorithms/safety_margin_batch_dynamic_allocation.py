
"""
IMPROVED SAFETY-MARGIN-AWARE BATCH-DYNAMIC GAS SPONSORSHIP ALLOCATION

Historical simulation only. No real transactions are sent and no actual
Paymaster balances are modified.

Features:
- Chronological ordering by block_number and log_index.
- Per-Paymaster chronological calibration/test split.
- Rolling P90 gas-cost prediction.
- Historical fallback predictions.
- FCFS, FAIRNESS and PROPOSED strategies.
- Five safety margins and three simulated budget levels.
- Dynamic user fairness and spending updates.
- Reservation budget validation.
- Global and per-Paymaster Jain fairness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "processed_userops.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

DETAIL_FILE = OUTPUT_DIR / "improved_safety_margin_allocation_results.csv"
SUMMARY_FILE = OUTPUT_DIR / "improved_safety_margin_allocation_summary.csv"
PAYMASTER_FILE = OUTPUT_DIR / "improved_safety_margin_allocation_by_paymaster.csv"
SPLIT_FILE = OUTPUT_DIR / "improved_safety_margin_split_summary.csv"

CALIBRATION_FRACTION = 0.20
BATCH_SIZE = 100
ROLLING_WINDOW = 50
MIN_HISTORY = 20

BUDGET_MULTIPLIERS = [1.5, 2.5, 5.0]
SAFETY_MULTIPLIERS = [1.00, 1.10, 1.25, 1.50, 2.00]

STRATEGIES = ["FCFS", "FAIRNESS", "PROPOSED"]

FAIRNESS_WEIGHT = 0.50
SPENDING_WEIGHT = 0.30
COST_WEIGHT = 0.20

BUDGET_TOLERANCE = 1e-18

CHRONOLOGY_COLUMNS = ["block_number", "log_index"]

REQUIRED_COLUMNS = [
    "userop_hash",
    "transaction_hash",
    "log_index",
    "paymaster",
    "sender",
    "block_number",
    "gas_cost_eth",
]


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

def load_dataset():

    print("Loading processed Optimism dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    for column in [
        "userop_hash",
        "transaction_hash",
        "paymaster",
        "sender",
    ]:

        if df[column].isna().any():
            raise ValueError(f"Missing values in {column}")

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    for column in [
        "block_number",
        "log_index",
        "gas_cost_eth",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        )

        if df[column].isna().any():
            raise ValueError(f"Missing numeric values in {column}")

        if not np.isfinite(df[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite values in {column}")

    if (df[["block_number", "log_index"]] < 0).any().any():
        raise ValueError("Negative block number or log index found.")

    if (df["gas_cost_eth"] < 0).any():
        raise ValueError("Negative gas cost found.")

    if df["userop_hash"].duplicated().any():
        raise ValueError("Duplicate UserOperation hashes found.")

    # Sort by actual event position, not transaction hash.
    df = df.sort_values(
        CHRONOLOGY_COLUMNS,
        kind="mergesort"
    ).reset_index(drop=True)

    if df.duplicated(CHRONOLOGY_COLUMNS).any():
        raise ValueError(
            "Duplicate block_number/log_index positions found."
        )

    print("Total operations:", len(df))
    print("Unique users:", df["sender"].nunique())
    print("Unique Paymasters:", df["paymaster"].nunique())
    print(
        "Observed gas spending:",
        f"{df['gas_cost_eth'].sum():.12f} ETH"
    )

    return df


# ============================================================
# 3. CHRONOLOGICAL CALIBRATION / TEST SPLIT
# ============================================================

def create_split(df):

    calibration_indices = []
    test_indices = []
    split_records = []

    for paymaster, group in df.groupby(
        "paymaster",
        sort=False
    ):

        indices = group.index.tolist()
        count = len(indices)

        if count <= 1:
            split_at = 0
        else:
            split_at = int(
                np.floor(count * CALIBRATION_FRACTION)
            )

            split_at = max(1, split_at)
            split_at = min(split_at, count - 1)

        calibration_indices.extend(indices[:split_at])
        test_indices.extend(indices[split_at:])

        split_records.append({
            "paymaster": paymaster,
            "total_operations": count,
            "calibration_operations": split_at,
            "test_operations": count - split_at,
            "has_calibration_history": split_at > 0,
            "zero_calibration_budget_note": (
                "No calibration history; zero simulated base budget"
                if split_at == 0
                else "Calibration-based simulated budget"
            )
        })

    calibration = df.loc[calibration_indices].copy()
    test = df.loc[test_indices].copy()

    calibration = calibration.sort_values(
        CHRONOLOGY_COLUMNS,
        kind="mergesort"
    ).reset_index(drop=True)

    test = test.sort_values(
        CHRONOLOGY_COLUMNS,
        kind="mergesort"
    ).reset_index(drop=True)

    split_summary = pd.DataFrame(split_records)

    if calibration.empty or test.empty:
        raise ValueError("Calibration or test data is empty.")

    if set(calibration["userop_hash"]) & set(test["userop_hash"]):
        raise AssertionError("Calibration/test overlap detected.")

    if len(calibration) + len(test) != len(df):
        raise AssertionError("Split does not account for all rows.")

    print("\nCHRONOLOGICAL SPLIT")
    print("Calibration operations:", len(calibration))
    print("Test operations:", len(test))

    print(
        "Paymasters without calibration history:",
        int((~split_summary["has_calibration_history"]).sum())
    )

    return calibration, test, split_summary


# ============================================================
# 4. GENERATE ROLLING P90 COST PREDICTIONS
# ============================================================

def generate_predictions(calibration, test):

    print("\nGenerating rolling P90 predictions...")

    history = {}

    for paymaster, group in calibration.groupby("paymaster"):

        history[paymaster] = (
            group["gas_cost_eth"]
            .astype(float)
            .tolist()
        )

    global_median = float(
        calibration["gas_cost_eth"].median()
    )

    global_p90 = float(
        calibration["gas_cost_eth"].quantile(0.90)
    )

    if not np.isfinite(global_median) or global_median <= 0:
        raise ValueError("Global calibration median must be positive.")

    if not np.isfinite(global_p90) or global_p90 <= 0:
        raise ValueError("Global calibration P90 must be positive.")

    global_scale = global_p90 / global_median

    records = []

    # Predict each test operation before adding its actual cost.
    for row in test.itertuples(index=False):

        paymaster = row.paymaster
        observed_history = history.get(paymaster, [])

        history_size = len(observed_history)

        if history_size >= MIN_HISTORY:

            recent = observed_history[-ROLLING_WINDOW:]

            predicted_cost = float(
                np.quantile(recent, 0.90)
            )

            method = f"rolling_p90_{ROLLING_WINDOW}"

        elif history_size > 0:

            predicted_cost = (
                float(np.median(observed_history))
                * global_scale
            )

            method = "scaled_historical_median"

        else:

            predicted_cost = global_p90
            method = "global_calibration_p90"

        if not np.isfinite(predicted_cost) or predicted_cost < 0:
            raise ValueError(
                f"Invalid prediction for {row.userop_hash}"
            )

        records.append({
            "userop_hash": row.userop_hash,
            "predicted_cost_eth": predicted_cost,
            "prediction_method": method,
            "prediction_history_size": history_size,
        })

        # Assumption: executed on-chain events are externally observable,
        # including operations rejected by the hypothetical allocation policy.
        # Current actual cost is added only after its prediction.
        history.setdefault(paymaster, []).append(
            float(row.gas_cost_eth)
        )

    predictions = pd.DataFrame(records)

    test = test.merge(
        predictions,
        on="userop_hash",
        how="left",
        validate="one_to_one"
    )

    if test["predicted_cost_eth"].isna().any():
        raise AssertionError("Missing cost predictions.")

    test = test.sort_values(
        CHRONOLOGY_COLUMNS,
        kind="mergesort"
    ).reset_index(drop=True)

    return test


# ============================================================
# 5. INITIALIZE SIMULATED BUDGETS AND USER HISTORY
# ============================================================

def initialize_state(calibration, paymasters):

    # Simulated budgets are derived from calibration-period spending.
    # These are NOT actual Paymaster wallet balances.

    base_budgets = (
        calibration.groupby("paymaster")["gas_cost_eth"]
        .sum()
        .to_dict()
    )

    base_budgets = {
        pm: float(base_budgets.get(pm, 0.0))
        for pm in paymasters
    }

    user_counts = {}
    user_spending = {}

    for (paymaster, sender), group in calibration.groupby(
        ["paymaster", "sender"]
    ):

        key = (paymaster, sender)

        user_counts[key] = len(group)

        user_spending[key] = float(
            group["gas_cost_eth"].sum()
        )

    return base_budgets, user_counts, user_spending


# ============================================================
# 6. JAIN FAIRNESS INDEX
# ============================================================

def jain_index(values):

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.nan

    denominator = len(values) * np.sum(values ** 2)

    if denominator == 0:
        return 0.0

    return float(
        np.sum(values) ** 2 / denominator
    )


def calculate_fairness_metrics(accepted, test, paymasters):

    # Global fairness across all test users.
    all_users = test["sender"].drop_duplicates().tolist()

    global_counts = (
        accepted.groupby("sender")
        .size()
        .reindex(all_users, fill_value=0)
    )

    global_fairness = jain_index(
        global_counts.to_numpy()
    )

    # Fairness within each Paymaster.
    per_paymaster_values = []

    for paymaster in paymasters:

        pm_test = test[
            test["paymaster"] == paymaster
        ]

        if pm_test.empty:
            continue

        pm_users = pm_test["sender"].drop_duplicates().tolist()

        pm_accepted = accepted[
            accepted["paymaster"] == paymaster
        ]

        counts = (
            pm_accepted.groupby("sender")
            .size()
            .reindex(pm_users, fill_value=0)
        )

        per_paymaster_values.append(
            jain_index(counts.to_numpy())
        )

    mean_pm_fairness = (
        float(np.nanmean(per_paymaster_values))
        if per_paymaster_values
        else np.nan
    )

    return global_fairness, mean_pm_fairness


# ============================================================
# 7. SCORE ALLOCATION CANDIDATES
# ============================================================

def chronological_key(row):

    return (
        int(row["block_number"]),
        int(row["log_index"]),
        str(row["userop_hash"])
    )


def score_candidates(
    candidates,
    user_counts,
    user_spending,
    paymaster,
    typical_cost
):

    costs = [
        float(row["predicted_cost_eth"])
        for row in candidates
    ]

    min_cost = min(costs)
    max_cost = max(costs)

    typical_cost = max(float(typical_cost), 1e-18)

    scored = []

    for row in candidates:

        sender = row["sender"]
        key = (paymaster, sender)

        previous_count = user_counts.get(key, 0)
        previous_spending = user_spending.get(key, 0.0)

        fairness_score = (
            1.0 / (1.0 + previous_count)
        )

        spending_score = (
            1.0 / (
                1.0 + previous_spending / typical_cost
            )
        )

        if max_cost > min_cost:

            cost_score = (
                max_cost - float(row["predicted_cost_eth"])
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
            "proposed_score": proposed_score,
        })

    return scored


# ============================================================
# 8. SELECT NEXT OPERATION
# ============================================================

def select_candidate(
    candidates,
    strategy,
    user_counts,
    user_spending,
    paymaster,
    typical_cost
):

    scored = score_candidates(
        candidates,
        user_counts,
        user_spending,
        paymaster,
        typical_cost
    )

    if strategy == "FCFS":

        return min(
            scored,
            key=lambda item: chronological_key(item["row"])
        )

    if strategy == "FAIRNESS":

        return min(
            scored,
            key=lambda item: (
                -item["fairness_score"],
                chronological_key(item["row"])
            )
        )

    if strategy == "PROPOSED":

        return min(
            scored,
            key=lambda item: (
                -item["proposed_score"],
                chronological_key(item["row"])
            )
        )

    raise ValueError(f"Unknown strategy: {strategy}")


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
    typical_cost
):

    remaining = {
        pm: base_budgets[pm] * budget_multiplier
        for pm in paymasters
    }

    initial_budget = remaining.copy()

    user_counts = initial_user_counts.copy()
    user_spending = initial_user_spending.copy()

    experiment_data = test.copy().reset_index(drop=True)

    experiment_data["batch_id"] = (
        np.arange(len(experiment_data)) // BATCH_SIZE
    )

    records = []

    for batch_id, batch in experiment_data.groupby(
        "batch_id",
        sort=True
    ):

        # Each Paymaster has an independent budget.
        for paymaster, pm_batch in batch.groupby(
            "paymaster",
            sort=True
        ):

            candidates = pm_batch.to_dict(
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
                    typical_cost
                )

                row = selected["row"]

                selected_hash = row["userop_hash"]

                candidates = [
                    item for item in candidates
                    if item["userop_hash"] != selected_hash
                ]

                predicted_cost = float(
                    row["predicted_cost_eth"]
                )

                adjusted_reserve = (
                    predicted_cost * safety_multiplier
                )

                actual_cost = float(
                    row["gas_cost_eth"]
                )

                remaining_before = remaining[paymaster]

                # Enforce the simulated reservation budget.
                accepted = (
                    adjusted_reserve
                    <= remaining_before + BUDGET_TOLERANCE
                )

                if accepted:

                    remaining[paymaster] = max(
                        0.0,
                        remaining_before - adjusted_reserve
                    )

                    # Update fairness history only after acceptance.
                    key = (paymaster, row["sender"])

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

                    "safety_multiplier": safety_multiplier,

                    "batch_id": int(batch_id),

                    "decision_order": decision_order,

                    "userop_hash": selected_hash,

                    "transaction_hash": row["transaction_hash"],

                    "block_number": int(row["block_number"]),

                    "log_index": int(row["log_index"]),

                    "paymaster": paymaster,

                    "sender": row["sender"],

                    "predicted_cost_eth": predicted_cost,

                    "safety_adjusted_reserve_eth": adjusted_reserve,

                    "actual_cost_eth": actual_cost,

                    "prediction_method": row["prediction_method"],

                    "prediction_history_size": int(
                        row["prediction_history_size"]
                    ),

                    "accepted": bool(accepted),

                    "actual_exceeded_adjusted_prediction": (
                        actual_cost > adjusted_reserve
                    ),

                    "fairness_score": selected["fairness_score"],

                    "spending_score": selected["spending_score"],

                    "cost_score": selected["cost_score"],

                    "proposed_score": selected["proposed_score"],

                    "remaining_budget_before_eth": remaining_before,

                    "remaining_budget_after_eth": remaining[paymaster],

                    "simulated_budget_eth": initial_budget[paymaster],

                })

    result = pd.DataFrame(records)

    if len(result) != len(test):
        raise AssertionError(
            "Not every test request received exactly one decision."
        )

    if result["userop_hash"].duplicated().any():
        raise AssertionError("A test operation was processed more than once.")

    # Explicitly validate reservation budget for every Paymaster.
    accepted = result[result["accepted"]]

    reserved_by_pm = (
        accepted.groupby("paymaster")[
            "safety_adjusted_reserve_eth"
        ].sum().to_dict()
    )

    for pm in paymasters:

        reserved = float(reserved_by_pm.get(pm, 0.0))
        budget = initial_budget[pm]

        if reserved > budget + BUDGET_TOLERANCE:

            raise AssertionError(
                f"Reservation budget exceeded for {pm}: "
                f"{reserved} > {budget}"
            )

    return result


# ============================================================
# 10. EVALUATE EXPERIMENT
# ============================================================

def evaluate_experiment(
    test,
    paymasters,
    base_budgets,
    result,
    strategy,
    budget_multiplier,
    safety_multiplier
):

    accepted = result[
        result["accepted"]
    ].copy()

    total_operations = len(test)
    total_users = test["sender"].nunique()

    sponsored = len(accepted)

    unique_sponsored_users = accepted["sender"].nunique()

    operation_coverage = (
        sponsored / total_operations * 100
        if total_operations else 0.0
    )

    user_coverage = (
        unique_sponsored_users / total_users * 100
        if total_users else 0.0
    )

    actual_total = float(
        accepted["actual_cost_eth"].sum()
    )

    reserved_total = float(
        accepted["safety_adjusted_reserve_eth"].sum()
    )

    total_budget = sum(
        base_budgets[pm] * budget_multiplier
        for pm in paymasters
    )

    actual_excess_total = 0.0
    exceeding_paymasters = 0

    pm_records = []

    for pm in paymasters:

        pm_budget = base_budgets[pm] * budget_multiplier

        pm_test = test[
            test["paymaster"] == pm
        ]

        pm_accepted = accepted[
            accepted["paymaster"] == pm
        ]

        pm_actual = float(
            pm_accepted["actual_cost_eth"].sum()
        )

        pm_reserved = float(
            pm_accepted["safety_adjusted_reserve_eth"].sum()
        )

        pm_excess = max(
            0.0,
            pm_actual - pm_budget
        )

        actual_excess_total += pm_excess

        if pm_excess > BUDGET_TOLERANCE:
            exceeding_paymasters += 1

        pm_users = pm_test["sender"].drop_duplicates().tolist()

        pm_counts = (
            pm_accepted.groupby("sender")
            .size()
            .reindex(pm_users, fill_value=0)
        )

        pm_fairness = (
            jain_index(pm_counts.to_numpy())
            if pm_users else np.nan
        )

        pm_records.append({

            "strategy": strategy,

            "budget_multiplier": budget_multiplier,

            "safety_multiplier": safety_multiplier,

            "paymaster": pm,

            "test_operations": len(pm_test),

            "test_users": len(pm_users),

            "simulated_budget_eth": pm_budget,

            "sponsored_operations": len(pm_accepted),

            "unique_sponsored_users": pm_accepted["sender"].nunique(),

            "actual_spending_eth": pm_actual,

            "reserved_spending_eth": pm_reserved,

            "remaining_reserved_budget_eth": max(
                0.0,
                pm_budget - pm_reserved
            ),

            "actual_budget_excess_eth": pm_excess,

            "reservation_budget_check_passed": (
                pm_reserved <= pm_budget + BUDGET_TOLERANCE
            ),

            "actual_budget_exceeded": (
                pm_excess > BUDGET_TOLERANCE
            ),

            "accepted_underestimated_operations": int(
                pm_accepted[
                    "actual_exceeded_adjusted_prediction"
                ].sum()
            ),

            "jain_fairness_index": pm_fairness,

        })

    global_fairness, mean_pm_fairness = (
        calculate_fairness_metrics(
            accepted,
            test,
            paymasters
        )
    )

    underestimated = int(
        accepted["actual_exceeded_adjusted_prediction"].sum()
    )

    summary = {

        "strategy": strategy,

        "budget_multiplier": budget_multiplier,

        "safety_multiplier": safety_multiplier,

        "batch_size": BATCH_SIZE,

        "test_operations": total_operations,

        "sponsored_operations": sponsored,

        "rejected_operations": total_operations - sponsored,

        "operation_coverage_percentage": operation_coverage,

        "total_test_users": total_users,

        "unique_sponsored_users": unique_sponsored_users,

        "user_coverage_percentage": user_coverage,

        "actual_spending_eth": actual_total,

        "safety_adjusted_reserved_spending_eth": reserved_total,

        "total_simulated_budget_eth": total_budget,

        "actual_budget_utilization_percentage": (
            actual_total / total_budget * 100
            if total_budget > 0 else 0.0
        ),

        "reserved_budget_utilization_percentage": (
            reserved_total / total_budget * 100
            if total_budget > 0 else 0.0
        ),

        "actual_budget_excess_eth": actual_excess_total,

        "paymasters_exceeding_budget": exceeding_paymasters,

        "accepted_operations_exceeding_adjusted_prediction": (
            underestimated
        ),

        "accepted_underestimation_rate_percentage": (
            underestimated / sponsored * 100
            if sponsored else 0.0
        ),

        "reserve_to_actual_spending_ratio": (
            reserved_total / actual_total
            if actual_total > 0 else np.nan
        ),

        "global_user_jain_fairness_index": global_fairness,

        "mean_per_paymaster_jain_fairness_index": mean_pm_fairness,

        "reservation_budget_checks_passed": all(
            row["reservation_budget_check_passed"]
            for row in pm_records
        ),

    }

    return summary, pd.DataFrame(pm_records)


# ============================================================
# 11. MAIN EXPERIMENT
# ============================================================

def main():

    df = load_dataset()

    calibration, test, split_summary = create_split(df)

    test = generate_predictions(
        calibration,
        test
    )

    paymasters = sorted(
        df["paymaster"].astype(str).unique().tolist()
    )

    (
        base_budgets,
        initial_user_counts,
        initial_user_spending
    ) = initialize_state(
        calibration,
        paymasters
    )

    typical_cost = float(
        calibration["gas_cost_eth"].median()
    )

    if typical_cost <= 0:
        typical_cost = 1e-18

    all_details = []
    all_summaries = []
    all_paymaster_summaries = []

    total_runs = (
        len(STRATEGIES)
        * len(BUDGET_MULTIPLIERS)
        * len(SAFETY_MULTIPLIERS)
    )

    run_number = 0

    print("\nStarting improved allocation experiments...")
    print("Total configurations:", total_runs)

    for budget_multiplier in BUDGET_MULTIPLIERS:

        for safety_multiplier in SAFETY_MULTIPLIERS:

            for strategy in STRATEGIES:

                run_number += 1

                print(
                    f"\nExperiment {run_number}/{total_runs}: "
                    f"{strategy}, "
                    f"Budget={budget_multiplier}x, "
                    f"Safety={safety_multiplier}x"
                )

                result = run_experiment(
                    test=test,
                    paymasters=paymasters,
                    base_budgets=base_budgets,
                    initial_user_counts=initial_user_counts,
                    initial_user_spending=initial_user_spending,
                    strategy=strategy,
                    budget_multiplier=budget_multiplier,
                    safety_multiplier=safety_multiplier,
                    typical_cost=typical_cost
                )

                summary, pm_summary = evaluate_experiment(
                    test=test,
                    paymasters=paymasters,
                    base_budgets=base_budgets,
                    result=result,
                    strategy=strategy,
                    budget_multiplier=budget_multiplier,
                    safety_multiplier=safety_multiplier
                )

                all_details.append(result)
                all_summaries.append(summary)
                all_paymaster_summaries.append(pm_summary)

    # Combine all experiments.

    detail_df = pd.concat(
        all_details,
        ignore_index=True
    )

    summary_df = pd.DataFrame(
        all_summaries
    )

    paymaster_df = pd.concat(
        all_paymaster_summaries,
        ignore_index=True
    )

    # Save results without overwriting previous experiment files.

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    detail_df.to_csv(
        DETAIL_FILE,
        index=False
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    paymaster_df.to_csv(
        PAYMASTER_FILE,
        index=False
    )

    split_summary.to_csv(
        SPLIT_FILE,
        index=False
    )

    # Display results.

    print("\n" + "=" * 110)
    print("IMPROVED SAFETY-MARGIN ALLOCATION RESULTS")
    print("=" * 110)

    display_columns = [

        "strategy",

        "budget_multiplier",

        "safety_multiplier",

        "sponsored_operations",

        "operation_coverage_percentage",

        "unique_sponsored_users",

        "actual_budget_excess_eth",

        "paymasters_exceeding_budget",

        "global_user_jain_fairness_index",

        "mean_per_paymaster_jain_fairness_index",

        "reservation_budget_checks_passed",

    ]

    print(
        summary_df[display_columns].to_string(
            index=False
        )
    )

    print("\nPrediction methods:")

    print(
        test["prediction_method"]
        .value_counts()
        .to_string()
    )

    print("\nCalibration/test split:")

    print(
        split_summary.to_string(index=False)
    )

    print("\nSaved files:")

    print(DETAIL_FILE)
    print(SUMMARY_FILE)
    print(PAYMASTER_FILE)
    print(SPLIT_FILE)

    print("\nAll experiments completed successfully.")

    print(
        "NOTE: These are historical simulations using simulated budgets, "
        "not real Paymaster balances."
    )


if __name__ == "__main__":
    main()

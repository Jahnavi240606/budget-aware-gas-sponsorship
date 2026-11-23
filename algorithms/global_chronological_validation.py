
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS AND EXPERIMENT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "processed_userops.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

CALIBRATION_RATIO = 0.20
ROLLING_WINDOW = 50
MIN_HISTORY = 20
BATCH_SIZE = 100

BUDGET_MULTIPLIERS = [1.5, 2.5, 5.0]
SAFETY_MULTIPLIERS = [1.0, 1.1, 1.25, 1.5, 2.0]

STRATEGIES = ["FCFS", "FAIRNESS", "PROPOSED"]


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "block_number",
        "log_index",
        "userop_hash",
        "sender",
        "paymaster",
        "actualGasCost",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["block_number"] = pd.to_numeric(
        df["block_number"], errors="raise"
    )

    df["log_index"] = pd.to_numeric(
        df["log_index"], errors="raise"
    )

    df["actualGasCost"] = pd.to_numeric(
        df["actualGasCost"], errors="raise"
    )

    df["actual_cost_eth"] = (
        df["actualGasCost"] / 1e18
    )

    df["sender"] = df["sender"].astype(str).str.lower()
    df["paymaster"] = df["paymaster"].astype(str).str.lower()

    df = df.sort_values(
        ["block_number", "log_index"]
    ).reset_index(drop=True)

    if df["userop_hash"].duplicated().any():
        raise ValueError(
            "Duplicate UserOperation hashes found."
        )

    return df


# ============================================================
# 3. GLOBAL CHRONOLOGICAL CALIBRATION / TEST SPLIT
# ============================================================

def global_chronological_split(df):

    split_index = int(
        len(df) * CALIBRATION_RATIO
    )

    split_index = max(
        1,
        min(split_index, len(df) - 1)
    )

    calibration = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    calibration = calibration.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return calibration, test


# ============================================================
# 4. GENERATE STRICTLY HISTORICAL COST PREDICTIONS
# ============================================================

def generate_cost_predictions(calibration, test):

    histories = {}

    for paymaster, group in calibration.groupby("paymaster"):

        histories[paymaster] = (
            group["actual_cost_eth"]
            .astype(float)
            .tolist()
        )

    global_median = calibration[
        "actual_cost_eth"
    ].median()

    global_p90 = calibration[
        "actual_cost_eth"
    ].quantile(0.90)

    if global_median <= 0:
        scale_factor = 1.0
    else:
        scale_factor = global_p90 / global_median

    predictions = []

    # Test operations are already globally chronological.
    for _, row in test.iterrows():

        paymaster = row["paymaster"]

        history = histories.get(paymaster, [])

        if len(history) >= MIN_HISTORY:

            recent_history = history[-ROLLING_WINDOW:]

            predicted_cost = float(
                np.quantile(recent_history, 0.90)
            )

            method = "rolling_p90_50"

        elif len(history) > 0:

            historical_median = float(
                np.median(history)
            )

            predicted_cost = (
                historical_median * scale_factor
            )

            method = "scaled_historical_median"

        else:

            predicted_cost = float(global_p90)

            method = "global_calibration_p90"

        predicted_cost = max(
            predicted_cost,
            0.0
        )

        predictions.append({
            "userop_hash": row["userop_hash"],
            "predicted_cost_eth": predicted_cost,
            "prediction_method": method,
            "history_size_before_prediction": len(history),
        })

        # Update history only after making this operation's
        # prediction. This prevents using its actual cost
        # to predict itself.
        histories.setdefault(paymaster, []).append(
            float(row["actual_cost_eth"])
        )

    prediction_df = pd.DataFrame(predictions)

    test = test.merge(
        prediction_df,
        on="userop_hash",
        how="left",
        validate="one_to_one"
    )

    return test


# ============================================================
# 5. INITIALIZE SIMULATED PAYMASTER BUDGETS AND USER HISTORY
# ============================================================

def initialize_simulation(calibration, test, budget_multiplier):

    calibration_spending = (
        calibration.groupby("paymaster")[
            "actual_cost_eth"
        ].sum().to_dict()
    )

    paymasters = set(test["paymaster"].unique())

    remaining_budgets = {}

    actual_spending = {}

    reserved_spending = {}

    for paymaster in paymasters:

        calibration_spend = float(
            calibration_spending.get(paymaster, 0.0)
        )

        remaining_budgets[paymaster] = (
            calibration_spend * budget_multiplier
        )

        actual_spending[paymaster] = 0.0

        reserved_spending[paymaster] = 0.0

    user_counts = (
        calibration.groupby("sender")
        .size()
        .to_dict()
    )

    user_spending = (
        calibration.groupby("sender")[
            "actual_cost_eth"
        ].sum().to_dict()
    )

    return (
        remaining_budgets,
        actual_spending,
        reserved_spending,
        user_counts,
        user_spending,
    )


# ============================================================
# 6. SELECT OPERATIONS WITHIN EACH CHRONOLOGICAL BATCH
# ============================================================

def order_batch(group, strategy, user_counts, user_spending):

    group = group.copy()

    group["historical_user_count"] = (
        group["sender"]
        .map(user_counts)
        .fillna(0)
    )

    group["historical_user_spending"] = (
        group["sender"]
        .map(user_spending)
        .fillna(0.0)
    )

    if strategy == "FCFS":

        return group.sort_values(
            ["block_number", "log_index"]
        )

    if strategy == "FAIRNESS":

        return group.sort_values(
            [
                "historical_user_count",
                "historical_user_spending",
                "predicted_cost_eth",
                "block_number",
                "log_index",
            ]
        )

    if strategy == "PROPOSED":

        # Lower rank means higher priority.
        group["fairness_rank"] = (
            group["historical_user_count"]
            .rank(method="average", pct=True)
        )

        group["spending_rank"] = (
            group["historical_user_spending"]
            .rank(method="average", pct=True)
        )

        group["cost_rank"] = (
            group["predicted_cost_eth"]
            .rank(method="average", pct=True)
        )

        group["priority_score"] = (
            0.50 * group["fairness_rank"]
            + 0.30 * group["spending_rank"]
            + 0.20 * group["cost_rank"]
        )

        return group.sort_values(
            [
                "priority_score",
                "block_number",
                "log_index",
            ]
        )

    raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================
# 7. RUN ONE ALLOCATION SIMULATION
# ============================================================

def run_simulation(
    calibration,
    test,
    strategy,
    budget_multiplier,
    safety_multiplier,
):

    (
        remaining_budgets,
        actual_spending,
        reserved_spending,
        user_counts,
        user_spending,
    ) = initialize_simulation(
        calibration,
        test,
        budget_multiplier
    )

    accepted_records = []

    total_operations = len(test)

    # Process the test period in chronological batches.
    for start in range(
        0,
        total_operations,
        BATCH_SIZE
    ):

        batch = test.iloc[
            start:start + BATCH_SIZE
        ].copy()

        # Keep each Paymaster's budget separate.
        for paymaster, paymaster_group in batch.groupby(
            "paymaster",
            sort=False
        ):

            ordered_group = order_batch(
                paymaster_group,
                strategy,
                user_counts,
                user_spending
            )

            for _, row in ordered_group.iterrows():

                sender = row["sender"]

                actual_cost = float(
                    row["actual_cost_eth"]
                )

                predicted_cost = float(
                    row["predicted_cost_eth"]
                )

                adjusted_reserve = (
                    predicted_cost * safety_multiplier
                )

                available_budget = float(
                    remaining_budgets.get(paymaster, 0.0)
                )

                # Accept only when the predicted reserve
                # fits within the remaining simulated budget.
                if adjusted_reserve <= available_budget:

                    remaining_budgets[paymaster] -= (
                        adjusted_reserve
                    )

                    reserved_spending[paymaster] += (
                        adjusted_reserve
                    )

                    actual_spending[paymaster] += (
                        actual_cost
                    )

                    user_counts[sender] = (
                        user_counts.get(sender, 0) + 1
                    )

                    user_spending[sender] = (
                        user_spending.get(sender, 0.0)
                        + actual_cost
                    )

                    accepted_records.append({
                        "userop_hash": row["userop_hash"],
                        "sender": sender,
                        "paymaster": paymaster,
                        "actual_cost_eth": actual_cost,
                        "predicted_cost_eth": predicted_cost,
                        "adjusted_reserve_eth": adjusted_reserve,
                    })

    accepted = pd.DataFrame(accepted_records)

    # --------------------------------------------------------
    # Aggregate results per Paymaster
    # --------------------------------------------------------

    paymaster_results = []

    all_paymasters = sorted(
        set(test["paymaster"].unique())
    )

    for paymaster in all_paymasters:

        budget = float(
            calibration.groupby("paymaster")[
                "actual_cost_eth"
            ].sum().get(paymaster, 0.0)
        ) * budget_multiplier

        actual = actual_spending.get(paymaster, 0.0)

        reserved = reserved_spending.get(paymaster, 0.0)

        overrun = max(
            actual - budget,
            0.0
        )

        paymaster_accepted = (
            accepted[
                accepted["paymaster"] == paymaster
            ]
            if not accepted.empty
            else pd.DataFrame()
        )

        paymaster_results.append({
            "paymaster": paymaster,
            "simulated_budget_eth": budget,
            "actual_spending_eth": actual,
            "reserved_spending_eth": reserved,
            "actual_budget_excess_eth": overrun,
            "accepted_operations": len(paymaster_accepted),
        })

    paymaster_df = pd.DataFrame(paymaster_results)

    # --------------------------------------------------------
    # Calculate global user fairness
    # --------------------------------------------------------

    if accepted.empty:

        unique_users = 0
        jain_index = 0.0

    else:

        user_operation_counts = (
            accepted.groupby("sender")
            .size()
            .to_numpy(dtype=float)
        )

        unique_users = len(user_operation_counts)

        denominator = (
            len(user_operation_counts)
            * np.sum(user_operation_counts ** 2)
        )

        if denominator == 0:
            jain_index = 0.0
        else:
            jain_index = (
                np.sum(user_operation_counts) ** 2
            ) / denominator

    total_budget = float(
        paymaster_df["simulated_budget_eth"].sum()
    )

    total_actual = float(
        paymaster_df["actual_spending_eth"].sum()
    )

    total_reserved = float(
        paymaster_df["reserved_spending_eth"].sum()
    )

    total_excess = float(
        paymaster_df["actual_budget_excess_eth"].sum()
    )

    accepted_count = len(accepted)

    actual_over_estimate_count = (
        int(
            (
                accepted["actual_cost_eth"]
                > accepted["adjusted_reserve_eth"]
            ).sum()
        )
        if not accepted.empty
        else 0
    )

    summary = {
        "strategy": strategy,
        "budget_multiplier": budget_multiplier,
        "safety_multiplier": safety_multiplier,

        "test_operations": total_operations,

        "accepted_operations": accepted_count,

        "operation_coverage_pct": (
            accepted_count / total_operations * 100
            if total_operations else 0.0
        ),

        "unique_users_sponsored": unique_users,

        "actual_spending_eth": total_actual,

        "reserved_spending_eth": total_reserved,

        "simulated_budget_eth": total_budget,

        "actual_budget_utilization_pct": (
            total_actual / total_budget * 100
            if total_budget else 0.0
        ),

        "reserved_budget_utilization_pct": (
            total_reserved / total_budget * 100
            if total_budget else 0.0
        ),

        "actual_budget_excess_eth": total_excess,

        "paymasters_exceeding_budget": int(
            (
                paymaster_df[
                    "actual_budget_excess_eth"
                ] > 1e-18
            ).sum()
        ),

        "accepted_ops_actual_cost_above_reserve": (
            actual_over_estimate_count
        ),

        "global_jain_fairness_index": jain_index,
    }

    return summary, paymaster_df


# ============================================================
# 8. MAIN EXPERIMENT
# ============================================================

def main():

    print("\nLoading Optimism UserOperation dataset...")

    df = load_data()

    calibration, test = global_chronological_split(df)

    print("\nGLOBAL CHRONOLOGICAL SPLIT")
    print("--------------------------")
    print(f"Total operations: {len(df)}")
    print(f"Calibration operations: {len(calibration)}")
    print(f"Test operations: {len(test)}")

    print(
        "\nCalibration period:",
        calibration.iloc[0]["block_number"],
        "to",
        calibration.iloc[-1]["block_number"]
    )

    print(
        "Test period:",
        test.iloc[0]["block_number"],
        "to",
        test.iloc[-1]["block_number"]
    )

    print("\nGenerating historical cost predictions...")

    test = generate_cost_predictions(
        calibration,
        test
    )

    print("\nPrediction methods:")
    print(
        test["prediction_method"]
        .value_counts()
        .to_string()
    )

    prediction_coverage = (
        test["actual_cost_eth"]
        <= test["predicted_cost_eth"]
    ).mean() * 100

    print(
        f"\nUnadjusted prediction coverage: "
        f"{prediction_coverage:.2f}%"
    )

    all_summaries = []
    all_paymaster_results = []

    total_runs = (
        len(STRATEGIES)
        * len(BUDGET_MULTIPLIERS)
        * len(SAFETY_MULTIPLIERS)
    )

    run_number = 0

    print("\nStarting allocation experiments...")

    for budget_multiplier in BUDGET_MULTIPLIERS:

        for safety_multiplier in SAFETY_MULTIPLIERS:

            for strategy in STRATEGIES:

                run_number += 1

                summary, paymaster_df = run_simulation(
                    calibration=calibration,
                    test=test,
                    strategy=strategy,
                    budget_multiplier=budget_multiplier,
                    safety_multiplier=safety_multiplier,
                )

                all_summaries.append(summary)

                paymaster_df["strategy"] = strategy
                paymaster_df["budget_multiplier"] = (
                    budget_multiplier
                )
                paymaster_df["safety_multiplier"] = (
                    safety_multiplier
                )

                all_paymaster_results.append(
                    paymaster_df
                )

                print(
                    f"Run {run_number}/{total_runs}: "
                    f"{strategy}, "
                    f"budget={budget_multiplier}x, "
                    f"safety={safety_multiplier}x, "
                    f"accepted={summary['accepted_operations']}, "
                    f"coverage={summary['operation_coverage_pct']:.2f}%, "
                    f"budget excess={summary['actual_budget_excess_eth']:.10f} ETH"
                )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    summary_df = pd.DataFrame(all_summaries)

    paymaster_results_df = pd.concat(
        all_paymaster_results,
        ignore_index=True
    )

    summary_path = (
        OUTPUT_DIR
        / "global_chronological_validation_summary.csv"
    )

    paymaster_path = (
        OUTPUT_DIR
        / "global_chronological_validation_by_paymaster.csv"
    )

    prediction_path = (
        OUTPUT_DIR
        / "global_chronological_validation_predictions.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    paymaster_results_df.to_csv(
        paymaster_path,
        index=False
    )

    test.to_csv(
        prediction_path,
        index=False
    )

    print("\n========================================")
    print("GLOBAL CHRONOLOGICAL VALIDATION COMPLETE")
    print("========================================")

    print("\nFinal summary:")
    print(
        summary_df[
            [
                "strategy",
                "budget_multiplier",
                "safety_multiplier",
                "accepted_operations",
                "operation_coverage_pct",
                "unique_users_sponsored",
                "actual_budget_excess_eth",
                "paymasters_exceeding_budget",
                "global_jain_fairness_index",
            ]
        ].to_string(index=False)
    )

    print("\nSaved files:")
    print(summary_path)
    print(paymaster_path)
    print(prediction_path)


if __name__ == "__main__":
    main()

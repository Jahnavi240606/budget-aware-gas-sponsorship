
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT = (
    ROOT / "data" / "processed"
    / "weight_tuning_allocation_summary.csv"
)

OUTPUT = (
    ROOT / "data" / "processed"
    / "weight_tuning_comparison_report.csv"
)

df = pd.read_csv(INPUT)

proposed = df[df["strategy"] == "PROPOSED"].copy()

# Average results across all budget/safety scenarios.
overall = (
    proposed.groupby("weight_config")
    .agg(
        average_operations=("sponsored_operations", "mean"),
        average_coverage=("operation_coverage_percentage", "mean"),
        average_users=("unique_sponsored_users", "mean"),
        average_fairness=("jain_fairness_index", "mean"),
        average_budget_excess=("actual_budget_excess_eth", "mean"),
        average_exceeding_paymasters=(
            "paymasters_exceeding_budget", "mean"
        )
    )
    .reset_index()
)

# Rank each configuration separately for each metric.
overall["fairness_rank"] = (
    overall["average_fairness"]
    .rank(ascending=False, method="min")
)

overall["coverage_rank"] = (
    overall["average_coverage"]
    .rank(ascending=False, method="min")
)

overall["users_rank"] = (
    overall["average_users"]
    .rank(ascending=False, method="min")
)

overall["budget_safety_rank"] = (
    overall["average_exceeding_paymasters"]
    .rank(ascending=True, method="min")
)

# Display overall comparison.
print("\nOVERALL WEIGHT COMPARISON")
print("=" * 90)

print(
    overall.sort_values("fairness_rank").to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)

# Compare configurations separately for every budget/safety pair.
print("\nBEST FAIRNESS CONFIGURATION BY SCENARIO")
print("=" * 90)

scenario_best = (
    proposed.sort_values(
        "jain_fairness_index",
        ascending=False
    )
    .groupby(
        ["budget_multiplier", "safety_multiplier"],
        as_index=False
    )
    .first()
)

print(
    scenario_best[
        [
            "budget_multiplier",
            "safety_multiplier",
            "weight_config",
            "sponsored_operations",
            "unique_sponsored_users",
            "actual_budget_excess_eth",
            "paymasters_exceeding_budget",
            "jain_fairness_index"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.8e}"
    )
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
overall.to_csv(OUTPUT, index=False)

print("\nSaved comparison report:")
print(OUTPUT)

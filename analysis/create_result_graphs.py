
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
input_file = Path("data/processed/batch_dynamic_allocation_summary.csv")
output_folder = Path("analysis/figures")
output_folder.mkdir(parents=True, exist_ok=True)

# Load experiment results
df = pd.read_csv(input_file)
df["budget_multiplier"] = df["budget_multiplier"].round(1)

strategies = ["FCFS", "FAIRNESS", "PROPOSED"]
budget_levels = sorted(df["budget_multiplier"].unique())

# Metrics and graph filenames
graphs = [
    (
        "operation_coverage_percentage",
        "Operation Coverage (%)",
        "Operation Coverage by Budget Level",
        "01_operation_coverage.png"
    ),
    (
        "unique_sponsored_users",
        "Unique Users Sponsored",
        "Unique Users Sponsored by Budget Level",
        "02_unique_users_sponsored.png"
    ),
    (
        "actual_budget_utilization_percentage",
        "Actual Budget Utilization (%)",
        "Actual Budget Utilization by Budget Level",
        "03_actual_budget_utilization.png"
    ),
    (
        "jain_fairness_index",
        "Jain Fairness Index",
        "Fairness Comparison by Budget Level",
        "04_jain_fairness.png"
    )
]

# Generate and save graphs
for metric, ylabel, title, filename in graphs:

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for strategy in strategies:
        subset = df[df["strategy"] == strategy].sort_values(
            "budget_multiplier"
        )

        ax.plot(
            subset["budget_multiplier"],
            subset[metric],
            marker="o",
            linewidth=2,
            markersize=7,
            label=strategy
        )

    ax.set_title(title)
    ax.set_xlabel("Simulated Budget Multiplier")
    ax.set_ylabel(ylabel)
    ax.set_xticks(budget_levels)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Allocation Strategy")

    fig.tight_layout()
    fig.savefig(
        output_folder / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)
    print(f"Saved: {output_folder / filename}")

print("\nAll four graphs generated successfully!")

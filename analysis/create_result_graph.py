
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load experiment results
df = pd.read_csv(
    "data/processed/batch_dynamic_allocation_summary.csv"
)

# Create output folder
output = Path("analysis/figures")
output.mkdir(parents=True, exist_ok=True)

df["budget_multiplier"] = df["budget_multiplier"].round(1)

graphs = [
    ("operation_coverage_percentage",
     "Operation Coverage (%)",
     "Operation Coverage",
     "01_operation_coverage.png"),

    ("unique_sponsored_users",
     "Unique Users Sponsored",
     "Unique Users Sponsored",
     "02_unique_users.png"),

    ("actual_budget_utilization_percentage",
     "Actual Budget Utilization (%)",
     "Actual Budget Utilization",
     "03_budget_utilization.png"),

    ("jain_fairness_index",
     "Jain Fairness Index",
     "Jain Fairness Index",
     "04_fairness_index.png")
]

strategies = ["FCFS", "FAIRNESS", "PROPOSED"]

for metric, ylabel, title, filename in graphs:

    fig, ax = plt.subplots(figsize=(9, 5))

    for strategy in strategies:
        data = df[df["strategy"] == strategy].sort_values(
            "budget_multiplier"
        )

        ax.plot(
            data["budget_multiplier"],
            data[metric],
            marker="o",
            linewidth=2,
            label=strategy
        )

    ax.set_title(title)
    ax.set_xlabel("Simulated Budget Multiplier")
    ax.set_ylabel(ylabel)
    ax.set_xticks([1.5, 2.5, 5.0])
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output / filename, dpi=300)
    plt.close()

    print("Created:", filename)

print("\nAll four graphs created successfully!")

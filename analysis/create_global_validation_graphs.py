
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "global_chronological_validation_summary.csv"
)

FIGURE_DIR = PROJECT_ROOT / "analysis" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

BUDGET_TO_PLOT = 2.5


# --------------------------------------------------
# 2. LOAD RESULTS
# --------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Results file not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "strategy",
    "budget_multiplier",
    "safety_multiplier",
    "operation_coverage_pct",
    "unique_users_sponsored",
    "actual_budget_excess_eth",
    "global_jain_fairness_index",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

df = df[
    df["budget_multiplier"] == BUDGET_TO_PLOT
].copy()

if df.empty:
    raise ValueError(
        "No results found for the selected budget."
    )

df = df.sort_values("safety_multiplier")

strategies = ["FCFS", "FAIRNESS", "PROPOSED"]


# --------------------------------------------------
# 3. HELPER FUNCTION
# --------------------------------------------------

def create_line_graph(
    metric,
    title,
    y_label,
    filename,
    scientific=False
):

    plt.figure(figsize=(10, 6))

    for strategy in strategies:

        strategy_df = df[
            df["strategy"] == strategy
        ].sort_values("safety_multiplier")

        plt.plot(
            strategy_df["safety_multiplier"],
            strategy_df[metric],
            marker="o",
            linewidth=2,
            markersize=7,
            label=strategy
        )

    plt.title(
        title,
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel(
        "Safety Multiplier",
        fontsize=11
    )

    plt.ylabel(
        y_label,
        fontsize=11
    )

    plt.xticks(
        sorted(df["safety_multiplier"].unique())
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    plt.legend()
    plt.tight_layout()

    if scientific:
        plt.ticklabel_format(
            axis="y",
            style="sci",
            scilimits=(0, 0)
        )

    output_path = FIGURE_DIR / filename

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Created: {output_path}")


# --------------------------------------------------
# 4. GRAPH 1: OPERATION COVERAGE
# --------------------------------------------------

create_line_graph(
    metric="operation_coverage_pct",
    title="Operation Coverage vs Safety Multiplier",
    y_label="Operation Coverage (%)",
    filename="05_global_operation_coverage.png"
)


# --------------------------------------------------
# 5. GRAPH 2: UNIQUE USERS SPONSORED
# --------------------------------------------------

create_line_graph(
    metric="unique_users_sponsored",
    title="Unique Sponsored Users vs Safety Multiplier",
    y_label="Number of Unique Users",
    filename="06_global_unique_users.png"
)


# --------------------------------------------------
# 6. GRAPH 3: ACTUAL BUDGET EXCESS
# --------------------------------------------------

create_line_graph(
    metric="actual_budget_excess_eth",
    title="Actual Budget Excess vs Safety Multiplier",
    y_label="Actual Budget Excess (ETH)",
    filename="07_global_budget_excess.png",
    scientific=True
)


# --------------------------------------------------
# 7. GRAPH 4: GLOBAL JAIN FAIRNESS
# --------------------------------------------------

create_line_graph(
    metric="global_jain_fairness_index",
    title="Global User Fairness vs Safety Multiplier",
    y_label="Global Jain Fairness Index",
    filename="08_global_jain_fairness.png"
)


# --------------------------------------------------
# 8. COMPLETION
# --------------------------------------------------

print("\nAll four global validation graphs created.")

print("\nGraph folder:")
print(FIGURE_DIR)

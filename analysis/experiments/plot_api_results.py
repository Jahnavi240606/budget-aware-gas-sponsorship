
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

input_file = BASE_DIR / "analysis" / "experiments" / "api_experiment_results.csv"
output_dir = BASE_DIR / "analysis" / "experiments" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file)

df["budget_eth"] = pd.to_numeric(df["budget_eth"])
df["accepted"] = pd.to_numeric(df["accepted"])
df["rejected"] = pd.to_numeric(df["rejected"])
df["remaining_eth"] = pd.to_numeric(df["remaining_eth"])

budgets = sorted(df["budget_eth"].unique())

for budget in budgets:
    subset = df[df["budget_eth"] == budget].copy()

    # Accepted operations
    plt.figure()
    plt.bar(subset["strategy"], subset["accepted"])
    plt.title(f"Accepted Operations - Budget {budget} ETH")
    plt.xlabel("Allocation Strategy")
    plt.ylabel("Accepted Operations")
    plt.tight_layout()
    plt.savefig(output_dir / f"accepted_{budget}.png", dpi=300)
    plt.close()

    # Rejected operations
    plt.figure()
    plt.bar(subset["strategy"], subset["rejected"])
    plt.title(f"Rejected Operations - Budget {budget} ETH")
    plt.xlabel("Allocation Strategy")
    plt.ylabel("Rejected Operations")
    plt.tight_layout()
    plt.savefig(output_dir / f"rejected_{budget}.png", dpi=300)
    plt.close()

    # Remaining simulated budget
    plt.figure()
    plt.bar(subset["strategy"], subset["remaining_eth"])
    plt.title(f"Remaining Simulated Budget - {budget} ETH")
    plt.xlabel("Allocation Strategy")
    plt.ylabel("Remaining Budget (ETH)")
    plt.tight_layout()
    plt.savefig(output_dir / f"remaining_{budget}.png", dpi=300)
    plt.close()

print("All graphs generated successfully!")
print(f"Saved in: {output_dir}")

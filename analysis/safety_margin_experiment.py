
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "data" / "processed" / "rolling_cost_predictions.csv"
OUTPUT = ROOT / "data" / "processed"

MULTIPLIERS = [1.00, 1.10, 1.25, 1.50, 2.00]

def evaluate(group, multiplier):
    actual = group["actual_cost_eth"]
    estimate = group["rolling_p90_50"] * multiplier

    return {
        "operations": len(group),
        "safety_multiplier": multiplier,
        "coverage_percent": (actual <= estimate).mean() * 100,
        "underestimation_percent": (actual > estimate).mean() * 100,
        "actual_total_eth": actual.sum(),
        "reserved_total_eth": estimate.sum(),
        "reserve_to_actual_ratio": (
            estimate.sum() / actual.sum()
            if actual.sum() > 0 else np.nan
        ),
        "mae_eth": (estimate - actual).abs().mean(),
    }


def main():
    df = pd.read_csv(INPUT)

    required = [
        "paymaster",
        "actual_cost_eth",
        "rolling_p90_50",
    ]

    if df[required].isna().any().any():
        raise ValueError("Missing values found.")

    overall_results = []
    paymaster_results = []

    for multiplier in MULTIPLIERS:
        result = evaluate(df, multiplier)
        overall_results.append(result)

        for paymaster, group in df.groupby("paymaster"):
            result = evaluate(group, multiplier)
            result["paymaster"] = paymaster
            paymaster_results.append(result)

    overall = pd.DataFrame(overall_results)
    by_paymaster = pd.DataFrame(paymaster_results)

    overall.to_csv(
        OUTPUT / "safety_margin_overall.csv",
        index=False
    )

    by_paymaster.to_csv(
        OUTPUT / "safety_margin_by_paymaster.csv",
        index=False
    )

    print("\nOVERALL SAFETY MARGIN RESULTS")
    print(overall.to_string(index=False))

    print("\nRESULTS SAVED SUCCESSFULLY.")


if __name__ == "__main__":
    main()

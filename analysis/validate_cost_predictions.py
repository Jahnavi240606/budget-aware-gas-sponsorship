
from pathlib import Path

import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rolling_cost_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

METHODS = [
    "expanding_p90",
    "rolling_p90_50",
    "rolling_p90_100",
]


def calculate_metrics(group, method):
    actual = group["actual_cost_eth"]
    predicted = group[method]

    errors = predicted - actual

    return {
        "method": method,
        "operations": len(group),
        "coverage_percent": (actual <= predicted).mean() * 100,
        "underestimation_percent": (actual > predicted).mean() * 100,
        "mae_eth": errors.abs().mean(),
        "rmse_eth": np.sqrt((errors ** 2).mean()),
        "actual_total_eth": actual.sum(),
        "predicted_total_eth": predicted.sum(),
        "predicted_to_actual_ratio": (
            predicted.sum() / actual.sum()
            if actual.sum() != 0
            else np.nan
        ),
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "userop_hash",
        "paymaster",
        "actual_cost_eth",
        *METHODS,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Missing values found in required prediction data."
        )

    if (df["actual_cost_eth"] < 0).any():
        raise ValueError("Negative actual gas costs found.")

    for method in METHODS:
        if (df[method] < 0).any():
            raise ValueError(
                f"Negative predictions found in {method}."
            )

    # Overall performance
    overall_results = []

    for method in METHODS:
        overall_results.append(
            calculate_metrics(df, method)
        )

    overall = pd.DataFrame(overall_results)

    # Per-Paymaster performance
    paymaster_results = []

    for paymaster, group in df.groupby("paymaster"):
        for method in METHODS:
            result = calculate_metrics(group, method)
            result["paymaster"] = paymaster
            paymaster_results.append(result)

    by_paymaster = pd.DataFrame(paymaster_results)

    # Save outputs
    overall_path = (
        OUTPUT_DIR / "cost_prediction_evaluation.csv"
    )

    paymaster_path = (
        OUTPUT_DIR / "cost_prediction_by_paymaster.csv"
    )

    overall.to_csv(overall_path, index=False)
    by_paymaster.to_csv(paymaster_path, index=False)

    print("\nOVERALL COST PREDICTION EVALUATION")
    print(overall.to_string(index=False))

    print("\nPAYMASTER COUNT:", df["paymaster"].nunique())
    print("TEST OPERATIONS:", len(df))

    print("\nFILES CREATED:")
    print(overall_path)
    print(paymaster_path)


if __name__ == "__main__":
    main()


from pathlib import Path
from typing import Literal
import re

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "processed_userops.csv"
)

EXPERIMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "batch_dynamic_allocation_summary.csv"
)

GLOBAL_VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "global_chronological_validation_summary.csv"
)

GLOBAL_PAYMASTER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "global_chronological_validation_by_paymaster.csv"
)


# ============================================================
# 2. RESEARCH PARAMETERS
# ============================================================

ROLLING_WINDOW = 50
MIN_HISTORY = 20

FAIRNESS_WEIGHT = 0.50
SPENDING_WEIGHT = 0.30
COST_WEIGHT = 0.20

BUDGET_TOLERANCE = 1e-18


# ============================================================
# 3. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Gas Sponsorship Allocation API",
    description=(
        "Backend API for Budget-Aware Gas Sponsorship "
        "Allocation for ERC-4337 Smart-Account Ecosystems. "
        "Allocation decisions are simulations, not live transactions."
    ),
    version="2.0.0",
)


# ============================================================
# 4. CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 5. REQUEST MODELS
# ============================================================

class SponsorshipCandidate(BaseModel):
    sender: str = Field(
        min_length=42,
        max_length=42,
        description="Ethereum sender address",
    )


class AllocationRequest(BaseModel):
    paymaster: str = Field(
        min_length=42,
        max_length=42,
        description="Ethereum Paymaster address",
    )

    candidates: list[SponsorshipCandidate] = Field(
        min_length=1,
        max_length=100,
        description="Candidates requesting simulated sponsorship",
    )

    simulated_budget_eth: float = Field(
        ge=0,
        le=1000,
        description="Simulated budget in ETH",
    )

    strategy: Literal[
        "FCFS",
        "FAIRNESS",
        "PROPOSED",
    ] = "PROPOSED"

    safety_multiplier: float = Field(
        default=1.50,
        ge=1.0,
        le=5.0,
    )


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def validate_address(address: str) -> str:
    """Validate and normalize an Ethereum address."""

    address = address.strip().lower()

    if not re.fullmatch(r"0x[a-f0-9]{40}", address):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid Ethereum address: {address}",
        )

    return address


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file and report useful errors."""

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Data file not found: {path.name}",
        )

    try:
        return pd.read_csv(path)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read {path.name}: {str(error)}",
        )


def dataframe_to_records(df: pd.DataFrame):
    """Convert DataFrame values to JSON-safe records."""

    clean_df = df.astype(object).where(
        pd.notna(df),
        None,
    )

    return clean_df.to_dict(orient="records")


def load_processed_dataset() -> pd.DataFrame:
    """Load and validate the processed UserOperation dataset."""

    df = load_csv(DATASET_PATH)

    required_columns = [
        "sender",
        "paymaster",
        "success",
        "actualGasCost",
        "gas_cost_eth",
        "block_number",
        "log_index",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing dataset columns: {missing}",
        )

    if df.empty:
        raise HTTPException(
            status_code=500,
            detail="The processed dataset is empty.",
        )

    df = df.copy()

    for column in ["sender", "paymaster"]:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    for column in [
        "gas_cost_eth",
        "block_number",
        "log_index",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "sender",
            "paymaster",
            "gas_cost_eth",
            "block_number",
            "log_index",
        ]
    )

    df = df[
        np.isfinite(df["gas_cost_eth"])
        & np.isfinite(df["block_number"])
        & np.isfinite(df["log_index"])
    ]

    df = df[df["gas_cost_eth"] >= 0]

    if df.empty:
        raise HTTPException(
            status_code=500,
            detail="No valid historical gas-cost observations found.",
        )

    df = df.sort_values(
        ["block_number", "log_index"],
        kind="mergesort",
    ).reset_index(drop=True)

    return df


# ============================================================
# 7. HISTORICAL GAS-COST PREDICTION
# ============================================================

def predict_paymaster_gas_cost(
    df: pd.DataFrame,
    paymaster: str,
):
    """
    Estimate gas cost from historical Paymaster operations.

    Uses rolling P90 when sufficient history exists.
    Uses scaled historical median for limited history.
    Uses dataset-wide P90 when no Paymaster history exists.
    """

    global_median = float(
        df["gas_cost_eth"].median()
    )

    global_p90 = float(
        df["gas_cost_eth"].quantile(0.90)
    )

    if (
        not np.isfinite(global_median)
        or not np.isfinite(global_p90)
        or global_median <= 0
        or global_p90 <= 0
    ):
        raise HTTPException(
            status_code=500,
            detail="Historical gas-cost reference is invalid.",
        )

    history = df.loc[
        df["paymaster"] == paymaster,
        "gas_cost_eth",
    ].astype(float).tolist()

    history_size = len(history)

    if history_size >= MIN_HISTORY:

        recent = history[-ROLLING_WINDOW:]

        predicted_cost = float(
            np.quantile(recent, 0.90)
        )

        method = "rolling_p90_50"

    elif history_size > 0:

        global_scale = global_p90 / global_median

        predicted_cost = (
            float(np.median(history))
            * global_scale
        )

        method = "scaled_historical_median"

    else:

        predicted_cost = global_p90

        method = "global_dataset_p90"

    if (
        not np.isfinite(predicted_cost)
        or predicted_cost <= 0
    ):
        raise HTTPException(
            status_code=500,
            detail="Could not generate a valid gas-cost prediction.",
        )

    return predicted_cost, method, history_size


# ============================================================
# 8. CANDIDATE-SPECIFIC GAS-COST PREDICTION
# ============================================================

def predict_candidate_gas_cost(
    sender: str,
    paymaster_history: pd.DataFrame,
    paymaster_reference: float,
    paymaster_median: float,
):
    """
    Predict gas cost for an individual candidate.

    Uses sender history under the selected Paymaster.

    If the sender has insufficient history, falls back to
    the Paymaster-level prediction.
    """

    sender_history = (
        paymaster_history.loc[
            paymaster_history["sender"] == sender,
            "gas_cost_eth",
        ]
        .astype(float)
        .tolist()
    )

    history_size = len(sender_history)

    if history_size >= MIN_HISTORY:

        recent = sender_history[-ROLLING_WINDOW:]

        predicted_cost = float(
            np.quantile(recent, 0.90)
        )

        method = "sender_rolling_p90_50"

    elif history_size > 0:

        sender_median = float(
            np.median(sender_history)
        )

        scale = (
            paymaster_reference / paymaster_median
        )

        predicted_cost = sender_median * scale

        method = "sender_scaled_median"

    else:

        predicted_cost = paymaster_reference

        method = "paymaster_fallback"

    if (
        not np.isfinite(predicted_cost)
        or predicted_cost <= 0
    ):
        predicted_cost = paymaster_reference
        method = "paymaster_fallback"

    return predicted_cost, method, history_size


# ============================================================
# 9. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Gas Sponsorship Allocation API is running!",
        "project": "Budget-Aware Gas Sponsorship Allocation",
        "status": "success",
    }


# ============================================================
# 10. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "api": "running",
    }


# ============================================================
# 11. PROJECT INFORMATION
# ============================================================

@app.get("/project")
def project_info():

    return {
        "project_name": (
            "Budget-Aware Gas Sponsorship Allocation "
            "for ERC-4337 Smart-Account Ecosystems"
        ),
        "blockchain": "Optimism",
        "chain_id": 10,
        "frontend": "Next.js",
        "backend": "FastAPI",
        "database": "PostgreSQL",
        "status": "development",
    }


# ============================================================
# 12. DATASET SUMMARY
# ============================================================

@app.get("/dataset/summary")
def dataset_summary():

    df = load_processed_dataset()

    total_operations = len(df)

    unique_users = int(
        df["sender"].nunique()
    )

    unique_paymasters = int(
        df["paymaster"].nunique()
    )

    success_values = (
        df["success"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    successful_mask = success_values.isin(
        ["true", "1", "yes"]
    )

    successful_operations = int(
        successful_mask.sum()
    )

    failed_operations = (
        total_operations - successful_operations
    )

    gas_cost = pd.to_numeric(
        df["actualGasCost"],
        errors="coerce",
    ).fillna(0)

    total_gas_spending_eth = float(
        gas_cost.sum() / 10**18
    )

    return {
        "source": "Processed Optimism blockchain dataset",
        "total_operations": total_operations,
        "unique_users": unique_users,
        "unique_paymasters": unique_paymasters,
        "successful_operations": successful_operations,
        "failed_operations": failed_operations,
        "total_gas_spending_eth": total_gas_spending_eth,
        "dataset_columns": df.columns.tolist(),
    }


# ============================================================
# 13. EXISTING ALLOCATION EXPERIMENTS
# ============================================================

@app.get("/allocation/experiments")
def allocation_experiments():

    df = load_csv(EXPERIMENT_PATH)

    return {
        "experiment_source": EXPERIMENT_PATH.name,
        "total_experiments": len(df),
        "results": dataframe_to_records(df),
    }


# ============================================================
# 14. GLOBAL CHRONOLOGICAL VALIDATION
# ============================================================

@app.get("/allocation/global-validation")
def global_validation():

    df = load_csv(GLOBAL_VALIDATION_PATH)

    return {
        "validation_source": GLOBAL_VALIDATION_PATH.name,
        "total_configurations": len(df),
        "results": dataframe_to_records(df),
    }


# ============================================================
# 15. GLOBAL VALIDATION BY PAYMASTER
# ============================================================

@app.get("/allocation/global-validation/by-paymaster")
def global_validation_by_paymaster():

    df = load_csv(GLOBAL_PAYMASTER_PATH)

    return {
        "source": GLOBAL_PAYMASTER_PATH.name,
        "total_records": len(df),
        "results": dataframe_to_records(df),
    }


# ============================================================
# 16. IMPROVED SPONSORSHIP ALLOCATION
# ============================================================

@app.post("/allocation/decide")
def decide_sponsorship(request: AllocationRequest):
    """
    Make sequential simulated sponsorship decisions.

    FCFS:
        Processes candidates in submission order.

    FAIRNESS:
        Prioritizes candidates with fewer previous sponsorships.

    PROPOSED:
        Combines fairness, historical spending, and
        candidate-specific predicted gas cost.

    No real Paymaster balances are accessed.
    No blockchain transactions are sent.
    """

    # --------------------------------------------------------
    # A. VALIDATE INPUTS
    # --------------------------------------------------------

    paymaster = validate_address(request.paymaster)

    candidates = [
        validate_address(candidate.sender)
        for candidate in request.candidates
    ]

    # --------------------------------------------------------
    # B. LOAD HISTORICAL DATA
    # --------------------------------------------------------

    df = load_processed_dataset()

    paymaster_history = df[
        df["paymaster"] == paymaster
    ].copy()

    paymaster_history = paymaster_history.sort_values(
        ["block_number", "log_index"],
        kind="mergesort",
    )

    # --------------------------------------------------------
    # C. PAYMASTER-LEVEL GAS PREDICTION
    # --------------------------------------------------------

    (
        paymaster_predicted_cost,
        prediction_method,
        prediction_history_size,
    ) = predict_paymaster_gas_cost(
        df,
        paymaster,
    )

    if not paymaster_history.empty:

        paymaster_median = float(
            paymaster_history["gas_cost_eth"].median()
        )

    else:

        paymaster_median = float(
            df["gas_cost_eth"].median()
        )

    paymaster_median = max(
        paymaster_median,
        1e-18,
    )

    paymaster_reference = max(
        paymaster_predicted_cost,
        1e-18,
    )

    # --------------------------------------------------------
    # D. INITIALIZE USER HISTORY
    # --------------------------------------------------------

    user_counts = (
        paymaster_history.groupby("sender")
        .size()
        .to_dict()
    )

    user_spending = (
        paymaster_history.groupby("sender")["gas_cost_eth"]
        .sum()
        .to_dict()
    )

    typical_cost = max(
        paymaster_predicted_cost,
        1e-18,
    )

    # --------------------------------------------------------
    # E. INITIALIZE SIMULATED BUDGET
    # --------------------------------------------------------

    initial_budget = float(
        request.simulated_budget_eth
    )

    remaining_budget = initial_budget

    # --------------------------------------------------------
    # F. INITIALIZE CANDIDATE QUEUE
    # --------------------------------------------------------

    pending = list(
        enumerate(candidates)
    )

    decisions = []

    decision_order = 0

    # --------------------------------------------------------
    # G. SEQUENTIAL ALLOCATION
    # --------------------------------------------------------

    while pending:

        scored_candidates = []

        for original_index, sender in pending:

            previous_count = int(
                user_counts.get(sender, 0)
            )

            previous_spending = float(
                user_spending.get(sender, 0.0)
            )

            # Fairness score:
            # Higher score for users with fewer previous
            # sponsorships.

            fairness_score = (
                1.0 / (1.0 + previous_count)
            )

            # Spending score:
            # Higher score for users with lower historical
            # sponsorship spending.

            spending_score = (
                1.0 / (
                    1.0 + previous_spending / typical_cost
                )
            )

            # Candidate-specific cost prediction.

            (
                candidate_cost,
                candidate_prediction_method,
                sender_history_size,
            ) = predict_candidate_gas_cost(
                sender,
                paymaster_history,
                paymaster_reference,
                paymaster_median,
            )

            # Cost score:
            # Lower predicted gas cost receives a higher score.

            cost_score = min(
                1.0,
                paymaster_reference / max(candidate_cost, 1e-18),
            )

            # Weighted proposed score.

            proposed_score = (
                FAIRNESS_WEIGHT * fairness_score
                + SPENDING_WEIGHT * spending_score
                + COST_WEIGHT * cost_score
            )

            scored_candidates.append({
                "original_index": original_index,
                "sender": sender,
                "fairness_score": fairness_score,
                "spending_score": spending_score,
                "cost_score": cost_score,
                "proposed_score": proposed_score,
                "predicted_cost": candidate_cost,
                "prediction_method": candidate_prediction_method,
                "sender_history_size": sender_history_size,
            })

        # ----------------------------------------------------
        # H. SELECT CANDIDATE USING REQUESTED STRATEGY
        # ----------------------------------------------------

        if request.strategy == "FCFS":

            selected = min(
                scored_candidates,
                key=lambda item: item["original_index"],
            )

        elif request.strategy == "FAIRNESS":

            selected = min(
                scored_candidates,
                key=lambda item: (
                    -item["fairness_score"],
                    item["original_index"],
                ),
            )

        else:

            selected = min(
                scored_candidates,
                key=lambda item: (
                    -item["proposed_score"],
                    item["original_index"],
                ),
            )

        # ----------------------------------------------------
        # I. REMOVE SELECTED CANDIDATE FROM QUEUE
        # ----------------------------------------------------

        selected_index = selected["original_index"]

        pending = [
            item
            for item in pending
            if item[0] != selected_index
        ]

        decision_order += 1

        # ----------------------------------------------------
        # J. CALCULATE RESERVATION
        # ----------------------------------------------------

        candidate_predicted_cost = selected["predicted_cost"]

        reserve_per_operation = (
            candidate_predicted_cost
            * request.safety_multiplier
        )

        budget_before = remaining_budget

        accepted = (
            reserve_per_operation
            <= remaining_budget + BUDGET_TOLERANCE
        )

        if accepted:

            remaining_budget = max(
                0.0,
                remaining_budget - reserve_per_operation,
            )

            sender = selected["sender"]

            # Update simulated user history after acceptance.

            user_counts[sender] = (
                user_counts.get(sender, 0) + 1
            )

            # Update simulated spending using predicted cost.
            # This is NOT an observed transaction cost.

            user_spending[sender] = (
                user_spending.get(sender, 0.0)
                + candidate_predicted_cost
            )

        # ----------------------------------------------------
        # K. RECORD DECISION
        # ----------------------------------------------------

        decisions.append({
            "decision_order": decision_order,
            "candidate_index": selected_index,
            "sender": selected["sender"],
            "accepted": bool(accepted),

            "reason": (
                "Sufficient simulated reservation budget"
                if accepted
                else "Insufficient simulated reservation budget"
            ),

            "predicted_gas_cost_eth": candidate_predicted_cost,

            "candidate_prediction_method": (
                selected["prediction_method"]
            ),

            "sender_history_size": (
                selected["sender_history_size"]
            ),

            "safety_multiplier": request.safety_multiplier,

            "required_reserve_eth": reserve_per_operation,

            "reserved_gas_eth": (
                reserve_per_operation if accepted else 0.0
            ),

            "budget_before_eth": budget_before,
            "budget_after_eth": remaining_budget,

            "fairness_score": selected["fairness_score"],
            "spending_score": selected["spending_score"],
            "cost_score": selected["cost_score"],
            "proposed_score": selected["proposed_score"],
        })

    # --------------------------------------------------------
    # L. FINAL BUDGET VALIDATION
    # --------------------------------------------------------

    total_reserved = sum(
        item["reserved_gas_eth"]
        for item in decisions
    )

    if (
        total_reserved
        > initial_budget + BUDGET_TOLERANCE
    ):
        raise HTTPException(
            status_code=500,
            detail="Internal reservation budget validation failed.",
        )

    accepted_count = sum(
        1 for item in decisions
        if item["accepted"]
    )

    rejected_count = (
        len(decisions) - accepted_count
    )

    # --------------------------------------------------------
    # M. RESPONSE
    # --------------------------------------------------------

    return {
        "mode": "SIMULATION_ONLY",

        "warning": (
            "This API does not send blockchain transactions, "
            "access actual Paymaster wallet balances, or guarantee "
            "that actual gas spending will remain within budget."
        ),

        "paymaster": paymaster,
        "strategy": request.strategy,

        "prediction_method": prediction_method,
        "prediction_history_size": prediction_history_size,

        "predicted_cost_eth": paymaster_predicted_cost,

        "safety_multiplier": request.safety_multiplier,

        "required_reserve_per_operation_eth": (
            paymaster_predicted_cost * request.safety_multiplier
        ),

        "initial_simulated_budget_eth": initial_budget,
        "total_reserved_eth": total_reserved,
        "remaining_simulated_budget_eth": remaining_budget,

        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "total_candidates": len(candidates),

        "decisions": decisions,
    }

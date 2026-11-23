
import pandas as pd
import requests
import time
from pathlib import Path

# ------------------------------------------------
# FILE LOCATIONS
# ------------------------------------------------

INPUT_FILE = Path("data/raw/userops.csv")
OUTPUT_FILE = Path("data/processed/processed_userops.csv")

RPC_URL = "https://mainnet.optimism.io"

# ------------------------------------------------
# LOAD REAL BLOCKCHAIN DATA
# ------------------------------------------------

print("ERC-4337 DATASET ENRICHMENT")
print("===========================")

if not INPUT_FILE.exists():
    raise FileNotFoundError("Raw dataset not found!")

df = pd.read_csv(INPUT_FILE)

print("Original records:", len(df))

# Check required columns
required_columns = [
    "block_number",
    "actualGasCost",
    "actualGasUsed",
    "sender",
    "paymaster",
    "success",
    "log_index"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Convert numerical columns
df["block_number"] = pd.to_numeric(
    df["block_number"], errors="raise"
)

df["actualGasCost"] = pd.to_numeric(
    df["actualGasCost"], errors="raise"
)

df["actualGasUsed"] = pd.to_numeric(
    df["actualGasUsed"], errors="raise"
)

df["log_index"] = pd.to_numeric(
    df["log_index"], errors="raise"
)

# Standardize addresses
df["sender"] = df["sender"].astype(str).str.lower()
df["paymaster"] = df["paymaster"].astype(str).str.lower()

# Convert success values into boolean values
df["success"] = (
    df["success"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "true": True,
        "false": False,
        "1": True,
        "0": False
    })
)

if df["success"].isna().any():
    raise ValueError(
        "Unexpected or missing values found in success column"
    )

# ------------------------------------------------
# ADD BLOCK TIMESTAMPS FROM OPTIMISM RPC
# ------------------------------------------------

session = requests.Session()
timestamp_cache = {}

def get_block_timestamp(block_number):

    if block_number in timestamp_cache:
        return timestamp_cache[block_number]

    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getBlockByNumber",
        "params": [hex(int(block_number)), False],
        "id": 1
    }

    for attempt in range(4):
        try:
            response = session.post(
                RPC_URL,
                json=payload,
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            if "error" in result:
                raise RuntimeError(result["error"])

            block = result.get("result")

            if block is None:
                raise RuntimeError("Block not found")

            timestamp = int(block["timestamp"], 16)

            timestamp_cache[block_number] = timestamp

            return timestamp

        except Exception:
            if attempt == 3:
                raise

            time.sleep(2 ** attempt)


print("\nRetrieving block timestamps...")

unique_blocks = sorted(df["block_number"].unique())

for index, block_number in enumerate(unique_blocks, start=1):

    get_block_timestamp(block_number)

    if index % 100 == 0:
        print(
            f"Processed {index}/{len(unique_blocks)} blocks"
        )

    time.sleep(0.05)

df["block_timestamp"] = df["block_number"].map(
    timestamp_cache
)

df["block_timestamp"] = pd.to_datetime(
    df["block_timestamp"],
    unit="s",
    utc=True
)

# ------------------------------------------------
# CALCULATE ACTUAL GAS COST IN ETH
# ------------------------------------------------

df["gas_cost_eth"] = (
    df["actualGasCost"] / 10**18
)

# ------------------------------------------------
# SORT CHRONOLOGICALLY
# ------------------------------------------------

df = df.sort_values(
    ["block_number", "log_index"]
).reset_index(drop=True)

# ------------------------------------------------
# USER HISTORICAL FEATURES
# ------------------------------------------------

user_groups = df.groupby("sender", sort=False)

# Number of earlier sponsored operations
df["previous_sponsorship_count"] = (
    user_groups.cumcount()
)

# Earlier total sponsorship cost
df["previous_sponsorship_cost"] = (
    user_groups["gas_cost_eth"].cumsum()
    - df["gas_cost_eth"]
)

# Time since previous sponsorship
df["time_since_last_sponsorship"] = (
    user_groups["block_timestamp"]
    .diff()
    .dt.total_seconds()
)

# ------------------------------------------------
# PAYMASTER HISTORICAL FEATURES
# ------------------------------------------------

paymaster_groups = df.groupby("paymaster", sort=False)

# Number of earlier operations for this Paymaster
df["paymaster_previous_operation_count"] = (
    paymaster_groups.cumcount()
)

# Earlier total spending by this Paymaster
df["paymaster_previous_spending"] = (
    paymaster_groups["gas_cost_eth"].cumsum()
    - df["gas_cost_eth"]
)

# ------------------------------------------------
# SUCCESSFUL-ONLY HISTORICAL FEATURES
# ------------------------------------------------

# Cost of successful operations only.
# Failed operations can still consume gas, so the original
# gas_cost_eth column is retained separately.

df["successful_gas_cost_eth"] = (
    df["gas_cost_eth"].where(df["success"], 0)
)

# Previous successful sponsorship count for each user
df["previous_successful_sponsorship_count"] = (
    df.groupby("sender", sort=False)["success"]
    .transform(
        lambda s: s.astype(int).cumsum() - s.astype(int)
    )
)

# Previous successful sponsorship cost for each user
df["previous_successful_sponsorship_cost"] = (
    df.groupby("sender", sort=False)["successful_gas_cost_eth"]
    .cumsum()
    - df["successful_gas_cost_eth"]
)

# Previous successful operation count for each Paymaster
df["paymaster_previous_successful_operation_count"] = (
    df.groupby("paymaster", sort=False)["success"]
    .transform(
        lambda s: s.astype(int).cumsum() - s.astype(int)
    )
)

# Previous successful spending for each Paymaster
df["paymaster_previous_successful_spending"] = (
    df.groupby("paymaster", sort=False)["successful_gas_cost_eth"]
    .cumsum()
    - df["successful_gas_cost_eth"]
)

# ------------------------------------------------
# SAVE PROCESSED DATASET
# ------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(OUTPUT_FILE, index=False)

# ------------------------------------------------
# DISPLAY SUMMARY
# ------------------------------------------------

print("\nDATASET ENRICHMENT COMPLETED")
print("============================")

print("Total operations:", len(df))
print("Unique users:", df["sender"].nunique())
print("Unique Paymasters:", df["paymaster"].nunique())

print(
    "Successful operations:",
    int(df["success"].sum())
)

print(
    "Failed operations:",
    int((~df["success"]).sum())
)

print("\nNew columns added:")
print("block_timestamp")
print("gas_cost_eth")
print("previous_sponsorship_count")
print("previous_sponsorship_cost")
print("time_since_last_sponsorship")
print("paymaster_previous_operation_count")
print("paymaster_previous_spending")
print("successful_gas_cost_eth")
print("previous_successful_sponsorship_count")
print("previous_successful_sponsorship_cost")
print("paymaster_previous_successful_operation_count")
print("paymaster_previous_successful_spending")

print("\nProcessed dataset saved:")
print(OUTPUT_FILE)

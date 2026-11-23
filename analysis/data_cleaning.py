
import pandas as pd
from pathlib import Path

# Location of the downloaded ERC-4337 dataset
DATA_FILE = Path("data/raw/userops.csv")

print("ERC-4337 DATASET PROCESSING")
print("==========================")

# Check whether the dataset exists
if not DATA_FILE.exists():
    print("Dataset not found!")
    print("Expected file:", DATA_FILE)
    print("Please download userops.csv from BigQuery.")
    raise SystemExit(1)

# Read the dataset
df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully!")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nAvailable columns:")
print(df.columns.tolist())

# Check required fields
required = [
    "sender",
    "paymaster",
    "actualGasCost",
    "actualGasUsed",
    "success"
]

missing = [column for column in required if column not in df.columns]

if missing:
    print("\nMissing required columns:", missing)
    raise SystemExit(1)

# Clean addresses
df["sender"] = df["sender"].astype(str).str.lower()
df["paymaster"] = df["paymaster"].astype(str).str.lower()

# Convert gas fields to numeric
df["actualGasCost"] = pd.to_numeric(
    df["actualGasCost"], errors="coerce"
)

df["actualGasUsed"] = pd.to_numeric(
    df["actualGasUsed"], errors="coerce"
)

# Remove rows with missing essential values
df = df.dropna(
    subset=["sender", "paymaster", "actualGasCost", "actualGasUsed"]
)

# Convert Wei to ETH
df["gas_cost_eth"] = df["actualGasCost"] / 10**18

# Save processed data
output = Path("data/processed/cleaned_userops.csv")
df.to_csv(output, index=False)

print("\nData cleaning completed!")
print("Valid operations:", len(df))
print("Total gas cost:", round(df["gas_cost_eth"].sum(), 8), "ETH")
print("Processed dataset saved to:", output)

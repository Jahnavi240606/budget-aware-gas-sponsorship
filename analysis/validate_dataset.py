
import pandas as pd

FILE = "data/processed/processed_userops.csv"

df = pd.read_csv(FILE)

print("ERC-4337 DATASET VALIDATION")
print("===========================")

print("\nDataset shape:")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nSuccess distribution:")
print(df["success"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate UserOperation hashes:")
print(df["userop_hash"].duplicated().sum())

print("\nUnique users:", df["sender"].nunique())
print("Unique Paymasters:", df["paymaster"].nunique())

print("\nTotal gas spending (ETH):")
print(round(df["gas_cost_eth"].sum(), 10))

print("\nSuccessful gas spending (ETH):")
print(round(df["successful_gas_cost_eth"].sum(), 10))

print("\nFailed operation gas spending (ETH):")
failed_cost = df.loc[~df["success"], "gas_cost_eth"].sum()
print(round(failed_cost, 10))

print("\nSuccessful-only user history:")
print(
    df["previous_successful_sponsorship_count"].describe()
)

print("\nSuccessful-only Paymaster spending:")
print(
    df["paymaster_previous_successful_spending"].describe()
)

print("\nChecking historical feature consistency:")

assert (
    df["previous_successful_sponsorship_count"]
    <= df["previous_sponsorship_count"]
).all()

assert (
    df["paymaster_previous_successful_operation_count"]
    <= df["paymaster_previous_operation_count"]
).all()

assert (
    df["successful_gas_cost_eth"] <= df["gas_cost_eth"] + 1e-15
).all()

print("Historical feature checks passed!")

print("\nVALIDATION COMPLETED")

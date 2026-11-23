
import csv
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

# Optimism Mainnet public RPC
RPC_URL = "https://mainnet.optimism.io"

# Official ERC-4337 EntryPoint addresses
ENTRY_POINTS = [
    "0x5ff137d4b0fdcd49dca30c7cf57e578a026d2789",  # v0.6
    "0x0000000071727de22e5e9d8baf0edac6f37da032",  # v0.7
    "0x4337084d9e255ff0702461cf8895ce9e3b5ff108"   # v0.8
]

# UserOperationEvent event signature
EVENT_TOPIC = (
    "0x49628fd1471006c1482da88028e9ce4dbb080b"
    "815c9b0344d39e5a8e6ec1419f"
)

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

# Initial collection settings
BLOCK_WINDOW = 20000
CHUNK_SIZE = 500

OUTPUT_FILE = Path("data/raw/userops.csv")


def rpc_call(method, params):
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1
    }

    request = urllib.request.Request(
        RPC_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.loads(response.read())

    if "error" in result:
        raise RuntimeError(result["error"])

    return result["result"]


def main():
    print("ERC-4337 REAL DATA COLLECTION")
    print("=============================")

    latest_block = int(rpc_call("eth_blockNumber", []), 16)

    start_block = max(0, latest_block - BLOCK_WINDOW)

    print("Latest block:", latest_block)
    print("Starting block:", start_block)

    # Check which EntryPoints are deployed
    deployed = []

    for address in ENTRY_POINTS:
        code = rpc_call(
            "eth_getCode",
            [address, "latest"]
        )

        if code != "0x":
            deployed.append(address)

    if not deployed:
        print("No EntryPoint contracts found.")
        return

    print("EntryPoints found:", len(deployed))

    all_rows = []

    # Collect event logs in small block ranges
    for start in range(start_block, latest_block + 1, CHUNK_SIZE):

        end = min(start + CHUNK_SIZE - 1, latest_block)

        print(f"Scanning blocks {start} to {end}")

        logs = rpc_call(
            "eth_getLogs",
            [{
                "fromBlock": hex(start),
                "toBlock": hex(end),
                "address": deployed,
                "topics": [EVENT_TOPIC]
            }]
        )

        for log in logs:
            topics = log["topics"]
            data = log["data"][2:]

            if len(topics) < 4 or len(data) < 256:
                continue

            sender = "0x" + topics[2][-40:]
            paymaster = "0x" + topics[3][-40:]

            # Keep only Paymaster-sponsored operations
            if paymaster.lower() == ZERO_ADDRESS:
                continue

            nonce = int(data[0:64], 16)
            success = bool(int(data[64:128], 16))
            actual_gas_cost = int(data[128:192], 16)
            actual_gas_used = int(data[192:256], 16)

            all_rows.append({
                "chain": "optimism",
                "chain_id": 10,
                "block_number": int(log["blockNumber"], 16),
                "transaction_hash": log["transactionHash"],
                "log_index": int(log["logIndex"], 16),
                "userop_hash": topics[1],
                "sender": sender.lower(),
                "paymaster": paymaster.lower(),
                "nonce": nonce,
                "success": success,
                "actualGasCost": actual_gas_cost,
                "actualGasUsed": actual_gas_used,
                "entrypoint": log["address"].lower()
            })

        time.sleep(0.1)

    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "chain",
        "chain_id",
        "block_number",
        "transaction_hash",
        "log_index",
        "userop_hash",
        "sender",
        "paymaster",
        "nonce",
        "success",
        "actualGasCost",
        "actualGasUsed",
        "entrypoint"
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nCOLLECTION COMPLETED")
    print("====================")
    print("Sponsored operations:", len(all_rows))
    print("Unique users:", len(set(row["sender"] for row in all_rows)))
    print("Unique Paymasters:", len(set(row["paymaster"] for row in all_rows)))
    print("CSV saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nCollection failed:", error)
        print("Check your internet connection and try again.")

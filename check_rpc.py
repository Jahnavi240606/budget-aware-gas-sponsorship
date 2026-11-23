
import json
import urllib.request

RPC_URL = "https://mainnet.optimism.io"

def rpc_call(method):
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": [],
        "id": 1
    }

    request = urllib.request.Request(
        RPC_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())

print("TESTING FREE OPTIMISM BLOCKCHAIN CONNECTION")

chain = rpc_call("eth_chainId")
block = rpc_call("eth_blockNumber")

print("Chain ID:", int(chain["result"], 16))
print("Latest block:", int(block["result"], 16))
print("Connection successful!")

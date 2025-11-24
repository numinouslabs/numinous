import os
import sys

from async_host import AsyncValidatorSigningProxy
from bittensor_wallet import Wallet

# Load wallet from mounted directory
wallet_name = os.environ["VALIDATOR_WALLET_NAME"]
wallet_path = os.environ["VALIDATOR_WALLET_PATH"]
wallet_hotkey = os.environ["VALIDATOR_WALLET_HOTKEY"]

wallet = Wallet(name=wallet_name, path=wallet_path, hotkey=wallet_hotkey)
gateway_url = os.environ["GATEWAY_URL"]
validator_version = os.environ.get("VALIDATOR_VERSION", "unknown")

print(f"[SIGNING-PROXY] Starting with wallet: {wallet.hotkey.ss58_address}", flush=True)
print(f"[SIGNING-PROXY] Gateway: {gateway_url}", flush=True)
print(f"[SIGNING-PROXY] Version: {validator_version}", flush=True)

# Create and start server using the class
proxy_server = AsyncValidatorSigningProxy(
    wallet=wallet,
    proxy_upstream_url=gateway_url,
    port=8888,
)

print("[SIGNING-PROXY] ✓ Ready on 0.0.0.0:8888", flush=True)
sys.stdout.flush()

proxy_server.start()

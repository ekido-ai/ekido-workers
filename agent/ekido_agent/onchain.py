"""Puts a provider's listing on-chain from its own wallet: ProviderRegistry.register(metadataHash).

After this, anyone can check what the coordinator shows about the provider against what the provider
itself committed, and see the provider even if the coordinator stopped listing it. It costs a little test
ETH for gas. Without any, joining still works; the listing just stays off-chain until the wallet is funded.

Standard library plus eth-account only, so the provider kit needs nothing new.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass

from eth_account import Account
from eth_utils import keccak, to_checksum_address

FAUCET_HINT = "Get free Base Sepolia ETH from a faucet (search 'Base Sepolia faucet'), then run join again to list on-chain."
GET_PROVIDER = keccak(text="getProvider(address)")[:4]
REGISTER = keccak(text="register(bytes32)")[:4]
UPDATE = keccak(text="updateMetadata(bytes32)")[:4]


class RpcError(RuntimeError):
    pass


@dataclass
class Outcome:
    status: str  # "registered", "updated", "already", "no-gas", "failed"
    detail: str
    tx: str | None = None


class Rpc:
    def __init__(self, url: str, timeout: float = 15.0):
        self.url = url
        self.timeout = timeout
        self._id = 0

    def call(self, method: str, params: list):
        self._id += 1
        body = json.dumps({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}).encode()
        # Some public RPCs refuse Python's default User-Agent.
        request = urllib.request.Request(self.url, data=body, headers={"content-type": "application/json", "user-agent": "gpuchain-agent/1"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            reply = json.loads(response.read())
        if "error" in reply:
            raise RpcError(str(reply["error"].get("message", reply["error"])))
        return reply["result"]


def _word(address_or_hash: str) -> bytes:
    raw = bytes.fromhex(address_or_hash.removeprefix("0x"))
    return raw.rjust(32, b"\x00")


def read_listing(rpc: Rpc, registry: str, wallet: str) -> tuple[bool, str]:
    """(registered, metadataHash) as the registry holds them now."""
    data = "0x" + (GET_PROVIDER + _word(wallet)).hex()
    out = bytes.fromhex(rpc.call("eth_call", [{"to": registry, "data": data}, "latest"]).removeprefix("0x"))
    if len(out) < 96:
        raise RpcError("unexpected getProvider reply")
    return out[31] == 1, "0x" + out[32:64].hex()


def build_tx(rpc: Rpc, chain_id: int, sender: str, registry: str, data: bytes) -> dict:
    registry = to_checksum_address(registry)  # eth-account refuses a lowercase "to".
    nonce = int(rpc.call("eth_getTransactionCount", [sender, "pending"]), 16)
    base = int(rpc.call("eth_getBlockByNumber", ["latest", False])["baseFeePerGas"], 16)
    tip = int(rpc.call("eth_maxPriorityFeePerGas", []), 16)
    call = {"from": sender, "to": registry, "data": "0x" + data.hex()}
    gas = int(rpc.call("eth_estimateGas", [call]), 16)
    return {
        "type": 2,
        "chainId": chain_id,
        "nonce": nonce,
        "to": registry,
        "value": 0,
        "data": "0x" + data.hex(),
        "gas": gas * 12 // 10,
        "maxPriorityFeePerGas": tip,
        "maxFeePerGas": base * 2 + tip,
    }


def wait_receipt(rpc: Rpc, tx_hash: str, timeout: float = 90.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        receipt = rpc.call("eth_getTransactionReceipt", [tx_hash])
        if receipt:
            return receipt
        time.sleep(2)
    raise RpcError(f"no receipt for {tx_hash} after {int(timeout)} s")


def list_on_chain(rpc: Rpc, chain_id: int, registry: str, private_key: str, meta_hash: str) -> Outcome:
    """Registers, or updates the hash if the terms changed. Never raises: joining must not fail because of this."""
    try:
        account = Account.from_key(private_key)
        if int(rpc.call("eth_chainId", []), 16) != chain_id:
            return Outcome("failed", "The RPC is on a different chain, so nothing was sent.")
        registered, current = read_listing(rpc, registry, account.address)
        if registered and current.lower() == meta_hash.lower():
            return Outcome("already", "Your listing on-chain already matches these terms.")
        if int(rpc.call("eth_getBalance", [account.address, "latest"]), 16) == 0:
            return Outcome("no-gas", FAUCET_HINT)
        data = (UPDATE if registered else REGISTER) + _word(meta_hash)
        tx = build_tx(rpc, chain_id, account.address, registry, data)
        signed = Account.sign_transaction(tx, private_key)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        tx_hash = rpc.call("eth_sendRawTransaction", ["0x" + bytes(raw).hex()])
        receipt = wait_receipt(rpc, tx_hash)
        if int(receipt["status"], 16) != 1:
            return Outcome("failed", "The registry rejected the transaction.", tx_hash)
        return Outcome("updated" if registered else "registered", "Your terms are now committed on-chain.", tx_hash)
    except Exception as exc:  # A flaky public RPC must not stop the provider from working.
        return Outcome("failed", f"Couldn't list on-chain this time ({type(exc).__name__}: {str(exc)[:160]}). Joining continues.")

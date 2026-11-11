"""Read payout trust facts from a verifier-configured JSON-RPC and escrow."""
from __future__ import annotations

import json
import re
import urllib.request
from decimal import Decimal

from eth_abi import decode
from eth_utils import keccak

ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}\Z")
BYTES32 = re.compile(r"0x[0-9a-fA-F]{64}\Z")
JOB_TUPLE = "(address,uint64,uint8,uint32,uint64,uint64,uint64,bool,uint256,uint256,uint256,bytes32,bytes32,bytes32,address[],uint256[],address)"


class ChainReadError(ValueError):
    pass


def _call_rpc(rpc_url: str, method: str, params: list):
    data = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    request = urllib.request.Request(rpc_url, data=data, headers={
        "Content-Type": "application/json",
        "User-Agent": "GPUChainVerifier/1.0",
    })
    with urllib.request.urlopen(request, timeout=15) as response:
        raw = response.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ChainReadError("RPC response exceeds the size limit")
    payload = json.loads(raw)
    if payload.get("id") != 1 or "result" not in payload or "error" in payload:
        raise ChainReadError(f"invalid RPC response to {method}")
    return payload["result"]


def _eth_call(rpc_url: str, contract: str, signature: str, arguments: bytes):
    selector = keccak(text=signature)[:4]
    result = _call_rpc(rpc_url, "eth_call", [{"to": contract, "data": "0x" + (selector + arguments).hex()}, "latest"])
    if not isinstance(result, str) or not result.startswith("0x"):
        raise ChainReadError("invalid eth_call result")
    try:
        return bytes.fromhex(result[2:])
    except ValueError as exc:
        raise ChainReadError("invalid eth_call hexadecimal") from exc


def read_chain_context(rpc_url: str, chain_id: int, contract: str, onchain_job_id: str, signers: list[str]) -> dict:
    """Read active job and signer eligibility from an independently configured chain.

    A caller may look up a proposed onchain_job_id, but must authenticate its
    binding to the off-chain job before signing. The payout checker supports a
    client-signed binding checked against the client returned here.
    """
    if type(chain_id) is not int or chain_id <= 0 or not isinstance(contract, str) or not ADDRESS.fullmatch(contract) or not isinstance(onchain_job_id, str) or not BYTES32.fullmatch(onchain_job_id):
        raise ChainReadError("invalid trusted chain configuration")
    if len(signers) > 16 * 257 or any(not isinstance(s, str) or not ADDRESS.fullmatch(s) for s in signers) or len({s.lower() for s in signers}) > 16:
        raise ChainReadError("invalid proposed verifier addresses")
    reported_chain = _call_rpc(rpc_url, "eth_chainId", [])
    try:
        if int(reported_chain, 16) != chain_id:
            raise ChainReadError("RPC is connected to a different chain")
        raw = decode([JOB_TUPLE], _eth_call(rpc_url, contract, "getJob(bytes32)", bytes.fromhex(onchain_job_id[2:])))[0]
        if raw[2] != 2:
            raise ChainReadError("escrow job is not active")
        set_id = raw[3]
        set_arg = set_id.to_bytes(32, "big")
        quorum = decode(["uint256"], _eth_call(rpc_url, contract, "quorumOf(uint32)", set_arg))[0]
        allowed = []
        for signer in sorted({s.lower() for s in signers}):
            args = set_arg + bytes.fromhex(signer[2:]).rjust(32, b"\0")
            if decode(["bool"], _eth_call(rpc_url, contract, "isVerifier(uint32,address)", args))[0]:
                allowed.append(signer)
        return {
            "chain_id": chain_id, "contract": contract.lower(), "onchain_job_id": onchain_job_id.lower(),
            "client": raw[0].lower(),
            "spec_hash": raw[11].hex(), "policy_hash": raw[12].hex(),
            "budget": str(Decimal(raw[8]) / Decimal(1_000_000)), "verifiers": allowed, "quorum": quorum,
        }
    except (ValueError, TypeError, IndexError, OverflowError) as exc:
        if isinstance(exc, ChainReadError):
            raise
        raise ChainReadError("malformed chain response") from exc

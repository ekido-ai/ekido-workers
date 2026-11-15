"""One command to rent out a GPU or CPU: signs up with the coordinator, then stays connected and takes work.

    python -m gpuchain_agent.join --coordinator wss://<coordinator address> --name "My RTX 4060" --price 0.002

- The provider wallet's key stays on this machine. It comes from PROVIDER_WALLET_KEY, or from a key file this
  command creates the first time (default ~/.gpuchain/provider.key). Only the address is ever sent or printed.
- The device is found automatically: CUDA if torch can see a GPU, otherwise the CPU. --device overrides it.
- Payment goes to the wallet, and only for work the verifiers accept. Testnet only: never use a wallet that
  holds real funds.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from eth_account import Account

from gpuchain_core.provider_meta import REGISTRIES, metadata_hash, provider_metadata
from gpuchain_core.signing import sign_document

from .agent import Agent, AgentConfig, DockerOptions
from .onchain import Rpc, list_on_chain

JOIN_KIND = "gpuchain.provider_join.v1"
CPU_DEFAULT_MIB = 4096  # A CPU has no memory reading the runtime can enforce, so a declared figure is used.


def load_or_create_key(key_file: Path) -> str:
    """The wallet key from the environment, or from a local file created on first use. Never printed."""
    key = os.environ.get("PROVIDER_WALLET_KEY")
    if key:
        return key
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key = "0x" + secrets.token_hex(32)
    key_file.write_text(key, encoding="utf-8")
    try:
        os.chmod(key_file, 0o600)  # Owner-only where the OS supports it.
    except OSError:
        pass  # The filesystem does not support chmod; the key file stays as created.
    print(f"Created a new testnet-only provider wallet. Its key is in {key_file}. Back it up, and never share it.")
    return key


def detect_device(requested: str, capacity_mib: int | None) -> tuple[str, int]:
    """Returns (device, capacity in MiB)."""
    if requested in ("auto", "cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                return "cuda", capacity_mib or int(total)
        except Exception:  # No torch, or a broken CUDA install: fall back to the CPU, unless CUDA was demanded.
            pass
        if requested == "cuda":
            raise SystemExit("--device cuda was asked for, but torch can't see a CUDA GPU here.")
    return "cpu", capacity_mib or CPU_DEFAULT_MIB


def join_document(name: str, wallet: str, price: str, device: str, capacity_mib: int, now: int | None = None) -> dict:
    return {
        "kind": JOIN_KIND,
        "name": name,
        "wallet": wallet,
        "price_per_step": price,
        "device": device,
        "capacity_mib": capacity_mib,
        "ts": int(now if now is not None else time.time()),
    }


def http_base(coordinator: str) -> str:
    return coordinator.replace("wss://", "https://").replace("ws://", "http://").rstrip("/")


def join(coordinator: str, document: dict, wallet_key: str) -> str:
    _, signature = sign_document(document, wallet_key)
    body = json.dumps({"document": document, "signature": signature}).encode()
    request = urllib.request.Request(http_base(coordinator) + "/providers/join", data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:500]
        raise SystemExit(f"The coordinator refused to sign this provider up ({error.code}): {detail}") from None
    except urllib.error.URLError as error:
        raise SystemExit(f"Couldn't reach the coordinator at {coordinator}: {error.reason}") from None
    print(("Signed up" if result.get("created") else "Already signed up") + f" as {result['name']} ({result['provider_id']}).")
    return result["provider_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coordinator", required=True, help="ws://host:8000 on a shared network, or wss://host through a tunnel.")
    parser.add_argument("--name", required=True, help="Shown on the dashboard, such as 'Vaibhav RTX 4060'.")
    parser.add_argument("--price", required=True, help="USDC per accepted training step, such as 0.002.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--capacity-mib", type=int, help="Memory to offer. Defaults to the whole GPU, or 4096 MiB on a CPU.")
    parser.add_argument("--key-file", type=Path, default=Path.home() / ".gpuchain" / "provider.key")
    parser.add_argument("--workdir", default="agent_runs")
    parser.add_argument("--chain", choices=("auto", "base-sepolia", "none"), default="auto", help="Where to commit your listing. auto: Base Sepolia when joining a public (wss://) coordinator.")
    parser.add_argument("--rpc", default="https://sepolia.base.org", help="Base Sepolia RPC used for the on-chain listing.")
    args = parser.parse_args()
    if not args.coordinator.startswith(("ws://", "wss://")):
        parser.error("--coordinator must start with ws:// or wss://")

    key = load_or_create_key(args.key_file)
    wallet = Account.from_key(key).address
    device, capacity = detect_device(args.device, args.capacity_mib)
    print(f"Wallet {wallet}. Offering the {'GPU' if device == 'cuda' else 'CPU'} with {capacity} MiB at {args.price} USDC a step.")
    provider_id = join(args.coordinator, join_document(args.name, wallet, args.price, device, capacity), key)
    if args.chain == "base-sepolia" or (args.chain == "auto" and args.coordinator.startswith("wss://")):
        meta = metadata_hash(provider_metadata(args.name, wallet, args.price, capacity))
        outcome = list_on_chain(Rpc(args.rpc), 84532, REGISTRIES[84532], key, meta)
        print(f"On-chain listing: {outcome.detail}" + (f" Transaction {outcome.tx}" if outcome.tx else ""))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = AgentConfig(
        coordinator=args.coordinator,
        provider_id=provider_id,
        wallet_key=key,
        worker_key="0x" + secrets.token_hex(32),
        workdir=Path(args.workdir).resolve(),
        mode="process",
        docker=DockerOptions(image="", gpu=device == "cuda", code_mounts=()),
        device=device,
        fault=None,
    )
    print("Connected. Leave this running to take work. Ctrl+C stops it.", flush=True)
    try:
        asyncio.run(Agent(config).run_forever())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()

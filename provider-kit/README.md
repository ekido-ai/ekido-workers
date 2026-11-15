# Ekido provider kit

Rent out an NVIDIA GPU, or a CPU, to the Ekido test network and earn test USDC for accepted work.

## What's in here

- `agent/`, `runtime/`, `core/`: the provider agent, the training runtime and the shared library. It's all plain Python source, so you can read every line.
- `setup.sh` / `setup.ps1`: these make a private environment in `.venv`, install PyTorch for your GPU (or the CPU), and download the model and dataset once.
- `join.sh` / `join.ps1`: these create your provider key, register your device, and start taking work.

## Steps

macOS or Linux:

```bash
./setup.sh
./join.sh --coordinator wss://gpuchain.duckdns.org/sepolia --name "My GPU" --price 0.002
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\join.ps1 --coordinator wss://gpuchain.duckdns.org/sepolia --name "My GPU" --price 0.002
```

The website's "Rent out your GPU" page builds the exact join command for you.

## Your key and your money

- **Your key:** on first run, `join` creates a test-network wallet key at `~/.gpuchain/provider.key` and prints only its address. **Never share that file.** Anyone holding it can sign work and withdraw as you.
- **What you earn:** your price for every accepted step. Work that's refused, and steps not yet saved in a checkpoint, earn nothing.
- **Getting it out:** earnings wait in the escrow contract under your address until you call `withdraw()`. See what you're owed on the website's On-chain page.

## Your listing on-chain

When you join the public network, `join` also commits your name, price and memory in the ProviderRegistry contract on Base Sepolia, from your own wallet. After that anyone can check that the coordinator shows your terms exactly as you set them, and see you even if it stopped listing you. This needs a little Base Sepolia test ETH for gas (free from a faucet). Without it, joining still works and `join` tells you how to list later. Use `--chain none` to skip it.

## Stopping

Press Ctrl+C. The job moves to another provider from its last checkpoint, and your accepted steps up to that checkpoint are still yours.

## Safety

Only approved workloads run, and only public data is used. This kit uses process mode, where jobs run as normal processes under your user, so only join a coordinator you trust. Docker mode adds isolation: no network, a read-only root and memory limits.

This is a prototype on a test network. It's unaudited and never meant for real money.

# Connecting a second GPU as a live provider

**Docker cleanup:** Agent startup no longer kills all Ekido containers by name; those names do not establish ownership. Abandoned training runtimes should stop when their leases expire, but orphaned probe containers may require manual inspection. The GPU experiment's helper cleanup is restricted to its own labelled containers. These changes have not yet been validated with Docker.

For a teammate's laptop with a different NVIDIA GPU. This is the only way to show recovery on a really different GPU live.

**Status:** tested on 5 Nov on one Windows laptop only. A coordinator bound the way `--lan` binds it, and one CPU agent using the laptop's own network address: it connected, qualified, and uploaded checkpoints, while the operator API refused that address with 401. **It hasn't run across two machines, on a second GPU, or as the full `demo_stack.py --lan` stack.** The setup steps below haven't been followed on a fresh machine either, so expect to fix something.

## What crosses the network

The other machine only needs `/agent/ws` and artifact uploads and downloads. Agents sign what they send, and artifact access uses connection-scoped tokens revoked when the agent disconnects. It can't start jobs, inject faults or read the operator API.

The connection is plain `ws://` and `http://`, so anyone on the same network can read the traffic: job details and checkpoints. Use your own phone hotspot, and public data only. Never use the venue's Wi-Fi.

## 1. On the other machine, while still online

Use Python 3.12 and a torch build that sees the GPU. From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu130   # Check this matches the GPU's driver.
pip install "transformers==5.22.1" "peft==0.21.2" safetensors datasets numpy eth-account websockets
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Download the model and data now, so nothing downloads on stage:

```bash
python -c "from transformers import AutoModel, AutoTokenizer as T; AutoModel.from_pretrained('distilbert-base-uncased'); T.from_pretrained('distilbert-base-uncased')"
python -c "from datasets import load_dataset; load_dataset('fancyzhx/ag_news')"
```

Then make a wallet for this provider. It's a testnet-only wallet, and the key stays on this machine:

```bash
python -c "from eth_account import Account; a = Account.create(); print('address', a.address); open('provider.key', 'w').write(a.key.hex())"
```

Send the **address** to Piyush. Never send `provider.key`, and don't put it in a zip.

Set the path once in each new terminal. On Windows, use `;` instead of `:`, and `set` or `$env:` instead of `export`:

```bash
export PYTHONPATH=packages/ekido_core:workers/agent:workers/runtime
export PROVIDER_WALLET_KEY=$(cat provider.key)
```

## 2. On the demo laptop

Join the same hotspot, then:

```powershell
.\.venv\Scripts\python.exe experiments\demo_stack.py --lan
ipconfig                       # Note the hotspot adapter's IPv4 address.
.\.venv\Scripts\python.exe experiments\add_remote_provider.py --name "Vaibhav RTX 4060" --wallet 0xTHE_ADDRESS --lan-ip THE_IP
```

Windows may ask whether to let Python through the firewall. Allow it on **private** networks only, and only if the hotspot is marked private. Whoever owns the laptop should do this by hand.

The script prints a command for the other machine, then waits.

## 3. Back on the other machine

Run the printed command. It looks like this:

```bash
python -m ekido_agent.agent --coordinator ws://THE_IP:8000 --provider-id THE_ID --workdir agent_runs --mode process --device cuda
```

The script on the demo laptop then says it connected, and runs a real probe. The dashboard shows the provider online, with the GPU its own agent reported.

## 4. Choosing when it runs

The scheduler picks by expected cost, so price decides the order. The script's default of 0.0025 USDC a step falls between Provider P (0.002) and Provider B (0.003). After P is rejected, the job moves to the second GPU.

## Say this in the demo

- "This provider is a separate physical machine with a different GPU." Show the GPU name on screen.
- The GPU name is what that provider's own agent reported. It isn't hardware attestation.
- It runs in process mode with weak isolation, so use it for team-run machines only.
- One successful move is evidence that a checkpoint moved and kept working. It doesn't prove the two runs are equivalent.

## If it doesn't connect

| Symptom | Check |
|---|---|
| Agent can't reach the coordinator | Both on the same hotspot? Firewall allowed? Can it open `http://THE_IP:8000/healthz`? |
| `AUTH_FAILED` | Is `PROVIDER_WALLET_KEY` the key for the address you registered? |
| Refused in qualification | Read the reason printed by the script |
| Dashboard's Start or Kill stops working | Open it at `http://127.0.0.1:3100` on the demo laptop, not by IP address |

## From a different network (untested)

Use this when the provider isn't on your hotspot. The agent connects out, so nothing needs opening on the provider's side. Only the coordinator needs a public address, so put a tunnel in front of it. **Not tried yet:** do one full run with a kill before you rely on it.

On the demo laptop, start the stack **without** `--lan`. The tunnel connects locally. Then run either of these (installing either tool is the laptop owner's decision):

```bash
cloudflared tunnel --url http://127.0.0.1:8000    # prints https://<random>.trycloudflare.com
ngrok http 8000                                   # prints https://<random>.ngrok-free.app
```

Then:

1. **Check the lock first.** `https://<host>/healthz` should return 200, and `https://<host>/jobs` should return 401. If `/jobs` returns 200, stop: the tunnel is hiding the outside client.
2. **Register the provider.** Run `experiments/add_remote_provider.py` as above, but use `--public-url wss://<host>` instead of `--lan-ip`. It prints the agent command with that address.
3. **On the provider's machine**, run the agent with `--coordinator wss://<host>`. Traffic is encrypted end to end through the tunnel.

**Limits:** a free quick tunnel can drop, or get a new address after a restart. Every checkpoint crosses the internet through your laptop. Anyone who has the address can reach `/healthz`, and attempt agent and verifier connections. Those still need valid signatures.

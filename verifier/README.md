# Verifier

An independent process that judges checkpoints and signs payouts. It runs apart from providers and from the coordinator, with its own signing key. Two of three verifiers must sign a verdict before the escrow accepts it.

## What it does

1. **Evaluates a candidate checkpoint** (`evaluate.py`). It loads the checkpoint, runs it on the held-out evaluation set in FP32 and compares it with the last accepted checkpoint using a paired bootstrap test. The test and its margin are fixed by the policy hash before training starts. The report is `PASS` or `FAIL`, with reason codes such as `NON_INFERIORITY_FAILED` and `WEIGHTS_UNCHANGED`, and is signed.
2. **Checks the payout evidence itself** (`payout_check.py`). It re-derives each provider's pay from signed manifests, signed reports and the agreed terms. The payout numbers in the request are only compared against what it computes.
3. **Reads the chain from its own RPC** (`trusted_chain.py`) for the job's budget, deadline and contract, instead of taking them from the coordinator.
4. **Signs only a checked verdict** (`secure_verdict.py`). It signs the EIP-712 verdict only when the numbers it computed match the ones it was sent, and refuses otherwise.
5. **Runs as a worker** (`worker.py`): pulls evaluation tasks with a lease and returns signed reports.

## Files

| Path | What |
|---|---|
| `ekido_verifier/evaluate.py` | Loads a checkpoint, scores it, compares it with the last accepted one, signs the report |
| `ekido_verifier/payout_check.py` | Recomputes payouts and the fee from signed evidence and refuses anything that doesn't match |
| `ekido_verifier/trusted_chain.py` | Reads job facts from a verifier-configured JSON-RPC and escrow address |
| `ekido_verifier/secure_verdict.py` | Fail-closed signing of passing verdicts |
| `ekido_verifier/worker.py` | The worker loop that claims tasks and returns reports |
| `tests/test_evaluate.py` | Poisoned, lazy and learning-rate-tampered candidates; wrong evaluation set; tampered reports |
| `tests/test_payout_check.py` | Forged payouts, recipients, rates, budgets and signatures; untrusted lineage and reports; fee not allowed by the spec |
| `tests/test_secure_verdict.py` | A wrong payout and an unreachable chain are refused; untrusted chains are refused |
| `tests/test_trusted_chain.py`, `test_checkpoint_download.py`, `test_identity_*.py` | Chain reads, checkpoint download names, evaluation-set identity |

## Limits

- **A failed-job verdict is not signed on this path.** `secure_verdict.py` refuses any verdict where `passed` is not true (`test_fails_closed_for_a_failed_job_until_refund_path_is_ready`). A rejected candidate gets a `FAIL` report from `evaluate.py`; the refund for a failed job needs a separately reviewed path that isn't here yet.
- Verifiers are run by the team in this prototype, so verification is permissioned, not trustless.
- A pass means no degradation was detected beyond the agreed margin, not that two models are identical.

## Tests

They run on CPU and need `torch`, `safetensors`, `numpy` and `eth-account`. From this folder:

```bash
PYTHONPATH=.:../../packages/ekido_core:../../workers/runtime python -m unittest discover -s tests
```

On Windows PowerShell, set `$env:PYTHONPATH = ".;..\..\packages\ekido_core;..\..\workers\runtime"` first.

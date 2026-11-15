# Bring your own code: custom training jobs (design, 7 Nov 2026)

Asked for by Piyush: anyone should be able to train their own model, with their own code and their own files, on someone else's GPU or CPU, while the provider's machine stays safe.

## What the client writes

One Python file, `job.py`, with three functions. Everything else (recovery, checkpoints, the quality test, payment) works as it does for the built-in jobs.

```python
def build_model(spec) -> torch.nn.Module:
    """forward(**batch) must return an object with .loss and .logits."""

def dataset_size(spec, files_dir) -> int:
    """How many training examples there are."""

def make_batch(spec, files_dir, indices) -> dict[str, torch.Tensor]:
    """The examples at these indices. Must include "labels". Must give the same batch for the same indices."""
```

- `files_dir` is a read-only folder holding the client's own files (datasets, tokenizers, configs).
- `spec["user_config"]` carries the client's settings.

**Why the shape is fixed:** recovery needs the same logical step to mean the same thing on every machine. So the platform, not the client's code, picks which examples go in each step, splits them into micro-batches, and saves and restores the weights. The client's code only builds the model and turns indices into tensors.

## How a provider's computer is protected

Client code is untrusted. The intended design is to run it inside the agent's Docker sandbox, which the agent already uses for built-in jobs. **Custom-job sandbox integration is not implemented or validated: do not run untrusted custom code.** The planned restrictions are:

- No network (`--network none`): code can't send data out, or download anything.
- Read-only system (`--read-only`). The code and files are mounted read-only. The only writable places are a scratch folder and the run's own work folder.
- A non-root user, every Linux capability dropped, and `no-new-privileges`.
- Limits on memory, CPUs and processes. A fork bomb or a memory leak hits the limit, not the laptop.
- Nothing else from the host is mounted.

**Implementation status:**
- The untested runtime checks `EKIDO_SANDBOX=1`. An environment variable is not proof of isolation; the process setting it must actually establish and enforce the sandbox.
- The agent's custom-job process-mode refusal, artifact download and sandbox mounts are still to be implemented.
- The job spec can contain SHA-256 hashes for code and files. End-to-end retrieval, mounting and execution of the agreed artifacts are not yet verified.

**Limits to state honestly:**
- A container shares the kernel and GPU driver with the host. It's much safer than running code directly, but it isn't a virtual machine. For strangers' code on valuable machines, use gVisor or a VM.
- The provider can read the client's code and files. Use public or non-sensitive data. This isn't private training.
- Providers need Docker for custom jobs. CPU-only and GPU providers both work, through Docker Desktop on Windows.

## The parts to build, in order

| # | Part | Status |
|---|---|---|
| 1 | Runtime: `custom-python/v1` in `workers/runtime/ekido_runtime/custom.py`. Loads `job.py` only inside the sandbox, checks the code and file hashes, then trains through the normal loop | **Written 7 Nov. No tests yet, and never run.** Write unit tests, including the refusal outside the sandbox, and an example `job.py` |
| 2 | Job spec rules for the new template (code hash, file hashes, user_config) in `ekido_core/job_spec.py` | **Written 7 Nov.** Existing tests pass, but there are no new tests for these rules |
| 3 | Agent: download the code and files by hash, mount them read-only into the container, and refuse process mode | To do |
| 4 | Coordinator: let the client upload code and files (operator API), stored by hash | To do |
| 5 | Verifier: evaluate custom jobs inside the same sandbox, on the client's own evaluation files | To do. Until then, custom jobs can't pass the quality test |
| 6 | Dashboard: a "Your own code" option on the New job form, with file pickers | To do |
| 7 | Economic rule: a client writes the evaluation, so a dishonest client could make the final test fail and get a refund after using compute. The evaluation must be fixed before training (it is, by hash), and providers should see it before accepting. Paying accepted steps on failure closes this properly | To decide |

## What it can't do yet

- Multi-GPU jobs, internet access during training, or packages outside the runtime image (torch, transformers, peft, numpy, safetensors).
- Non-classification losses on the quality test. Training can use any loss, but the verifier currently scores accuracy from `.logits` against `labels`.

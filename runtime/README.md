# Training runtime

The part of GPUChain that runs on a provider's GPU. This folder holds the training loop, the run entry point that writes and restores checkpoints, and the workloads it can train. The memory probe tries a candidate config for real, so the elastic compiler can approve it.

## The step model

A job is priced, checkpointed and verified in **logical steps**. One logical step is one optimizer update over `logical_batch_size` examples, and it means the same thing on every GPU.

A GPU with less memory can't always fit the whole logical batch, so the batch is split:

```
logical_batch_size = micro_batch x accumulation
```

`32 x 1` and `8 x 4` are the same logical step. The second runs four forward and backward passes, adds up the gradients, and then updates the weights once. `check_exec_config` refuses any pair that doesn't multiply to the logical batch size.

What stays fixed, whatever the micro-batch:

| Thing | How it's fixed | Code |
|---|---|---|
| Which examples a step sees | A function of the job seed and the step number | `logical_batch_indices` in `gpuchain_core/logical.py` |
| The gradient | Each micro-batch's mean loss is weighted by its share of the logical batch, so the sum is the full-batch mean | `accumulate_gradients` in `trainer.py` |
| Gradient clipping | Applied once, after accumulation | `run_logical_step` in `trainer.py` |
| Learning rate | A function of the step number, so there's no scheduler state to move between machines | `learning_rate` in `gpuchain_core/logical.py` |
| Weights and optimizer state | Kept at their base dtype; only autocast (`none`, `fp16`, `bf16`) changes | `amp_dtype`, `make_scaler` in `trainer.py` |

What can differ: floating-point rounding, and dropout masks when the micro-batch size changes, because each micro-batch draws its own mask from a seed of (job seed, step, micro-batch index). The tests measure the rounding difference; they don't claim the runs are bit-identical.

## Checkpoints (GPCK/1)

A checkpoint is a folder with three files:

| File | What |
|---|---|
| `trainable.safetensors` | The trainable weights |
| `optimizer.safetensors` | AdamW state per parameter (`exp_avg`, `exp_avg_sq`, `step`), keyed by parameter name. A parameter with no state has no entries |
| `manifest.json` | Step, job spec hash, exec config, canary loss, environment, the SHA-256 and size of both files, and `parent_manifest_hash` |

Pickle is never used, on save or load. The format lives in `packages/gpuchain_core/gpuchain_core/gpck.py`.

- **Write order:** tensors first, manifest last, through a temp file and a rename. A crash before the manifest exists leaves nothing that can be picked up as a checkpoint.
- **Parent link:** each manifest stores the hash of the one before it, so checkpoints form a chain back to the start of the job.
- **Load:** `load_checkpoint` checks the job spec hash, file sizes and SHA-256, then tensor names, shapes, dtypes and NaN/Inf, before any tensor reaches a model. A failure raises `CheckpointError` with a reason code such as `HASH_MISMATCH`, `MALFORMED_CHECKPOINT`, `INVALID_LINEAGE` or `NAN_OR_INF`.
- **Canary:** after a restore the runtime recomputes an eval-mode loss on a fixed batch and compares it with the value saved in the manifest. A mismatch is refused as `CANARY_MISMATCH`.
- **Empty optimizer state:** AdamW creates state for a parameter only once it has received a gradient. A trainable parameter with no state yet, before the first step or because it never got a gradient, is stored as no tensors at all: none of `<name>.exp_avg`, `<name>.exp_avg_sq` or `<name>.step` appears in `optimizer.safetensors`. A checkpoint taken before the first step has an empty optimizer file. On restore that parameter is left without state, so AdamW starts it fresh when its first gradient arrives, as it would have in an uninterrupted run.
- **Partial optimizer state:** a parameter must have all three fields or none. State with only some of them is refused on save (`BAD_CHECKPOINT`) and on load (`MALFORMED_CHECKPOINT`).
- **What the loader can't tell:** the format does not record which parameters should have state. A file that leaves out a parameter's state entirely loads as "no state" for it. The file's SHA-256 is in the manifest, so the omission is fixed by the manifest hash, but it is not refused.

Signature, fencing and lineage across attempts are checked by the caller, not here.

## Running one attempt

```bash
python -m gpuchain_runtime.run <config.json>
```

The process talks to the provider agent only through files in its work folder: it appends to `events.jsonl`, exits if the agent's `lease` file goes stale, and writes a checkpoint at the next step boundary when `checkpoint_now` appears. Exit codes: 0 finished, 2 lease lost, 3 out of memory, 4 refused checkpoint, 5 other error. Fault injection for the demo and tests (`OOM`, `POISON`, `NO_OP`, `LR_TAMPER`, `CRASH_MID_CHECKPOINT`) is always reported as injected in the events.

## Fitting a job to a GPU

When a job moves to a different GPU, the elastic compiler (`packages/gpuchain_core/gpuchain_core/compiler.py`) picks a new `micro_batch`, `accumulation` and precision. The job spec never changes, and every candidate keeps `micro_batch x accumulation` equal to the logical batch size.

- **Gates first:** a target that is offline, too old, on the wrong torch version or short of free memory is refused before any probe runs.
- **Only a real probe approves:** estimates may order the candidates, but a config is accepted only after `gpuchain_runtime.probe` has run it on the target and reported its peak memory and speed. The peak must fit under the capacity with a safety margin, and the measured speed must finish the remaining steps before the deadline.
- **Precision:** BF16 only where the GPU supports it natively, FP16 on older GPUs such as a T4, none on CPU.
- **Budget:** if the probe budget runs out, a config that was already proven is kept. With nothing proven, the target is refused.
- **Record:** `build_adaptation_manifest` lists every changed setting with its reason and the rejected candidates. Its hash, the chosen exec config and the driver version reported by `nvidia-smi` are written into each checkpoint manifest by `run.py`.

```bash
python -m gpuchain_runtime.probe <config.json> <result.json>
```

The probe runs as its own process, so a failed attempt can't leave GPU memory stuck. It runs one warm-up step and a few timed steps, then writes `ok`, `oom`, `peak_mem_mib`, `steps_per_s` and the device. Out of memory is a valid answer and still exits 0. On CPU there is no memory cap to enforce, so `peak_mem_mib` is reported as 0.0.

## Files

| Path | What |
|---|---|
| `gpuchain_runtime/trainer.py` | `run_logical_step`, gradient accumulation, AdamW setup, the canary loss used to check a restore |
| `gpuchain_runtime/run.py` | One training attempt: restore, train, checkpoint, report events |
| `gpuchain_runtime/probe.py` | Runs a candidate config for a few steps and measures memory and speed |
| `gpuchain_runtime/workloads.py` | The workload allowlist: `synthetic-linear/v1`, `distilbert-lora-agnews/v1`, `custom-python/v1` |
| `gpuchain_runtime/custom.py` | Loader for client code, which refuses to run outside the provider's Docker sandbox |
| `gpuchain_runtime/synthetic.py` | A small classifier and dataset that need no GPU and no downloads |
| `tests/test_runtime.py` | Gradient invariance, step counting, save and resume, malicious checkpoints, precision modes |
| `tests/test_probe.py` | Probe on CPU: a working config, a bad config, a workload off the allowlist |
| `tests/test_run.py` | The run process end to end on CPU: checkpoint chain, kill and resume, refused checkpoints, injected faults, stale lease |

## Tests

They run on CPU and need `torch`, `safetensors` and `numpy`. From this folder:

```bash
PYTHONPATH=.:../../packages/gpuchain_core python -m unittest discover -s tests
```

On Windows PowerShell, set `$env:PYTHONPATH = ".;..\..\packages\gpuchain_core"` first.

What they check:

- Gradients for micro-batch 16, 8, 4, 2 and 1 match micro-batch 32 to 1e-9 in FP64 and 1e-4 in FP32.
- Unequal micro-batches (20 + 12) still give the full-batch gradient, and the common divide-by-accumulation mistake is detectably wrong.
- 30 steps at `32 x 1` and 30 steps at `8 x 4` report the same step numbers and learning rates, make 30 optimizer updates each, and end with weights within 1e-8 in FP64.
- A config where `micro_batch x accumulation` isn't the logical batch size is refused.
- Save then resume with the same config gives weights exactly equal to an uninterrupted run, with and without dropout.
- Resume with micro-batch 32 changed to 4 stays within 1e-8 of the uninterrupted run in FP64.
- A parameter that never received a gradient saves and restores with no optimizer state, and the resumed run ends with weights exactly equal to an uninterrupted one. A checkpoint taken before the first step does the same.
- A flipped byte, truncated file, missing file, garbage manifest, another job's checkpoint, wrong shape, extra or missing tensor, wrong dtype, NaN, Inf and an oversized file are each refused, and a pickle renamed as a checkpoint never executes.

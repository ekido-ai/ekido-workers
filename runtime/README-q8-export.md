# Local Q8 export metadata

The existing export command can merge the checkpoint on a CUDA device and
record the F16 and Q8 files locally:

```sh
python -m ekido_runtime.export --spec spec.json --checkpoint checkpoints/step-400 \
  --out export --device cuda --quantize Q8_0 --llama-cpp /path/to/llama.cpp
```

The CLI holds an OS lock for the export directory. `export.json` records the
canonical spec and source-checkpoint manifest hashes. After a converter or
quantizer succeeds, each local GGUF is checked for a plausible header, streamed
through SHA-256 and added to `artifacts` with relative path, bytes, format and
the quantization setting passed to the tool. Metadata is replaced atomically;
if a later export step fails, earlier successful artifacts may remain recorded,
so callers must require a successful process exit before treating the whole job
as complete. A header and a tool argument do not prove the full GGUF or its
quantization is correct.

This file is **unsigned local metadata**, not the Coordinator's agreed signed
export manifest. It does not prove quality, job authorization or actual GPU
execution, encrypt an artifact, upload it, or authorize publication. Callers
must keep the directory private, check the finished file independently, and
use the eventual signed network contract and quality verdict before serving.
Direct library callers must hold `export_lock` for the entire export themselves.

Real provider export, the five-recipe Q8 score comparison, signed manifest and
staging serving remain to be tested; see `docs/q8-export-risks.md`.

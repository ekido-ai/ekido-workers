# Seed-wise distillation generation (standalone building block)

`ekido_runtime.distill_gen.run_seed_generation` calls the existing
`distill.generate_rows` filters one seed at a time, checkpoints completed seeds,
and emits `generated.jsonl` and `generation_report.json` only after all seeds
finish and at least one row is accepted. The caller passes a job ID, recipe,
seed and held-out rows, a loaded teacher and tokenizer, immutable requested
teacher and generator revisions, the requested quantization setting, and a
private job work directory. The report records hashes, settings and counts;
it does not include the seed or held-out text. The work directory and generated
file do contain customer-derived data and require access control and cleanup.
The filter counters include zeroes; annotation removals can overlap with a
later rejection, so counters are not necessarily mutually exclusive.

The checkpoint binds the job ID, input hashes and generation settings; on retry,
a different identity or damaged checkpoint fails closed. A kernel-held lock
rejects a second writer and is released if the process exits. A seed interrupted
before its checkpoint can be retried, and completed seeds are not generated a
second time. Held-out inputs, original seeds and duplicates across seeds are
filtered by normalized exact match. Seed-specific RNG initialization supports
retries, but bit-for-bit output across different GPU architectures is not
promised. If no row is usable, the run fails instead of publishing a report.

The caller must verify the loaded model, actual quantization, deployed code
revision, job authorization and work-directory ownership: the values here are
requested settings and the checkpoint hash detects accidental corruption, not
an adversarial provider rewriting both the state and its hash. A process failure
between writing the final data and report can leave the data file without a
report; consumers must require a successful completion and verify its hash
against the report. Retrying the same completed checkpoint repairs that case.

This helper is not yet wired into the agent, checkpoint transport, Coordinator
or staging, and is not the published `distill-gen/v1` network contract. Those
parts depend on the contract and must be verified separately, including real T4
kill/resume, model provenance, settlement and customer isolation. See
`docs/distill-gen-risks.md` for the initial failure analysis.

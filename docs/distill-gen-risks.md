# Distillation generation: risks before implementation

Related to [internal #133](https://github.com/ekido-ai/ekido-internal/issues/133).
The network contract is scheduled for publication on 25 November; this initial
work prepares an isolated seed-at-a-time runner and does not claim integration.

1. **Invalid or malicious inputs:** malformed seed rows, a wrong recipe, or
   teacher output containing answer leaks must be rejected and counted without
   contaminating the generated dataset.
2. **Interrupted generation:** a device may die during a seed. Resume must use
   only a complete, integrity-checked checkpoint, never duplicate an already
   committed seed or skip a partially processed one.
3. **Concurrent attempts:** two attempts might write the same destination and
   corrupt progress. A second writer must fail rather than interleave files.
4. **Cross-customer or changed inputs:** a resumed checkpoint for a different
   dataset, recipe or generation configuration must be rejected before it can
   combine another customer's seeds or output with this run.
5. **Evidence and retries:** partially generated output or a failed/refunded
   attempt must not be published as successful final evidence. Reports must
   identify the inputs and output bytes without exposing customer seed content.

The reviewer should add another failure mode during review. Tests should exercise
these cases before any staging integration is claimed.

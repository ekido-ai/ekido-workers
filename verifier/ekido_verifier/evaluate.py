"""The Quality Guard's evaluator. Runs apart from providers, with its own signing key.

It judges outcomes, not execution: a pass means this checkpoint shows no degradation beyond the
agreed margin on the frozen evaluation set. It doesn't prove the provider did the training honestly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from ekido_core.canonical import document_hash
from ekido_core.data_identity import check_split, validate_spec_identity
from ekido_core.gpck import CheckpointError, load_checkpoint, manifest_hash
from ekido_core.logical import derive_seed
from ekido_core.quality import compare_paired, decide, macro_f1, margin_for
from ekido_core.signing import sign_document, verify_document
from ekido_runtime.trainer import trainable_named_params
from ekido_runtime.workloads import build_workload

# Fixed for evaluation: the eval path never uses autocast, dropout or a device-dependent batch size.
EVAL_EXEC = {"micro_batch": 1, "accumulation": 1, "amp": "none", "grad_checkpointing": False, "attention_impl": "sdpa", "loader_workers": 0}


@dataclass(frozen=True)
class EvalSet:
    """Fixed before the job runs. Providers are never sent this. See plan.md on public datasets."""

    inputs: Mapping[str, torch.Tensor]
    labels: torch.Tensor
    hash: str
    # llm-lora/v1 only: the test rows as prompts and reference answers, judged by the recipe's pass rule.
    recipe_id: str | None = None
    prompts: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    max_prompt_tokens: int = 0


@dataclass(frozen=True)
class Evaluation:
    correct: np.ndarray
    predictions: np.ndarray
    loss: float
    score: float | None = None  # llm-lora/v1: the recipe metric's mean, reported next to the pass rate.


def hash_eval_set(inputs: Mapping[str, torch.Tensor], labels: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for name in sorted(inputs):
        digest.update(name.encode())
        digest.update(inputs[name].cpu().numpy().tobytes())
    digest.update(labels.cpu().numpy().tobytes())
    return digest.hexdigest()


def load_eval_set(spec: dict, examples: int | None = None, data_dir: Path | None = None) -> EvalSet:
    if spec["template"] == "llm-lora/v1":
        from ekido_core.recipes import eval_set_hash, recipe
        from ekido_runtime.llm import load_rows

        if data_dir is None:
            raise ValueError("llm-lora/v1 needs the folder that holds dataset.jsonl")
        # The customer's own test rows, never generated ones. Every row is used: the policy fixes this exact set.
        rows = load_rows(Path(data_dir) / "dataset.jsonl", spec["dataset"]["sha256"], split="test")
        if len(rows) != spec["dataset"]["test_rows"]:
            raise ValueError(f"dataset has {len(rows)} test rows, the spec says {spec['dataset']['test_rows']}")
        r = recipe(spec["recipe_id"])
        source, target = r.columns
        return EvalSet(
            {}, torch.zeros(len(rows)), eval_set_hash(r.recipe_id, rows), recipe_id=r.recipe_id,
            prompts=tuple(r.prompt(row[source]) for row in rows), references=tuple(row[target] for row in rows),
            max_prompt_tokens=max(1, spec["max_seq_length"] - r.max_answer_tokens),
        )

    if spec["template"] == "synthetic-linear/v1":
        from ekido_runtime.synthetic import SyntheticData

        # Same rule as the training data but a separate sample, so it measures generalization.
        source = SyntheticData(size=examples or 512, seed=spec["seed"], split=1)
        inputs = {"features": source.x}
        return EvalSet(inputs, source.y, hash_eval_set(inputs, source.y))

    if spec["template"] == "distilbert-lora-agnews/v1":
        from datasets import load_dataset
        from transformers import AutoTokenizer

        has_identity = validate_spec_identity(spec)
        tokenizer = AutoTokenizer.from_pretrained(spec["model"]["id"], revision=spec["model"]["revision"])
        dataset_spec = spec["dataset"]
        test = load_dataset(dataset_spec["id"], split="test", **({"revision": dataset_spec["revision"]} if has_identity else {}))
        if has_identity:
            check_split(test, dataset_spec["test_content_sha256"], "evaluation data")
        test = test.shuffle(seed=20261104)
        if examples:
            test = test.select(range(min(examples, len(test))))
        encoded = tokenizer(list(test["text"]), padding="max_length", truncation=True, max_length=spec["max_seq_length"], return_tensors="pt")
        inputs = {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
        labels = torch.as_tensor(list(test["label"]))
        return EvalSet(inputs, labels, hash_eval_set(inputs, labels))

    raise ValueError(f"workload not on the allowlist: {spec['template']}")


@torch.no_grad()
def predict(model: torch.nn.Module, eval_set: EvalSet, device: str, batch_size: int = 64) -> Evaluation:
    """FP32, no autocast and no dropout, always in the same order."""
    if eval_set.recipe_id is not None:
        return _predict_text(model, eval_set, device)
    model.eval()
    predictions, losses = [], []
    n = eval_set.labels.shape[0]
    for start in range(0, n, batch_size):
        batch = {k: v[start : start + batch_size].to(device) for k, v in eval_set.inputs.items()}
        labels = eval_set.labels[start : start + batch_size].to(device)
        output = model(**batch, labels=labels)
        predictions.append(output.logits.argmax(dim=-1).cpu().numpy())
        losses.append(float(output.loss.item()) * labels.shape[0])
    predicted = np.concatenate(predictions)
    return Evaluation(correct=predicted == eval_set.labels.numpy(), predictions=predicted, loss=sum(losses) / n)


def _predict_text(model: torch.nn.Module, eval_set: EvalSet, device: str) -> Evaluation:
    """Generates an answer per test prompt and marks each one passed or failed under the recipe's rule."""
    from ekido_core.recipes import passes, recipe, score
    from ekido_runtime.llm import generate_answers

    r = recipe(eval_set.recipe_id)
    answers = generate_answers(model, model.ekido_tokenizer, eval_set.prompts, max_new_tokens=r.max_answer_tokens, max_prompt_tokens=eval_set.max_prompt_tokens, device=device)
    passed = np.array([passes(r.recipe_id, a, ref) for a, ref in zip(answers, eval_set.references)], dtype=bool)
    mean = score(r.recipe_id, answers, list(eval_set.references))
    return Evaluation(correct=passed, predictions=np.array(answers, dtype=object), loss=1.0 - mean, score=mean)


def load_model(spec: dict, checkpoint_dir: Path | None, device: str) -> tuple[torch.nn.Module, dict | None, dict[str, torch.Tensor] | None]:
    """No checkpoint gives the untrained baseline. Loading goes through full GPCK validation."""
    workload = build_workload(spec, EVAL_EXEC, device, with_data=False)
    if checkpoint_dir is None:
        return workload.model, None, None
    manifest, trainable, _ = load_checkpoint(checkpoint_dir, workload.param_shapes(), expected_immutable_hash=document_hash(spec))
    named = trainable_named_params(workload.model)
    with torch.no_grad():
        for name, param in named.items():
            param.copy_(trainable[name].to(param.device))
    return workload.model, manifest, trainable


def _failed_report(job_id, checkpoint_hash, reference_hash, policy, policy_hash, is_final, device, reasons) -> dict:
    return {
        "schema_version": 1,
        "job_id": job_id,
        "checkpoint_hash": checkpoint_hash,
        "reference_checkpoint_hash": reference_hash,
        "policy_hash": policy_hash,
        "eval_set_hash": policy["eval_set_hash"],
        "eval_version_hash": policy["eval_version_hash"],
        "is_final": is_final,
        "n_examples": 1,
        "metrics": {"accuracy": 0.0, "macro_f1": 0.0, "loss": 0.0},
        "comparison": {"reference_accuracy": 0.0, "difference": 0.0, "ci_lower": -1.0, "margin": margin_for(policy, is_final), "confidence": policy["confidence"], "resamples": 1, "mcnemar_p": None},
        "checks": {"signature_valid": "INVALID_SIGNATURE" not in reasons, "lineage_valid": "INVALID_LINEAGE" not in reasons, "no_nan_inf": "NAN_OR_INF" not in reasons, "weights_changed": True},
        "verdict": "FAIL",
        "reasons": sorted(reasons),
        "device": device,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def evaluate_checkpoint(
    *,
    job_id: str,
    spec: dict,
    policy: dict,
    eval_set: EvalSet,
    checkpoint_dir: Path,
    reference_dir: Path | None,
    expected_signer: str | None,
    is_final: bool,
    device: str = "cpu",
    provider_signature: str | None = None,
    expected_parent: str | None = None,
    reference_cache: dict | None = None,
) -> dict:
    """Produces an unsigned quality report. A checkpoint that can't even be loaded gets a FAIL, not an exception.

    reference_cache keeps earlier predictions, so an unchanged reference isn't evaluated again.
    """
    policy_hash = document_hash(policy)
    if policy["eval_set_hash"] != eval_set.hash:
        raise ValueError("this evaluation set isn't the one fixed in the policy, so results wouldn't be comparable")

    reference_model, reference_manifest, reference_weights = load_model(spec, reference_dir, device)
    reference_hash = manifest_hash(reference_manifest) if reference_manifest else None
    cache_key = (eval_set.hash, reference_hash or f"baseline:{document_hash(spec)}")
    reference_eval = reference_cache.get(cache_key) if reference_cache is not None else None
    if reference_eval is None:
        reference_eval = predict(reference_model, eval_set, device)
        if reference_cache is not None:
            reference_cache[cache_key] = reference_eval
    del reference_model

    try:
        model, manifest, weights = load_model(spec, Path(checkpoint_dir), device)
    except CheckpointError as exc:
        return _failed_report(job_id, "0" * 64, reference_hash, policy, policy_hash, is_final, device, [exc.code])
    checkpoint_hash = manifest_hash(manifest)

    signature_valid = True
    if expected_signer is not None:
        signature_valid = bool(provider_signature) and verify_document(manifest, provider_signature, expected_signer)
    lineage_valid = expected_parent is None or manifest.get("parent_manifest_hash") == expected_parent
    weights_changed = True
    if reference_weights is not None:
        weights_changed = any(not torch.equal(weights[name], reference_weights[name]) for name in weights)

    candidate = predict(model, eval_set, device)
    if reference_cache is not None:
        # An accepted candidate is often the next reference, so this saves evaluating it twice.
        reference_cache[(eval_set.hash, checkpoint_hash)] = candidate
    seed = derive_seed("quality", job_id, checkpoint_hash)
    comparison = compare_paired(candidate.correct.tolist(), reference_eval.correct.tolist(), confidence=policy["confidence"], resamples=policy["bootstrap_resamples"], seed=seed)
    decision = decide(
        comparison,
        margin=margin_for(policy, is_final),
        min_absolute=policy["min_absolute"],
        is_final=is_final,
        signature_valid=signature_valid,
        lineage_valid=lineage_valid,
        no_nan_inf=True,  # load_checkpoint already refused non-finite tensors.
        weights_changed=weights_changed,
    )
    if eval_set.recipe_id is not None:
        from ekido_core.recipes import recipe

        # "accuracy" is the share of test examples that passed; "score" is the recipe metric's mean.
        metrics = {"accuracy": comparison.candidate_accuracy, "score": candidate.score, "metric": recipe(eval_set.recipe_id).metric}
    else:
        labels = eval_set.labels.numpy().tolist()
        metrics = {"accuracy": comparison.candidate_accuracy, "macro_f1": macro_f1(candidate.predictions.tolist(), labels), "loss": candidate.loss}
    return {
        "schema_version": 1,
        "job_id": job_id,
        "checkpoint_hash": checkpoint_hash,
        "reference_checkpoint_hash": reference_hash,
        "policy_hash": policy_hash,
        "eval_set_hash": eval_set.hash,
        "eval_version_hash": policy["eval_version_hash"],
        "is_final": is_final,
        "n_examples": comparison.n,
        "metrics": metrics,
        "comparison": {
            "reference_accuracy": comparison.reference_accuracy,
            "difference": comparison.difference,
            "ci_lower": comparison.ci_lower,
            "margin": margin_for(policy, is_final),
            "confidence": comparison.confidence,
            "resamples": comparison.resamples,
            "mcnemar_p": comparison.mcnemar_p,
        },
        "correctness_hash": hashlib.sha256(candidate.correct.astype(np.uint8).tobytes()).hexdigest(),
        "checks": {"signature_valid": signature_valid, "lineage_valid": lineage_valid, "no_nan_inf": True, "weights_changed": weights_changed},
        "verdict": decision.verdict,
        "reasons": decision.reasons,
        "device": device,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def sign_report(report: dict, private_key: str) -> dict:
    signer, _ = sign_document(report, private_key)
    with_signer = dict(report, signer=signer)
    _, signature = sign_document(with_signer, private_key)
    return dict(with_signer, signature=signature)


def quorum_verdict(reports: Sequence[dict], checkpoint_hash: str, quorum: int) -> str | None:
    """PASS or FAIL once `quorum` distinct verifiers agree on this checkpoint, else None.

    Reports can differ slightly across devices, so only the verdict has to match. A failure quorum
    is checked first, so ties resolve to the safe outcome.
    """
    counts: dict[str, set[str]] = {"PASS": set(), "FAIL": set()}
    for report in reports:
        if report["checkpoint_hash"] == checkpoint_hash and verify_document(report, report["signature"], report["signer"]):
            counts[report["verdict"]].add(report["signer"].lower())
    if len(counts["FAIL"]) >= quorum:
        return "FAIL"
    if len(counts["PASS"]) >= quorum:
        return "PASS"
    return None

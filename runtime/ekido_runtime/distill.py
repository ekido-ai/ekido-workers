"""distill/v1: the teacher generates, the student trains.

A customer with only a handful of examples (the Week-1 shape is 30) cannot train a
useful model directly. So the teacher — Qwen2.5-7B-Instruct, run on a provider GPU —
writes more training rows in the recipe's own format: for each seed row it composes a
new input whose correct target is the seed row's target. The student (a Qwen2.5 base,
through the ordinary llm-lora/v1 workload) trains on the generated rows.

The test set is always the user's own examples: generated rows are training data only,
are deduplicated against the seeds and the test set, and never leak into eval.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from ekido_core.job_spec import LLM_BASES
from ekido_core.recipes import recipe

from . import train as train_module

TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# Pinned so a change on Hugging Face can never silently swap the teacher. Last modified there on 12 Jan 2025.
TEACHER_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def generation_prompt(recipe_id: str, seed_row: dict) -> str:
    r = recipe(recipe_id)
    source, target = r.columns
    return (
        f"{r.instruction}\n"
        f"Write one new {r.input_name} for which the correct {r.answer_name} is: {seed_row[target]}\n"
        f"Example {r.input_name}: {seed_row[source]}\n"
        f"New {r.input_name}:"
    )


def _clean(text: str, seed_input: str) -> str | None:
    for line in text.splitlines():
        candidate = line.strip().strip('"').strip()
        # Teachers sometimes append the requested target to the new input, e.g.
        # "... The correct category for this ticket is: refund."  That exposes
        # the answer inside the student's input and must not become training data.
        candidate = re.split(
            r"(?:^|\s+)(?:the\s+)?correct\s+(?:category|sentiment|label|answer)"
            r"(?:\s+for\s+[^:\n]{1,60})?\s*(?:is\s*:|is\s+|:|=)",
            candidate, maxsplit=1, flags=re.IGNORECASE,
        )[0].strip().strip('"').strip()
        if re.match(r"^(?:category|sentiment|label|answer)\s*:", candidate, flags=re.IGNORECASE):
            continue
        if len(candidate) >= 3 and candidate.casefold() != seed_input.strip().casefold():
            return candidate
    return None


def _trailing_label(candidate: str, labels: tuple[str, ...]) -> bool:
    """Reject an answer presented as a standalone final sentence, not normal text."""
    if not labels:
        return False
    choices = '|'.join(re.escape(label) for label in labels)
    return bool(re.search(r'(?:^|[.!?]\s+)(?:' + choices + r')\s*[.!?]*\s*$', candidate, re.IGNORECASE))


@torch.no_grad()
def generate_rows(teacher_model, teacher_tokenizer, recipe_id: str, seed_rows: list[dict], *,
                  per_seed: int, device: str, max_new_tokens: int = 96, attempts: int = 3,
                  filter_counts: dict | None = None) -> list[dict]:
    """New rows in the recipe's format. Sampling is on: identical greedy rows are useless."""
    r = recipe(recipe_id)
    source, target = r.columns
    seen = {re.sub(r"\s+", " ", row[source].strip().lower()) for row in seed_rows}
    rows: list[dict] = []
    was_training = teacher_model.training
    teacher_model.eval()
    pad_id = teacher_tokenizer.pad_token_id or teacher_tokenizer.eos_token_id
    try:
        for seed_row in seed_rows:
            made = 0
            for _attempt in range(per_seed * attempts):
                if made >= per_seed:
                    break
                prompt = generation_prompt(recipe_id, seed_row)
                encoded = teacher_tokenizer(prompt, return_tensors="pt").to(device)
                output = teacher_model.generate(
                    **encoded, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=0.8, top_p=0.95, pad_token_id=pad_id,
                )
                text = teacher_tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
                candidate = _clean(text, seed_row[source])
                if candidate is None:
                    continue
                if _trailing_label(candidate, r.labels):
                    if filter_counts is not None:
                        filter_counts['trailing_label_rows_dropped'] = filter_counts.get('trailing_label_rows_dropped', 0) + 1
                    continue
                key = re.sub(r"\s+", " ", candidate.lower())
                if key in seen:
                    continue
                seen.add(key)
                rows.append({source: candidate, target: seed_row[target]})
                made += 1
    finally:
        if was_training:
            teacher_model.train()
    return rows


def build_student_spec(recipe_id: str, *, student_model: str, student_revision: str,
                       digest: str, nbytes: int, train_rows: int, test_rows: int,
                       steps: int, seed: int = 42) -> dict:
    logical_batch = max(1, min(8, train_rows))
    return {
        "template": "llm-lora/v1",
        "recipe_id": recipe_id,
        "model": {"id": student_model, "revision": student_revision},
        "dataset": {"sha256": digest, "bytes": nbytes, "train_rows": train_rows, "test_rows": test_rows},
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]},
        "max_seq_length": 512,
        "logical_batch_size": logical_batch,
        "total_logical_steps": steps,
        "seed": seed,
        "checkpoint_every_steps": steps,
        "optimizer": {"name": "adamw", "lr": 2e-4, "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8, "max_grad_norm": 1.0},
        "lr_schedule": {"name": "linear_warmup_linear_decay", "warmup_steps": min(10, max(1, steps // 10))},
    }


def run_distillation(recipe_id: str, seed_rows: list[dict], test_rows: list[dict], out_dir: Path, *,
                     teacher_model=None, teacher_tokenizer=None, student_model=None, student_tokenizer=None,
                     teacher_id: str = TEACHER_MODEL, teacher_pin: str | None = None, student_id: str = LLM_BASES[0],
                     student_revision: str | None = None, per_seed: int = 3, steps: int = 50,
                     device: str | None = None, exec_config: dict | None = None,
                     teacher_4bit: bool = False, generation_seed: int = 42, log=print) -> dict:
    """The whole distill/v1 run. Injected models are how CI tests it with tiny ones; a
    provider run leaves them out and both models are downloaded by id."""
    from .llm import load_model, load_tokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    owns_teacher = teacher_model is None
    if teacher_pin is None and teacher_id == TEACHER_MODEL:
        teacher_pin = TEACHER_REVISION
    if (teacher_tokenizer is None or teacher_model is None) and teacher_pin is None:
        raise ValueError(f"pin a revision for teacher {teacher_id}: an unpinned model can change between runs")
    if teacher_tokenizer is None:
        teacher_tokenizer = load_tokenizer(teacher_id, teacher_pin)
    if teacher_model is None:
        teacher_model = load_model(teacher_id, teacher_pin, device=device, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
                                   load_in_4bit=teacher_4bit)

    torch.manual_seed(generation_seed)
    filter_counts = {'trailing_label_rows_dropped': 0}
    generated = generate_rows(teacher_model, teacher_tokenizer, recipe_id, seed_rows,
                              per_seed=per_seed, device=device, filter_counts=filter_counts)
    teacher_revision = getattr(getattr(teacher_model, "config", None), "_commit_hash", None)
    if owns_teacher:
        del teacher_model
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    source, _target = recipe(recipe_id).columns
    normalise = lambda value: re.sub(r"\s+", " ", value.strip()).casefold()
    test_inputs = {normalise(row[source]) for row in test_rows}
    generated = [row for row in generated if normalise(row[source]) not in test_inputs]
    log(f"teacher generated {len(generated)} usable rows from {len(seed_rows)} seeds")
    if not generated:
        raise RuntimeError("the teacher produced no usable rows; check the seed examples")

    if student_revision is None:
        if student_model is None:
            from huggingface_hub import HfApi

            student_revision = HfApi().model_info(student_id).sha
        else:
            student_revision = "0" * 39 + "1"  # injected model: the revision is never fetched
    blob = out_dir / "student_dataset.jsonl"
    digest, nbytes = train_module.write_dataset(blob, generated, test_rows)
    spec = build_student_spec(recipe_id, student_model=student_id, student_revision=student_revision,
                              digest=digest, nbytes=nbytes, train_rows=len(generated),
                              test_rows=len(test_rows), steps=steps)
    (out_dir / "student_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    evidence = train_module.run_training(
        spec, blob, out_dir / "student",
        device=device, model=student_model, tokenizer=student_tokenizer,
        exec_config=exec_config, eval_before=True, log=log,
    )
    result = {
        "workload": "distill/v1",
        "recipe_id": recipe_id,
        "teacher": teacher_id,
        "teacher_revision": teacher_revision,
        "generation_filter_counts": filter_counts,
        "generation_settings": {"seed": generation_seed, "per_seed": per_seed, "attempts_per_output": 3,
                                "max_new_tokens": 96, "temperature": 0.8, "top_p": 0.95,
                                "do_sample": True, "teacher_4bit": teacher_4bit},
        "student": student_id,
        "seed_rows": len(seed_rows),
        "generated_rows": len(generated),
        "test_rows": len(test_rows),
        "student_eval": evidence["eval"],
        "student_base_eval": evidence["base_eval"],
    }
    (out_dir / "distill_evidence.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run one distill/v1 job: teacher generates, student trains.")
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--seed", type=Path, required=True, help="JSONL of the user's seed rows (recipe columns)")
    parser.add_argument("--test", type=Path, required=True, help="JSONL of the user's own test rows (never trained on)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teacher", default=TEACHER_MODEL)
    parser.add_argument("--teacher-revision", default=None, help="commit to pin; required for a teacher other than the default")
    parser.add_argument("--student", choices=LLM_BASES, default=LLM_BASES[0])
    parser.add_argument("--per-seed", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--teacher-4bit", action="store_true", help="load the teacher in 4-bit (7B on a 16 GB card)")
    parser.add_argument("--tiny", action="store_true", help="CI smoke run with tiny models")
    args = parser.parse_args(argv)

    seed_rows = [json.loads(line) for line in args.seed.read_text(encoding="utf-8").splitlines() if line.strip()]
    test_rows = [json.loads(line) for line in args.test.read_text(encoding="utf-8").splitlines() if line.strip()]
    teacher_model = teacher_tokenizer = student_model = student_tokenizer = None
    if args.tiny:
        from .tiny import build_tiny_model, build_tiny_tokenizer

        r = recipe(args.recipe)
        texts = [str(row[r.columns[0]]) + " " + str(row[r.columns[1]]) for row in seed_rows + test_rows]
        teacher_tokenizer = build_tiny_tokenizer(texts)
        teacher_model = build_tiny_model(teacher_tokenizer)
        student_tokenizer = teacher_tokenizer
        student_model = build_tiny_model(teacher_tokenizer)
    result = run_distillation(
        args.recipe, seed_rows, test_rows, args.out,
        teacher_model=teacher_model, teacher_tokenizer=teacher_tokenizer,
        student_model=student_model, student_tokenizer=student_tokenizer,
        teacher_id="tiny-qwen2" if args.tiny else args.teacher,
        teacher_pin=None if args.tiny else args.teacher_revision,
        student_id=args.student, per_seed=args.per_seed, steps=args.steps, device=args.device,
        teacher_4bit=args.teacher_4bit,
    )
    print(f"done: {result['generated_rows']} generated rows, student {result['student_eval']['pass_rate']:.1%} pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

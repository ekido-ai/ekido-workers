"""A deterministic check of the rows a teacher generated for distillation (ticket #141).

The teacher writes `generated.jsonl`: rows in the recipe's own two columns (the new input, and the target of the seed it was made from). A
provider is paid for those rows, and a student is trained on them, so a verifier re-checks them. This check is **pure Python and
deterministic**: no GPU, no model, no network, no randomness; the same bytes always give the same answer. It does not re-run the
generation (that would need the teacher); it re-runs the *filters* the generator promised to apply, so a generator that skipped them,
or poisoned the rows, is caught.

What it checks, each with a reason code in the verdict:

- `FILE_TOO_BIG`, `TOO_MANY_ROWS`, `NO_ROWS`, `NOT_UTF8`, `BAD_JSON`: the file itself;
- `SCHEMA`: every row is a JSON object with exactly the recipe's two columns, both non-empty text;
- `INPUT_TOO_SHORT`: a generated input of fewer than 3 characters;
- `BAD_LABEL`: for a classification recipe, the target is not one of the recipe's labels;
- `TEST_OVERLAP`: the input equals one of the customer's test inputs (after normalising spaces and case), which would leak the test set
  into training;
- `SEED_COPY`: the input equals one of the customer's own seed inputs (nothing new was generated);
- `DUPLICATE`: the same input appears twice (normalised);
- the leak filters of the generator (`ekido_runtime.distill`, workers #18), re-run: `TRAILING_LABEL` (the answer is written at the end
  of the input), `CLASSIFICATION_TEXT` (the input still talks about categories or classifying), `INVOICE_TARGET_MISMATCH` (an invoice
  whose fields are not all present in its text, or that copies the target object), `FAQ_ANSWER_ECHO` (the answer appears inside the
  question), `FAQ_PROMPT_MARKER` (a leftover "New question:").

The filters here are a copy of the generator's, kept in step by `tests/test_generated_rows.py`, which runs both on the same inputs and fails
if they ever disagree. They are copied so that the verifier never needs `torch` (the generator's module imports it).

The default is strict: **one** violating row fails the whole file, because the generator is supposed to have dropped such rows, and a provider
must not be paid for a file that contains them. `max_violations` can loosen that for a recipe if measurements ever justify it.

Given a `generation_report.json`, `check_report` also compares what the report claims (the row count and the SHA-256 of the rows) with the
file. Nothing here trusts the provider's own numbers.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Iterable

from ekido_core.recipes import recipe

MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_ROWS = 50_000
MIN_INPUT_CHARS = 3
INVOICE_KEYS = ("vendor", "invoice_no", "date", "total", "gstin")

REASONS = (
    "FILE_TOO_BIG", "TOO_MANY_ROWS", "NO_ROWS", "NOT_UTF8", "BAD_JSON", "SCHEMA", "INPUT_TOO_SHORT", "BAD_LABEL", "TEST_OVERLAP", "SEED_COPY",
    "DUPLICATE", "TRAILING_LABEL", "CLASSIFICATION_TEXT", "INVOICE_TARGET_MISMATCH", "FAQ_ANSWER_ECHO", "FAQ_PROMPT_MARKER",
)


@dataclass
class Verdict:
    passed: bool
    rows: int = 0
    violations: int = 0  # rows with at least one problem
    counts: dict[str, int] = field(default_factory=dict)  # reason code -> how many problems (1 for a file-level problem)
    row_hash: str = ""

    @property
    def reasons(self) -> list[str]:
        return sorted(self.counts)

    def as_dict(self) -> dict:
        return {"passed": self.passed, "rows": self.rows, "violations": self.violations, "reasons": self.reasons, "counts": dict(sorted(self.counts.items())), "row_hash": self.row_hash}


def normalise(value: str) -> str:
    """The same normalisation the generator uses for its own dedupe: spaces collapsed, case folded."""
    return re.sub(r"\s+", " ", value.strip()).casefold()


def input_hash(value: str) -> str:
    """The SHA-256 of a normalised input, for callers that hold only hashes of the test inputs (the spec carries a hash)."""
    return hashlib.sha256(normalise(value).encode("utf-8")).hexdigest()


# --- the generator's leak filters (a copy of ekido_runtime.distill; a test keeps them equal) --------------------------------------------

def trailing_label(candidate: str, labels: tuple[str, ...]) -> bool:
    if not labels:
        return False
    choices = "|".join(re.escape(label) for label in labels)
    if re.search(r"(?:^|[.!?।॥]\s+)(?:" + choices + r")\s*[.!?।॥]*\s*$", candidate, re.IGNORECASE):
        return True
    title_choices = "|".join(re.escape(label.title()) for label in labels)
    return bool(re.search(r"\s+(?:" + title_choices + r")\s*[.!?।॥]*\s*$", candidate))


def residual_classification_text(candidate: str, labels: tuple[str, ...]) -> bool:
    return bool(labels and re.search(r"\b(?:categor(?:y|ies|iz(?:e|ed|ing)|is(?:e|ed|ing))|classif(?:y|ied|ication|ying))\b", candidate, re.IGNORECASE))


def invoice_values_present(candidate: str, target: str) -> bool:
    if re.search(r'["\'](?:vendor|invoice_no|date|total|gstin)["\']\s*:', candidate, re.IGNORECASE):
        return False
    try:
        values = json.loads(target)
    except (TypeError, ValueError):
        return False
    if not isinstance(values, dict) or any(key not in values or values[key] is None for key in INVOICE_KEYS):
        return False
    text = normalise(candidate)
    return all(normalise(str(values[key])) and re.search(r"(?<!\w)" + re.escape(normalise(str(values[key]))) + r"(?!\w)", text) for key in INVOICE_KEYS)


def faq_answer_echo(candidate: str, target: str) -> bool:
    answer = normalise(target)
    return bool(answer and answer in normalise(candidate))


def faq_prompt_marker(candidate: str) -> bool:
    return bool(re.search(r"\bnew\s+question\s*:", candidate, re.IGNORECASE))


# --- the check ----------------------------------------------------------------------------------------------------------------------------

def canonical_row_hash(rows: Iterable[dict]) -> str:
    """The SHA-256 over the rows in order, each as compact, key-sorted JSON on its own line: what a report's `row_hash` should be."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def check_rows(rows: list, recipe_id: str, *, test_inputs: Iterable[str] = (), test_input_hashes: Iterable[str] = (), seed_inputs: Iterable[str] = (),
               max_violations: int = 0) -> Verdict:
    """Check parsed rows. `test_inputs` (or only their `test_input_hashes`) are the customer's own test inputs; `seed_inputs` their seeds."""
    r = recipe(recipe_id)  # raises for an unknown recipe: that is a caller bug, not a bad file
    source, target = r.columns
    labels = tuple(r.labels)
    excluded = {input_hash(v) for v in test_inputs} | {str(h).lower() for h in test_input_hashes}
    seeds = {normalise(v) for v in seed_inputs}
    counts: dict[str, int] = {}

    def bad(code: str) -> None:
        counts[code] = counts.get(code, 0) + 1

    if len(rows) > MAX_ROWS:
        counts["TOO_MANY_ROWS"] = 1
        return Verdict(False, len(rows), 1, counts, "")
    if not rows:
        counts["NO_ROWS"] = 1
        return Verdict(False, 0, 1, counts, "")
    seen: set[str] = set()
    bad_rows = 0  # rows with at least one problem (a row with two problems counts once)
    for row in rows:
        if not (isinstance(row, dict) and set(row) == {source, target} and all(isinstance(row[k], str) and row[k].strip() for k in (source, target))):
            bad("SCHEMA")
            bad_rows += 1
            continue
        text, answer = row[source], row[target]
        failed = False

        def flag(code: str) -> None:
            nonlocal failed
            failed = True
            bad(code)

        if len(text.strip()) < MIN_INPUT_CHARS:
            flag("INPUT_TOO_SHORT")
        if labels and answer not in labels:
            flag("BAD_LABEL")
        key = normalise(text)
        if input_hash(text) in excluded:
            flag("TEST_OVERLAP")
        if key in seeds:
            flag("SEED_COPY")
        if key in seen:
            flag("DUPLICATE")
        seen.add(key)
        if trailing_label(text, labels):
            flag("TRAILING_LABEL")
        if residual_classification_text(text, labels):
            flag("CLASSIFICATION_TEXT")
        if recipe_id == "invoice-fields" and not invoice_values_present(text, answer):
            flag("INVOICE_TARGET_MISMATCH")
        if recipe_id == "faq-bot-hinglish" and faq_answer_echo(text, answer):
            flag("FAQ_ANSWER_ECHO")
        if recipe_id == "faq-bot-hinglish" and faq_prompt_marker(text):
            flag("FAQ_PROMPT_MARKER")
        if failed:
            bad_rows += 1
    well_formed = all(isinstance(row, dict) for row in rows)
    return Verdict(bad_rows <= max_violations, len(rows), bad_rows, counts, canonical_row_hash(rows) if well_formed else "")


def check_generated_jsonl(data: bytes, recipe_id: str, **kwargs) -> Verdict:
    """Check the bytes of `generated.jsonl`. A file that is too big, not UTF-8 or not JSON lines fails with its own reason before any row is read."""
    if len(data) > MAX_FILE_BYTES:
        return Verdict(False, 0, 1, {"FILE_TOO_BIG": 1})
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return Verdict(False, 0, 1, {"NOT_UTF8": 1})
    rows: list = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            return Verdict(False, len(rows), 1, {"BAD_JSON": 1})
        if len(rows) > MAX_ROWS:
            return Verdict(False, len(rows), 1, {"TOO_MANY_ROWS": 1})
    return check_rows(rows, recipe_id, **kwargs)


def check_report(report: object, verdict: Verdict) -> list[str]:
    """What a `generation_report.json` claims that the file does not bear out. Empty means nothing contradicts it.

    Only claims about the file itself are compared (`generated_rows` or `rows`, and `row_hash`); the report's own filter counts are the
    provider's word and are not used to excuse anything."""
    if not isinstance(report, dict):
        return ["the report is not a JSON object"]
    problems = []
    claimed_rows = report.get("generated_rows", report.get("rows"))
    if claimed_rows is not None and claimed_rows != verdict.rows:
        problems.append(f"the report says {claimed_rows} rows but the file has {verdict.rows}")
    claimed_hash = report.get("row_hash")
    if claimed_hash is not None and verdict.row_hash and str(claimed_hash).lower() != verdict.row_hash:
        problems.append("the report's row_hash is not the hash of the rows in the file")
    return problems

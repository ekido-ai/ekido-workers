"""Seed-at-a-time generation with local, integrity-checked progress.

This is a standalone building block, not the pending coordinator contract. Its
work directory contains customer-derived data and must be private to the job.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

FILTER_NAMES = ('clean_rejected', 'answer_annotations_removed', 'trailing_label_rows_dropped',
                'classification_text_rows_dropped', 'invoice_target_mismatch_rows_dropped',
                'faq_answer_echo_rows_dropped', 'faq_prompt_marker_rows_dropped',
                'generation_duplicate_rows_dropped', 'duplicate_or_held_out_rows_dropped')


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')


def _hash(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _normalise(value):
    return re.sub(r'\s+', ' ', value.strip()).casefold()


@contextmanager
def _writer_lock(directory):
    """Nonblocking OS lock; the kernel releases it even if the process dies."""
    fd = os.open(directory / '.writer.lock', os.O_RDWR | os.O_CREAT, 0o600)
    locked = False
    try:
        if os.fstat(fd).st_size == 0:
            os.write(fd, b'0')
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('generation work directory already has a writer') from exc
        locked = True
        yield
    finally:
        if locked:
            if os.name == 'nt':
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _atomic_write(path, data):
    fd, temporary = tempfile.mkstemp(prefix='.distill-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_seed_generation(*, job_id, recipe_id, seed_rows, test_rows, out_dir, teacher_model,
                        teacher_tokenizer, teacher_id, teacher_revision, generator_revision, teacher_4bit=True, per_seed=3,
                        generation_seed=42, max_new_tokens=96, attempts=3, device='cpu',
                        generate=None, set_seed=None):
    """Generate training rows, committing progress after each complete seed.

    `generate` and `set_seed` are injectable for tests; the default generator is
    the existing filtered teacher implementation. A crash during one seed can
    repeat that seed; completed seeds and their accepted rows are never appended
    twice. Exact output across different hardware is not guaranteed.
    """
    from ekido_core.recipes import recipe

    if not isinstance(job_id, str) or not job_id.strip() or len(job_id) > 200:
        raise ValueError('job_id must be nonempty text')
    if not isinstance(recipe_id, str):
        raise ValueError('invalid recipe')
    r = recipe(recipe_id)
    source, target = r.columns
    if not isinstance(seed_rows, list) or not 1 <= len(seed_rows) <= 100:
        raise ValueError('need 1 to 100 seed rows')
    if not isinstance(test_rows, list) or len(test_rows) > 10000:
        raise ValueError('invalid test rows')
    if (type(per_seed) is not int or not 1 <= per_seed <= 20 or type(attempts) is not int or not 1 <= attempts <= 10
            or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 512
            or type(generation_seed) is not int or not 0 <= generation_seed < 2**31):
        raise ValueError('invalid generation settings')
    if (not isinstance(teacher_id, str) or not teacher_id.strip() or not isinstance(teacher_revision, str)
            or not re.fullmatch(r'[0-9a-f]{40}', teacher_revision) or not isinstance(generator_revision, str)
            or not re.fullmatch(r'[0-9a-f]{40}', generator_revision) or type(teacher_4bit) is not bool):
        raise ValueError('teacher and generator must have immutable revisions and a quantization setting')
    for row in seed_rows + test_rows:
        if (not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip()
                                            or len(row[k]) > 8192 for k in (source, target))):
            raise ValueError('invalid seed or test row')
        if r.labels and row[target].strip().casefold() not in r.labels:
            raise ValueError('row target is not a recipe label')
    if generate is None:
        from .distill import generate_rows
        generate = generate_rows
    if set_seed is None:
        import torch
        set_seed = torch.manual_seed

    settings = {'job_id': job_id, 'recipe_id': recipe_id, 'seed_rows_sha256': _hash(seed_rows),
                'test_rows_sha256': _hash(test_rows), 'teacher_requested_id': teacher_id,
                'teacher_requested_revision': teacher_revision, 'generator_requested_revision': generator_revision,
                'teacher_requested_4bit': teacher_4bit,
                'per_seed': per_seed, 'generation_seed': generation_seed, 'max_new_tokens': max_new_tokens,
                'attempts': attempts}
    identity = _hash(settings)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with _writer_lock(directory):
        checkpoint = directory / 'seed-progress.json'
        state = {'identity': identity, 'completed_seeds': 0, 'rows': [],
                 'filter_counts': {name: 0 for name in FILTER_NAMES}}
        if checkpoint.exists():
            if checkpoint.stat().st_size > 50_000_000:
                raise ValueError('checkpoint too large')
            envelope = json.loads(checkpoint.read_text(encoding='utf-8'))
            if not isinstance(envelope, dict) or set(envelope) != {'state', 'sha256'} or _hash(envelope['state']) != envelope['sha256']:
                raise ValueError('checkpoint integrity failure')
            state = envelope['state']
            if (not isinstance(state, dict) or set(state) != {'identity', 'completed_seeds', 'rows', 'filter_counts'}
                    or state['identity'] != identity or type(state['completed_seeds']) is not int
                    or not 0 <= state['completed_seeds'] <= len(seed_rows) or not isinstance(state['rows'], list)
                    or not isinstance(state['filter_counts'], dict)
                    or any(not isinstance(k, str) or type(v) is not int or v < 0
                           for k, v in state['filter_counts'].items())):
                raise ValueError('checkpoint does not belong to this job or settings')
        excluded = {_normalise(row[source]) for row in seed_rows + test_rows}
        seen = set(excluded)
        for row in state['rows']:
            if (not isinstance(row, dict) or set(row) != {source, target}
                    or not isinstance(row[source], str) or not row[source].strip()
                    or not isinstance(row[target], str) or not row[target].strip()
                    or len(row[source]) > 8192 or len(row[target]) > 8192
                    or row[target] not in {seed[target] for seed in seed_rows[:state['completed_seeds']]}):
                raise ValueError('invalid checkpoint row')
            key = _normalise(row[source])
            if key in seen:
                raise ValueError('checkpoint contains duplicate or held-out input')
            seen.add(key)
        if len(state['rows']) > state['completed_seeds'] * per_seed:
            raise ValueError('checkpoint has too many rows')
        for index in range(state['completed_seeds'], len(seed_rows)):
            counts = dict(state['filter_counts'])
            set_seed(generation_seed + index)
            rows = generate(teacher_model, teacher_tokenizer, recipe_id, [seed_rows[index]], per_seed=per_seed,
                            device=device, max_new_tokens=max_new_tokens, attempts=attempts, filter_counts=counts)
            if not isinstance(rows, list) or len(rows) > per_seed:
                raise ValueError('generator returned invalid rows')
            if any(not isinstance(k, str) or type(v) is not int or v < 0 for k, v in counts.items()):
                raise ValueError('generator returned invalid filter counts')
            accepted = []
            for row in rows:
                if (not isinstance(row, dict) or set(row) != {source, target} or not isinstance(row[source], str)
                        or not row[source].strip() or len(row[source]) > 8192 or row[target] != seed_rows[index][target]):
                    raise ValueError('generator returned invalid or mismatched target')
                key = _normalise(row[source])
                if key in seen:
                    counts['duplicate_or_held_out_rows_dropped'] = counts.get('duplicate_or_held_out_rows_dropped', 0) + 1
                    continue
                seen.add(key)
                accepted.append({source: row[source], target: row[target]})
            state = {'identity': identity, 'completed_seeds': index + 1, 'rows': state['rows'] + accepted,
                     'filter_counts': counts}
            _atomic_write(checkpoint, _json_bytes({'state': state, 'sha256': _hash(state)}))
        if not state['rows']:
            raise RuntimeError('no usable generated rows')
        output = b''.join(_json_bytes(row) for row in state['rows'])
        report = {'settings': settings, 'completed_seeds': state['completed_seeds'], 'generated_rows': len(state['rows']),
                  'filter_counts': state['filter_counts'], 'generated_sha256': hashlib.sha256(output).hexdigest(),
                  'generated_bytes': len(output)}
        _atomic_write(directory / 'generated.jsonl', output)
        _atomic_write(directory / 'generation_report.json', _json_bytes(report))
        return report

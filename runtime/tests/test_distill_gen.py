import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ekido_runtime.distill_gen import _writer_lock, run_seed_generation


class SeedGeneration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.seeds = [{'text': 'seed one', 'label': 'refund'}, {'text': 'seed two', 'label': 'order'}]
        self.tests = [{'text': 'held out', 'label': 'order'}]
        self.calls = []
        self.seeding = []

    def generate(self, _model, _tokenizer, _recipe, seeds, **kwargs):
        self.calls.append(seeds[0]['text'])
        kwargs['filter_counts']['test_filter'] = kwargs['filter_counts'].get('test_filter', 0) + 1
        return [{'text': 'new ' + seeds[0]['text'], 'label': seeds[0]['label']}]

    def run_job(self, **changes):
        args = dict(job_id='job-a', recipe_id='ticket-sorter-hinglish', seed_rows=self.seeds, test_rows=self.tests,
                    out_dir=self.out, teacher_model=None, teacher_tokenizer=None, teacher_id='test/teacher',
                    teacher_revision='a' * 40, generator_revision='b' * 40, per_seed=1,
                    generate=self.generate, set_seed=self.seeding.append)
        return run_seed_generation(**{**args, **changes})

    def test_crash_resumes_at_next_seed_without_duplicate_output(self):
        def interrupted(model, tokenizer, recipe, seeds, **kwargs):
            if seeds[0]['text'] == 'seed two':
                raise RuntimeError('device died')
            return self.generate(model, tokenizer, recipe, seeds, **kwargs)
        with self.assertRaisesRegex(RuntimeError, 'device died'):
            self.run_job(generate=interrupted)
        self.assertFalse((self.out / 'generated.jsonl').exists())
        self.assertFalse((self.out / 'generation_report.json').exists())
        self.assertEqual(json.loads((self.out / 'seed-progress.json').read_text())['state']['completed_seeds'], 1)
        report = self.run_job()
        self.assertEqual(self.calls, ['seed one', 'seed two'])
        self.assertEqual(self.seeding, [42, 43, 43])
        self.assertEqual(report['completed_seeds'], 2)
        self.assertEqual(report['filter_counts']['test_filter'], 2)
        data = (self.out / 'generated.jsonl').read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), report['generated_sha256'])
        self.assertEqual([json.loads(line)['text'] for line in data.splitlines()], ['new seed one', 'new seed two'])
        self.run_job()
        self.assertEqual(self.calls, ['seed one', 'seed two'])

    def test_rejects_changed_job_inputs_settings_and_corrupt_checkpoint(self):
        self.run_job()
        for changes in ({'job_id': 'other'}, {'generation_seed': 43}, {'teacher_4bit': False},
                        {'generator_revision': 'c' * 40},
                        {'seed_rows': [{'text': 'other customer', 'label': 'refund'}]},
                        {'test_rows': [{'text': 'different held out', 'label': 'order'}]}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'does not belong'):
                self.run_job(**changes)
        path = self.out / 'seed-progress.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        data['state']['completed_seeds'] = 0
        path.write_text(json.dumps(data), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            self.run_job()

    def test_rejects_competing_writer(self):
        with _writer_lock(self.out):
            with self.assertRaisesRegex(RuntimeError, 'already has a writer'):
                self.run_job()
        self.assertEqual(self.run_job()['completed_seeds'], 2)

    def test_drops_seed_holdout_and_prior_duplicate_inputs(self):
        def generate(_model, _tokenizer, _recipe, seeds, **_kwargs):
            rows = ['held out', 'shared new'] if seeds[0]['text'] == 'seed one' else [' SHARED   NEW ', 'unique']
            return [{'text': text, 'label': seeds[0]['label']} for text in rows]
        report = self.run_job(generate=generate, per_seed=2)
        self.assertEqual(report['generated_rows'], 2)
        self.assertEqual(report['filter_counts']['duplicate_or_held_out_rows_dropped'], 2)
        self.assertEqual([json.loads(line)['text'] for line in (self.out / 'generated.jsonl').read_text().splitlines()],
                         ['shared new', 'unique'])

    def test_invalid_rows_or_mismatched_generated_targets_never_publish(self):
        with self.assertRaises(ValueError):
            self.run_job(seed_rows=[{'text': 'seed', 'label': 'unknown'}])
        with self.assertRaises(ValueError):
            self.run_job(teacher_revision='main')
        def bad(_model, _tokenizer, _recipe, _seeds, **_kwargs):
            return [{'text': 'new request', 'label': 'order'}]
        with self.assertRaisesRegex(ValueError, 'mismatched'):
            self.run_job(generate=bad)
        self.assertFalse((self.out / 'generated.jsonl').exists())
        self.assertFalse((self.out / 'seed-progress.json').exists())

    def test_recover_after_output_write_but_before_report(self):
        from ekido_runtime import distill_gen
        original = distill_gen._atomic_write
        def interrupt_report(path, data):
            if path.name == 'generation_report.json':
                raise OSError('disk interrupted')
            return original(path, data)
        with mock.patch.object(distill_gen, '_atomic_write', side_effect=interrupt_report):
            with self.assertRaisesRegex(OSError, 'disk interrupted'):
                self.run_job()
        self.assertTrue((self.out / 'generated.jsonl').exists())
        self.assertFalse((self.out / 'generation_report.json').exists())
        report = self.run_job()
        self.assertEqual(self.calls, ['seed one', 'seed two'])
        self.assertEqual(report['generated_sha256'], hashlib.sha256((self.out / 'generated.jsonl').read_bytes()).hexdigest())

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch is required for the actual teacher filter')
    def test_existing_filter_runs_with_teacher_stub(self):
        import torch
        class Encoded(dict):
            def to(self, _device): return self
        class Tokenizer:
            pad_token_id, eos_token_id = 0, 1
            def __init__(self): self.outputs = iter(['The parcel was late. Complaint.', 'My parcel is delayed'])
            def __call__(self, _prompt, return_tensors): return Encoded(input_ids=torch.tensor([[1]]))
            def decode(self, *_args, **_kwargs): return next(self.outputs)
        model = mock.Mock(training=False)
        model.generate.return_value = torch.tensor([[1, 2]])
        report = self.run_job(seed_rows=[{'text': 'seed one', 'label': 'complaint'}], teacher_model=model,
                              teacher_tokenizer=Tokenizer(), generate=None)
        self.assertEqual(report['generated_rows'], 1)
        self.assertEqual(report['filter_counts']['trailing_label_rows_dropped'], 1)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch is required for the actual teacher filter')
    def test_teacher_filter_counts_empty_and_duplicate_candidates(self):
        import torch
        class Encoded(dict):
            def to(self, _device): return self
        class Tokenizer:
            pad_token_id, eos_token_id = 0, 1
            def __init__(self): self.outputs = iter(['seed one', 'new ticket one', 'new ticket one', 'new ticket two'])
            def __call__(self, _prompt, return_tensors): return Encoded(input_ids=torch.tensor([[1]]))
            def decode(self, *_args, **_kwargs): return next(self.outputs)
        model = mock.Mock(training=False)
        model.generate.return_value = torch.tensor([[1, 2]])
        report = self.run_job(seed_rows=[self.seeds[0]], teacher_model=model, teacher_tokenizer=Tokenizer(),
                              per_seed=2, generate=None)
        self.assertEqual(report['generated_rows'], 2)
        self.assertEqual(report['filter_counts']['clean_rejected'], 1)
        self.assertEqual(report['filter_counts']['generation_duplicate_rows_dropped'], 1)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                         'tiny-model dependencies are required')
    def test_tiny_teacher_reaches_a_complete_seed_boundary(self):
        from ekido_runtime.tiny import build_tiny_model, build_tiny_tokenizer
        tokenizer = build_tiny_tokenizer(['seed one', 'held out', 'refund', 'order'])
        model = build_tiny_model(tokenizer)
        try:
            report = self.run_job(seed_rows=[self.seeds[0]], teacher_model=model, teacher_tokenizer=tokenizer,
                                  generate=None, max_new_tokens=8, attempts=1)
        except RuntimeError as exc:
            self.assertEqual(str(exc), 'no usable generated rows')
            self.assertFalse((self.out / 'generation_report.json').exists())
        else:
            self.assertEqual(report['completed_seeds'], 1)
        state = json.loads((self.out / 'seed-progress.json').read_text(encoding='utf-8'))['state']
        self.assertEqual(state['completed_seeds'], 1)


if __name__ == '__main__':
    unittest.main()

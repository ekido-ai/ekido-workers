import unittest
from unittest.mock import patch

from ekido_verifier.worker import Verifier, _trim_checkpoint_cache


class IdentityCache(unittest.TestCase):
    def test_checkpoint_eviction_preserves_active_job_baselines(self):
        baseline = ('eval', 'baseline:job:spec')
        cache = {baseline: object()}
        cache.update({('eval', f'checkpoint-{i}'): i for i in range(20)})
        _trim_checkpoint_cache(cache, limit=8)
        self.assertIn(baseline, cache)
        self.assertEqual(len(cache), 9)
        self.assertNotIn(('eval', 'checkpoint-0'), cache)
        self.assertIn(('eval', 'checkpoint-19'), cache)

    def test_different_full_specs_do_not_share_eval_set(self):
        verifier = Verifier("http://coordinator", "0x" + "a" * 64, "cpu", 128)
        first = {"template": "distilbert-lora-agnews/v1", "seed": 1, "max_seq_length": 128, "model": {"revision": "a" * 40}, "dataset": {"test_content_sha256": "1" * 64}}
        second = {**first, "dataset": {"test_content_sha256": "2" * 64}}
        with patch("ekido_verifier.worker.load_eval_set", side_effect=[object(), object()]) as loader:
            a = verifier._eval_set(first)
            self.assertIs(verifier._eval_set(first), a)
            self.assertIsNot(verifier._eval_set(second), a)
            self.assertEqual(loader.call_count, 2)

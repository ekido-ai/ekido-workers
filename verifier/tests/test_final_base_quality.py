"""The customer's final recipe must meet its floor and beat its base model."""
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from ekido_verifier.evaluate import EvalSet, Evaluation, evaluate_checkpoint


def evaluation(count):
    correct = np.array([True] * count + [False] * (100 - count))
    return Evaluation(correct=correct, predictions=correct, loss=0.0, score=count / 100)


class FinalBaseQuality(unittest.TestCase):
    def setUp(self):
        self.spec = {'template': 'llm-lora/v1', 'model': {'id': 'test', 'revision': 'a' * 40}}
        self.eval_set = EvalSet({}, torch.zeros(100), 'e' * 64, recipe_id='review-sentiment')
        self.policy = {'eval_set_hash': self.eval_set.hash, 'eval_version_hash': 'f' * 64,
                       'confidence': 0.95, 'bootstrap_resamples': 200,
                       'non_inferiority_margin': 0.02, 'min_absolute': 0.8}

    def run_case(self, candidate, base, reference, cache=None):
        models = {'base': evaluation(base), 'previous': evaluation(reference), 'candidate': evaluation(candidate)}
        def load(_spec, path, _device):
            if path is None:
                return 'base', None, None
            if str(path) == 'previous':
                return 'previous', {'kind': 'previous'}, None
            return 'candidate', {'kind': 'candidate'}, None
        with mock.patch('ekido_verifier.evaluate.load_model', side_effect=load) as loader, \
                mock.patch('ekido_verifier.evaluate.predict', side_effect=lambda model, *_: models[model]), \
                mock.patch('ekido_verifier.evaluate.manifest_hash', side_effect=lambda manifest: manifest['kind']):
            result = evaluate_checkpoint(job_id='job', spec=self.spec, policy=self.policy,
                                         eval_set=self.eval_set, checkpoint_dir=Path('candidate'),
                                         reference_dir=Path('previous'), expected_signer=None,
                                         is_final=True, reference_cache=cache)
        return result, loader

    def test_both_conditions_are_required(self):
        for candidate, base, expected in [(80, 80, 'FAIL'), (70, 50, 'FAIL'),
                                          (90, 50, 'PASS'), (40, 50, 'FAIL')]:
            with self.subTest(candidate=candidate, base=base):
                report, _ = self.run_case(candidate, base, reference=40)
                self.assertEqual(report['verdict'], expected)
                self.assertEqual(report['base_comparison']['reference_accuracy'], base / 100)
        tied, _ = self.run_case(80, 80, reference=40)
        self.assertIn('BASE_MODEL_NOT_BEATEN', tied['reasons'])

    def test_base_predictions_are_reused(self):
        cache = {}
        self.run_case(90, 50, reference=40, cache=cache)
        _, loader = self.run_case(90, 50, reference=40, cache=cache)
        self.assertFalse(any(call.args[1] is None for call in loader.call_args_list))

    def test_weaker_policy_cannot_bypass_recipe_floor(self):
        self.policy['min_absolute'] = 0.1
        report, _ = self.run_case(70, 50, reference=40)
        self.assertEqual(report['verdict'], 'FAIL')
        self.assertIn('BELOW_ABSOLUTE_FLOOR', report['reasons'])


if __name__ == '__main__':
    unittest.main()

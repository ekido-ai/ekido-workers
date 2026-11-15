"""CPU tests of the training/evaluation identity boundary with fake datasets."""
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch

from ekido_core.data_identity import hash_text_labels
from ekido_runtime.workloads import build_workload
from ekido_verifier.evaluate import load_eval_set


class Rows(list):
    def __getitem__(self, key):
        return [row[key] for row in self] if isinstance(key, str) else super().__getitem__(key)

    def select(self, indices):
        return Rows(self[i] for i in indices)

    def shuffle(self, seed):
        return Rows(reversed(self))


class IdentityIntegration(unittest.TestCase):
    def setUp(self):
        self.train = Rows({"text": f"train {i}", "label": i % 4} for i in range(20))
        self.test = Rows({"text": f"test {i}", "label": i % 4} for i in range(8))
        train_hash, test_hash = hash_text_labels(self.train), hash_text_labels(self.test)
        self.spec = {"template": "distilbert-lora-agnews/v1", "seed": 1, "max_seq_length": 8, "model": {"id": "model", "revision": "a" * 40}, "dataset": {"id": "dataset", "split": "train", "revision": "b" * 40, "fingerprint": train_hash, "content_sha256": train_hash, "test_content_sha256": test_hash}, "lora": {"r": 2, "alpha": 4, "dropout": 0.0, "target_modules": ["q_lin"]}}
        self.fetch = Mock(side_effect=lambda _name, split, **_kwargs: self.train if split == "train" else self.test)
        tokenizer = Mock(side_effect=lambda texts, **_kwargs: {"input_ids": torch.zeros((len(texts), 8), dtype=torch.long), "attention_mask": torch.ones((len(texts), 8), dtype=torch.long)})
        model = Mock()
        model.to.return_value = model
        self.modules = {
            "datasets": types.SimpleNamespace(load_dataset=self.fetch),
            "transformers": types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)), AutoModelForSequenceClassification=types.SimpleNamespace(from_pretrained=Mock(return_value=model))),
            "peft": types.SimpleNamespace(LoraConfig=Mock(), TaskType=types.SimpleNamespace(SEQ_CLS="classification"), get_peft_model=lambda base, _config: base),
        }

    def test_training_checks_full_split_and_uses_pinned_revision(self):
        with patch.dict(sys.modules, self.modules):
            workload = build_workload(self.spec, {"attention_impl": "sdpa"}, "cpu")
            self.assertEqual(workload.dataset_size, 20)
            self.fetch.assert_called_with("dataset", split="train", revision="b" * 40)
            self.train[0]["text"] = "changed"
            with self.assertRaisesRegex(ValueError, "training data mismatch"):
                build_workload(self.spec, {"attention_impl": "sdpa"}, "cpu")

    def test_evaluation_checks_test_split_and_uses_pinned_revision(self):
        with patch.dict(sys.modules, self.modules):
            result = load_eval_set(self.spec, examples=4)
            self.assertEqual(len(result.labels), 4)
            self.fetch.assert_called_with("dataset", split="test", revision="b" * 40)
            self.test[0]["label"] = 3
            with self.assertRaisesRegex(ValueError, "evaluation data mismatch"):
                load_eval_set(self.spec, examples=4)

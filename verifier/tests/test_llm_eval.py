"""llm-lora/v1 end to end on CPU: the real training loop writes checkpoints for a tiny randomly-built model, and the
verifier scores them on the customer's test rows. Nothing is downloaded."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAVE_ML = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "peft"))
if HAVE_ML:
    import torch
    from transformers import AutoModelForCausalLM, Qwen2Config

    from ekido_core.recipes import eval_set_hash
    from ekido_runtime.run import run
    from ekido_runtime.synthetic import exec_config, synthetic_spec
    from ekido_verifier.evaluate import evaluate_checkpoint, load_eval_set

JOB = "00000000-0000-0000-0000-0000000000a1"
ALPHABET = "\n abcdefghijklmnopqrstuvwxyz0123456789:.,?'-"


class CharTokenizer:
    """One token per character, so text survives a round trip. Ids 0 and 1 are padding and end-of-text."""

    pad_token_id, eos_token_id = 0, 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + max(0, ALPHABET.find(c)) for c in text.lower()]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(ALPHABET[i - 2] for i in ids if 2 <= i < 2 + len(ALPHABET))


def tiny_loader(spec, exec_config, device):
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=256, eos_token_id=1, pad_token_id=0, bos_token_id=1)
    torch.manual_seed(0)
    return CharTokenizer(), AutoModelForCausalLM.from_config(config)


@unittest.skipUnless(HAVE_ML, "torch, transformers and peft are needed")
class LlmEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        train = [{"split": "train", "text": f"accha hai {i}", "label": "positive"} for i in range(8)]
        train += [{"split": "train", "text": f"bekaar hai {i}", "label": "negative"} for i in range(8)]
        cls.test_rows = [{"split": "test", "text": f"theek hai {i}", "label": "neutral"} for i in range(20)]
        cls.data = "\n".join(json.dumps(r) for r in train + cls.test_rows).encode()
        spec = synthetic_spec(total_steps=6, logical_batch_size=8, lr=0.01)
        spec.update(
            template="llm-lora/v1", recipe_id="review-sentiment", max_seq_length=64, checkpoint_every_steps=3,
            model={"id": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "a" * 40},
            dataset={"sha256": hashlib.sha256(cls.data).hexdigest(), "bytes": len(cls.data), "train_rows": 16, "test_rows": 20},
            lora={"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
        )
        cls.spec = spec
        cls.patch = patch("ekido_runtime.llm._pretrained", tiny_loader)
        cls.patch.start()
        workdir = cls.root / "run"
        workdir.mkdir()
        (workdir / "dataset.jsonl").write_bytes(cls.data)  # The agent puts it here before a real run.
        run({"job_id": JOB, "attempt": 1, "fencing_token": 1, "workdir": str(workdir), "device": "cpu", "spec": spec,
             "exec": exec_config(4, logical_batch_size=8), "signer": "0x" + "a" * 40, "progress_seconds": 60})
        events = [json.loads(line) for line in (workdir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        cls.checkpoints = [e for e in events if e["kind"] == "checkpoint.ready"]
        cls.eval_set = load_eval_set(spec, data_dir=workdir)

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        cls.tmp.cleanup()

    def policy(self, **overrides):
        policy = {"schema_version": 1, "metric": "accuracy", "min_absolute": 0.0, "non_inferiority_margin": 0.05, "confidence": 0.95,
                  "bootstrap_resamples": 500, "eval_set_hash": self.eval_set.hash, "eval_version_hash": "e" * 64}
        policy.update(overrides)
        return policy

    def evaluate(self, checkpoint, reference=None, **kw):
        return evaluate_checkpoint(job_id=JOB, spec=self.spec, policy=kw.pop("policy", self.policy()), eval_set=self.eval_set,
                                   checkpoint_dir=Path(checkpoint["path"]), reference_dir=Path(reference["path"]) if reference else None,
                                   expected_signer=None, is_final=kw.pop("final", False), **kw)

    def test_training_wrote_checkpoints(self):
        self.assertEqual([c["step"] for c in self.checkpoints], [3, 6])

    def test_the_eval_set_is_the_customers_test_rows(self):
        self.assertEqual(len(self.eval_set.prompts), 20)
        self.assertEqual(self.eval_set.hash, eval_set_hash("review-sentiment", self.test_rows))
        self.assertTrue(self.eval_set.prompts[0].endswith("Review: theek hai 0\nSentiment:"))
        self.assertEqual(set(self.eval_set.references), {"neutral"})

    def test_a_report_scores_every_test_row(self):
        report = self.evaluate(self.checkpoints[-1], self.checkpoints[0], final=True)
        self.assertEqual(report["n_examples"], 20)
        self.assertEqual(report["metrics"]["metric"], "accuracy")
        self.assertTrue(0.0 <= report["metrics"]["accuracy"] <= 1.0)
        self.assertTrue(0.0 <= report["metrics"]["score"] <= 1.0)
        self.assertIn(report["verdict"], ("PASS", "FAIL"))
        self.assertEqual(report["eval_set_hash"], self.eval_set.hash)

    def test_scoring_is_repeatable(self):
        first = self.evaluate(self.checkpoints[-1])
        second = self.evaluate(self.checkpoints[-1])
        self.assertEqual(first["correctness_hash"], second["correctness_hash"])
        self.assertEqual(first["metrics"], second["metrics"])

    def test_a_floor_the_model_misses_fails_the_final_check(self):
        report = self.evaluate(self.checkpoints[-1], final=True, policy=self.policy(min_absolute=1.01))
        self.assertEqual(report["verdict"], "FAIL")

    def test_a_different_test_set_is_refused(self):
        with self.assertRaises(ValueError):
            self.evaluate(self.checkpoints[-1], policy=self.policy(eval_set_hash="0" * 64))

    def test_a_wrong_row_count_is_refused(self):
        spec = dict(self.spec, dataset=dict(self.spec["dataset"], test_rows=21))
        with self.assertRaises(ValueError):
            load_eval_set(spec, data_dir=self.root / "run")


if __name__ == "__main__":
    unittest.main()

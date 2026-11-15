"""llm-lora/v1 on a tiny randomly-built Qwen2 model: no downloads, CPU only, a few seconds."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from peft import LoraConfig  # noqa: F401
    from transformers import AutoModelForCausalLM, Qwen2Config

    from ekido_runtime.llm import IGNORE, build_llm, encode
except ImportError:  # The suite runs without the ML stack in some environments.
    torch = None


class CharTokenizer:
    """Just enough of a Hugging Face tokenizer: one token per character, with pad and end-of-text ids."""

    pad_token_id, eos_token_id = 0, 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + (ord(c) % 120) for c in text]}


def tiny_loader(spec, exec_config, device):
    config = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=256)
    torch.manual_seed(0)  # Same base weights every time, like a pinned checkpoint.
    return CharTokenizer(), AutoModelForCausalLM.from_config(config)


def spec_for(digest, train_rows, seed=7):
    return {
        "template": "llm-lora/v1", "recipe_id": "review-sentiment", "seed": seed, "max_seq_length": 96,
        "model": {"id": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "a" * 40},
        "dataset": {"sha256": digest, "bytes": 1, "train_rows": train_rows, "test_rows": 20},
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
    }


@unittest.skipIf(torch is None, "torch, transformers and peft are needed")
class LlmWorkload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        rows = [{"split": "train", "text": f"accha product hai {i}", "label": "positive"} for i in range(8)]
        rows += [{"split": "train", "text": f"bekaar quality {i}", "label": "negative"} for i in range(8)]
        rows += [{"split": "test", "text": f"theek hai {i}", "label": "neutral"} for i in range(20)]
        data = "\n".join(json.dumps(r) for r in rows).encode()
        (Path(self.tmp.name) / "dataset.jsonl").write_bytes(data)
        self.digest = hashlib.sha256(data).hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, **overrides):
        spec = spec_for(self.digest, 16, **overrides)
        return build_llm(spec, {}, "cpu", Path(self.tmp.name), loader=tiny_loader)

    def test_only_the_answer_is_learned(self):
        batch = encode(CharTokenizer(), [{"prompt": "Review: ok\nSentiment:", "completion": " neutral"}], 40)
        labels, mask = batch["labels"][0], batch["attention_mask"][0]
        answer_len = len(" neutral") + 1  # plus end-of-text
        self.assertEqual(int((labels != IGNORE).sum()), answer_len)
        self.assertEqual(int(mask.sum()), len("Review: ok\nSentiment:") + answer_len)
        self.assertEqual(batch["input_ids"].shape, (1, 40))

    def test_long_prompts_lose_their_start_not_the_answer(self):
        batch = encode(CharTokenizer(), [{"prompt": "x" * 100, "completion": " positive"}], 32)
        self.assertEqual(int((batch["labels"][0] != IGNORE).sum()), len(" positive") + 1)

    def test_only_training_rows_are_used_and_the_hash_is_checked(self):
        _, size, batch_fn, canary = self.build()
        self.assertEqual(size, 16)
        self.assertEqual(canary["input_ids"].shape, (16, 96))
        spec = spec_for("0" * 64, 16)
        with self.assertRaises(ValueError):
            build_llm(spec, {}, "cpu", Path(self.tmp.name), loader=tiny_loader)

    def test_lora_starts_from_the_job_seed(self):
        def lora_weights(model):
            return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}

        first, second, other = (lora_weights(self.build(seed=s)[0]) for s in (7, 7, 8))
        self.assertTrue(first)
        self.assertTrue(all(torch.equal(first[n], second[n]) for n in first))
        self.assertFalse(all(torch.equal(first[n], other[n]) for n in first))

    def test_training_lowers_the_loss_and_touches_only_lora(self):
        model, size, batch_fn, _ = self.build()
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(trainable and all("lora_" in n for n in trainable))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-2)
        batch = batch_fn(range(size))
        start = model(**batch).loss.item()
        for _ in range(60):
            optimizer.zero_grad()
            model(**batch).loss.backward()
            optimizer.step()
        # A random tiny base learns slowly through LoRA alone; a steady 5% drop on 16 examples shows it trains.
        self.assertLess(model(**batch).loss.item(), start * 0.95)


if __name__ == "__main__":
    unittest.main()

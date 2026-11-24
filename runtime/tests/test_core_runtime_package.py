"""The runtime on the tiny CPU config: no downloads, no GPU.

Covers the three Week-1 core tickets end to end at small scale:
- C6: generate_answers (the baseline contract), the probe, training with GPCK/1
  checkpoints, and resume reproducing the uninterrupted run (the logical-step invariant).
- C10: merging a checkpoint into the base model and saving safetensors.
- C14: a (tiny) teacher generating rows and a student training on them.
"""

import json
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace
from pathlib import Path

import torch

from ekido_core.job_spec import LLM_BASES
from ekido_runtime import distill, export, train
from ekido_runtime.llm import generate_answers, load_model
from ekido_runtime.model_probe import probe_step
from ekido_runtime.tiny import build_tiny_model, build_tiny_tokenizer

COMMIT = "a" * 40
EXEC = {"micro_batch": 2, "accumulation": 2, "amp": "none", "grad_checkpointing": False}

POS = ["yeh phone bahut accha hai", "quality ekdam badhiya hai", "mujhe yeh product bahut pasand aaya",
       "delivery time par hui aur packing acchi thi", "paise vasool hai yeh cheez", "service bahut acchi lagi mujhe",
       "ekdam mast hai bhai", "dobara zaroor khareedunga"]
NEG = ["yeh phone bahut bekar hai", "quality ekdam ghatiya hai", "mujhe yeh product bilkul pasand nahi aaya",
       "delivery bahut late hui aur packing kharab thi", "paise barbaad ho gaye", "service bahut buri lagi mujhe",
       "ekdam bakwaas hai bhai", "dobara kabhi nahi khareedunga"]
NEU = ["yeh phone theek thaak hai", "quality ausat hai", "product chalega kaam mera",
       "delivery aam si thi", "na accha na bura", "theek hai bas",
       "jaisa socha tha waisa hi hai", "koi khaas baat nahi hai"]


def rows(texts, label):
    return [{"text": text, "label": label} for text in texts]


def make_data(tmp: Path):
    train_rows = rows(POS[:6], "positive") + rows(NEG[:6], "negative") + rows(NEU[:6], "neutral")
    # exactly 20 test rows (the spec's minimum): 2 held out per label + 14 more
    extra = [{"text": f"sample review number {i}", "label": ["positive", "negative", "neutral"][i % 3]} for i in range(14)]
    test_rows = rows(POS[6:], "positive") + rows(NEG[6:], "negative") + rows(NEU[6:], "neutral") + extra
    assert len(test_rows) == 20
    blob = tmp / "data.jsonl"
    digest, nbytes = train.write_dataset(blob, train_rows, test_rows)
    return blob, train_rows, test_rows, digest, nbytes


def make_spec(digest, nbytes, n_train, n_test, steps=4):
    return {
        "template": "llm-lora/v1",
        "recipe_id": "review-sentiment",
        "model": {"id": LLM_BASES[0], "revision": COMMIT},
        "dataset": {"sha256": digest, "bytes": nbytes, "train_rows": n_train, "test_rows": n_test},
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
        "max_seq_length": 64,
        "logical_batch_size": 4,
        "total_logical_steps": steps,
        "seed": 7,
        "checkpoint_every_steps": 2,
        "optimizer": {"name": "adamw", "lr": 1e-2, "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8, "max_grad_norm": 1.0},
        "lr_schedule": {"name": "linear_warmup_linear_decay", "warmup_steps": 1},
    }


def tiny_pair(train_rows, test_rows):
    texts = [row["text"] + " " + row["label"] for row in train_rows + test_rows]
    tokenizer = build_tiny_tokenizer(texts)
    return build_tiny_model(tokenizer), tokenizer


class Runtime(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.blob, self.train_rows, self.test_rows, digest, nbytes = make_data(self.tmp)
        self.spec = make_spec(digest, nbytes, len(self.train_rows), len(self.test_rows))

    def test_generate_answers_contract(self):
        model, tokenizer = tiny_pair(self.train_rows, self.test_rows)
        answers = generate_answers(model, tokenizer, ["Review: yeh phone\nSentiment:", "Review: bekar\nSentiment:"],
                                   max_new_tokens=4, max_prompt_tokens=32, device="cpu")
        self.assertEqual(len(answers), 2)
        self.assertTrue(all(isinstance(a, str) for a in answers))

    def test_4bit_teacher_uses_supported_compute_dtype(self):
        with mock.patch('transformers.BitsAndBytesConfig') as config, \
                mock.patch('transformers.AutoModelForCausalLM.from_pretrained'), \
                mock.patch.object(torch.cuda, 'get_device_capability', return_value=(7, 5)):
            load_model('teacher', device='cuda:0', load_in_4bit=True)
        self.assertEqual(config.call_args.kwargs['bnb_4bit_compute_dtype'], torch.float16)
        with self.assertRaisesRegex(ValueError, 'requires a CUDA device'):
            load_model('teacher', device='cpu', load_in_4bit=True)

    def test_probe_reports_a_real_measurement(self):
        model, tokenizer = tiny_pair(self.train_rows, self.test_rows)
        model = train.apply_lora(model, self.spec)
        before = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}
        forwards = []
        hook = model.register_forward_hook(lambda *_args: forwards.append(True))
        original_step = torch.optim.AdamW.step
        observed = []
        def checked_step(optimizer, *args, **kwargs):
            result = original_step(optimizer, *args, **kwargs)
            observed.append(bool(optimizer.state))
            return result
        with mock.patch.object(torch.optim.AdamW, 'step', checked_step):
            result = probe_step(model, tokenizer, [row["text"] for row in self.train_rows],
                                micro_batch=2, accumulation=2, max_seq_length=64, device="cpu")
        self.assertTrue(result.ok, result.error)
        self.assertGreater(result.peak_mem_mib, 0)
        self.assertEqual(observed, [True])
        self.assertEqual(len(forwards), 2)
        hook.remove()
        for name, param in model.named_parameters():
            if name in before:
                self.assertTrue(torch.equal(param, before[name]), name)

    def test_probe_classifies_cuda_library_allocation_failure(self):
        model, tokenizer = tiny_pair(self.train_rows, self.test_rows)
        model = train.apply_lora(model, self.spec)
        before = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}
        with mock.patch.object(model, 'forward', side_effect=RuntimeError('CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate')):
            result = probe_step(model, tokenizer, [self.train_rows[0]['text']],
                                micro_batch=1, max_seq_length=64, device='cpu')
        self.assertFalse(result.ok)
        self.assertTrue(result.oom)
        for name, param in model.named_parameters():
            if name in before:
                self.assertTrue(torch.equal(param, before[name]), name)

    def test_compiler_approves_a_config_for_cpu(self):
        model, tokenizer = tiny_pair(self.train_rows, self.test_rows)
        model = train.apply_lora(model, self.spec)
        config = train.decide_exec_config(self.spec, model, tokenizer,
                                          [row["text"] for row in self.train_rows[:4]], device="cpu")
        self.assertEqual(config["micro_batch"] * config["accumulation"], self.spec["logical_batch_size"])

    def test_training_checkpoints_and_resume_matches(self):
        # Uninterrupted: 4 steps.
        model_a, tokenizer = tiny_pair(self.train_rows, self.test_rows)
        with mock.patch.object(train, "_collate", wraps=train._collate) as collate:
            evidence_a = train.run_training(self.spec, self.blob, self.tmp / "run-a",
                                            device="cpu", model=model_a, tokenizer=tokenizer,
                                            exec_config=EXEC, measure_training=True, log=lambda *a: None)
        batches = [call.args[0] for call in collate.call_args_list]
        metrics = evidence_a["training_metrics"]
        self.assertEqual(metrics["steps"], 4)
        self.assertEqual(metrics["non_padding_tokens"], sum(len(ids) for batch in batches for ids, _ in batch))
        self.assertEqual(metrics["input_tokens_including_padding"], sum(max(len(ids) for ids, _ in batch) * len(batch) for batch in batches))
        self.assertGreater(metrics["seconds"], 0)
        self.assertAlmostEqual(metrics["non_padding_tokens_per_second"], metrics["non_padding_tokens"] / metrics["seconds"])
        self.assertIsNone(metrics["gpu_name"])
        self.assertIsNone(metrics["peak_allocated_bytes"])
        self.assertEqual(evidence_a["eval"]["examples"], 20)
        self.assertEqual(len(evidence_a["checkpoints"]), 2)
        manifest = json.loads((self.tmp / "run-a" / "checkpoints" / "step-4" / "manifest.json").read_text())
        self.assertEqual(manifest["format"], "GPCK/1")
        self.assertEqual(manifest["logical_step"], 4)
        final_a = {n: p.detach().clone() for n, p in model_a.named_parameters() if p.requires_grad}

        # Stopped after step 2 (same spec — a spec never changes), resumed in a fresh
        # model: must land on the same weights.
        model_b, _ = tiny_pair(self.train_rows, self.test_rows)
        first_segment = train.run_training(self.spec, self.blob, self.tmp / "run-b1",
                                           device="cpu", model=model_b, tokenizer=tokenizer,
                                           exec_config=EXEC, stop_after=2, log=lambda *a: None)
        self.assertNotIn("training_metrics", first_segment)
        model_c, _ = tiny_pair(self.train_rows, self.test_rows)
        resumed = train.run_training(self.spec, self.blob, self.tmp / "run-b2",
                                     device="cpu", model=model_c, tokenizer=tokenizer,
                                     resume=self.tmp / "run-b1" / "checkpoints" / "step-2",
                                     exec_config=EXEC, measure_training=True, log=lambda *a: None)
        self.assertEqual(resumed["training_metrics"]["steps"], 2)
        self.assertLess(resumed["training_metrics"]["non_padding_tokens"], metrics["non_padding_tokens"])
        final_c = {n: p.detach().clone() for n, p in model_c.named_parameters() if p.requires_grad}
        self.assertEqual(set(final_a), set(final_c))
        for name in final_a:
            self.assertTrue(torch.allclose(final_a[name], final_c[name], atol=1e-6), name)

    def test_dataset_mismatch_is_refused(self):
        bad = dict(self.spec, dataset=dict(self.spec["dataset"], sha256="b" * 64))
        model, tokenizer = tiny_pair(self.train_rows, self.test_rows)
        with self.assertRaises(ValueError):
            train.run_training(bad, self.blob, self.tmp / "run-bad",
                               device="cpu", model=model, tokenizer=tokenizer,
                               exec_config=EXEC, log=lambda *a: None)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for GPU metrics")
    def test_training_reports_cuda_metrics(self):
        model, tokenizer = tiny_pair(self.train_rows, self.test_rows)
        evidence = train.run_training(self.spec, self.blob, self.tmp / "measured-gpu",
                                      device="cuda", model=model, tokenizer=tokenizer,
                                      exec_config=EXEC, stop_after=1, measure_training=True, log=lambda *a: None)
        metrics = evidence["training_metrics"]
        self.assertEqual(metrics["steps"], 1)
        self.assertGreater(metrics["non_padding_tokens"], 0)
        self.assertGreater(metrics["seconds"], 0)
        self.assertEqual(metrics["gpu_name"], torch.cuda.get_device_name())
        self.assertGreater(metrics["peak_allocated_bytes"], 0)
        self.assertGreaterEqual(metrics["peak_reserved_bytes"], metrics["peak_allocated_bytes"])

    def test_export_merges_checkpoint_to_safetensors(self):
        model, tokenizer = tiny_pair(self.train_rows, self.test_rows)
        train.run_training(self.spec, self.blob, self.tmp / "run-x",
                           device="cpu", model=model, tokenizer=tokenizer,
                           exec_config=EXEC, log=lambda *a: None)
        fresh, _ = tiny_pair(self.train_rows, self.test_rows)
        out = self.tmp / "merged"
        info = export.merge_checkpoint(self.spec, self.tmp / "run-x" / "checkpoints" / "step-4", out,
                                       base_model=fresh, tokenizer=tokenizer)
        self.assertEqual(info["logical_step"], 4)
        from transformers import AutoModelForCausalLM
        loaded = AutoModelForCausalLM.from_pretrained(out)
        self.assertEqual(sum(p.numel() for p in loaded.parameters()), sum(p.numel() for p in fresh.parameters()))

    def test_export_finds_release_build_tools(self):
        build = self.tmp / 'llama' / 'build' / 'bin' / 'Release'
        build.mkdir(parents=True)
        name = 'llama-quantize.exe' if __import__('os').name == 'nt' else 'llama-quantize'
        binary = build / name
        binary.touch()
        self.assertEqual(export._llama_binary(self.tmp / 'llama', 'llama-quantize'), binary)

    def _legacy_llama_binary(self, _root, name):
        if name == 'llama-completion':
            raise FileNotFoundError(name)
        return self.tmp / 'llama-cli'

    def test_export_raw_completion_disables_chat_template(self):
        with mock.patch.object(export, '_llama_binary', return_value=self.tmp / 'llama-completion'), \
                mock.patch.object(export.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=' order\n')) as run:
            self.assertEqual(export.llama_cli_generate(self.tmp, self.tmp / 'model.gguf', 'prompt', n_predict=4), 'order')
        command = run.call_args.args[0]
        self.assertIn('--no-conversation', command)
        self.assertIn('--no-display-prompt', command)
        self.assertIn('--simple-io', command)
        self.assertEqual(command[command.index('-c') + 1], '512')
        self.assertNotIn('--single-turn', command)

    def test_export_legacy_refuses_implicit_chat_fallback(self):
        with mock.patch.object(export, '_llama_binary', side_effect=self._legacy_llama_binary), \
                mock.patch.object(export.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stdout='invalid argument -no-cnv')) as run:
            with self.assertRaisesRegex(RuntimeError, 'llama-cli failed'):
                export.llama_cli_generate(self.tmp, self.tmp / 'model.gguf', 'prompt')
        self.assertEqual(run.call_count, 1)
        self.assertIn('-no-cnv', ' '.join(map(str, run.call_args.args[0])))

    def test_export_rejects_empty_cli_output(self):
        with mock.patch.object(export, '_llama_binary', side_effect=self._legacy_llama_binary), \
                mock.patch.object(export.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='')):
            with self.assertRaisesRegex(RuntimeError, 'no captured answer'):
                export.llama_cli_generate(self.tmp, self.tmp / 'model.gguf', 'prompt')

    def test_export_reads_answer_after_echoed_prompt(self):
        output = '\x1b[32mLoading model\x1b[0m\n> prompt\norder\nExiting...\n'
        with mock.patch.object(export, '_llama_binary', side_effect=self._legacy_llama_binary), \
                mock.patch.object(export.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=output)):
            self.assertEqual(export.llama_cli_generate(self.tmp, self.tmp / 'model.gguf', 'prompt'), 'order')

    def test_export_rejects_startup_banner_without_answer(self):
        with mock.patch.object(export, '_llama_binary', side_effect=self._legacy_llama_binary), \
                mock.patch.object(export.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='Loading model\nExiting...')):
            with self.assertRaisesRegex(RuntimeError, 'no captured answer'):
                export.llama_cli_generate(self.tmp, self.tmp / 'model.gguf', 'prompt')

    def test_export_reads_multiline_prompt_from_t4_terminal_capture(self):
        prompt = ('Sort this customer support ticket into exactly one category: complaint, order, question, refund or other.\n'
                  'Ticket: Mera order abhi tak nahi aaya, 5 din ho gaye\nCategory:')
        fixture = Path(__file__).parent / 'fixtures' / 'llama-cli-t4.txt'
        output = fixture.read_bytes().decode('utf-8').replace('\r\n', '\n').replace('\n', '\r\n')
        with mock.patch.object(export, '_llama_binary', side_effect=self._legacy_llama_binary), \
                mock.patch.object(export.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=output)):
            self.assertEqual(export.llama_cli_generate(self.tmp, self.tmp / 'model.gguf', prompt), 'order')
    def test_distill_pins_the_default_teacher_and_refuses_an_unpinned_one(self):
        self.assertRegex(distill.TEACHER_REVISION, r'^[0-9a-f]{40}$')
        with self.assertRaisesRegex(ValueError, 'pin a revision'):
            distill.run_distillation('review-sentiment', [], [], self.tmp / 'out', teacher_id='someone/other-teacher')

    def test_export_step_that_hangs_is_stopped(self):
        with mock.patch.object(export.subprocess, 'run', side_effect=export.subprocess.TimeoutExpired('convert', 1)):
            with self.assertRaisesRegex(RuntimeError, 'did not finish'):
                export._run(['convert', 'x'])

    def test_distill_filters_generated_test_inputs(self):
        outputs = [{'text': '  SAME  test  ', 'label': 'positive'},
                   {'text': 'novel example', 'label': 'negative'}]
        with mock.patch.object(distill, 'generate_rows', return_value=outputs), \
                mock.patch.object(train, 'run_training', return_value={'eval': {}, 'base_eval': {}}):
            result = distill.run_distillation(
                'review-sentiment', [{'text': 'seed text', 'label': 'positive'}],
                [{'text': 'same test', 'label': 'positive'}], self.tmp / 'filter',
                teacher_model=object(), teacher_tokenizer=object(), student_model=object(),
                student_tokenizer=object(), device='cpu', log=lambda *a: None)
        self.assertEqual(result['generated_rows'], 1)
        self.assertEqual(result['generation_settings']['seed'], 42)
        self.assertEqual(result['generation_settings']['temperature'], 0.8)
        self.assertIsNone(result['teacher_requested_revision'])  # injected, not loaded by this run
        self.assertEqual(result['generation_filter_counts']['trailing_label_rows_dropped'], 0)
        lines = (self.tmp / 'filter' / 'student_dataset.jsonl').read_text(encoding='utf-8').splitlines()
        self.assertEqual(json.loads(lines[0])['text'], 'novel example')

    def test_distill_records_requested_pin_when_model_reports_no_commit(self):
        teacher = SimpleNamespace(config=SimpleNamespace(_commit_hash=None))
        generated = [{'text': 'novel example', 'label': 'positive'}]
        with mock.patch('ekido_runtime.llm.load_model', return_value=teacher) as load, \
                mock.patch('ekido_runtime.llm.load_tokenizer', return_value=object()), \
                mock.patch.object(distill, 'generate_rows', return_value=generated), \
                mock.patch.object(train, 'run_training', return_value={'eval': {}, 'base_eval': {}}):
            result = distill.run_distillation(
                'review-sentiment', [{'text': 'seed text', 'label': 'positive'}],
                [{'text': 'test text', 'label': 'positive'}], self.tmp / 'requested-pin',
                student_model=object(), student_tokenizer=object(), device='cpu', log=lambda *a: None)
        self.assertEqual(load.call_args.args[1], distill.TEACHER_REVISION)
        self.assertIsNone(result['teacher_revision'])
        self.assertEqual(result['teacher_requested_revision'], distill.TEACHER_REVISION)
        saved = json.loads((self.tmp / 'requested-pin' / 'distill_evidence.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['teacher_requested_revision'], distill.TEACHER_REVISION)
        self.assertIsNone(saved['teacher_revision'])

    def test_distill_cli_passes_selected_student(self):
        seeds, tests = self.tmp / 'seeds.jsonl', self.tmp / 'test.jsonl'
        seeds.write_text('{"text":"seed","label":"positive"}\n', encoding='utf-8')
        tests.write_text('{"text":"test","label":"positive"}\n', encoding='utf-8')
        result = {'generated_rows': 1, 'student_eval': {'pass_rate': 0.5}}
        with mock.patch.object(distill, 'run_distillation', return_value=result) as run:
            self.assertEqual(distill.main(['--recipe', 'review-sentiment', '--seed', str(seeds),
                                           '--test', str(tests), '--out', str(self.tmp / 'result'),
                                           '--student', LLM_BASES[1]]), 0)
        self.assertEqual(run.call_args.kwargs['student_id'], LLM_BASES[1])

    def test_long_completion_is_bounded_and_ends_with_eos(self):
        class Tokenizer:
            eos_token_id = 99
            def __call__(self, text, add_special_tokens=False):
                return {'input_ids': list(range(len(text)))}
        ids, labels = train.encode_example(Tokenizer(), {'prompt': 'abc', 'completion': 'x' * 30}, 8)
        self.assertEqual(len(ids), 8)
        self.assertEqual(ids[-1], 99)
        self.assertEqual(labels, ids)

    def test_distill_tiny_teacher_to_student(self):
        seeds = rows(POS[:2], "positive") + rows(NEG[:2], "negative") + rows(NEU[:2], "neutral")
        teacher, tokenizer = tiny_pair(self.train_rows, self.test_rows)
        student, _ = tiny_pair(self.train_rows, self.test_rows)
        result = distill.run_distillation(
            "review-sentiment", seeds, self.test_rows, self.tmp / "distill",
            teacher_model=teacher, teacher_tokenizer=tokenizer,
            student_model=student, student_tokenizer=tokenizer,
            teacher_id="tiny-qwen2", per_seed=1, steps=2,
            log=lambda *a: None,
        )
        self.assertGreaterEqual(result["generated_rows"], 1)
        self.assertEqual(result["student_eval"]["examples"], 20)
        self.assertTrue((self.tmp / "distill" / "distill_evidence.json").is_file())


if __name__ == "__main__":
    unittest.main()

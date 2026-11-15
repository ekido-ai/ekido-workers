"""The correctness tests the whole design rests on. Run on CPU, so they need no GPU.

From the repo root:
  PYTHONPATH=packages/ekido_core:workers/runtime python -m unittest discover -s workers/runtime/tests
"""

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from ekido_core.canonical import document_hash
from ekido_core.gpck import (
    OPTIMIZER_FILE,
    TRAINABLE_FILE,
    CheckpointError,
    load_checkpoint,
    manifest_hash,
    optimizer_state_to_tensors,
    restore_optimizer_state,
    write_checkpoint,
)
from ekido_core.logical import logical_batch_indices
from ekido_runtime.synthetic import SyntheticClassifier, SyntheticData, exec_config, synthetic_spec
from ekido_runtime.trainer import (
    accumulate_gradients,
    amp_dtype,
    canary_loss,
    canary_matches,
    create_optimizer,
    run_logical_step,
    trainable_named_params,
)


def make_model(dtype=torch.float32, dropout=0.0, seed=7):
    torch.manual_seed(seed)
    return SyntheticClassifier(dropout=dropout).to(dtype)


def gradients(model):
    return {n: p.grad.clone() for n, p in model.named_parameters()}


def max_relative_difference(a, b):
    worst = 0.0
    for name in a:
        worst = max(worst, float((a[name] - b[name]).abs().max() / (b[name].abs().max() + 1e-30)))
    return worst


class GradientInvariance(unittest.TestCase):
    """Plan section I: changing micro-batch must not change the logical step's gradient."""

    def grads_for(self, micro, dtype, sizes=None):
        data = SyntheticData(dtype=dtype)
        model = make_model(dtype)
        indices = logical_batch_indices(1337, 0, 32, data.size)
        model.zero_grad()
        accumulate_gradients(model, data.batch, indices, micro, "none", "cpu", None, 1337, 0)
        return gradients(model)

    def test_fp64_exact_to_1e9(self):
        reference = self.grads_for(32, torch.float64)
        for micro in (16, 8, 4, 2, 1):
            self.assertLessEqual(max_relative_difference(self.grads_for(micro, torch.float64), reference), 1e-9, msg=f"micro={micro}")

    def test_fp32_within_1e4(self):
        reference = self.grads_for(32, torch.float32)
        for micro in (16, 8, 4, 2, 1):
            self.assertLessEqual(max_relative_difference(self.grads_for(micro, torch.float32), reference), 1e-4, msg=f"micro={micro}")

    def test_unequal_micro_batches_stay_correct(self):
        # 32 examples split as 20 + 12 still has to equal the full-batch gradient.
        data = SyntheticData(dtype=torch.float64)
        indices = logical_batch_indices(1337, 0, 32, data.size)

        model = make_model(torch.float64)
        model.zero_grad()
        for part in (indices[:20], indices[20:]):
            batch = data.batch(part)
            (model(**batch).loss * (len(part) / 32)).backward()
        unequal = gradients(model)

        self.assertLessEqual(max_relative_difference(unequal, self.grads_for(32, torch.float64)), 1e-9)

    def test_naive_averaging_would_fail(self):
        # Guards the test itself: the common divide-by-accumulation bug must be detectably wrong here.
        data = SyntheticData(dtype=torch.float64)
        indices = logical_batch_indices(1337, 0, 32, data.size)
        model = make_model(torch.float64)
        model.zero_grad()
        for part in (indices[:20], indices[20:]):
            (model(**data.batch(part)).loss / 2).backward()
        self.assertGreater(max_relative_difference(gradients(model), self.grads_for(32, torch.float64)), 1e-6)

    def test_examples_are_identical_for_any_micro_batch(self):
        for step in (0, 5, 100):
            self.assertEqual(len(set(logical_batch_indices(1337, step, 32, 2048))), 32)


def train(model, optimizer, data, spec, exec_cfg, start, stop):
    results = []
    for step in range(start, stop):
        results.append(run_logical_step(model, optimizer, data.batch, spec, exec_cfg, step, data.size, "cpu"))
    return results


def save(directory, model, optimizer, step, spec, exec_cfg, data=None):
    named = trainable_named_params(model)
    names = {id(p): n for n, p in named.items()}
    trainable = {n: p.detach().cpu().clone() for n, p in named.items()}
    canary = canary_loss(model, data.batch(range(64)), "cpu") if data else 0.0
    manifest = {
        "job_id": "00000000-0000-0000-0000-000000000001",
        "attempt": 1,
        "fencing_token": 1,
        "logical_step": step,
        "epoch": 0,
        "immutable_hash": document_hash(spec),
        "exec": exec_cfg,
        "scaler": None,
        "canary": {"loss_fp32": canary},
        "parent_manifest_hash": None,
    }
    return write_checkpoint(Path(directory), trainable, optimizer_state_to_tensors(optimizer, names), manifest)


def restore(directory, spec, dtype=torch.float32, dropout=0.0):
    # Deliberately a different initial seed: every weight must come from the checkpoint.
    model = make_model(dtype, dropout, seed=999)
    named = trainable_named_params(model)
    shapes = {n: tuple(p.shape) for n, p in named.items()}
    manifest, trainable, optimizer_tensors = load_checkpoint(directory, shapes, dtype=dtype, expected_immutable_hash=document_hash(spec))
    with torch.no_grad():
        for name, param in named.items():
            param.copy_(trainable[name])
    optimizer = create_optimizer(list(named.values()), spec["optimizer"])
    restore_optimizer_state(optimizer, named, optimizer_tensors)
    return model, optimizer, manifest


def flat(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


class LogicalSteps(unittest.TestCase):
    spec = synthetic_spec()

    def test_micro_batch_32_and_8x4_reach_the_same_logical_step(self):
        # 8 x 4 accumulation is still 30 logical steps and 30 optimizer updates, not 120.
        dtype = torch.float64
        data = SyntheticData(dtype=dtype)
        self.assertEqual(exec_config(8)["accumulation"], 4)

        runs = {}
        for micro in (32, 8):
            model = make_model(dtype)
            optimizer = create_optimizer(list(model.parameters()), self.spec["optimizer"])
            results = train(model, optimizer, data, self.spec, exec_config(micro), 0, 30)
            updates = {int(state["step"]) for state in optimizer.state.values()}
            runs[micro] = (results, updates, flat(model))

        full, accumulated = runs[32], runs[8]
        self.assertEqual([r.step for r in full[0]], [r.step for r in accumulated[0]])
        self.assertEqual([r.lr for r in full[0]], [r.lr for r in accumulated[0]])
        self.assertEqual(full[1], {30})
        self.assertEqual(accumulated[1], {30})
        # Dropout is off and this is FP64, so changing only the execution plan leaves almost no trace.
        self.assertLess(float((full[2] - accumulated[2]).abs().max()), 1e-8)

    def test_training_actually_learns(self):
        data = SyntheticData()
        model = make_model()
        optimizer = create_optimizer(list(model.parameters()), self.spec["optimizer"])
        results = train(model, optimizer, data, self.spec, exec_config(32), 0, 40)
        self.assertLess(sum(r.loss for r in results[-5:]) / 5, sum(r.loss for r in results[:5]) / 5)

    def test_invalid_exec_config_refused(self):
        data = SyntheticData()
        model = make_model()
        optimizer = create_optimizer(list(model.parameters()), self.spec["optimizer"])
        bad = dict(exec_config(8), accumulation=3)
        with self.assertRaises(ValueError):
            run_logical_step(model, optimizer, data.batch, self.spec, bad, 0, data.size, "cpu")


class ResumeCorrectness(unittest.TestCase):
    spec = synthetic_spec()

    def test_resume_same_config_matches_uninterrupted_exactly(self):
        for dropout in (0.0, 0.3):
            with self.subTest(dropout=dropout), tempfile.TemporaryDirectory() as tmp:
                data = SyntheticData()
                cfg = exec_config(8)

                control = make_model(dropout=dropout)
                control_opt = create_optimizer(list(control.parameters()), self.spec["optimizer"])
                train(control, control_opt, data, self.spec, cfg, 0, 30)

                first = make_model(dropout=dropout)
                first_opt = create_optimizer(list(first.parameters()), self.spec["optimizer"])
                train(first, first_opt, data, self.spec, cfg, 0, 15)
                save(tmp, first, first_opt, 15, self.spec, cfg)

                resumed, resumed_opt, manifest = restore(tmp, self.spec, dropout=dropout)
                train(resumed, resumed_opt, data, self.spec, cfg, manifest["logical_step"], 30)
                # Same seeds, same order, same state, same machine, so exact equality is the right test.
                self.assertTrue(torch.equal(flat(control), flat(resumed)))

    def test_resume_with_changed_micro_batch_tracks_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            dtype = torch.float64
            data = SyntheticData(dtype=dtype)

            control = make_model(dtype)
            control_opt = create_optimizer(list(control.parameters()), self.spec["optimizer"])
            train(control, control_opt, data, self.spec, exec_config(32), 0, 30)

            first = make_model(dtype)
            first_opt = create_optimizer(list(first.parameters()), self.spec["optimizer"])
            train(first, first_opt, data, self.spec, exec_config(32), 0, 15)
            save(tmp, first, first_opt, 15, self.spec, exec_config(32))

            resumed, resumed_opt, _ = restore(tmp, self.spec, dtype=dtype)
            train(resumed, resumed_opt, data, self.spec, exec_config(4), 15, 30)
            # Dropout is off and this is FP64, so changing only the execution plan leaves almost no trace.
            self.assertLess(float((flat(control) - flat(resumed)).abs().max()), 1e-8)

    def test_canary_catches_a_broken_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = SyntheticData()
            model = make_model()
            optimizer = create_optimizer(list(model.parameters()), self.spec["optimizer"])
            train(model, optimizer, data, self.spec, exec_config(8), 0, 10)
            manifest = save(tmp, model, optimizer, 10, self.spec, exec_config(8), data)

            restored, _, _ = restore(tmp, self.spec)
            batch = data.batch(range(64))
            self.assertTrue(canary_matches(manifest["canary"]["loss_fp32"], canary_loss(restored, batch, "cpu")))
            # A freshly initialized model is what a silently failed restore would look like.
            self.assertFalse(canary_matches(manifest["canary"]["loss_fp32"], canary_loss(make_model(seed=999), batch, "cpu")))


class OptimizerStateOnSave(unittest.TestCase):
    def test_a_parameter_with_no_optimizer_state_is_stored_as_no_tensors(self):
        model = make_model()
        named = trainable_named_params(model)
        optimizer = torch.optim.AdamW(list(named.values()), lr=1e-3)
        self.assertEqual(optimizer_state_to_tensors(optimizer, {id(p): n for n, p in named.items()}), {})

    def test_partially_initialized_optimizer_state_is_refused(self):
        model = make_model()
        named = trainable_named_params(model)
        optimizer = torch.optim.AdamW(list(named.values()), lr=1e-3)
        first = next(iter(named.values()))
        optimizer.state[first] = {"step": torch.tensor(1.0)}
        with self.assertRaises(CheckpointError):
            optimizer_state_to_tensors(optimizer, {id(p): n for n, p in named.items()})


class SpareParameter(SyntheticClassifier):
    """Has a trainable parameter the forward pass never uses, so AdamW never creates state for it."""

    def __init__(self):
        super().__init__()
        self.spare = torch.nn.Parameter(torch.zeros(3))


def make_spare(seed=7):
    torch.manual_seed(seed)
    return SpareParameter()


class EmptyOptimizerState(unittest.TestCase):
    spec = synthetic_spec()

    def save(self, directory, model, optimizer, step):
        named = trainable_named_params(model)
        names = {id(p): n for n, p in named.items()}
        trainable = {n: p.detach().cpu().clone() for n, p in named.items()}
        manifest = {"logical_step": step, "immutable_hash": document_hash(self.spec), "parent_manifest_hash": None}
        return write_checkpoint(Path(directory), trainable, optimizer_state_to_tensors(optimizer, names), manifest)

    def restore(self, directory):
        # Deliberately a different initial seed: every weight must come from the checkpoint.
        model = make_spare(seed=999)
        named = trainable_named_params(model)
        shapes = {n: tuple(p.shape) for n, p in named.items()}
        manifest, trainable, optimizer_tensors = load_checkpoint(directory, shapes, expected_immutable_hash=document_hash(self.spec))
        with torch.no_grad():
            for name, param in named.items():
                param.copy_(trainable[name])
        optimizer = create_optimizer(list(named.values()), self.spec["optimizer"])
        restore_optimizer_state(optimizer, named, optimizer_tensors)
        return model, optimizer, manifest

    def test_save_then_restore_when_a_parameter_never_got_a_gradient(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = SyntheticData()
            cfg = exec_config(8)

            control = make_spare()
            control_opt = create_optimizer(list(control.parameters()), self.spec["optimizer"])
            train(control, control_opt, data, self.spec, cfg, 0, 20)

            first = make_spare()
            first_opt = create_optimizer(list(first.parameters()), self.spec["optimizer"])
            train(first, first_opt, data, self.spec, cfg, 0, 10)
            self.assertNotIn(first.spare, first_opt.state)
            self.save(tmp, first, first_opt, 10)

            resumed, resumed_opt, manifest = self.restore(tmp)
            self.assertNotIn(resumed.spare, resumed_opt.state)
            self.assertEqual(len(resumed_opt.state), len(first_opt.state))
            train(resumed, resumed_opt, data, self.spec, cfg, manifest["logical_step"], 20)
            self.assertTrue(torch.equal(flat(control), flat(resumed)))

    def test_checkpoint_before_the_first_step_saves_and_restores(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = make_spare()
            optimizer = create_optimizer(list(model.parameters()), self.spec["optimizer"])
            self.save(tmp, model, optimizer, 0)
            resumed, resumed_opt, _ = self.restore(tmp)
            self.assertEqual(len(resumed_opt.state), 0)
            self.assertTrue(torch.equal(flat(model), flat(resumed)))

    def test_state_cut_down_to_some_fields_is_refused_on_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = SyntheticData()
            model = make_spare()
            optimizer = create_optimizer(list(model.parameters()), self.spec["optimizer"])
            train(model, optimizer, data, self.spec, exec_config(8), 0, 2)
            manifest = self.save(tmp, model, optimizer, 2)

            path = Path(tmp) / OPTIMIZER_FILE
            names = {id(p): n for n, p in trainable_named_params(model).items()}
            tensors = optimizer_state_to_tensors(optimizer, names)
            dropped = next(name for name in tensors if name.endswith(".exp_avg"))
            del tensors[dropped]
            save_file(tensors, str(path))
            from ekido_core.gpck import sha256_file

            manifest["files"][OPTIMIZER_FILE] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            (Path(tmp) / "manifest.json").write_text(json.dumps(manifest))

            shapes = {n: tuple(p.shape) for n, p in trainable_named_params(model).items()}
            with self.assertRaises(CheckpointError) as caught:
                load_checkpoint(tmp, shapes, expected_immutable_hash=document_hash(self.spec))
            self.assertEqual(caught.exception.code, "MALFORMED_CHECKPOINT")


class MaliciousCheckpoints(unittest.TestCase):
    """Plan section I: everything here must be refused before any tensor reaches a model."""

    spec = synthetic_spec()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        data = SyntheticData()
        self.model = make_model()
        self.optimizer = create_optimizer(list(self.model.parameters()), self.spec["optimizer"])
        train(self.model, self.optimizer, data, self.spec, exec_config(32), 0, 2)
        self.manifest = save(self.dir, self.model, self.optimizer, 2, self.spec, exec_config(32))
        self.shapes = {n: tuple(p.shape) for n, p in trainable_named_params(self.model).items()}

    def refuse(self, code):
        with self.assertRaises(CheckpointError) as caught:
            load_checkpoint(self.dir, self.shapes, expected_immutable_hash=document_hash(self.spec))
        self.assertEqual(caught.exception.code, code)

    def rewrite_manifest(self):
        from ekido_core.gpck import sha256_file

        for name in (TRAINABLE_FILE, OPTIMIZER_FILE):
            path = self.dir / name
            self.manifest["files"][name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        (self.dir / "manifest.json").write_text(json.dumps(self.manifest))

    def test_valid_checkpoint_loads(self):
        load_checkpoint(self.dir, self.shapes, expected_immutable_hash=document_hash(self.spec))

    def test_flipped_byte(self):
        path = self.dir / TRAINABLE_FILE
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 0xFF
        path.write_bytes(raw)
        self.refuse("HASH_MISMATCH")

    def test_truncated_file(self):
        path = self.dir / OPTIMIZER_FILE
        path.write_bytes(path.read_bytes()[:50])
        self.refuse("HASH_MISMATCH")

    def test_missing_file(self):
        (self.dir / OPTIMIZER_FILE).unlink()
        self.refuse("MALFORMED_CHECKPOINT")

    def test_missing_manifest(self):
        (self.dir / "manifest.json").unlink()
        self.refuse("MALFORMED_CHECKPOINT")

    def test_garbage_manifest(self):
        (self.dir / "manifest.json").write_text("{not json")
        self.refuse("MALFORMED_CHECKPOINT")

    def test_wrong_job(self):
        self.manifest["immutable_hash"] = "0" * 64
        self.rewrite_manifest()
        self.refuse("INVALID_LINEAGE")

    def test_pickle_renamed_as_checkpoint_never_executes(self):
        marker = self.dir / "pwned"

        class Exploit:
            def __reduce__(self):
                return (Path.touch, (marker,))

        (self.dir / TRAINABLE_FILE).write_bytes(pickle.dumps({"state": Exploit()}))
        self.rewrite_manifest()
        self.refuse("MALFORMED_CHECKPOINT")
        self.assertFalse(marker.exists(), "the pickle payload executed")

    def replace_trainable(self, tensors):
        save_file(tensors, str(self.dir / TRAINABLE_FILE))
        self.rewrite_manifest()

    def good_state(self):
        return {n: p.detach().cpu().clone() for n, p in trainable_named_params(self.model).items()}

    def test_wrong_shape(self):
        state = self.good_state()
        name = next(iter(state))
        state[name] = torch.zeros(3, 3)
        self.replace_trainable(state)
        self.refuse("MALFORMED_CHECKPOINT")

    def test_extra_tensor(self):
        self.replace_trainable({**self.good_state(), "backdoor": torch.zeros(1)})
        self.refuse("MALFORMED_CHECKPOINT")

    def test_missing_tensor(self):
        state = self.good_state()
        state.pop(next(iter(state)))
        self.replace_trainable(state)
        self.refuse("MALFORMED_CHECKPOINT")

    def test_wrong_dtype(self):
        self.replace_trainable({n: t.double() for n, t in self.good_state().items()})
        self.refuse("MALFORMED_CHECKPOINT")

    def test_nan(self):
        state = self.good_state()
        name = next(iter(state))
        state[name] = state[name].clone()
        state[name].flatten()[0] = float("nan")
        self.replace_trainable(state)
        self.refuse("NAN_OR_INF")

    def test_inf(self):
        state = self.good_state()
        name = next(iter(state))
        state[name] = state[name].clone()
        state[name].flatten()[0] = float("inf")
        self.replace_trainable(state)
        self.refuse("NAN_OR_INF")

    def test_oversized_file(self):
        with self.assertRaises(CheckpointError) as caught:
            load_checkpoint(self.dir, self.shapes, max_file_bytes=10)
        self.assertEqual(caught.exception.code, "MALFORMED_CHECKPOINT")

    def test_manifest_hash_covers_content(self):
        changed = dict(self.manifest, logical_step=999)
        self.assertNotEqual(manifest_hash(self.manifest), manifest_hash(changed))

    def test_manifest_hash_ignores_signature(self):
        self.assertEqual(manifest_hash(self.manifest), manifest_hash(dict(self.manifest, signature="0x" + "a" * 130)))


class Precision(unittest.TestCase):
    def test_modes(self):
        self.assertEqual(amp_dtype("bf16"), torch.bfloat16)
        self.assertEqual(amp_dtype("fp16"), torch.float16)
        self.assertIsNone(amp_dtype("none"))
        with self.assertRaises(ValueError):
            amp_dtype("fp8")


if __name__ == "__main__":
    unittest.main()

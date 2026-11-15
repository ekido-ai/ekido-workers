"""End-to-end tests of the runtime process, on CPU with the synthetic workload."""

import json
import os
import subprocess
import sys
import tempfile
import time
import math
import unittest
from pathlib import Path
from unittest import mock
import torch

from gpuchain_core.gpck import manifest_hash, read_manifest
from gpuchain_runtime.run import EXIT_BAD_CHECKPOINT, EXIT_ERROR, EXIT_LEASE, EXIT_OK, EXIT_OOM, run, scaler_from_manifest, scaler_to_manifest
from gpuchain_runtime.synthetic import exec_config, synthetic_spec


def spec(total=20, every=5, **extra):
    s = synthetic_spec(total_steps=total)
    s.update(template="synthetic-linear/v1", checkpoint_every_steps=every, **extra)
    return s


def events_of(workdir):
    return [json.loads(line) for line in (Path(workdir) / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def config(workdir, s=None, attempt=1, token=1, micro=8, **extra):
    return {
        "job_id": "00000000-0000-0000-0000-000000000001",
        "attempt": attempt,
        "fencing_token": token,
        "workdir": str(workdir),
        "device": "cpu",
        "spec": s or spec(),
        "exec": exec_config(micro),
        "signer": "0x" + "a" * 40,
        "progress_seconds": 0,
        **extra,
    }


class RunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def checkpoints(self, workdir):
        return [e for e in events_of(workdir) if e["kind"] == "checkpoint.ready"]

    def test_full_run_reports_and_checkpoints(self):
        w = self.root / "a"
        self.assertEqual(run(config(w)), EXIT_OK)
        kinds = [e["kind"] for e in events_of(w)]
        self.assertEqual(kinds[0], "run.starting")
        self.assertEqual(kinds[-1], "run.exited")
        self.assertEqual([c["step"] for c in self.checkpoints(w)], [5, 10, 15, 20])

    def test_checkpoints_chain_to_their_parents(self):
        w = self.root / "a"
        run(config(w))
        checkpoints = self.checkpoints(w)
        self.assertIsNone(checkpoints[0]["manifest"]["parent_manifest_hash"])
        for parent, child in zip(checkpoints, checkpoints[1:]):
            self.assertEqual(child["manifest"]["parent_manifest_hash"], parent["manifest_hash"])
        for checkpoint in checkpoints:
            self.assertEqual(manifest_hash(read_manifest(Path(checkpoint["path"]))), checkpoint["manifest_hash"])

    def test_kill_and_resume_with_a_different_micro_batch(self):
        s = spec()
        first = self.root / "first"
        self.assertEqual(run(config(first, s, stop_at_step=10)), EXIT_OK)
        tip = self.checkpoints(first)[-1]

        second = self.root / "second"
        code = run(config(second, s, attempt=2, token=2, micro=2, resume_from=tip["path"]))
        self.assertEqual(code, EXIT_OK)

        restored = next(e for e in events_of(second) if e["kind"] == "run.restored")
        self.assertEqual(restored["step"], 10)
        self.assertTrue(restored["canary_ok"])
        resumed = self.checkpoints(second)
        self.assertEqual([c["step"] for c in resumed], [15, 20])
        self.assertEqual(resumed[0]["manifest"]["parent_manifest_hash"], tip["manifest_hash"])
        self.assertEqual(resumed[0]["manifest"]["exec"]["micro_batch"], 2)
        self.assertEqual(resumed[0]["manifest"]["fencing_token"], 2)

    def test_real_gradscaler_state_survives_checkpoint_and_resume_on_cpu(self):
        """Exercise the FP16 scaler protocol without requiring a GPU."""
        s = spec(total=10, every=5)
        first, second = self.root / "scaled_first", self.root / "scaled_second"
        with mock.patch("gpuchain_runtime.run.make_scaler", side_effect=lambda *_: torch.amp.GradScaler("cpu", growth_factor=3.0, backoff_factor=0.25, growth_interval=7)):
            self.assertEqual(run(config(first, s, stop_at_step=5)), EXIT_OK)
            checkpoint = self.checkpoints(first)[-1]
            self.assertEqual(checkpoint["manifest"]["scaler"]["growth_factor"], 3.0)
            self.assertEqual(checkpoint["manifest"]["scaler"]["growth_interval"], 7)
            self.assertEqual(run(config(second, s, attempt=2, token=2, resume_from=checkpoint["path"])), EXIT_OK)
        self.assertTrue(next(e for e in events_of(second) if e["kind"] == "run.restored")["canary_ok"])

    def test_legacy_two_field_gradscaler_state_loads(self):
        legacy = {"scale": 65536.0, "growth_tracker": 150}
        scaler = torch.amp.GradScaler("cpu")
        scaler.load_state_dict(scaler_from_manifest(legacy))
        self.assertEqual(scaler_to_manifest(scaler), {**legacy, "growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 2000})

    def test_incomplete_or_invalid_extended_scaler_state_is_refused(self):
        for fields in ({"growth_factor": 3.0}, {"growth_factor": 3.0, "backoff_factor": 1.0, "growth_interval": 5}, {"growth_factor": float("nan"), "backoff_factor": 0.5, "growth_interval": 5}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                scaler_from_manifest({"scale": 65536.0, "growth_tracker": 1, **fields})

    def test_resume_refuses_corrupt_checkpoint(self):
        first = self.root / "first"
        run(config(first, stop_at_step=5))
        tip = Path(self.checkpoints(first)[-1]["path"])
        raw = bytearray((tip / "trainable.safetensors").read_bytes())
        raw[-1] ^= 0xFF
        (tip / "trainable.safetensors").write_bytes(raw)

        second = self.root / "second"
        self.assertEqual(run(config(second, attempt=2, token=2, resume_from=str(tip))), EXIT_BAD_CHECKPOINT)
        exited = events_of(second)[-1]
        self.assertEqual((exited["reason"], exited["code"]), ("BAD_CHECKPOINT", "HASH_MISMATCH"))

    def test_failed_canary_is_named_for_what_failed(self):
        first = self.root / "first"
        self.assertEqual(run(config(first)), EXIT_OK)
        tip = self.checkpoints(first)[0]["path"]
        second = self.root / "second"
        with mock.patch("gpuchain_runtime.run.canary_matches", return_value=False):
            self.assertEqual(run(config(second, attempt=2, token=2, resume_from=tip)), EXIT_BAD_CHECKPOINT)
        exited = events_of(second)[-1]
        self.assertEqual((exited["reason"], exited["code"]), ("BAD_CHECKPOINT", "CANARY_MISMATCH"))

    def test_resume_refuses_checkpoint_from_another_job(self):
        first = self.root / "first"
        run(config(first, spec(), stop_at_step=5))
        other = spec(total=99)
        second = self.root / "second"
        self.assertEqual(run(config(second, other, attempt=2, token=2, resume_from=self.checkpoints(first)[-1]["path"])), EXIT_BAD_CHECKPOINT)

    def test_invalid_exec_config_exits_with_reason(self):
        w = self.root / "a"
        bad = config(w)
        bad["exec"]["accumulation"] = 3
        self.assertEqual(run(bad), EXIT_ERROR)
        self.assertIn("InvalidExecConfig", events_of(w)[-1]["detail"])

    def test_unlisted_workload_refused(self):
        w = self.root / "a"
        self.assertEqual(run(config(w, spec() | {"template": "evil/v1"})), EXIT_ERROR)
        self.assertIn("allowlist", events_of(w)[-1]["detail"])

    def test_emergency_checkpoint_on_request(self):
        w = self.root / "a"
        w.mkdir()
        (w / "checkpoint_now").touch()
        run(config(w, spec(total=7, every=100)))
        checkpoints = self.checkpoints(w)
        self.assertEqual(checkpoints[0]["step"], 1)
        self.assertTrue(checkpoints[0]["emergency"])
        self.assertFalse((w / "checkpoint_now").exists())

    def test_poison_is_labelled_and_changes_the_model(self):
        clean, bad = self.root / "clean", self.root / "bad"
        run(config(clean, spec(total=10, every=10)))
        run(config(bad, spec(total=10, every=10), fault={"kind": "POISON", "at_step": 5}))
        self.assertTrue(any(e["kind"] == "fault.applied" and e["injected"] for e in events_of(bad)))
        self.assertNotEqual(self.checkpoints(clean)[0]["manifest"]["files"], self.checkpoints(bad)[0]["manifest"]["files"])

    def test_injected_oom_reports_without_filling_memory(self):
        w = self.root / "oom"
        self.assertEqual(run(config(w, spec(total=20, every=5), fault={"kind": "OOM", "at_step": 3})), EXIT_OOM)
        last = events_of(w)[-1]
        self.assertEqual((last["kind"], last["reason"]), ("run.exited", "OOM"))
        self.assertTrue(any(e["kind"] == "fault.applied" and e["fault"] == "OOM" and e["injected"] for e in events_of(w)))

    def test_poisoning_provider_keeps_running_and_reporting(self):
        # The whole point: it looks healthy from the outside, so only an evaluation can catch it.
        w = self.root / "bad"
        self.assertEqual(run(config(w, spec(total=20, every=5), fault={"kind": "POISON", "at_step": 0})), EXIT_OK)
        self.assertEqual(events_of(w)[-1]["reason"], "FINISHED")
        self.assertEqual([c["step"] for c in self.checkpoints(w)], [5, 10, 15, 20])
        losses = [e["loss"] for e in events_of(w) if e["kind"] == "run.progress"]
        self.assertTrue(all(math.isfinite(loss) for loss in losses), "losses must stay finite")

    def test_diverged_training_reports_plainly(self):
        w = self.root / "lr"
        bad_spec = spec(total=30, every=5)
        bad_spec["optimizer"]["lr"] = 1e308  # Certain to overflow.
        bad_spec["optimizer"]["max_grad_norm"] = 1e308
        self.assertEqual(run(config(w, bad_spec)), EXIT_ERROR)
        self.assertEqual(events_of(w)[-1]["reason"], "DIVERGED")

    def test_no_op_provider_leaves_weights_unchanged(self):
        s = spec(total=10, every=5)
        first = self.root / "first"
        run(config(first, s, stop_at_step=5))
        tip = self.checkpoints(first)[-1]
        lazy = self.root / "lazy"
        run(config(lazy, s, attempt=2, token=2, resume_from=tip["path"], fault={"kind": "NO_OP"}))
        self.assertEqual(self.checkpoints(lazy)[-1]["manifest"]["files"]["trainable.safetensors"]["sha256"], tip["manifest"]["files"]["trainable.safetensors"]["sha256"])

    def _spawn(self, cfg):
        path = self.root / f"cfg_{time.time_ns()}.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        return subprocess.Popen([sys.executable, "-m", "gpuchain_runtime.run", str(path)], env=env)

    def test_crash_mid_checkpoint_leaves_earlier_checkpoint_usable(self):
        s = spec()
        w = self.root / "a"
        proc = self._spawn(config(w, s, fault={"kind": "CRASH_MID_CHECKPOINT", "at_step": 10}))
        self.assertEqual(proc.wait(timeout=120), EXIT_ERROR)
        checkpoints = self.checkpoints(w)
        self.assertEqual([c["step"] for c in checkpoints], [5])
        self.assertFalse((w / "checkpoints" / "step_00000010" / "manifest.json").exists())
        second = self.root / "b"
        self.assertEqual(run(config(second, s, attempt=2, token=2, resume_from=checkpoints[-1]["path"])), EXIT_OK)

    def test_stale_lease_kills_a_zombie_process(self):
        w = self.root / "a"
        lease = self.root / "lease"
        lease.touch()
        # A long job with a lease nobody renews: exactly the state after an agent dies.
        proc = self._spawn(config(w, spec(total=1_000_000, every=1_000_000), lease_file=str(lease), lease_timeout=2.0))
        self.assertEqual(proc.wait(timeout=120), EXIT_LEASE)
        self.assertEqual(events_of(w)[-1]["reason"], "LEASE_LOST")


if __name__ == "__main__":
    unittest.main()

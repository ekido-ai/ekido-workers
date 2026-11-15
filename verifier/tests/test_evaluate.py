"""The Quality Guard's whole job, end to end on CPU: an honest resume passes and bad work gets rejected."""

import json
import tempfile
import unittest
from pathlib import Path

from ekido_core.canonical import document_hash
from ekido_core.signing import sign_document
from ekido_runtime.run import run
from ekido_runtime.synthetic import exec_config, synthetic_spec
from ekido_verifier.evaluate import evaluate_checkpoint, load_eval_set, quorum_verdict, sign_report

JOB = "00000000-0000-0000-0000-000000000001"
PROVIDER_KEY = "0x" + "11" * 32
V1, V2, V3 = ("0x" + c * 64 for c in "abc")


def spec():
    s = synthetic_spec(total_steps=60, lr=0.01)
    s.update(template="synthetic-linear/v1", checkpoint_every_steps=20)
    return s


def policy(eval_set, margin=0.02, floor=0.0):
    return {
        "schema_version": 1,
        "metric": "accuracy",
        "min_absolute": floor,
        "non_inferiority_margin": margin,
        "confidence": 0.95,
        "bootstrap_resamples": 1000,
        "eval_set_hash": eval_set.hash,
        "eval_version_hash": "e" * 64,
    }


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.spec = spec()
        cls.eval_set = load_eval_set(cls.spec, examples=2000)
        cls.policy = policy(cls.eval_set)
        cfg = cls.config("honest", micro=8)
        run(cfg)
        cls.honest = cls.checkpoints("honest")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def config(cls, name, micro=8, attempt=1, token=1, **extra):
        return {
            "job_id": JOB, "attempt": attempt, "fencing_token": token, "workdir": str(cls.root / name),
            "device": "cpu", "spec": cls.spec, "exec": exec_config(micro), "signer": "0x" + "a" * 40, "progress_seconds": 60, **extra,
        }

    @classmethod
    def checkpoints(cls, name):
        lines = (cls.root / name / "events.jsonl").read_text(encoding="utf-8").splitlines()
        return [e for e in map(json.loads, lines) if e["kind"] == "checkpoint.ready"]

    def evaluate(self, checkpoint, reference, *, final=False, pol=None, **kw):
        return evaluate_checkpoint(
            job_id=JOB, spec=self.spec, policy=pol or self.policy, eval_set=self.eval_set,
            checkpoint_dir=Path(checkpoint["path"]), reference_dir=Path(reference["path"]) if reference else None,
            expected_signer=None, is_final=final, **kw,
        )

    def resumed_from(self, name, step_index, **fault_and_config):
        tip = self.honest[step_index]
        run(self.config(name, attempt=2, token=2, resume_from=tip["path"], **fault_and_config))
        return tip, self.checkpoints(name)


class Verdicts(Fixture):
    def test_training_beats_the_untrained_baseline(self):
        report = self.evaluate(self.honest[-1], None)
        self.assertEqual(report["verdict"], "PASS")
        self.assertGreater(report["metrics"]["accuracy"], report["comparison"]["reference_accuracy"] + 0.1)
        self.assertIsNone(report["reference_checkpoint_hash"])

    def test_honest_recovery_with_changed_micro_batch_passes(self):
        tip = self.honest[0]
        run(self.config("adapted", micro=2, attempt=2, token=2, resume_from=tip["path"]))
        report = self.evaluate(self.checkpoints("adapted")[-1], tip, final=True)
        self.assertEqual(report["verdict"], "PASS", msg=str(report["reasons"]))

    def test_poisoned_candidate_is_rejected(self):
        tip, bad = self.resumed_from("poison", 1, fault={"kind": "POISON", "at_step": 40})
        report = self.evaluate(bad[-1], tip)
        self.assertEqual(report["verdict"], "FAIL")
        self.assertIn("NON_INFERIORITY_FAILED", report["reasons"])
        self.assertLess(report["comparison"]["ci_lower"], -report["comparison"]["margin"])

    def test_lr_tampering_is_rejected(self):
        tip, bad = self.resumed_from("tamper", 1, fault={"kind": "LR_TAMPER", "at_step": 40})
        # A 50x LR damages this small model, measurably, within the remaining steps.
        report = self.evaluate(bad[-1], tip)
        self.assertEqual(report["verdict"], "FAIL", msg=f"{report['metrics']} {report['comparison']}")

    def test_lazy_provider_is_rejected(self):
        tip, lazy = self.resumed_from("lazy", 1, fault={"kind": "NO_OP"})
        report = self.evaluate(lazy[-1], tip)
        self.assertEqual(report["verdict"], "FAIL")
        self.assertIn("WEIGHTS_UNCHANGED", report["reasons"])

    def test_a_wider_mid_training_margin_never_loosens_the_final_check(self):
        tip, bad = self.resumed_from("wide_margin", 1, fault={"kind": "POISON", "at_step": 40})
        wide = dict(self.policy, probation_margin=0.99)
        middle = self.evaluate(bad[-1], tip, pol=wide, final=False)
        self.assertEqual(middle["comparison"]["margin"], 0.99)
        self.assertNotIn("NON_INFERIORITY_FAILED", middle["reasons"])
        final = self.evaluate(bad[-1], tip, pol=wide, final=True)
        self.assertEqual(final["comparison"]["margin"], self.policy["non_inferiority_margin"])
        self.assertIn("NON_INFERIORITY_FAILED", final["reasons"])

    def test_floor_applies_only_to_final(self):
        strict = policy(self.eval_set, floor=0.999)
        self.assertEqual(self.evaluate(self.honest[-1], self.honest[0], pol=strict, final=False)["verdict"], "PASS")
        report = self.evaluate(self.honest[-1], self.honest[0], pol=strict, final=True)
        self.assertEqual(report["reasons"], ["BELOW_ABSOLUTE_FLOOR"])

    def test_wrong_eval_set_is_refused(self):
        wrong = dict(self.policy, eval_set_hash="0" * 64)
        with self.assertRaises(ValueError):
            self.evaluate(self.honest[-1], None, pol=wrong)

    def test_corrupt_checkpoint_is_a_fail_report_not_a_crash(self):
        run(self.config("corrupt"))
        checkpoint = self.checkpoints("corrupt")[-1]
        path = Path(checkpoint["path"]) / "trainable.safetensors"
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 0xFF
        path.write_bytes(raw)
        report = self.evaluate(checkpoint, None)
        self.assertEqual((report["verdict"], report["reasons"]), ("FAIL", ["HASH_MISMATCH"]))

    def test_bad_provider_signature_fails_despite_good_accuracy(self):
        checkpoint = self.honest[-1]
        manifest = json.loads((Path(checkpoint["path"]) / "manifest.json").read_text(encoding="utf-8"))
        signer, signature = sign_document(manifest, PROVIDER_KEY)

        good = evaluate_checkpoint(job_id=JOB, spec=self.spec, policy=self.policy, eval_set=self.eval_set, checkpoint_dir=Path(checkpoint["path"]), reference_dir=None, expected_signer=signer, is_final=False, provider_signature=signature)
        self.assertEqual(good["verdict"], "PASS")

        bad = evaluate_checkpoint(job_id=JOB, spec=self.spec, policy=self.policy, eval_set=self.eval_set, checkpoint_dir=Path(checkpoint["path"]), reference_dir=None, expected_signer="0x" + "9" * 40, is_final=False, provider_signature=signature)
        self.assertEqual(bad["reasons"], ["INVALID_SIGNATURE"])

    def test_broken_lineage_fails(self):
        report = self.evaluate(self.honest[1], None, expected_parent="f" * 64)
        self.assertIn("INVALID_LINEAGE", report["reasons"])

    def test_evaluation_repeats_exactly(self):
        a, b = self.evaluate(self.honest[-1], None), self.evaluate(self.honest[-1], None)
        self.assertEqual(a["correctness_hash"], b["correctness_hash"])
        self.assertEqual(a["comparison"]["ci_lower"], b["comparison"]["ci_lower"])

    def test_report_hash_is_stable_json(self):
        document_hash(self.evaluate(self.honest[-1], None))  # Raises if anything isn't canonical-JSON safe.


class Quorum(Fixture):
    def reports(self, verdicts):
        base = self.evaluate(self.honest[-1], None)
        return base["checkpoint_hash"], [sign_report(dict(base, verdict=v), k) for v, k in verdicts]

    def test_two_of_three_pass(self):
        checkpoint, reports = self.reports([("PASS", V1), ("PASS", V2)])
        self.assertEqual(quorum_verdict(reports, checkpoint, 2), "PASS")

    def test_below_quorum_is_undecided(self):
        checkpoint, reports = self.reports([("PASS", V1), ("FAIL", V2)])
        self.assertIsNone(quorum_verdict(reports, checkpoint, 2))

    def test_failure_quorum(self):
        checkpoint, reports = self.reports([("FAIL", V1), ("FAIL", V2), ("PASS", V3)])
        self.assertEqual(quorum_verdict(reports, checkpoint, 2), "FAIL")

    def test_one_verifier_cannot_vote_twice(self):
        checkpoint, reports = self.reports([("PASS", V1), ("PASS", V1)])
        self.assertIsNone(quorum_verdict(reports, checkpoint, 2))

    def test_report_for_another_checkpoint_does_not_count(self):
        checkpoint, reports = self.reports([("PASS", V1), ("PASS", V2)])
        self.assertIsNone(quorum_verdict(reports, "0" * 64, 2))

    def test_tampered_report_does_not_count(self):
        checkpoint, reports = self.reports([("FAIL", V1), ("PASS", V2)])
        reports[0]["verdict"] = "PASS"  # Flipped after signing.
        self.assertIsNone(quorum_verdict(reports, checkpoint, 2))


if __name__ == "__main__":
    unittest.main()

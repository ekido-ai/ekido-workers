import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from ekido_core.signing import recover_verdict_signer
from ekido_verifier.payout_check import PayoutEvidenceError
from ekido_verifier.secure_verdict import sign_checked_verdict
from ekido_verifier.worker import Verifier, verdict_from_inputs
import test_payout_check


class SecureVerdict(unittest.TestCase):
    def setUp(self):
        self.fixture = test_payout_check.PayoutCheck("test_accepts_valid_lineage_with_a_restarted_workers_old_key")
        self.fixture.setUp()
        self.key = "0x" + "d" * 64
        self.fixture.inputs["expiry"] = 1900000000

    def sign(self):
        f = self.fixture
        return sign_checked_verdict(f.inputs, f.spec, f.policy, rpc_url="http://independent-rpc", chain_id=31337, escrow_address=f.chain["contract"], key=self.key, own_final_reports=f.own_reports)

    def test_signs_correct_payout_using_configured_chain(self):
        f = self.fixture
        with patch("ekido_verifier.secure_verdict.read_chain_context", return_value=f.chain) as reader:
            signature = self.sign()
        self.assertEqual(recover_verdict_signer(verdict_from_inputs(f.inputs), 31337, f.chain["contract"], signature).lower(), f.v1)
        self.assertEqual(reader.call_args.args[:3], ("http://independent-rpc", 31337, f.chain["contract"]))

    def test_refuses_wrong_payout_and_unavailable_chain(self):
        f = self.fixture
        f.inputs["payouts"] = {"0x" + "9" * 40: "5"}
        f.inputs["refund"] = "0"
        with patch("ekido_verifier.secure_verdict.read_chain_context", return_value=f.chain):
            with self.assertRaises(PayoutEvidenceError): self.sign()
        with patch("ekido_verifier.secure_verdict.read_chain_context", side_effect=OSError("RPC down")):
            with self.assertRaises(OSError): self.sign()

    def test_fails_closed_for_a_failed_job_until_refund_path_is_ready(self):
        self.fixture.inputs["passed"] = False
        with self.assertRaises(PayoutEvidenceError): self.sign()

    def failing_inputs(self):
        f = self.fixture
        failure_hash = f.inputs['final_checkpoint_hash']
        v1_key, v2_key = f.keys[2:4]
        reports = []
        for report, key in zip(f.inputs['reports'], (v1_key, v2_key)):
            body = {k: v for k, v in report.items() if k != 'signature'}
            body['verdict'] = 'FAIL'
            body['reasons'] = ['BASE_MODEL_NOT_BEATEN']
            reports.append(test_payout_check.signed(body, key))
        f.inputs.update(passed=False, final_checkpoint_hash=None, failure_checkpoint_hash=failure_hash,
                        lineage=[], payouts={}, refund='5', reports=reports)
        from ekido_core.canonical import document_hash
        f.own_reports = {('job', failure_hash): document_hash(reports[0])}
        return f

    def test_signs_full_refund_only_for_own_signed_failure_quorum(self):
        f = self.failing_inputs()
        with patch('ekido_verifier.secure_verdict.read_chain_context', return_value=f.chain):
            signature = self.sign()
        self.assertEqual(recover_verdict_signer(verdict_from_inputs(f.inputs), 31337, f.chain['contract'], signature).lower(), f.v1)

    def test_refuses_forged_or_incomplete_refund_evidence(self):
        for change in ('refund', 'payout', 'binding', 'duplicate', 'bad_signature', 'missing_own', 'wrong_job', 'wrong_checkpoint', 'non_quality_failure'):
            with self.subTest(change=change):
                self.setUp()
                f = self.failing_inputs()
                if change == 'refund': f.inputs['refund'] = '4'
                if change == 'payout': f.inputs['payouts'] = {f.wallet: '1'}
                if change == 'binding': f.inputs['job_binding']['signature'] = '0x00'
                if change == 'duplicate': f.inputs['reports'][1] = f.inputs['reports'][0]
                if change == 'bad_signature': f.inputs['reports'][1]['signature'] = '0x00'
                if change == 'missing_own': f.own_reports = {}
                if change == 'wrong_job': f.inputs['job_id'] = 'another-job'
                if change == 'wrong_checkpoint': f.inputs['failure_checkpoint_hash'] = 'a' * 64
                if change == 'non_quality_failure':
                    body = {k: v for k, v in f.inputs['reports'][1].items() if k != 'signature'}
                    body['reasons'] = ['INVALID_SIGNATURE']
                    f.inputs['reports'][1] = test_payout_check.signed(body, f.keys[3])
                with patch('ekido_verifier.secure_verdict.read_chain_context', return_value=f.chain):
                    with self.assertRaises(PayoutEvidenceError): self.sign()

    def test_failed_report_context_survives_restart(self):
        f = self.failing_inputs()
        with tempfile.TemporaryDirectory() as directory:
            options = dict(rpc_url='http://independent-rpc', chain_id=31337,
                           escrow_address=f.chain['contract'], state_dir=directory)
            worker = Verifier('http://coordinator', self.key, 'cpu', 512, **options)
            worker._record_final_report(f.inputs['reports'][0], f.spec, f.policy)
            restarted = Verifier('http://coordinator', self.key, 'cpu', 512, **options)
            with patch('ekido_verifier.secure_verdict.read_chain_context', return_value=f.chain):
                self.assertIsNotNone(restarted.sign_verdict_task(f.inputs))

    def test_live_worker_requires_own_report_and_checks_payout(self):
        f = self.fixture
        worker = Verifier("http://coordinator", self.key, "cpu", 512, rpc_url="http://independent-rpc", chain_id=31337, escrow_address=f.chain["contract"])
        self.assertIsNone(worker.sign_verdict_task(f.inputs))
        report_key = (f.inputs["job_id"], f.inputs["final_checkpoint_hash"])
        worker.final_context[report_key] = (f.spec, f.policy)
        worker.final_reports.update(f.own_reports)
        with patch("ekido_verifier.secure_verdict.read_chain_context", return_value=f.chain):
            self.assertIsNotNone(worker.sign_verdict_task(f.inputs))
            f.inputs["payouts"] = {"0x" + "9" * 40: "5"}
            f.inputs["refund"] = "0"
            self.assertIsNone(worker.sign_verdict_task(f.inputs))

    def test_default_and_demo_mode_refuse_untrusted_domains(self):
        f = self.fixture
        self.assertIsNone(Verifier("http://coordinator", self.key, "cpu", 512).sign_verdict_task(f.inputs))
        worker = Verifier("http://coordinator", self.key, "cpu", 512, unsafe_local_demo=True)
        worker.signed_pass.add((f.inputs["job_id"], f.inputs["final_checkpoint_hash"]))
        f.inputs["chain_id"] = 84532
        self.assertIsNone(worker.sign_verdict_task(f.inputs))

    def test_final_context_survives_restart_and_tampering_is_refused(self):
        f = self.fixture
        with tempfile.TemporaryDirectory() as directory:
            options = dict(rpc_url="http://independent-rpc", chain_id=31337, escrow_address=f.chain["contract"], state_dir=directory)
            worker = Verifier("http://coordinator", self.key, "cpu", 512, **options)
            worker._record_final_report(f.inputs["reports"][0], f.spec, f.policy)
            restarted = Verifier("http://coordinator", self.key, "cpu", 512, **options)
            with patch("ekido_verifier.secure_verdict.read_chain_context", return_value=f.chain):
                self.assertIsNotNone(restarted.sign_verdict_task(f.inputs))
                path = next(Path(directory).glob("*.json"))
                saved = json.loads(path.read_text(encoding="utf-8"))
                saved["spec"]["total_logical_steps"] = 999
                path.write_text(json.dumps(saved), encoding="utf-8")
                restarted = Verifier("http://coordinator", self.key, "cpu", 512, **options)
                self.assertIsNone(restarted.sign_verdict_task(f.inputs))

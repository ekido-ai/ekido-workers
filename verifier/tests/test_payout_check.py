"""Adversarial tests for the new, not-yet-wired independent payout checker."""
import copy
import unittest

from eth_account import Account
from gpuchain_core.canonical import document_hash
from gpuchain_core.payout_terms import workload_hash
from gpuchain_core.signing import sign_document
from gpuchain_verifier.payout_check import MAX_MANIFESTS, MAX_PROVIDERS, MAX_REPORTS, PayoutEvidenceError, check_passing_evidence


def address(key):
    return Account.from_key(key).address.lower()


def signed(body, key):
    _, signature = sign_document(body, key)
    return dict(body, signature=signature)


class PayoutCheck(unittest.TestCase):
    fee = None

    def setUp(self):
        extra = {"protocol_fee": self.fee} if self.fee else {}
        wallet_key, old_key, new_key, v1_key, v2_key, client_key = ("0x" + c * 64 for c in "cabde8")
        self.keys = (old_key, new_key, v1_key, v2_key, client_key)
        self.wallet, old, new, self.v1, self.v2 = map(address, (wallet_key, old_key, new_key, v1_key, v2_key))
        quote = signed({"kind": "gpuchain.provider_quote.v1", "provider_id": "p", "wallet": self.wallet, "price_per_step": "0.001", "job_id": "job", "onchain_job_id": "0x" + "2" * 64, "chain_id": 31337, "contract": "0x" + "1" * 40, "workload_hash": workload_hash({"total_logical_steps": 50, **extra})}, wallet_key)
        self.spec = {"total_logical_steps": 50, **extra, "provider_terms": [{"provider_id": "p", "wallet": self.wallet, "price_per_step": "0.001", "quote": quote}]}
        self.policy = {"metric": "accuracy", "eval_set_hash": "e" * 64, "eval_version_hash": "f" * 64}
        self.chain = {"chain_id": 31337, "contract": "0x" + "1" * 40, "onchain_job_id": "0x" + "2" * 64, "client": address(client_key), "spec_hash": document_hash(self.spec), "policy_hash": document_hash(self.policy), "budget": "5", "verifiers": [self.v1, self.v2], "quorum": 2}
        self.delegations = {}
        for key in (old_key, new_key):
            signer = address(key)
            doc = {"kind": "gpuchain.enrollment.v1", "provider_id": "p", "worker_key": signer}
            self.delegations[signer] = {"provider_id": "p", "signature": sign_document(doc, wallet_key)[1]}
        first = signed({"job_id": "job", "immutable_hash": self.chain["spec_hash"], "logical_step": 20, "parent_manifest_hash": None, "signer": old}, old_key)
        second = signed({"job_id": "job", "immutable_hash": self.chain["spec_hash"], "logical_step": 50, "parent_manifest_hash": document_hash(first), "signer": new}, new_key)
        self.manifests = [first, second]
        lineage = [document_hash(m) for m in self.manifests]
        self.inputs = {"passed": True, "chain_id": 31337, "contract": self.chain["contract"], "onchain_job_id": self.chain["onchain_job_id"], "job_id": "job", "budget": "5", "final_checkpoint_hash": lineage[-1], "lineage": lineage, "payouts": {self.wallet: "0.05"}, "refund": "4.95", "reports": []}
        self.inputs["job_binding"] = signed({"kind": "gpuchain.job_binding.v1", "job_id": "job", "onchain_job_id": self.chain["onchain_job_id"], "chain_id": 31337, "contract": self.chain["contract"], "spec_hash": self.chain["spec_hash"], "policy_hash": self.chain["policy_hash"], "budget": "5"}, client_key)
        handover = []
        for key in (v1_key, v2_key):
            handover.append(signed({"job_id": "job", "checkpoint_hash": lineage[0], "reference_checkpoint_hash": None, "policy_hash": document_hash(self.policy), "eval_set_hash": self.policy["eval_set_hash"], "eval_version_hash": self.policy["eval_version_hash"], "is_final": False, "verdict": "PASS", "signer": address(key)}, key))
            self.inputs["reports"].append(signed({"job_id": "job", "checkpoint_hash": lineage[-1], "reference_checkpoint_hash": lineage[0], "policy_hash": document_hash(self.policy), "eval_set_hash": self.policy["eval_set_hash"], "eval_version_hash": self.policy["eval_version_hash"], "is_final": True, "verdict": "PASS", "signer": address(key)}, key))
        self.inputs["payout_evidence"] = {"manifests": self.manifests, "delegations": self.delegations, "handover_reports": {lineage[0]: handover}}
        self.own_reports = {("job", lineage[-1]): document_hash(self.inputs["reports"][0])}

    def check(self):
        return check_passing_evidence(self.inputs, self.manifests, self.delegations, spec=self.spec, policy=self.policy, chain=self.chain, own_address=self.v1, own_final_reports=self.own_reports)

    def test_accepts_valid_lineage_with_a_restarted_workers_old_key(self):
        self.assertIsNone(self.check())

    def test_refuses_excessive_evidence_before_checking_signatures(self):
        for field in ("manifests", "delegations", "providers", "reports"):
            with self.subTest(field=field):
                self.setUp()
                if field == "manifests": self.manifests = [self.manifests[0]] * (MAX_MANIFESTS + 1)
                if field == "delegations": self.delegations = {str(i): {} for i in range(MAX_MANIFESTS + 1)}
                if field == "providers": self.spec["provider_terms"] = [self.spec["provider_terms"][0]] * (MAX_PROVIDERS + 1)
                if field == "reports": self.inputs["reports"] = [self.inputs["reports"][0]] * (MAX_REPORTS + 1)
                with self.assertRaises(PayoutEvidenceError): self.check()

    def test_rejects_forged_payout_and_untrusted_terms(self):
        for change in ("recipient", "amount", "refund", "rate", "wallet", "quote_signature", "budget", "domain", "job", "binding_signature"):
            with self.subTest(change=change):
                self.setUp()
                if change == "recipient": self.inputs["payouts"] = {"0x" + "9" * 40: "0.05"}
                if change == "amount": self.inputs["payouts"][self.wallet] = "4.05"
                if change == "refund": self.inputs["refund"] = "0"
                if change == "rate": self.spec["provider_terms"][0]["price_per_step"] = "0.5"
                if change == "wallet": self.spec["provider_terms"][0]["wallet"] = "0x" + "9" * 40
                if change == "quote_signature": self.spec["provider_terms"][0]["quote"]["signature"] = "0x00"
                if change == "budget": self.inputs["budget"] = "10"
                if change == "domain": self.inputs["contract"] = "0x" + "9" * 40
                if change == "job": self.inputs["onchain_job_id"] = "0x" + "9" * 64
                if change == "binding_signature": self.inputs["job_binding"]["signature"] = "0x00"
                with self.assertRaises(PayoutEvidenceError): self.check()

    def test_rejects_untrusted_lineage_and_reports(self):
        for change in ("step", "parent", "signature", "delegation", "missing_manifest", "report_signature", "duplicate_report", "not_final", "policy", "handover"):
            with self.subTest(change=change):
                self.setUp()
                if change == "step": self.manifests[0]["logical_step"] = 21
                if change == "parent": self.manifests[1]["parent_manifest_hash"] = "0" * 64
                if change == "signature": self.manifests[0]["signature"] = "0x00"
                if change == "delegation": self.delegations[self.manifests[0]["signer"]]["signature"] = "0x00"
                if change == "missing_manifest": self.manifests.pop(0)
                if change == "report_signature": self.inputs["reports"][0]["signature"] = "0x00"
                if change == "duplicate_report": self.inputs["reports"][1] = copy.deepcopy(self.inputs["reports"][0])
                if change == "not_final": self.inputs["reports"][0]["is_final"] = False
                if change == "policy": self.policy["metric"] = "other"
                if change == "handover": self.inputs["payout_evidence"]["handover_reports"] = {}
                with self.assertRaises(PayoutEvidenceError): self.check()

    def test_rejects_invalid_quote_even_if_client_commits_to_it(self):
        old_key, new_key, v1_key, v2_key, client_key = self.keys
        self.spec["provider_terms"][0]["quote"]["signature"] = "0x00"
        self.chain["spec_hash"] = document_hash(self.spec)
        binding = {k: v for k, v in self.inputs["job_binding"].items() if k != "signature"}
        binding["spec_hash"] = self.chain["spec_hash"]
        self.inputs["job_binding"] = signed(binding, client_key)
        previous = None
        for index, key in enumerate((old_key, new_key)):
            manifest = {k: v for k, v in self.manifests[index].items() if k != "signature"}
            manifest.update(immutable_hash=self.chain["spec_hash"], parent_manifest_hash=previous)
            self.manifests[index] = signed(manifest, key)
            previous = document_hash(manifest)
        lineage = [document_hash(m) for m in self.manifests]
        self.inputs.update(lineage=lineage, final_checkpoint_hash=lineage[-1])
        for index, key in enumerate((v1_key, v2_key)):
            report = {k: v for k, v in self.inputs["reports"][index].items() if k != "signature"}
            report["checkpoint_hash"] = lineage[-1]
            self.inputs["reports"][index] = signed(report, key)
        self.own_reports = {("job", lineage[-1]): document_hash(self.inputs["reports"][0])}
        with self.assertRaisesRegex(PayoutEvidenceError, "invalid provider quote"):
            self.check()

    def test_rejects_validly_signed_report_for_wrong_evaluation_set(self):
        report = dict(self.inputs["reports"][1], eval_set_hash="0" * 64)
        self.inputs["reports"][1] = signed(report, self.keys[3])
        with self.assertRaisesRegex(PayoutEvidenceError, "evaluation context"):
            self.check()


TREASURY = "0x915f91f109980c5e810507fed50d5d364d15e074"


class PayoutCheckWithFee(PayoutCheck):
    """The same evidence with an 8% protocol fee in the funded spec."""

    fee = {"bps": 800, "recipient": TREASURY}

    def setUp(self):
        super().setUp()
        self.inputs["payouts"] = {self.wallet: "0.05", TREASURY: "0.004"}
        self.inputs["refund"] = "4.946"

    def test_accepts_valid_lineage_with_a_restarted_workers_old_key(self):
        self.assertIsNone(self.check())

    def test_refuses_a_fee_the_spec_does_not_allow(self):
        for change in ("raised", "missing", "redirected", "spec_without_fee"):
            with self.subTest(change=change):
                self.setUp()
                if change == "raised": self.inputs["payouts"][TREASURY] = "0.005"; self.inputs["refund"] = "4.945"
                if change == "missing": del self.inputs["payouts"][TREASURY]; self.inputs["refund"] = "4.95"
                if change == "redirected": self.inputs["payouts"] = {self.wallet: "0.05", "0x" + "9" * 40: "0.004"}
                if change == "spec_without_fee": self.spec.pop("protocol_fee")
                with self.assertRaises(PayoutEvidenceError): self.check()

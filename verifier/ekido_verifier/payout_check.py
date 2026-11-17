"""Independent verification primitives for the planned signed payout evidence protocol.

Callers must supply chain facts from their own configured RPC/escrow, never from a
coordinator's task. The worker uses this checker for its default signing path.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation

from ekido_core.canonical import document_hash
from ekido_core.payout_terms import workload_hash
from ekido_core.protocol_fee import add_fee, fee_problems, fee_terms, provider_budget
from ekido_core.signing import verify_document


class PayoutEvidenceError(ValueError):
    pass


# Bound signature-verification work before accepting evidence from a task.
MAX_MANIFESTS = 4096
MAX_PROVIDERS = 256
MAX_REPORTS = 16


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PayoutEvidenceError(message)


def check_failed_evidence(inputs: dict, *, spec: dict, policy: dict, chain: dict,
                          own_address: str, own_final_reports: dict) -> None:
    """Check a full refund against the funded job and a quorum of signed final failures."""
    try:
        _require(inputs['passed'] is False, 'expected a failing verdict')
        _require(inputs['chain_id'] == chain['chain_id'] and inputs['contract'].lower() == chain['contract'].lower(), 'wrong signing domain')
        _require(inputs['onchain_job_id'].lower() == chain['onchain_job_id'].lower(), 'wrong on-chain job')
        _require(document_hash(spec) == chain['spec_hash'].removeprefix('0x').lower(), 'spec does not match funded job')
        _require(document_hash(policy) == chain['policy_hash'].removeprefix('0x').lower(), 'policy does not match funded job')
        budget = Decimal(str(chain['budget']))
        _require(budget.is_finite() and budget > 0 and Decimal(str(inputs['budget'])) == budget and Decimal(str(inputs['refund'])) == budget, 'refund must equal the funded budget')
        _require(inputs['payouts'] == {} and inputs['lineage'] == [] and inputs['final_checkpoint_hash'] is None, 'failed verdict cannot pay providers or accept a checkpoint')
        binding = inputs['job_binding']
        _require(binding['kind'] == 'gpuchain.job_binding.v1' and binding['job_id'] == inputs['job_id'] and binding['onchain_job_id'].lower() == chain['onchain_job_id'].lower() and binding['chain_id'] == chain['chain_id'] and binding['contract'].lower() == chain['contract'].lower(), 'client job binding does not match')
        _require(binding['spec_hash'] == document_hash(spec) and binding['policy_hash'] == document_hash(policy) and Decimal(str(binding['budget'])) == budget, 'client job binding has different terms')
        _require(verify_document(binding, binding['signature'], chain['client']), 'invalid client job binding')
        failure_hash = inputs['failure_checkpoint_hash']
        _require(isinstance(failure_hash, str) and len(failure_hash) == 64 and all(c in '0123456789abcdef' for c in failure_hash), 'invalid failed checkpoint hash')
        reports = inputs['reports']
        _require(isinstance(reports, list) and 0 < len(reports) <= MAX_REPORTS, 'invalid report count')
        allowed = {v.lower() for v in chain['verifiers']}
        quorum = chain['quorum']
        _require(type(quorum) is int and 1 <= quorum <= len(allowed), 'invalid verifier quorum')
        own_hash = own_final_reports.get((inputs['job_id'], failure_hash))
        signers, references, own_matches = set(), set(), False
        for report in reports:
            signer = report['signer'].lower()
            _require(signer in allowed and signer not in signers, 'unknown or duplicate verifier')
            _require(report['job_id'] == inputs['job_id'] and report['checkpoint_hash'] == failure_hash and report['policy_hash'] == document_hash(policy) and report['is_final'] is True and report['verdict'] == 'FAIL', 'failure report context does not match')
            _require(report['eval_set_hash'] == policy['eval_set_hash'] and report['eval_version_hash'] == policy['eval_version_hash'], 'failure evaluation context differs from policy')
            reasons = report.get('reasons')
            _require(isinstance(reasons, list) and bool({'BELOW_ABSOLUTE_FLOOR', 'BASE_MODEL_NOT_BEATEN'}.intersection(reasons)), 'failure report does not identify a recipe quality failure')
            _require(verify_document(report, report['signature'], signer), 'invalid quality report signature')
            references.add(report['reference_checkpoint_hash'])
            if signer == own_address.lower():
                own_matches = bool(own_hash) and document_hash(report) == own_hash
            signers.add(signer)
        _require(len(signers) >= quorum and own_matches and len(references) == 1, 'missing or inconsistent failure report quorum')
    except (KeyError, TypeError, AttributeError, InvalidOperation) as exc:
        raise PayoutEvidenceError('malformed or incomplete failure evidence') from exc


def check_passing_evidence(inputs: dict, manifests: list[dict], delegations: dict, *, spec: dict, policy: dict, chain: dict, own_address: str, own_final_reports: dict) -> None:
    """Refuse an inconsistent passing verdict using preauthenticated chain facts.

    chain requires chain_id, contract, onchain_job_id, client, spec_hash, policy_hash,
    budget (USDC), verifiers and quorum. spec.provider_terms binds provider_id,
    wallet and price_per_step to the hash fixed by the funded job.
    """
    try:
        _require(isinstance(manifests, list) and 0 < len(manifests) <= MAX_MANIFESTS, "invalid lineage size")
        _require(isinstance(delegations, dict) and len(delegations) <= MAX_MANIFESTS, "invalid delegation count")
        _require(isinstance(spec["provider_terms"], list) and 0 < len(spec["provider_terms"]) <= MAX_PROVIDERS, "invalid provider count")
        _require(isinstance(inputs["reports"], list) and 0 < len(inputs["reports"]) <= MAX_REPORTS, "invalid report count")
        _require(inputs["passed"] is True, "expected a passing verdict")
        _require(inputs["chain_id"] == chain["chain_id"] and inputs["contract"].lower() == chain["contract"].lower(), "wrong signing domain")
        _require(inputs["onchain_job_id"].lower() == chain["onchain_job_id"].lower(), "wrong on-chain job")
        _require(document_hash(spec) == chain["spec_hash"].removeprefix("0x").lower(), "spec does not match funded job")
        _require(document_hash(policy) == chain["policy_hash"].removeprefix("0x").lower(), "policy does not match funded job")
        budget = Decimal(str(chain["budget"]))
        _require(budget.is_finite() and budget > 0 and Decimal(str(inputs["budget"])) == budget, "wrong budget")
        binding = inputs["job_binding"]
        _require(binding["kind"] == "gpuchain.job_binding.v1" and binding["job_id"] == inputs["job_id"] and binding["onchain_job_id"].lower() == chain["onchain_job_id"].lower() and binding["chain_id"] == chain["chain_id"] and binding["contract"].lower() == chain["contract"].lower(), "client job binding does not match")
        _require(binding["spec_hash"] == document_hash(spec) and binding["policy_hash"] == document_hash(policy) and Decimal(str(binding["budget"])) == budget, "client job binding has different terms")
        _require(verify_document(binding, binding["signature"], chain["client"]), "invalid client job binding")
        own_report_hash = own_final_reports.get((inputs["job_id"], inputs["final_checkpoint_hash"]))
        _require(bool(own_report_hash), "verifier has no recorded final report")
        terms = {}
        for term in spec["provider_terms"]:
            pid, wallet, rate = term["provider_id"], term["wallet"].lower(), Decimal(str(term["price_per_step"]))
            _require(pid not in terms and rate.is_finite() and rate >= 0 and rate * 1_000_000 == (rate * 1_000_000).to_integral_value(), "invalid provider terms")
            quote = term["quote"]
            _require(quote["kind"] == "gpuchain.provider_quote.v1" and quote["provider_id"] == pid and quote["wallet"].lower() == wallet and Decimal(str(quote["price_per_step"])) == rate, "provider quote does not match the agreed rate")
            _require(quote["workload_hash"] == workload_hash(spec), "provider quote does not match the workload")
            _require(quote["job_id"] == inputs["job_id"] and quote["onchain_job_id"].lower() == chain["onchain_job_id"].lower() and quote["chain_id"] == chain["chain_id"] and quote["contract"].lower() == chain["contract"].lower(), "provider quote does not match the job")
            _require(verify_document(quote, quote["signature"], wallet), "invalid provider quote")
            terms[pid] = (wallet, rate)
        _require(bool(manifests), "missing lineage")
        hashes, steps, wallets, rates, boundaries = [], {}, {}, {}, []
        previous_hash, previous_step = None, 0
        previous_signer = None
        for manifest in manifests:
            digest = document_hash(manifest)
            signer = manifest["signer"].lower()
            proof = delegations[signer]
            provider = proof["provider_id"]
            _require(provider in terms, "provider has no agreed terms")
            wallet, rate = terms[provider]
            enrollment = {"kind": "gpuchain.enrollment.v1", "provider_id": provider, "worker_key": signer}
            _require(verify_document(enrollment, proof["signature"], wallet), "invalid worker delegation")
            _require(verify_document(manifest, manifest["signature"], signer), "invalid checkpoint signature")
            _require(manifest["job_id"] == inputs["job_id"] and manifest["immutable_hash"] == chain["spec_hash"].removeprefix("0x").lower(), "checkpoint belongs to different job terms")
            step = manifest["logical_step"]
            _require(type(step) is int and step > previous_step and manifest["parent_manifest_hash"] == previous_hash, "broken checkpoint lineage")
            _require(digest not in hashes, "duplicate checkpoint")
            if previous_signer is not None and signer != previous_signer:
                boundaries.append(previous_hash)
            hashes.append(digest)
            steps[signer] = steps.get(signer, 0) + step - previous_step
            wallets[signer], rates[signer] = wallet, rate
            previous_hash, previous_step = digest, step
            previous_signer = signer
        _require(hashes == inputs["lineage"] and hashes[-1] == inputs["final_checkpoint_hash"], "lineage does not match verdict")
        _require(previous_step == spec["total_logical_steps"], "incomplete training")
        allowed = {v.lower() for v in chain["verifiers"]}
        quorum = chain["quorum"]
        _require(type(quorum) is int and 1 <= quorum <= len(allowed), "invalid verifier quorum")
        handovers = inputs.get("payout_evidence", {}).get("handover_reports", {})
        _require(isinstance(handovers, dict) and len(handovers) <= 256 and set(boundaries).issubset(handovers) and set(handovers).issubset(hashes[:-1]), "missing or unexpected handover evidence")
        prior_reference = None
        for boundary in (digest for digest in hashes[:-1] if digest in handovers):
            handover_reports = handovers[boundary]
            _require(isinstance(handover_reports, list) and quorum <= len(handover_reports) <= MAX_REPORTS, "invalid handover quorum")
            handover_signers = set()
            for report in handover_reports:
                signer = report["signer"].lower()
                _require(signer in allowed and signer not in handover_signers, "unknown or duplicate handover verifier")
                _require(report["job_id"] == inputs["job_id"] and report["checkpoint_hash"] == boundary and report["policy_hash"] == document_hash(policy) and report["is_final"] is False and report["verdict"] == "PASS" and report["reference_checkpoint_hash"] == prior_reference, "handover report context does not match")
                _require(report["eval_set_hash"] == policy["eval_set_hash"] and report["eval_version_hash"] == policy["eval_version_hash"], "handover evaluation context differs from policy")
                _require(verify_document(report, report["signature"], signer), "invalid handover report signature")
                handover_signers.add(signer)
            prior_reference = boundary
        signers = set()
        references = set()
        own_report_matches = False
        for report in inputs["reports"]:
            signer = report["signer"].lower()
            _require(signer in allowed and signer not in signers, "unknown or duplicate verifier")
            _require(report["job_id"] == inputs["job_id"] and report["checkpoint_hash"] == hashes[-1] and report["policy_hash"] == document_hash(policy) and report["is_final"] is True and report["verdict"] == "PASS", "report context does not match final job")
            _require(report["eval_set_hash"] == policy["eval_set_hash"] and report["eval_version_hash"] == policy["eval_version_hash"], "final evaluation context differs from policy")
            _require(verify_document(report, report["signature"], signer), "invalid quality report signature")
            references.add(report["reference_checkpoint_hash"])
            if signer == own_address.lower():
                own_report_matches = document_hash(report) == own_report_hash
            signers.add(signer)
        _require(len(signers) >= quorum and own_report_matches and len(references) == 1, "missing or inconsistent final report quorum")
        reference = next(iter(references))
        _require(reference == prior_reference, "final reference does not match the last verified handover")
        # The fee comes only from the funded spec, whose hash is on-chain; the coordinator can't add or raise it.
        _require(not fee_problems(spec), "invalid protocol fee terms")
        terms = fee_terms(spec)
        payouts, remaining = {}, (provider_budget(budget, terms[0]) if terms else budget)
        for signer in sorted(steps):
            amount = min(remaining, steps[signer] * rates[signer])
            if amount > 0:
                wallet = wallets[signer]
                payouts[wallet] = payouts.get(wallet, Decimal(0)) + amount
                remaining -= amount
        if terms:
            payouts, remaining = add_fee(payouts, budget, terms)
        proposed = {}
        for wallet, amount in inputs["payouts"].items():
            normalized = wallet.lower()
            _require(normalized not in proposed, "duplicate payout wallet")
            value = Decimal(str(amount))
            _require(value.is_finite() and value > 0, "invalid payout")
            proposed[normalized] = value
        _require(proposed == payouts and Decimal(str(inputs["refund"])) == remaining, "payout or refund differs from agreed terms")
    except (KeyError, TypeError, AttributeError, InvalidOperation) as exc:
        raise PayoutEvidenceError("malformed or incomplete payout evidence") from exc

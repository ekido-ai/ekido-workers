"""Fail-closed signing of passing verdicts with independent chain facts.

The worker uses this path by default. Failed-job refunds require a separately
reviewed path and are refused here.
"""
from __future__ import annotations

from decimal import Decimal

from eth_account import Account
from ekido_core.settlement import build_verdict
from ekido_core.signing import sign_verdict

from .payout_check import MAX_REPORTS, PayoutEvidenceError, check_passing_evidence
from .trusted_chain import read_chain_context


def sign_checked_verdict(inputs: dict, spec: dict, policy: dict, *, rpc_url: str, chain_id: int, escrow_address: str, key: str, own_final_reports: dict) -> str:
    """Read configured chain, check all evidence, then sign only the checked verdict."""
    try:
        if inputs["passed"] is not True:
            raise PayoutEvidenceError("failed-job refund signing is not implemented in this path")
        if not isinstance(inputs["reports"], list) or not 0 < len(inputs["reports"]) <= MAX_REPORTS:
            raise PayoutEvidenceError("invalid report count")
        evidence = inputs["payout_evidence"]
        handovers = evidence.get("handover_reports", {})
        if not isinstance(handovers, dict) or len(handovers) > 256:
            raise PayoutEvidenceError("invalid handover evidence")
        signers = [report["signer"] for report in inputs["reports"]]
        for reports in handovers.values():
            if not isinstance(reports, list) or not 0 < len(reports) <= MAX_REPORTS:
                raise PayoutEvidenceError("invalid handover report count")
            signers.extend(report["signer"] for report in reports)
        chain = read_chain_context(rpc_url, chain_id, escrow_address, inputs["onchain_job_id"], signers)
        check_passing_evidence(inputs, evidence["manifests"], evidence["delegations"], spec=spec, policy=policy, chain=chain, own_address=Account.from_key(key).address, own_final_reports=own_final_reports)
        verdict = build_verdict(
            onchain_job_id=inputs["onchain_job_id"], passed=True,
            final_checkpoint_hash=inputs["final_checkpoint_hash"], lineage=inputs["lineage"], reports=inputs["reports"],
            payouts={address: Decimal(amount) for address, amount in inputs["payouts"].items()},
            refund=Decimal(inputs["refund"]), budget=Decimal(inputs["budget"]), expiry=inputs["expiry"],
        )
        return sign_verdict(verdict, chain_id, escrow_address, key)
    except (KeyError, TypeError, AttributeError) as exc:
        raise PayoutEvidenceError("malformed or incomplete secure verdict task") from exc

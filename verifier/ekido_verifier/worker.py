"""A verifier process: pulls evaluation tasks, judges them, and returns signed reports.

    python -m ekido_verifier.worker --coordinator http://127.0.0.1:8000 --key 0x... --device cuda

Each verifier has its own key. In this prototype the team operates them all, so this is a permissioned
set, and the product says so.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from decimal import Decimal
from pathlib import Path

from eth_account import Account

from ekido_core.settlement import build_verdict
from ekido_core.checkpoint_files import checkpoint_files
from ekido_core.transfer import download_blob
from ekido_core.canonical import document_hash
from ekido_core import jobkeys
from ekido_core.checkpoint_crypto import CheckpointError, ciphertext_name, decrypt_checkpoint_files
from ekido_core.signing import sign_document, sign_verdict, verify_document

from .evaluate import base_cache_key, evaluate_checkpoint, load_eval_set, sign_report
from .secure_verdict import sign_checked_verdict

log = logging.getLogger("gpuchain.verifier")


def _trim_checkpoint_cache(cache: dict, limit: int = 8) -> None:
    """Bound checkpoint entries without discarding baselines for unfinished jobs."""
    checkpoints = [key for key in cache if not str(key[1]).startswith('baseline:')]
    for key in checkpoints[:-limit] if limit else checkpoints:
        cache.pop(key, None)


def verdict_from_inputs(inputs: dict) -> dict:
    """Must stay identical to the coordinator's derivation, or signatures won't reach quorum."""
    return build_verdict(
        onchain_job_id=inputs["onchain_job_id"],
        passed=inputs["passed"],
        final_checkpoint_hash=inputs["final_checkpoint_hash"],
        lineage=inputs["lineage"],
        reports=inputs["reports"],
        payouts={address: Decimal(amount) for address, amount in inputs["payouts"].items()},
        refund=Decimal(inputs["refund"]),
        budget=Decimal(inputs["budget"]),
        expiry=inputs["expiry"],
    )


def check_verdict_inputs(inputs: dict, own_address: str, own_final_passes: set) -> str | None:
    """A verifier's independent check on what the coordinator proposed. Returns why it's refused, or None."""
    if inputs["passed"]:
        if (inputs["job_id"], inputs["final_checkpoint_hash"]) not in own_final_passes:
            return "this verifier didn't itself pass that final checkpoint"
        if not inputs["lineage"] or inputs["lineage"][-1] != inputs["final_checkpoint_hash"]:
            return "the lineage doesn't end at the final checkpoint"
    elif inputs["payouts"]:
        return "a failed job can't pay anyone"
    return None


class Verifier:
    def __init__(self, coordinator: str, key: str, device: str, eval_examples: int | None, poll_seconds: float = 1.0, *, rpc_url: str | None = None, chain_id: int | None = None, escrow_address: str | None = None, unsafe_local_demo: bool = False, state_dir: str | None = None):
        self.base = coordinator.rstrip("/")
        self.key = key
        self.address = Account.from_key(key).address
        self.device = device
        self.eval_examples = eval_examples
        self.poll_seconds = poll_seconds
        self.token: str | None = None
        # Tokenizing the evaluation set is slow, so it's kept between tasks.
        self._eval_sets: dict[str, object] = {}
        self._reference_cache: dict = {}
        self.signed_pass: set[tuple[str, str]] = set()  # Final checkpoints this verifier itself passed.
        self.final_reports: dict[tuple[str, str], str] = {}
        self.final_context: dict[tuple[str, str], tuple[dict, dict]] = {}
        self.rpc_url, self.chain_id, self.escrow_address = rpc_url, chain_id, escrow_address
        self.unsafe_local_demo = unsafe_local_demo
        self.state_dir = Path(state_dir) if state_dir else None

    def _record_final_report(self, report: dict, spec: dict, policy: dict) -> None:
        report_key = (report["job_id"], report["checkpoint_hash"])
        context = json.loads(json.dumps((spec, policy)))
        if self.state_dir is not None:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            name = document_hash({"job_id": report_key[0], "checkpoint_hash": report_key[1], "verifier": self.address.lower()}) + ".json"
            fd, temporary = tempfile.mkstemp(prefix=".verifier-", suffix=".tmp", dir=self.state_dir)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as output:
                    saved = {"report": report, "spec": context[0], "policy": context[1]}
                    saved["signature"] = sign_document(saved, self.key)[1]
                    json.dump(saved, output)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, self.state_dir / name)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        if report['verdict'] == 'PASS':
            self.signed_pass.add(report_key)
        self.final_reports[report_key] = document_hash(report)
        self.final_context[report_key] = context

    def _restore_final_report(self, report_key: tuple[str, str]) -> None:
        if self.state_dir is None:
            return
        name = document_hash({"job_id": report_key[0], "checkpoint_hash": report_key[1], "verifier": self.address.lower()}) + ".json"
        try:
            with (self.state_dir / name).open(encoding="utf-8") as source:
                saved = json.load(source)
            report, spec, policy = saved["report"], saved["spec"], saved["policy"]
            if not verify_document(saved, saved["signature"], self.address) or (report["job_id"], report["checkpoint_hash"]) != report_key or report["is_final"] is not True or report["verdict"] not in ("PASS", "FAIL") or report["signer"].lower() != self.address.lower() or report["policy_hash"] != document_hash(policy) or not verify_document(report, report["signature"], self.address):
                raise ValueError("invalid saved final report")
            self.final_reports[report_key] = document_hash(report)
            self.final_context[report_key] = (spec, policy)
        except FileNotFoundError:
            return
        except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
            log.warning("refusing saved verifier context: %s", exc)

    def _call(self, method: str, path: str, body=None, timeout: float = 120):
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"null")

    def login(self) -> None:
        message = {"kind": "gpuchain.verifier.session.v1", "address": self.address, "ts": int(time.time())}
        _, signature = sign_document(message, self.key)
        self.token = self._call("POST", "/verifier/session", {"message": message, "signature": signature})["token"]

    def _download(self, digest: str, target: Path, expected_bytes: int) -> None:
        download_blob(self.base, self.token, digest, target, expected_bytes, timeout=300)

    def _job_key_identity(self, job_id: str) -> str:
        """The job's key for the evaluation this verifier has claimed (ticket A12). It makes a one-off key pair, sends only the public half,
        and decrypts the coordinator's answer. The result lives in memory for this task and is never written or logged. If the coordinator
        refuses (HTTP error) nothing is reported for the task: a missing key is not a verdict."""
        ours = jobkeys.create_job_key()
        reply = self._call("POST", "/verifier/tasks/job-key", {"address": self.address, "job_id": job_id, "recipient": ours.recipient})
        sealed = base64.b64decode(reply["ciphertext_b64"], validate=True)
        return jobkeys.decrypt_bytes(sealed, ours.identity).decode("ascii")

    def _fetch_checkpoint(self, manifest: dict, directory: Path, identity: str | None = None) -> Path:
        files = checkpoint_files(manifest)
        directory.mkdir(parents=True, exist_ok=True)
        if manifest.get("format") == "GPCK/2":
            if not identity:
                raise ValueError("the job key is needed to read an encrypted checkpoint")
            blobs = directory.parent / (directory.name + "_blobs")
            blobs.mkdir(parents=True, exist_ok=True)
            try:
                for name, entry in files.items():
                    self._download(entry["sha256"], blobs / ciphertext_name(name), entry["bytes"])  # checked against the manifest's hash of the stored bytes
                decrypt_checkpoint_files(blobs, manifest, identity, directory)
            except CheckpointError as exc:
                if exc.code == "DECRYPT_FAILED":
                    raise  # a key and a blob that do not fit could be either side's fault: report nothing rather than guess
                # The blob or the manifest does not hold together (a size or hash that differs): leave no tensor files, so the
                # normal evaluation refuses the checkpoint and reports FAIL with its usual reason.
                log.warning("checkpoint refused before evaluation: %s", exc.code)
            finally:
                shutil.rmtree(blobs, ignore_errors=True)
        else:
            for name, entry in files.items():
                self._download(entry["sha256"], directory / name, entry["bytes"])
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return directory

    def _eval_set(self, spec: dict):
        key = document_hash(spec)
        if key not in self._eval_sets:
            started = time.perf_counter()
            if spec.get("template") == "llm-lora/v1":
                # The customer's dataset is one blob in the artifact store; the download is checked against its hash.
                with tempfile.TemporaryDirectory(prefix="dataset_") as tmp:
                    self._download(spec["dataset"]["sha256"], Path(tmp) / "dataset.jsonl", spec["dataset"]["bytes"])
                    self._eval_sets[key] = load_eval_set(spec, data_dir=Path(tmp))
            else:
                self._eval_sets[key] = load_eval_set(spec, self.eval_examples)
            log.info("evaluation set ready in %.1fs", time.perf_counter() - started)
        return self._eval_sets[key]

    def handle(self, task: dict) -> dict:
        with tempfile.TemporaryDirectory(prefix="verify_") as tmp:
            root = Path(tmp)
            encrypted = any((m or {}).get("format") == "GPCK/2" for m in (task["manifest"], task.get("reference_manifest")))
            identity = self._job_key_identity(task["job_id"]) if encrypted else None  # once per task, for the candidate and the reference
            checkpoint = self._fetch_checkpoint(task["manifest"], root / "candidate", identity)
            reference = self._fetch_checkpoint(task["reference_manifest"], root / "reference", identity) if task.get("reference_manifest") else None
            identity = None
            started = time.perf_counter()
            report = evaluate_checkpoint(
                job_id=task["job_id"],
                spec=task["spec"],
                policy=task["policy"],
                eval_set=self._eval_set(task["spec"]),
                checkpoint_dir=checkpoint,
                reference_dir=reference,
                expected_signer=task.get("expected_signer"),
                provider_signature=task["manifest"].get("signature"),
                is_final=task["is_final"],
                device=self.device,
                reference_cache=self._reference_cache,
            )
            # A candidate often becomes the next reference, so keep only recent entries.
            _trim_checkpoint_cache(self._reference_cache)
            log.info("step %s: %s %s in %.1fs", task["manifest"].get("logical_step"), report["verdict"], report["reasons"], time.perf_counter() - started)
            signed_report = sign_report(report, self.key)
            if report["is_final"]:
                self._record_final_report(signed_report, task["spec"], task["policy"])
                self._reference_cache.pop(base_cache_key(task['job_id'], task['spec'], report['eval_set_hash']), None)
            return signed_report

    def sign_verdict_task(self, inputs: dict) -> str | None:
        """Signs a payment only if it matches this verifier's own evaluation. Returns None to refuse."""
        if not self.unsafe_local_demo:
            if not (self.rpc_url and self.chain_id and self.escrow_address):
                log.warning("refusing verdict: verifier has no trusted chain configuration")
                return None
            report_key = (inputs.get("job_id"), inputs.get("final_checkpoint_hash") if inputs.get("passed") is True else inputs.get("failure_checkpoint_hash"))
            if report_key not in self.final_context:
                self._restore_final_report(report_key)
            context = self.final_context.get(report_key)
            if context is None:
                log.warning("refusing verdict: no locally observed final evaluation context")
                return None
            try:
                return sign_checked_verdict(inputs, *context, rpc_url=self.rpc_url, chain_id=self.chain_id, escrow_address=self.escrow_address, key=self.key, own_final_reports=self.final_reports)
            except (ValueError, OSError) as exc:
                log.warning("refusing unchecked verdict: %s", exc)
                return None
        if inputs.get("chain_id") != 31337:
            log.warning("refusing unsafe demo verdict outside chain 31337")
            return None
        refusal = check_verdict_inputs(inputs, self.address, self.signed_pass)
        if refusal:
            log.warning("refusing to sign the verdict for job %s: %s", inputs.get("job_id"), refusal)
            return None
        verdict = verdict_from_inputs(inputs)
        return sign_verdict(verdict, inputs["chain_id"], inputs["contract"], self.key)

    def run_forever(self) -> None:
        while True:
            try:
                if self.token is None:
                    self.login()
                    log.info("verifier %s connected", self.address)
                verdict_task = self._call("POST", "/verifier/verdicts/claim", {"address": self.address})
                if verdict_task:
                    signature = self.sign_verdict_task(verdict_task)
                    if signature:
                        self._call("POST", "/verifier/verdicts", {"job_id": verdict_task["job_id"], "signature": signature})
                    continue
                task = self._call("POST", "/verifier/tasks/claim", {"address": self.address})
                if not task:
                    time.sleep(self.poll_seconds)
                    continue
                self._call("POST", "/verifier/evaluations", {"task_id": task["task_id"], "report": self.handle(task)})
            except urllib.error.HTTPError as exc:
                log.warning("coordinator refused: %s %s", exc.code, exc.read()[:200])
                self.token = None
                time.sleep(self.poll_seconds * 3)
            except (urllib.error.URLError, OSError) as exc:
                log.warning("coordinator unreachable: %s", exc)
                time.sleep(self.poll_seconds * 3)
            except ValueError as exc:
                # Invalid evidence must not crash the verifier or count as an approval.
                log.warning("refusing invalid verifier task: %s", exc)
                time.sleep(self.poll_seconds * 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--key", default=os.environ.get("VERIFIER_KEY"), help="Or set VERIFIER_KEY.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-examples", type=int, help="Must match the set fixed in the job's policy.")
    parser.add_argument("--rpc-url", default=os.environ.get("VERIFIER_RPC_URL"), help="Verifier's independently configured trusted RPC.")
    parser.add_argument("--chain-id", type=int, default=os.environ.get("VERIFIER_CHAIN_ID"), help="Trusted chain ID.")
    parser.add_argument("--escrow-address", default=os.environ.get("VERIFIER_ESCROW_ADDRESS"), help="Trusted escrow address.")
    parser.add_argument("--state-dir", default=os.environ.get("VERIFIER_STATE_DIR"), help="Private local directory for signed final evaluation context across restarts.")
    parser.add_argument("--unsafe-local-demo-verdicts", action="store_true", help="INSECURE: trust coordinator payouts for local chain 31337 demonstrations only.")
    args = parser.parse_args()
    if not args.key:
        parser.error("--key or VERIFIER_KEY is required")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.unsafe_local_demo_verdicts:
        log.warning("INSECURE local demo verdicts enabled: coordinator can redirect payouts")
    Verifier(args.coordinator, args.key, args.device, args.eval_examples, rpc_url=args.rpc_url, chain_id=args.chain_id, escrow_address=args.escrow_address, unsafe_local_demo=args.unsafe_local_demo_verdicts, state_dir=args.state_dir).run_forever()


if __name__ == "__main__":
    main()

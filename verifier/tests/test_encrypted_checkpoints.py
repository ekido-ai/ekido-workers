"""A12/A13, verifier side: an encrypted (GPCK/2) checkpoint is read with the job key the coordinator gives for a claimed task.
The network and the evaluation are replaced; what is tested is how the verifier gets and uses the key and the blobs."""

import base64
import hashlib
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from ekido_core import jobkeys
from ekido_core.checkpoint_crypto import CheckpointError, ciphertext_name

from ekido_verifier import worker
from ekido_verifier.worker import Verifier

KEY = "0x" + "b" * 64
JOB = "11111111-2222-4333-8444-555555555555"
NAMES = ("trainable.safetensors", "optimizer.safetensors")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class World:
    def __init__(self, job_key=None):
        self.key = job_key or jobkeys.create_job_key()
        self.store: dict[str, bytes] = {}
        self.plain = {n: os.urandom(200 + i) for i, n in enumerate(NAMES)}
        files = {}
        for name, data in self.plain.items():
            blob = jobkeys.encrypt_bytes(data, self.key.recipient)
            self.store[sha(blob)] = blob
            files[name] = {"sha256": sha(blob), "bytes": len(blob), "plaintext_sha256": sha(data), "plaintext_bytes": len(data), "encryption": "age-v1"}
        self.manifest = {"format": "GPCK/2", "files": files, "logical_step": 10}

    def download(self, digest, target, expected_bytes):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(self.store[digest])


def verifier(world, coordinator_answers=None):
    v = Verifier("http://coordinator.invalid", KEY, "cpu", None)
    v._download = world.download
    v.calls = []

    def call(method, path, body=None, timeout=120):
        v.calls.append((method, path, body))
        if path == "/verifier/tasks/job-key":
            if isinstance(coordinator_answers, Exception):
                raise coordinator_answers
            sealed = jobkeys.encrypt_bytes((coordinator_answers or world.key).identity.encode("ascii"), body["recipient"])
            return {"ciphertext_b64": base64.b64encode(sealed).decode()}
        raise AssertionError("unexpected call " + path)

    v._call = call
    return v


class AskingForTheKey(unittest.TestCase):
    def test_it_sends_only_a_public_recipient_and_decrypts_the_answer(self):
        world = World()
        v = verifier(world)
        identity = v._job_key_identity(JOB)
        self.assertEqual(identity, world.key.identity)
        (method, path, body), = v.calls
        self.assertEqual((method, path), ("POST", "/verifier/tasks/job-key"))
        self.assertEqual(set(body), {"address", "job_id", "recipient"})
        self.assertEqual((body["address"].lower(), body["job_id"]), (v.address.lower(), JOB))
        self.assertTrue(body["recipient"].startswith("age1"))
        self.assertNotIn("AGE-SECRET-KEY", json.dumps(body))

    def test_a_refusal_from_the_coordinator_is_not_swallowed(self):
        refusal = urllib.error.HTTPError("u", 403, "NOT_ASSIGNED", {}, None)
        with self.assertRaises(urllib.error.HTTPError):
            verifier(World(), refusal)._job_key_identity(JOB)

    def test_a_reply_encrypted_to_someone_else_is_refused(self):
        v = verifier(World())
        v._call = lambda *a, **k: {"ciphertext_b64": base64.b64encode(jobkeys.encrypt_bytes(b"x", jobkeys.create_job_key().recipient)).decode()}
        with self.assertRaises(jobkeys.JobKeyError):
            v._job_key_identity(JOB)


class FetchingCheckpoints(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_blobs_are_fetched_decrypted_and_the_encrypted_copies_removed(self):
        world = World()
        folder = verifier(world)._fetch_checkpoint(world.manifest, self.root / "candidate", world.key.identity)
        for name, data in world.plain.items():
            self.assertEqual((folder / name).read_bytes(), data)
        self.assertEqual(json.loads((folder / "manifest.json").read_text()), world.manifest)
        self.assertFalse((self.root / "candidate_blobs").exists())
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["candidate"])

    def test_an_encrypted_checkpoint_without_the_key_is_refused(self):
        world = World()
        with self.assertRaises(ValueError):
            verifier(world)._fetch_checkpoint(world.manifest, self.root / "c")

    def test_a_key_that_does_not_open_the_blobs_reports_nothing(self):
        world = World()
        with self.assertRaises(CheckpointError) as caught:
            verifier(world)._fetch_checkpoint(world.manifest, self.root / "c", jobkeys.create_job_key().identity)
        self.assertEqual(caught.exception.code, "DECRYPT_FAILED")
        self.assertFalse((self.root / "c_blobs").exists())

    def test_a_manifest_that_lies_about_the_plaintext_leaves_no_tensor_files_so_evaluation_fails_it(self):
        world = World()
        lying = json.loads(json.dumps(world.manifest))
        lying["files"]["optimizer.safetensors"]["plaintext_sha256"] = "0" * 64
        folder = verifier(world)._fetch_checkpoint(lying, self.root / "c", world.key.identity)
        self.assertEqual([p.name for p in folder.glob("*.safetensors")], [])  # nothing for the evaluator to load
        self.assertTrue((folder / "manifest.json").exists())

    def test_a_gpck_1_checkpoint_is_fetched_exactly_as_before(self):
        plain = {n: os.urandom(40 + i) for i, n in enumerate(NAMES)}
        store = {sha(d): d for d in plain.values()}
        manifest = {"format": "GPCK/1", "files": {n: {"sha256": sha(d), "bytes": len(d)} for n, d in plain.items()}}
        v = Verifier("http://coordinator.invalid", KEY, "cpu", None)
        v._download = lambda digest, target, expected: (Path(target).parent.mkdir(parents=True, exist_ok=True), Path(target).write_bytes(store[digest]))
        folder = v._fetch_checkpoint(manifest, self.root / "v1")
        for name, data in plain.items():
            self.assertEqual((folder / name).read_bytes(), data)


class HandlingATask(unittest.TestCase):
    def task(self, manifest, reference=None):
        return {"job_id": JOB, "task_id": "t", "manifest": manifest, "reference_manifest": reference, "spec": {}, "policy": {}, "is_final": False, "expected_signer": None}

    def run_handle(self, v, task):
        seen = {}

        def fake_evaluate(**kwargs):
            seen["candidate"] = sorted(p.name for p in Path(kwargs["checkpoint_dir"]).iterdir())
            seen["reference"] = sorted(p.name for p in Path(kwargs["reference_dir"]).iterdir()) if kwargs.get("reference_dir") else None
            seen["kwargs"] = sorted(kwargs)
            return {"verdict": "PASS", "reasons": [], "is_final": False, "eval_set_hash": "x", "job_id": JOB, "checkpoint_hash": "0" * 64, "signer": v.address}

        with patch.object(worker, "evaluate_checkpoint", fake_evaluate), patch.object(worker, "sign_report", lambda report, key: report), patch.object(v, "_eval_set", lambda spec: object()):
            v.handle(task)
        return seen

    def test_the_key_is_asked_for_once_per_task_and_the_evaluator_gets_plaintext_files(self):
        world, other = World(), None
        other = World(world.key)  # a reference checkpoint of the same job (the same key)
        world.store.update(other.store)
        v = verifier(world)
        seen = self.run_handle(v, self.task(world.manifest, other.manifest))
        self.assertEqual([c[1] for c in v.calls], ["/verifier/tasks/job-key"])  # one request for both checkpoints
        self.assertEqual(seen["candidate"], ["manifest.json", "optimizer.safetensors", "trainable.safetensors"])
        self.assertEqual(seen["reference"], ["manifest.json", "optimizer.safetensors", "trainable.safetensors"])
        self.assertNotIn("identity", seen["kwargs"])  # the key is not handed to the evaluator

    def test_a_gpck_1_task_never_asks_for_a_key(self):
        plain = {n: os.urandom(30) for n in NAMES}
        store = {sha(d): d for d in plain.values()}
        manifest = {"format": "GPCK/1", "files": {n: {"sha256": sha(d), "bytes": len(d)} for n, d in plain.items()}}
        v = Verifier("http://coordinator.invalid", KEY, "cpu", None)
        v._download = lambda digest, target, expected: (Path(target).parent.mkdir(parents=True, exist_ok=True), Path(target).write_bytes(store[digest]))
        calls = []
        v._call = lambda *a, **k: calls.append(a)
        self.run_handle(v, self.task(manifest))
        self.assertEqual(calls, [])

    def test_when_the_coordinator_refuses_the_key_nothing_is_evaluated_or_reported(self):
        world = World()
        v = verifier(world, urllib.error.HTTPError("u", 403, "NOT_ASSIGNED", {}, None))
        with self.assertRaises(urllib.error.HTTPError):
            self.run_handle(v, self.task(world.manifest))

    def test_the_secret_key_never_appears_in_a_log(self):
        world = World()
        with self.assertLogs(level="DEBUG") as logs:
            import logging

            logging.getLogger("probe").debug("start")
            lying = json.loads(json.dumps(world.manifest))
            lying["files"]["trainable.safetensors"]["plaintext_bytes"] += 1
            self.run_handle(verifier(world), self.task(lying))
        self.assertNotIn(world.key.identity, "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()

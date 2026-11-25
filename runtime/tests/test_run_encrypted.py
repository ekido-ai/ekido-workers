"""A13: the runtime writes GPCK/2 checkpoints when the job has a recipient, and resumes from a decrypted one."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ekido_core import jobkeys
from ekido_core.checkpoint_crypto import ciphertext_name, decrypt_checkpoint_files
from ekido_core.gpck import manifest_hash, read_manifest

from ekido_runtime.run import EXIT_OK, run

from test_run import config, events_of


class EncryptedRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.key = jobkeys.create_job_key()

    def checkpoints(self, workdir):
        return [e for e in events_of(workdir) if e["kind"] == "checkpoint.ready"]

    def test_every_checkpoint_is_gpck_2_with_encrypted_blobs_and_the_chain_still_holds(self):
        w = self.root / "a"
        self.assertEqual(run(config(w, job_recipient=self.key.recipient)), EXIT_OK)
        checkpoints = self.checkpoints(w)
        self.assertEqual([c["step"] for c in checkpoints], [5, 10, 15, 20])
        for event in checkpoints:
            folder = Path(event["path"])
            manifest = event["manifest"]
            self.assertEqual(manifest["format"], "GPCK/2")
            for name, entry in manifest["files"].items():
                blob = folder / ciphertext_name(name)
                self.assertTrue(jobkeys.is_age_ciphertext(blob.read_bytes()))
                self.assertEqual(entry["encryption"], "age-v1")
            self.assertEqual(manifest_hash(read_manifest(folder)), event["manifest_hash"])
        for parent, child in zip(checkpoints, checkpoints[1:]):
            self.assertEqual(child["manifest"]["parent_manifest_hash"], parent["manifest_hash"])

    def test_without_a_recipient_nothing_changes(self):
        w = self.root / "plain"
        run(config(w))
        for event in self.checkpoints(w):
            self.assertEqual(event["manifest"]["format"], "GPCK/1")
            self.assertEqual([p.suffix for p in Path(event["path"]).glob("*.age")], [])

    def test_resuming_from_a_decrypted_checkpoint_continues_the_same_chain(self):
        first = self.root / "first"
        run(config(first, job_recipient=self.key.recipient))
        ten = next(c for c in self.checkpoints(first) if c["step"] == 10)
        resume = self.root / "resume"
        # What the agent does on a new device: it gets the blobs, decrypts them with the job key, and keeps the real manifest.
        decrypt_checkpoint_files(Path(ten["path"]), ten["manifest"], self.key.identity, resume)
        shutil.copy(Path(ten["path"]) / "manifest.json", resume / "manifest.json")
        second = self.root / "second"
        self.assertEqual(run(config(second, attempt=2, token=2, resume_from=str(resume), job_recipient=self.key.recipient)), EXIT_OK)
        restored = [e for e in events_of(second) if e["kind"] == "run.restored"]
        self.assertEqual((restored[0]["step"], restored[0]["canary_ok"]), (10, True))
        later = self.checkpoints(second)
        self.assertEqual([c["step"] for c in later], [15, 20])
        self.assertEqual(later[0]["manifest"]["parent_manifest_hash"], ten["manifest_hash"])  # the parent is the real GPCK/2 manifest
        self.assertEqual(later[0]["manifest"]["format"], "GPCK/2")

    def test_resuming_from_the_blobs_alone_is_refused(self):
        first = self.root / "first"
        run(config(first, job_recipient=self.key.recipient))
        ten = next(c for c in self.checkpoints(first) if c["step"] == 10)
        blobs_only = self.root / "blobs"
        blobs_only.mkdir()
        for name in ("trainable.safetensors", "optimizer.safetensors"):
            shutil.copy(Path(ten["path"]) / ciphertext_name(name), blobs_only / name)  # ciphertext under the plaintext names
        shutil.copy(Path(ten["path"]) / "manifest.json", blobs_only / "manifest.json")
        self.assertNotEqual(run(config(self.root / "second", attempt=2, token=2, resume_from=str(blobs_only))), EXIT_OK)

    def test_a_bad_recipient_commits_no_checkpoint_and_the_run_fails(self):
        w = self.root / "bad"
        code = run(config(w, job_recipient="age1notarealkey"))
        self.assertNotEqual(code, EXIT_OK)
        self.assertEqual(list(w.glob("checkpoints/*/manifest.json")), [])  # the manifest is written last, so nothing can be committed


if __name__ == "__main__":
    unittest.main()

"""A13, agent side: encrypted (GPCK/2) checkpoints are uploaded as blobs and resumed by decrypting with the in-memory job key.
Standard library plus ekido_core only: the agent never imports torch."""

import asyncio
import base64
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ekido_core import jobkeys
from ekido_core.checkpoint_crypto import CheckpointError, ciphertext_name

from ekido_agent.agent import Agent, AgentConfig

JOB = "11111111-2222-4333-8444-555555555555"
NAMES = ("trainable.safetensors", "optimizer.safetensors")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class World:
    """A fake object store, a job key and an encrypted checkpoint in it."""

    def __init__(self):
        self.key = jobkeys.create_job_key()
        self.store: dict[str, bytes] = {}
        self.plain = {name: os.urandom(300 + i) for i, name in enumerate(NAMES)}
        files = {}
        for name, data in self.plain.items():
            blob = jobkeys.encrypt_bytes(data, self.key.recipient)
            self.store[sha(blob)] = blob
            files[name] = {"sha256": sha(blob), "bytes": len(blob), "plaintext_sha256": sha(data), "plaintext_bytes": len(data), "encryption": "age-v1"}
        self.manifest = {"format": "GPCK/2", "files": files, "logical_step": 10}

    def download(self, digest, target, expected_bytes):
        data = self.store[digest]
        if len(data) != expected_bytes:
            raise OSError("size differs")
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(data)


def make_agent(workdir):
    agent = Agent(AgentConfig("ws://coordinator", "provider", "0x" + "11" * 32, "0x" + "22" * 32, Path(workdir), mode="process"))
    agent.sent = []

    async def capture(kind, payload=None, *, job_id=None, fencing_token=None):
        agent.sent.append({"type": kind, "payload": payload or {}, "job_id": job_id, "fencing_token": fencing_token})

    agent.send = capture
    return agent


class PrepareResume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.world = World()
        self.agent = make_agent(self.work)
        self.agent._download = self.world.download

    def test_blobs_are_fetched_by_hash_decrypted_and_the_real_manifest_is_kept(self):
        self.agent._prepare_resume(self.work, self.world.manifest, self.world.key.identity)
        for name, data in self.world.plain.items():
            self.assertEqual((self.work / "resume" / name).read_bytes(), data)
        self.assertEqual(json.loads((self.work / "resume" / "manifest.json").read_text()), self.world.manifest)
        self.assertFalse((self.work / "resume_blobs").exists())  # the encrypted copies are not left behind

    def test_a_wrong_key_a_missing_key_and_a_tampered_blob_are_refused_and_leave_no_plaintext(self):
        with self.assertRaises(CheckpointError):
            self.agent._prepare_resume(self.work, self.world.manifest, jobkeys.create_job_key().identity)
        with self.assertRaises(ValueError):
            self.agent._prepare_resume(self.work, self.world.manifest, None)
        digest = self.world.manifest["files"]["trainable.safetensors"]["sha256"]
        blob = bytearray(self.world.store[digest])
        blob[len(blob) // 2] ^= 1
        self.world.store[digest] = bytes(blob)
        with self.assertRaises(CheckpointError):
            self.agent._prepare_resume(self.work, self.world.manifest, self.world.key.identity)
        self.assertEqual([p.name for p in (self.work / "resume").glob("*.safetensors")], [])
        self.assertFalse((self.work / "resume_blobs").exists())

    def test_a_gpck_1_checkpoint_is_downloaded_exactly_as_before(self):
        plain = {name: os.urandom(50 + i) for i, name in enumerate(NAMES)}
        store = {sha(d): d for d in plain.values()}
        manifest = {"format": "GPCK/1", "files": {n: {"sha256": sha(d), "bytes": len(d)} for n, d in plain.items()}}
        self.agent._download = lambda digest, target, expected: (Path(target).parent.mkdir(parents=True, exist_ok=True), Path(target).write_bytes(store[digest]))
        self.agent._prepare_resume(self.work, manifest)
        for name, data in plain.items():
            self.assertEqual((self.work / "resume" / name).read_bytes(), data)


class Upload(unittest.IsolatedAsyncioTestCase):
    async def test_an_encrypted_checkpoint_uploads_the_blobs_never_the_plaintext(self):
        with tempfile.TemporaryDirectory() as tmp:
            world = World()
            agent = make_agent(tmp)
            agent.upload_token = "t"
            folder = Path(tmp) / "step"
            folder.mkdir()
            for name in NAMES:
                (folder / name).write_bytes(world.plain[name])  # the runtime keeps the plaintext locally
                digest = world.manifest["files"][name]["sha256"]
                (folder / ciphertext_name(name)).write_bytes(world.store[digest])
            uploaded = []
            agent._upload = lambda path, digest: uploaded.append((Path(path).name, digest))
            runner = Mock(workdir=Path(tmp))
            ok = await agent.upload_checkpoint(runner, {"manifest": world.manifest, "path": str(folder), "step": 10}, JOB, 3)
            self.assertTrue(ok)
            self.assertEqual(sorted(uploaded), sorted((ciphertext_name(n), world.manifest["files"][n]["sha256"]) for n in NAMES))
            sent = agent.sent[-1]
            self.assertEqual(sent["type"], "checkpoint.uploaded")
            self.assertEqual(sent["payload"]["manifest"], world.manifest)

    async def test_a_gpck_1_checkpoint_still_uploads_its_plain_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            folder = Path(tmp) / "step"
            folder.mkdir()
            manifest = {"format": "GPCK/1", "files": {}}
            for name in NAMES:
                data = os.urandom(40)
                (folder / name).write_bytes(data)
                manifest["files"][name] = {"sha256": sha(data), "bytes": len(data)}
            uploaded = []
            agent._upload = lambda path, digest: uploaded.append(Path(path).name)
            self.assertTrue(await agent.upload_checkpoint(Mock(workdir=Path(tmp)), {"manifest": manifest, "path": str(folder), "step": 5}, JOB, 3))
            self.assertEqual(sorted(uploaded), sorted(NAMES))


class StartRunWaitsForTheKey(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.world = World()
        self.agent = make_agent(self.tmp.name)
        self.agent._download = self.world.download
        self.config = {"job_id": JOB, "fencing_token": 4, "attempt": 2, "spec": {}, "resume_manifest": self.world.manifest, "job_recipient": self.world.key.recipient}

    async def start(self):
        await self.agent.on_command({"type": "cmd.start_run", "payload": {"config": dict(self.config)}})

    async def deliver_key(self, job_key=None):
        request = next(m for m in self.agent.sent if m["type"] == "job_key.request")
        sealed = jobkeys.encrypt_bytes((job_key or self.world.key).identity.encode("ascii"), request["payload"]["recipient"])
        await self.agent.on_command({"type": "job_key.release", "job_id": JOB, "fencing_token": 4, "payload": {"ciphertext_b64": base64.b64encode(sealed).decode()}})

    async def until_asking(self):
        for _ in range(100):
            if any(m["type"] == "job_key.request" for m in self.agent.sent):
                return
            await asyncio.sleep(0.01)
        self.fail("the agent never asked for the key")

    async def test_the_message_loop_is_not_blocked_and_the_run_starts_once_the_key_arrives(self):
        with patch("ekido_agent.agent.Runner") as runner_class:
            runner_class.return_value = Mock()
            self.agent.follow_run = Mock(side_effect=lambda *a: asyncio.sleep(0))
            await self.start()  # returns at once, though the key has not come
            self.assertIsNotNone(self.agent._start_task)
            await self.until_asking()
            self.assertFalse(runner_class.return_value.start.called)
            await self.deliver_key()  # handled while the start is waiting: no deadlock
            await asyncio.wait_for(self.agent._start_task, 10)
            runner_class.return_value.start.assert_called_once()
            for name, data in self.world.plain.items():
                found = list(Path(self.tmp.name).rglob(f"resume/{name}"))
                self.assertEqual([f.read_bytes() for f in found], [data])  # the runtime would read the decrypted file here
        self.assertEqual([m["payload"].get("reason") for m in self.agent.sent if m["type"] == "run.exited"], [])

    async def test_a_key_that_never_arrives_fails_the_start_clearly_and_no_runtime_starts(self):
        async def never(*args, **kwargs):
            return None

        self.agent.wait_for_job_key = never
        with patch("ekido_agent.agent.Runner") as runner_class:
            await self.start()
            await asyncio.wait_for(self.agent._start_task, 10)
            self.assertFalse(runner_class.called and runner_class.return_value.start.called)
        failed = [m for m in self.agent.sent if m["type"] == "run.exited"]
        self.assertEqual((failed[0]["payload"]["reason"], failed[0]["fencing_token"]), ("START_FAILED", 4))
        self.assertIn("job key", failed[0]["payload"]["detail"])

    async def test_the_wrong_key_fails_the_start_and_no_runtime_starts(self):
        with patch("ekido_agent.agent.Runner") as runner_class:
            await self.start()
            await self.until_asking()
            await self.deliver_key(job_key=jobkeys.create_job_key())  # a key that is not this job's
            await asyncio.wait_for(self.agent._start_task, 10)
            self.assertFalse(runner_class.return_value.start.called)
        failed = [m for m in self.agent.sent if m["type"] == "run.exited"]
        self.assertEqual(failed[0]["payload"]["reason"], "START_FAILED")
        self.assertNotIn("AGE-SECRET-KEY", json.dumps(self.agent.sent))

    async def test_stopping_cancels_a_start_that_is_still_waiting(self):
        with patch("ekido_agent.agent.Runner") as runner_class:
            await self.start()
            await self.until_asking()
            task = self.agent._start_task
            await self.agent.stop_run(None, "replaced")
            await asyncio.sleep(0.05)
            self.assertTrue(task.cancelled() or task.done())
            self.assertFalse(runner_class.return_value.start.called)
            self.assertIsNone(self.agent._start_task)

    async def test_an_unencrypted_resume_does_not_wait_and_does_not_use_a_task(self):
        plain = {n: os.urandom(30) for n in NAMES}
        store = {sha(d): d for d in plain.values()}
        self.config["resume_manifest"] = {"format": "GPCK/1", "files": {n: {"sha256": sha(d), "bytes": len(d)} for n, d in plain.items()}}
        self.agent._download = lambda digest, target, expected: (Path(target).parent.mkdir(parents=True, exist_ok=True), Path(target).write_bytes(store[digest]))
        with patch("ekido_agent.agent.Runner") as runner_class:
            runner_class.return_value = Mock()
            self.agent.follow_run = Mock(side_effect=lambda *a: asyncio.sleep(0))
            await self.start()
            self.assertIsNone(self.agent._start_task)
            runner_class.return_value.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()

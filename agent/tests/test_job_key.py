"""A12, agent side: a fresh keypair per assignment, the job key kept in memory only and forgotten when the run ends."""

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from ekido_core import jobkeys

from ekido_agent.agent import Agent, AgentConfig


JOB = "11111111-2222-4333-8444-555555555555"


def make_agent(workdir="."):
    agent = Agent(AgentConfig("ws://coordinator", "provider", "0x" + "11" * 32, "0x" + "22" * 32, Path(workdir), mode="process"))
    agent.sent = []

    async def capture(kind, payload=None, *, job_id=None, fencing_token=None):
        agent.sent.append({"type": kind, "payload": payload or {}, "job_id": job_id, "fencing_token": fencing_token})

    agent.send = capture
    return agent


def coordinator_release(job_key, recipient):
    """What the coordinator does: the job identity, encrypted to the device's one-off recipient."""
    return base64.b64encode(jobkeys.encrypt_bytes(job_key.identity.encode("ascii"), recipient)).decode("ascii")


class KeyExchange(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.agent = make_agent()
        self.job_key = jobkeys.create_job_key()

    async def request(self, job_id="job-1", token=1):
        await self.agent.request_job_key(job_id, token)
        return self.agent.sent[-1]

    async def test_a_request_names_a_valid_one_off_recipient(self):
        message = await self.request()
        self.assertEqual((message["type"], message["job_id"], message["fencing_token"]), ("job_key.request", "job-1", 1))
        recipient = message["payload"]["recipient"]
        self.assertTrue(recipient.startswith("age1"))
        self.assertEqual(set(message["payload"]), {"recipient"})  # the secret half is never in the message
        self.assertEqual(jobkeys.decrypt_bytes(jobkeys.encrypt_bytes(b"x", recipient), self.agent._pending_key[2].identity), b"x")

    async def test_every_assignment_gets_a_fresh_keypair(self):
        first = (await self.request("job-1", 1))["payload"]["recipient"]
        second = (await self.request("job-1", 2))["payload"]["recipient"]
        third = (await self.request("job-2", 1))["payload"]["recipient"]
        self.assertEqual(len({first, second, third}), 3)

    async def test_the_released_key_is_read_and_held_only_for_the_running_job(self):
        recipient = (await self.request())["payload"]["recipient"]
        self.agent.job_id, self.agent.fencing_token = "job-1", 1
        self.agent.receive_job_key("job-1", 1, coordinator_release(self.job_key, recipient))
        key = self.agent.job_key()
        self.assertEqual((key.recipient, key.identity), (self.job_key.recipient, self.job_key.identity))
        self.assertIsNone(self.agent._pending_key)
        self.agent.job_id = "job-9"  # a different run is current: the key is not handed out
        self.assertIsNone(self.agent.job_key())

    async def test_a_reply_for_another_job_or_attempt_is_ignored(self):
        recipient = (await self.request("job-1", 3))["payload"]["recipient"]
        release = coordinator_release(self.job_key, recipient)
        for job_id, token in (("job-2", 3), ("job-1", 2), (None, None)):
            self.agent.receive_job_key(job_id, token, release)
        self.assertIsNone(self.agent._job_key)
        self.assertIsNotNone(self.agent._pending_key)  # still waiting for the right reply

    async def test_a_reply_encrypted_to_someone_else_is_not_accepted(self):
        await self.request()
        stranger = jobkeys.create_job_key()
        with self.assertLogs("gpuchain.agent", "WARNING") as logs:
            self.agent.receive_job_key("job-1", 1, coordinator_release(self.job_key, stranger.recipient))
        self.assertIsNone(self.agent._job_key)
        self.assertNotIn(self.job_key.identity, "\n".join(logs.output))

    async def test_garbage_replies_are_not_accepted(self):
        await self.request()
        for junk in ("", "!!!not base64!!!", base64.b64encode(b"not age").decode(), 5, None):
            with self.subTest(junk=junk):
                self.agent.receive_job_key("job-1", 1, junk)
                self.assertIsNone(self.agent._job_key)

    async def test_the_key_is_forgotten_when_the_run_stops(self):
        recipient = (await self.request())["payload"]["recipient"]
        self.agent.receive_job_key("job-1", 1, coordinator_release(self.job_key, recipient))
        self.assertIsNotNone(self.agent._job_key)
        await self.agent.stop_run(None, "test")
        self.assertIsNone(self.agent._job_key)
        self.assertIsNone(self.agent._pending_key)

    async def test_commands_from_the_coordinator_are_routed(self):
        recipient = (await self.request())["payload"]["recipient"]
        self.agent.job_id, self.agent.fencing_token = "job-1", 1
        await self.agent.on_command({"type": "job_key.release", "job_id": "job-1", "fencing_token": 1, "payload": {"ciphertext_b64": coordinator_release(self.job_key, recipient)}})
        self.assertEqual(self.agent.job_key().identity, self.job_key.identity)
        with self.assertLogs("gpuchain.agent", "WARNING") as logs:
            await self.agent.on_command({"type": "error", "payload": {"error": "KEY_REFUSED", "reason": "JOB_ENDED"}})
        self.assertIn("JOB_ENDED", "\n".join(logs.output))

    async def test_no_secret_reaches_a_log_or_a_message(self):
        with self.assertLogs("gpuchain.agent", "DEBUG") as logs:
            import logging

            logging.getLogger("gpuchain.agent").debug("start")
            recipient = (await self.request())["payload"]["recipient"]
            ours = self.agent._pending_key[2].identity
            self.agent.receive_job_key("job-1", 1, coordinator_release(self.job_key, recipient))
            self.agent.receive_job_key("job-1", 1, "garbage")
        everything = "\n".join(logs.output) + json.dumps(self.agent.sent)
        for secret in (ours, self.job_key.identity):
            self.assertNotIn(secret, everything)
        self.assertNotIn(repr(self.agent._job_key[2]) if self.agent._job_key else "", self.job_key.identity)
        self.assertNotIn(self.job_key.identity, repr(self.agent._job_key))


class StartRun(unittest.IsolatedAsyncioTestCase):
    async def test_starting_a_run_asks_for_the_key_before_anything_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            agent.follow_run = AsyncMock()
            with patch("ekido_agent.agent.Runner") as runner_class:
                runner_class.return_value = Mock()
                await agent.start_run({"job_id": JOB, "fencing_token": 4, "attempt": 1, "spec": {}})
            kinds = [m["type"] for m in agent.sent]
            self.assertEqual(kinds, ["job_key.request"])
            self.assertEqual((agent.sent[0]["job_id"], agent.sent[0]["fencing_token"]), (JOB, 4))
            runner_class.return_value.start.assert_called_once()  # the run still starts; encrypted data is A13 and A14


if __name__ == "__main__":
    unittest.main()

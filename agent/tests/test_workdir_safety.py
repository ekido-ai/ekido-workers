import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from gpuchain_agent.workdir import run_directory
from gpuchain_agent.agent import Agent, AgentConfig


class WorkdirSafety(unittest.TestCase):
    def test_accepts_canonical_id_and_positive_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            job_id = str(uuid4())
            self.assertEqual(run_directory(Path(directory), job_id, 2), Path(directory).resolve() / job_id[:12] / "attempt_2")

    def test_refuses_traversal_and_invalid_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            for job, attempt in (("../../outside", 1), (str(uuid4()), "../../outside"), (str(uuid4()), 0), (str(uuid4()), True)):
                with self.subTest(job=job, attempt=attempt), self.assertRaises(ValueError):
                    run_directory(Path(directory), job, attempt)

    def test_refuses_linked_parent_directory(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            job_id = str(uuid4())
            link = Path(directory) / job_id[:12]
            try:
                link.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("directory symlink creation unavailable")
            with self.assertRaises(ValueError):
                run_directory(Path(directory), job_id, 1)

    def test_refuses_a_detected_link_without_filesystem_privileges(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaises(ValueError):
                run_directory(Path(directory), str(uuid4()), 1)


class AgentWorkdirSafety(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_command_neither_stops_existing_work_nor_deletes_files(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(AgentConfig(coordinator="ws://localhost", provider_id="p", wallet_key="0x" + "1" * 64, worker_key="0x" + "2" * 64, workdir=Path(directory), mode="process"))
            agent.stop_run = AsyncMock()
            agent.send = AsyncMock()
            await agent.start_run({"job_id": "../../outside", "attempt": 1, "fencing_token": 1})
            agent.stop_run.assert_not_awaited()

    async def test_existing_attempt_is_preserved_instead_of_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            job_id = str(uuid4())
            target = run_directory(Path(directory), job_id, 1)
            target.mkdir(parents=True)
            sentinel = target / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            agent = Agent(AgentConfig(coordinator="ws://localhost", provider_id="p", wallet_key="0x" + "1" * 64, worker_key="0x" + "2" * 64, workdir=Path(directory), mode="process"))
            agent.stop_run = AsyncMock()
            agent.send = AsyncMock()
            await agent.start_run({"job_id": job_id, "attempt": 1, "fencing_token": 1})
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            agent.stop_run.assert_not_awaited()
            agent.send.assert_awaited_once()

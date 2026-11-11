"""Standard library only, so these run anywhere."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import websockets

from gpuchain_agent.agent import Agent, AgentConfig
from gpuchain_agent.spawner import DockerOptions, Runner, container_name, docker_run_args, reap_orphans
from gpuchain_agent.tailer import EventTailer
from gpuchain_agent.telemetry import native_bf16, parse_nvidia_smi


class Telemetry(unittest.TestCase):
    def test_parses_a_real_line(self):
        line = "NVIDIA GeForce RTX 3060 Laptop GPU, GPU-abc, 6144, 769, 29, 66, 25.03, 8.6, 581.80"
        sample = parse_nvidia_smi(line)[0]
        self.assertEqual(sample.name, "NVIDIA GeForce RTX 3060 Laptop GPU")
        self.assertEqual((sample.memory_total_mib, sample.memory_used_mib, sample.util_pct, sample.temp_c, sample.power_w), (6144, 769, 29, 66, 25.03))
        self.assertEqual((sample.compute_cap, sample.driver), ("8.6", "581.80"))

    def test_unknown_values_stay_unknown_not_zero(self):
        sample = parse_nvidia_smi("Tesla T4, GPU-x, 15360, 0, [N/A], [Not Supported], [N/A], [N/A], 550.1")[0]
        self.assertIsNone(sample.util_pct)
        self.assertIsNone(sample.temp_c)
        self.assertIsNone(sample.power_w)
        self.assertIsNone(sample.compute_cap)
        self.assertEqual(sample.memory_used_mib, 0)

    def test_garbage_is_skipped(self):
        self.assertEqual(parse_nvidia_smi("command not found\n\n"), [])

    def test_multiple_gpus(self):
        lines = "A, u1, 1, 1, 1, 1, 1, 8.0, d\nB, u2, 2, 2, 2, 2, 2, 7.5, d"
        self.assertEqual([s.name for s in parse_nvidia_smi(lines)], ["A", "B"])

    def test_bf16_native_only_from_ampere(self):
        self.assertTrue(native_bf16("8.6"))
        self.assertTrue(native_bf16("9.0"))
        self.assertFalse(native_bf16("7.5"))  # T4 emulates it, which is what this check exists for.
        self.assertIsNone(native_bf16(None))
        self.assertIsNone(native_bf16("bad"))


class DockerSecurity(unittest.TestCase):
    args = docker_run_args("n", "/work", DockerOptions())

    def flag_value(self, flag):
        return self.args[self.args.index(flag) + 1]

    def test_isolation_flags_are_present(self):
        self.assertEqual(self.flag_value("--network"), "none")
        self.assertEqual(self.flag_value("--cap-drop"), "ALL")
        self.assertEqual(self.flag_value("--security-opt"), "no-new-privileges")
        self.assertIn("--read-only", self.args)
        self.assertEqual(self.flag_value("--user"), "1000")

    def test_resource_limits_are_present(self):
        for flag in ("--memory", "--cpus", "--pids-limit"):
            self.assertIn(flag, self.args)

    def test_never_privileged_or_host_network(self):
        joined = " ".join(self.args)
        for bad in ("--privileged", "--network host", "docker.sock", "--pid host"):
            self.assertNotIn(bad, joined)

    def test_runs_only_the_runtime_module(self):
        self.assertEqual(self.args[-4:], ["python", "-m", "gpuchain_runtime.run", "/run_data/config.json"])

    def test_code_mounts_are_read_only(self):
        args = docker_run_args("n", "/work", DockerOptions(code_mounts=(("/src", "/app"),)))
        self.assertIn("/src:/app:ro", args)

    def test_gpu_can_be_off_for_cpu_fallback(self):
        self.assertNotIn("--gpus", docker_run_args("n", "/work", DockerOptions(gpu=False)))

    def test_container_name_is_unique_per_attempt(self):
        self.assertNotEqual(container_name("abc", 1), container_name("abc", 2))

    def test_startup_does_not_kill_other_agents_containers(self):
        with patch("gpuchain_agent.spawner.subprocess.run") as docker:
            self.assertEqual(reap_orphans(), [])
            docker.assert_not_called()


class RunnerConfig(unittest.TestCase):
    def test_docker_paths_use_container_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            final = Runner("job", 1, Path(tmp)).write_config({"resume_from": r"C:\host\ckpt", "x": 1})
            self.assertEqual(final["workdir"], "/run_data")
            self.assertEqual(final["lease_file"], "/run_data/lease")
            self.assertEqual(final["resume_from"], "/run_data/resume")
            self.assertEqual(json.loads((Path(tmp) / "config.json").read_text())["x"], 1)
            self.assertTrue((Path(tmp) / "lease").exists())

    def test_process_paths_use_host_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            final = Runner("job", 1, Path(tmp), mode="process").write_config({})
            self.assertEqual(final["workdir"], tmp)

    def test_unknown_mode_refused(self):
        with self.assertRaises(ValueError):
            Runner("job", 1, Path("."), mode="vm")

    def test_checkpoint_request_creates_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            Runner("job", 1, Path(tmp)).request_checkpoint()
            self.assertTrue((Path(tmp) / "checkpoint_now").exists())


class Tailer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "events.jsonl"

    def append(self, data: bytes):
        with open(self.path, "ab") as handle:
            handle.write(data)

    def test_missing_file_is_empty(self):
        self.assertEqual(EventTailer(self.path).read_new(), [])

    def test_reads_only_new_lines(self):
        tailer = EventTailer(self.path)
        self.append(b'{"kind":"a"}\n')
        self.assertEqual([e["kind"] for e in tailer.read_new()], ["a"])
        self.assertEqual(tailer.read_new(), [])
        self.append(b'{"kind":"b"}\n')
        self.assertEqual([e["kind"] for e in tailer.read_new()], ["b"])

    def test_waits_for_a_complete_line(self):
        tailer = EventTailer(self.path)
        self.append(b'{"kind":"pa')
        self.assertEqual(tailer.read_new(), [])
        self.append(b'rtial"}\n')
        self.assertEqual(tailer.read_new()[0]["kind"], "partial")

    def test_malformed_lines_are_counted_not_fatal(self):
        tailer = EventTailer(self.path)
        self.append(b'not json\n[1,2]\n{"no":"kind"}\n{"kind":"ok"}\n')
        self.assertEqual([e["kind"] for e in tailer.read_new()], ["ok"])
        self.assertEqual(tailer.malformed, 3)


class Sending(unittest.IsolatedAsyncioTestCase):
    async def test_a_closed_socket_doesnt_stop_the_run_being_followed(self):
        class ClosedSocket:
            async def send(self, _):
                raise websockets.ConnectionClosed(None, None)

        agent = Agent(AgentConfig("ws://coordinator", "provider", "0x" + "11" * 32, "0x" + "22" * 32, Path(".")))
        agent.ws = ClosedSocket()
        with self.assertLogs("gpuchain.agent", "WARNING"):
            await agent.send("run.progress", {"step": 1})


class ResumeDownloads(unittest.TestCase):
    def test_unexpected_file_names_are_refused_before_any_download(self):
        for name in ("../escaped.py", "nested/file", r"..\escaped.py", r"C:\escaped.py", "/escaped.py", "extra.safetensors"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                agent = Agent(AgentConfig("ws://coordinator", "provider", "0x" + "11" * 32, "0x" + "22" * 32, Path(tmp)))
                agent._download = Mock()
                files = {n: {"sha256": "a" * 64, "bytes": 8} for n in ("trainable.safetensors", "optimizer.safetensors", name)}
                with self.assertRaises(ValueError):
                    agent._prepare_resume(Path(tmp), {"files": files})
                agent._download.assert_not_called()


if __name__ == "__main__":
    unittest.main()

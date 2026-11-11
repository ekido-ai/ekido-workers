"""Starts and stops the training runtime, in a locked-down container or as a plain process.

The container limits protect the provider's host from the job. Nothing here protects the job's data
from the provider, because that would need a TEE. Process mode has weak isolation and is only for
hosts the team runs itself.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

CONTAINER_WORKDIR = "/run_data"
CONTAINER_CONFIG = f"{CONTAINER_WORKDIR}/config.json"
HF_CACHE_VOLUME = "gpuchain_hf_cache"


@dataclass(frozen=True)
class DockerOptions:
    image: str = "gpuchain-runtime:latest"
    memory: str = "6g"
    cpus: str = "6"
    pids_limit: int = 512
    gpu: bool = True
    code_mounts: tuple[tuple[str, str], ...] = ()  # (host path, container path). Development only.


def docker_run_args(name: str, host_workdir: str, options: DockerOptions, command: tuple[str, ...] = ("gpuchain_runtime.run", CONTAINER_CONFIG)) -> list[str]:
    """The full docker command. A test pins the security flags, so they can't be dropped by accident."""
    args = [
        "docker", "run", "--rm",
        "--name", name,
        "--network", "none",  # Model and data come pre-cached. Artifacts move through the agent.
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--read-only",
        "--tmpfs", "/tmp:size=512m",
        "--user", "1000",
        "--pids-limit", str(options.pids_limit),
        "--memory", options.memory,
        "--cpus", options.cpus,
        "-e", "HF_HUB_OFFLINE=1",
        "-e", "HF_DATASETS_OFFLINE=1",
        "-e", "PYTHONPATH=/app/packages/gpuchain_core:/app/workers/runtime",
        "-v", f"{host_workdir}:{CONTAINER_WORKDIR}",
        "-v", f"{HF_CACHE_VOLUME}:/hf_cache",
    ]
    if options.gpu:
        args += ["--gpus", "all"]
    for host, container in options.code_mounts:
        args += ["-v", f"{host}:{container}:ro"]
    return args + [options.image, "python", "-m", *command]


def probe_command(root: Path, mode: str, options: DockerOptions, suffix: str) -> tuple[list[str], dict | None]:
    """Same isolation as a real run. A fresh process each time, so a failed attempt leaves no memory behind."""
    if mode == "docker":
        args = docker_run_args(f"gpuchain-probe-{suffix}", str(Path(root).resolve()), options, ("gpuchain_runtime.probe", CONTAINER_CONFIG, f"{CONTAINER_WORKDIR}/result.json"))
        return args, None
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    return [sys.executable, "-m", "gpuchain_runtime.probe", str(root / "config.json"), str(root / "result.json")], env


def container_name(job_id: str, attempt: int) -> str:
    return f"gpuchain-run-{job_id[:12]}-{attempt}"


class Runner:
    """One training attempt, on this machine."""

    def __init__(self, job_id: str, attempt: int, workdir: Path, *, mode: str = "docker", docker: DockerOptions | None = None):
        if mode not in ("docker", "process"):
            raise ValueError(f"unknown mode: {mode}")
        self.job_id = job_id
        self.attempt = attempt
        self.workdir = Path(workdir)
        self.mode = mode
        self.docker = docker or DockerOptions()
        self.name = container_name(job_id, attempt)
        self.process: subprocess.Popen | None = None
        self.lease_file = self.workdir / "lease"

    def write_config(self, config: dict) -> dict:
        """Paths in the config are the ones the runtime sees, which differ inside a container."""
        self.workdir.mkdir(parents=True, exist_ok=True)
        root = CONTAINER_WORKDIR if self.mode == "docker" else str(self.workdir)
        final = dict(config, workdir=root, lease_file=f"{root}/lease")
        if final.get("resume_from"):
            final["resume_from"] = f"{root}/resume" if self.mode == "docker" else str(self.workdir / "resume")
        self.renew_lease()
        (self.workdir / "config.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
        return final

    def renew_lease(self) -> None:
        self.lease_file.touch()

    def request_checkpoint(self) -> None:
        (self.workdir / "checkpoint_now").touch()

    def start(self, config: dict, *, extra_env: dict | None = None) -> None:
        self.write_config(config)
        if self.mode == "docker":
            args = docker_run_args(self.name, str(self.workdir.resolve()), self.docker)
            env = None
        else:
            args = [sys.executable, "-m", "gpuchain_runtime.run", str(self.workdir / "config.json")]
            env = {**os.environ, **(extra_env or {})}
        # The child keeps its own copy of the log handle, so the parent's can be closed right after the start.
        with open(self.workdir / "runtime.log", "ab") as log:
            self.process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)

    def poll(self) -> int | None:
        return None if self.process is None else self.process.poll()

    def kill(self) -> None:
        """Abrupt on purpose: a real provider failure gives no warning."""
        if self.mode == "docker":
            subprocess.run(["docker", "kill", self.name], capture_output=True, timeout=30, check=False)
        if self.process is not None and self.process.poll() is None:
            self.process.kill()


def reap_orphans(timeout: float = 30) -> list[str]:
    """Do not reap unowned containers: names alone cannot identify this agent's work.

    Runtime leases cause abandoned training to stop on their own. A future
    ownership-aware cleanup can handle other leftovers after proving ownership.
    """
    return []

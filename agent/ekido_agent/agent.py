"""The provider agent: connects out to the coordinator and runs what it's told.

    python -m ekido_agent.agent --coordinator ws://127.0.0.1:8000 --provider-id ... --wallet-key ... --mode docker

Telemetry and hardware details it reports are self-reported by the provider, and are labelled so.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import math
import os
import secrets
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import websockets
from eth_account import Account

from ekido_core import jobkeys
from ekido_core.signing import sign_document
from ekido_core.checkpoint_files import checkpoint_files
from ekido_core.transfer import download_blob, upload_blob

from .spawner import CONTAINER_WORKDIR, DockerOptions, Runner, probe_command, reap_orphans
from .tailer import EventTailer
from .telemetry import read_gpus
from .workdir import run_directory

log = logging.getLogger("gpuchain.agent")

LEASE_RENEW_SECONDS = 2.0
TAIL_POLL_SECONDS = 0.25
TELEMETRY_SECONDS = 2.0
PROBE_TIMEOUT_SECONDS = 170.0
PARTITION_SECONDS = 25.0
UPLOAD_ATTEMPTS = 3


def enrollment_document(provider_id: str, worker_key: str) -> dict:
    # Must match the coordinator's agent_gateway.enrollment_document.
    return {"kind": "gpuchain.enrollment.v1", "provider_id": provider_id, "worker_key": worker_key.lower()}


def sanitize(value):
    """NaN and Infinity aren't valid JSON or canonical JSON. A diverged run reports them as unknown."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return value


@dataclass
class AgentConfig:
    coordinator: str
    provider_id: str
    wallet_key: str
    worker_key: str
    workdir: Path
    mode: str = "docker"
    docker: DockerOptions | None = None
    device: str | None = None
    # A deliberately bad provider for the demo. The coordinator's INJECTED_FAULT badge must match this.
    fault: dict | None = None
    reconnect_seconds: float = 2.0


class Agent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.account = Account.from_key(config.worker_key)
        self.seq = 0
        self.ws = None
        self.upload_token: str | None = None
        self.runner: Runner | None = None
        self.run_task: asyncio.Task | None = None
        self.job_id: str | None = None
        self.fencing_token: int | None = None
        self.silent_until = 0.0  # Heartbeat-loss fault: keeps training but goes quiet, like a network partition.
        self.injected_exit = False
        self.corrupt_next_upload = False
        self.send_lock = asyncio.Lock()
        # The job key (A12). Held in memory only, for one assignment, and forgotten when the run ends. A fresh keypair is
        # made for every assignment; the coordinator encrypts the job's key to its public half, so only this process can read it.
        self._pending_key: tuple[str, int, jobkeys.JobKey] | None = None  # (job_id, fencing_token, our one-off keypair)
        self._job_key: tuple[str, int, jobkeys.JobKey] | None = None  # (job_id, fencing_token, the job's key)

    @property
    def http_base(self) -> str:
        return self.config.coordinator.replace("wss://", "https://").replace("ws://", "http://").rstrip("/")

    # ------------------------------------------------------------ messages

    async def send(self, message_type: str, payload: dict | None = None, *, job_id: str | None = None, fencing_token: int | None = None) -> None:
        if time.monotonic() < self.silent_until or self.ws is None:
            return
        async with self.send_lock:
            self.seq += 1
            message = {
                "v": 1,
                "seq": self.seq,
                "ts": datetime.now(timezone.utc).isoformat(),
                "type": message_type,
                "payload": sanitize(payload or {}),
                "job_id": job_id,
                "fencing_token": fencing_token,
            }
            _, message["sig"] = sign_document(message, self.config.worker_key)
            try:
                await self.ws.send(json.dumps(message))
            except websockets.ConnectionClosed:
                # The socket can close before the reconnect loop notices. The run must keep being followed.
                log.warning("not sent, connection closed: %s", message_type)

    async def hello(self) -> None:
        _, enrollment = sign_document(enrollment_document(self.config.provider_id, self.account.address), self.config.wallet_key)
        if self.config.device == "cpu":
            hardware = {"name": "CPU", "uuid": f"cpu-{self.config.provider_id}", "memory_total_mib": None, "compute_cap": "cpu", "driver": None}
        else:
            gpus = await asyncio.to_thread(read_gpus)
            hardware = gpus[0].to_dict() if gpus else None
        await self.send(
            "agent.hello",
            {
                "provider_id": self.config.provider_id,
                "worker_key": self.account.address,
                "enrollment_sig": enrollment,
                "mode": self.config.mode,
                "hardware": hardware,
            },
        )

    # ------------------------------------------------------------ main loop

    async def run_forever(self) -> None:
        if self.config.mode == "docker":
            orphans = await asyncio.to_thread(reap_orphans)
            if orphans:
                log.warning("removed leftover containers: %s", orphans)
        while True:
            try:
                await self.connect_once()
            except (OSError, websockets.WebSocketException) as exc:
                log.warning("connection lost: %s", exc)
            self.ws = None
            await asyncio.sleep(self.config.reconnect_seconds)

    async def connect_once(self) -> None:
        url = self.config.coordinator.rstrip("/") + "/agent/ws"
        async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
            self.ws = ws
            self.seq = int(time.time() * 1000)  # Stays above whatever this agent sent before a restart.
            await self.hello()
            background = asyncio.create_task(self.heartbeat_loop())
            try:
                async for raw in ws:
                    await self.on_command(json.loads(raw))
            finally:
                background.cancel()

    async def heartbeat_loop(self) -> None:
        while True:
            await self.send("agent.heartbeat")
            gpus = await asyncio.to_thread(read_gpus)
            if gpus:
                g = gpus[0]
                await self.send("agent.telemetry", {"util_pct": g.util_pct, "mem_used_mib": g.memory_used_mib, "temp_c": g.temp_c, "power_w": g.power_w, "source": g.source})
            await asyncio.sleep(TELEMETRY_SECONDS)

    async def on_command(self, message: dict) -> None:
        kind, payload = message.get("type"), message.get("payload") or {}
        if kind == "agent.welcome":
            self.upload_token = payload.get("upload_token")
            log.info("connected as provider %s", self.config.provider_id)
        elif kind == "cmd.probe":
            asyncio.create_task(self.do_probe(payload))
        elif kind == "cmd.start_run":
            await self.start_run(payload["config"])
        elif kind == "cmd.stop_run":
            await self.stop_run(message.get("fencing_token"), payload.get("reason"))
        elif kind == "cmd.inject_fault":
            await self.inject_fault(payload.get("kind"))
        elif kind == "job_key.release":
            self.receive_job_key(message.get("job_id"), message.get("fencing_token"), payload.get("ciphertext_b64"))
        elif kind == "error" and payload.get("error") == "KEY_REFUSED":
            log.warning("the coordinator refused the job key: %s", payload.get("reason"))  # a fixed code, never key material
        elif kind == "error":
            log.warning("coordinator reported: %s", payload)

    # ------------------------------------------------------------ job key (A12)

    async def request_job_key(self, job_id: str, token: int) -> None:
        """Make a fresh keypair for this assignment and ask the coordinator for the job's key, naming our public half."""
        ours = jobkeys.create_job_key()
        self._pending_key = (job_id, token, ours)
        await self.send("job_key.request", {"recipient": ours.recipient}, job_id=job_id, fencing_token=token)

    def receive_job_key(self, job_id, token, ciphertext_b64) -> None:
        """The job's key, encrypted to the keypair we made. A reply for another job or attempt is ignored."""
        pending = self._pending_key
        if pending is None or (pending[0], pending[1]) != (job_id, token) or not isinstance(ciphertext_b64, str):
            return
        try:
            identity = jobkeys.decrypt_bytes(base64.b64decode(ciphertext_b64, validate=True), pending[2].identity).decode("ascii")
            key = jobkeys.JobKey(recipient=str(jobkeys.x25519.Identity.from_str(identity).to_public()), identity=identity)
        except Exception:  # any failure (bad base64, wrong key, damaged data) means no key; the message says nothing more
            log.warning("could not read the job key sent for job %s", job_id)
            return
        self._job_key = (job_id, token, key)
        self._pending_key = None

    def job_key(self) -> "jobkeys.JobKey | None":
        """The key for the run in progress, or None. Never stored on disk, never logged."""
        held = self._job_key
        if held is not None and (held[0], held[1]) == (self.job_id, self.fencing_token):
            return held[2]
        return None

    def forget_job_key(self) -> None:
        self._pending_key = self._job_key = None

    # ------------------------------------------------------------- probing

    async def do_probe(self, payload: dict) -> None:
        request_id, config = payload["request_id"], payload["config"]
        result = await asyncio.to_thread(self._run_probe, config)
        await self.send("probe.result", {"request_id": request_id, "result": result})

    def _run_probe(self, config: dict) -> dict:
        self.config.workdir.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="probe_", dir=self.config.workdir))
        try:
            (root / "config.json").write_text(json.dumps(dict(config, device=self.config.device)), encoding="utf-8")
            spec = config.get("spec") or {}
            dataset = spec.get("dataset") if isinstance(spec.get("dataset"), dict) else {}
            if spec.get("template") == "llm-lora/v1" and dataset.get("sha256"):
                # The probe trains a few steps on the customer's own rows, so it needs them like a run does (verified by hash).
                self._download(dataset["sha256"], root / "dataset.jsonl", dataset["bytes"])
            command, env = probe_command(root, self.config.mode, self.config.docker or DockerOptions(), secrets.token_hex(4))
            import subprocess

            subprocess.run(command, env=env, capture_output=True, timeout=PROBE_TIMEOUT_SECONDS, check=False)
            result_file = root / "result.json"
            if result_file.is_file():
                return json.loads(result_file.read_text(encoding="utf-8"))
            return {"ok": False, "oom": False, "error": "Probe process wrote no result"}
        except (OSError, ValueError, Exception) as exc:  # A probe failing is an answer, never a crash.
            return {"ok": False, "oom": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
        finally:
            shutil.rmtree(root, ignore_errors=True)

    # ------------------------------------------------------------- running

    def _download(self, digest: str, target: Path, expected_bytes: int) -> None:
        download_blob(self.http_base, self.upload_token, digest, target, expected_bytes, timeout=120)

    def _prepare_resume(self, workdir: Path, manifest: dict) -> None:
        """The runtime validates everything again on load. This only fetches the files."""
        files = checkpoint_files(manifest)
        resume = workdir / "resume"
        resume.mkdir(parents=True, exist_ok=True)
        for name, entry in files.items():
            self._download(entry["sha256"], resume / name, entry["bytes"])
        (resume / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    async def start_run(self, config: dict) -> None:
        try:
            job_id, token, attempt = config["job_id"], config["fencing_token"], config["attempt"]
            if type(token) is not int or token < 1:
                raise ValueError("invalid fencing token")
            workdir = run_directory(self.config.workdir, job_id, attempt)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            log.warning("refusing invalid run command: %s", exc)
            return
        if workdir.exists():
            await self.send("run.exited", {"reason": "START_FAILED", "detail": "attempt directory already exists"}, job_id=job_id, fencing_token=token)
            return
        await self.stop_run(None, "starting a new run")
        try:
            workdir.parent.mkdir(parents=True, exist_ok=True)
            workdir = run_directory(self.config.workdir, job_id, attempt)
            workdir.mkdir(exist_ok=False)
        except (OSError, ValueError) as exc:
            await self.send("run.exited", {"reason": "START_FAILED", "detail": str(exc)[:300]}, job_id=job_id, fencing_token=token)
            return

        await self.request_job_key(job_id, token)
        run_config = dict(config)
        manifest = run_config.pop("resume_manifest", None)
        run_config.pop("resume_from_hash", None)
        run_config.pop("provider_kind", None)
        run_config["device"] = self.config.device
        run_config["lease_timeout"] = 15.0
        if self.config.fault:
            run_config["fault"] = self.config.fault
        try:
            if manifest:
                await asyncio.to_thread(self._prepare_resume, workdir, manifest)
                run_config["resume_from"] = "resume"
            spec = run_config.get("spec") or {}
            if spec.get("template") == "llm-lora/v1":
                # The customer's rows travel as one blob; the download is checked against its hash and size.
                dataset = spec["dataset"]
                await asyncio.to_thread(self._download, dataset["sha256"], workdir / "dataset.jsonl", dataset["bytes"])
            self.runner = Runner(job_id, attempt, workdir, mode=self.config.mode, docker=self.config.docker)
            self.job_id, self.fencing_token = job_id, token
            self.injected_exit = False
            env = {"PYTHONPATH": os.pathsep.join(sys.path)} if self.config.mode == "process" else None
            await asyncio.to_thread(self.runner.start, run_config, extra_env=env)
        except (OSError, urllib.error.URLError, ValueError) as exc:
            await self.send("run.exited", {"reason": "START_FAILED", "detail": str(exc)[:300]}, job_id=job_id, fencing_token=token)
            return
        self.run_task = asyncio.create_task(self.follow_run(self.runner, job_id, token))

    async def stop_run(self, fencing_token: int | None, reason: str | None) -> None:
        """A stale order for an older attempt is ignored, so it can't kill a newer run here."""
        held = {entry[1] for entry in (self._pending_key, self._job_key) if entry is not None}
        if fencing_token is None or fencing_token in held:
            self.forget_job_key()  # even if the runtime never started, a stopped assignment must not keep its key
        if self.runner is None or (fencing_token is not None and fencing_token != self.fencing_token):
            return
        log.info("stopping run: %s", reason)
        runner, task = self.runner, self.run_task
        self.runner, self.run_task, self.job_id, self.fencing_token = None, None, None, None
        self.forget_job_key()
        if task is not None:
            task.cancel()
        await asyncio.to_thread(runner.kill)

    async def follow_run(self, runner: Runner, job_id: str, token: int) -> None:
        tailer = EventTailer(runner.workdir / "events.jsonl")
        last_lease = 0.0
        exit_reported = False
        while True:
            now = time.monotonic()
            if now - last_lease >= LEASE_RENEW_SECONDS:
                runner.renew_lease()
                last_lease = now
            for event in tailer.read_new():
                if event["kind"] == "run.exited":
                    exit_reported = True
                if not await self.forward(runner, event, job_id, token):
                    await self.abandon_run(runner, job_id, token)
                    return
            code = runner.poll()
            if code is not None:
                for event in tailer.read_new():
                    if event["kind"] == "run.exited":
                        exit_reported = True
                    if not await self.forward(runner, event, job_id, token):
                        await self.abandon_run(runner, job_id, token)
                        return
                if not exit_reported:
                    # Killed from outside, so it couldn't say why. This is what a real crash looks like.
                    await self.send("run.exited", {"reason": "PROCESS_EXIT", "code": code, "injected": self.injected_exit}, job_id=job_id, fencing_token=token)
                if self.runner is runner:
                    self.runner, self.run_task, self.job_id, self.fencing_token = None, None, None, None
                    self.forget_job_key()
                return
            await asyncio.sleep(TAIL_POLL_SECONDS)

    async def abandon_run(self, runner: Runner, job_id: str, token: int) -> None:
        """A checkpoint couldn't be stored. Later ones would chain to it and be refused, so stop now and fail over."""
        log.info("stopping run: its checkpoint couldn't be stored")
        await asyncio.to_thread(runner.kill)
        await self.send("run.exited", {"reason": "BAD_CHECKPOINT", "injected": self.injected_exit}, job_id=job_id, fencing_token=token)
        if self.runner is runner:
            self.runner, self.run_task, self.job_id, self.fencing_token = None, None, None, None
            self.forget_job_key()

    async def forward(self, runner: Runner, event: dict, job_id: str, token: int) -> bool:
        """Returns False if the run can't usefully continue."""
        kind = event["kind"]
        fields = {k: v for k, v in event.items() if k not in ("kind", "ts")}
        if kind == "checkpoint.ready":
            return await self.upload_checkpoint(runner, event, job_id, token)
        if kind in ("run.started", "run.progress", "run.exited", "run.restored", "fault.applied"):
            if kind == "run.exited":
                fields["injected"] = self.injected_exit
            await self.send(kind, fields, job_id=job_id, fencing_token=token)
        return True

    def _host_path(self, runner: Runner, runtime_path: str) -> Path:
        """The runtime reports paths as it sees them, which differ from the host's inside a container."""
        if self.config.mode == "docker" and runtime_path.startswith(CONTAINER_WORKDIR):
            return runner.workdir / runtime_path[len(CONTAINER_WORKDIR) :].lstrip("/")
        return Path(runtime_path)

    def _upload(self, path: Path, digest: str) -> None:
        upload_blob(self.http_base, self.upload_token, path, digest, timeout=120)

    async def upload_checkpoint(self, runner: Runner, event: dict, job_id: str, token: int) -> bool:
        """Blobs first, then the signed manifest, which is the commit. A crash in between leaves nothing committed.

        Returns False if it couldn't be stored. A dropped connection gets retried. A refusal doesn't,
        because the store rejects content that doesn't match its hash, and sending it again can't help.
        """
        manifest, directory = event["manifest"], self._host_path(runner, event["path"])
        for name, entry in manifest["files"].items():
            path = directory / name
            if self.corrupt_next_upload:
                self.corrupt_next_upload = False
                raw = bytearray(path.read_bytes())
                raw[-1] ^= 0xFF
                path.write_bytes(raw)
                await self.send("fault.applied", {"fault": "CORRUPT_CHECKPOINT", "step": event.get("step"), "injected": True}, job_id=job_id, fencing_token=token)
            for attempt in range(UPLOAD_ATTEMPTS):
                try:
                    await asyncio.to_thread(self._upload, path, entry["sha256"])
                    break
                except (OSError, urllib.error.URLError) as exc:
                    refused = isinstance(exc, urllib.error.HTTPError)
                    if refused or attempt == UPLOAD_ATTEMPTS - 1:
                        log.warning("checkpoint at step %s not uploaded: %s", event.get("step"), exc)
                        await self.send("run.warning", {"rule": "CHECKPOINT_UPLOAD_FAILED", "detail": str(exc)[:200], "step": event.get("step")}, job_id=job_id, fencing_token=token)
                        return False
                    await asyncio.sleep(1 + attempt)
        _, signature = sign_document(manifest, self.config.worker_key)
        await self.send("checkpoint.uploaded", {"manifest": manifest, "signature": signature}, job_id=job_id, fencing_token=token)
        return True

    # -------------------------------------------------------------- faults

    async def inject_fault(self, kind: str | None) -> None:
        log.warning("INJECTED FAULT: %s", kind)
        runner = self.runner
        if kind == "KILL_CONTAINER" and runner:
            self.injected_exit = True
            await asyncio.to_thread(runner.kill)
        elif kind == "KILL_AGENT":
            os._exit(137)  # The runtime's lease then goes stale and it kills itself.
        elif kind == "HEARTBEAT_LOSS":
            # A partition that heals: long enough to be declared dead, and then its late messages get fenced.
            self.silent_until = time.monotonic() + PARTITION_SECONDS
        elif kind == "DISCONNECT" and self.ws is not None:
            await self.ws.close()  # The run keeps going, and the agent reconnects, as after a real network drop.
        elif kind == "OOM" and runner:
            self.injected_exit = True
            (runner.workdir / "inject_oom").touch()
        elif kind == "CRASH_MID_CHECKPOINT" and runner:
            self.injected_exit = True
            (runner.workdir / "inject_crash_checkpoint").touch()
        elif kind == "CORRUPT_CHECKPOINT":
            self.injected_exit = True
            self.corrupt_next_upload = True
        elif kind == "THERMAL_WARNING" and runner:
            await self.send("run.warning", {"rule": "THERMAL", "detail": "Injected thermal warning", "injected": True}, job_id=self.job_id, fencing_token=self.fencing_token)
            runner.request_checkpoint()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--provider-id", required=True)
    parser.add_argument("--wallet-key", default=os.environ.get("PROVIDER_WALLET_KEY"), help="Or set PROVIDER_WALLET_KEY.")
    parser.add_argument("--worker-key", default=os.environ.get("PROVIDER_WORKER_KEY"), help="Generated when omitted.")
    parser.add_argument("--workdir", default="agent_runs")
    parser.add_argument("--mode", choices=("docker", "process"), default="docker")
    parser.add_argument("--image", default="ekido-runtime:spike")
    parser.add_argument("--repo-root", help="Mounts code from here, read-only. Development only.")
    parser.add_argument("--device", help="Force cpu or cuda.")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--fault", help="Injected-fault provider only, for example POISON or LR_TAMPER.")
    parser.add_argument("--fault-at-step", type=int, default=0)
    args = parser.parse_args()
    if not args.wallet_key:
        parser.error("--wallet-key or PROVIDER_WALLET_KEY is required")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    mounts = ((args.repo_root, "/app"),) if args.repo_root else ()
    config = AgentConfig(
        coordinator=args.coordinator,
        provider_id=args.provider_id,
        wallet_key=args.wallet_key,
        worker_key=args.worker_key or "0x" + secrets.token_hex(32),
        workdir=Path(args.workdir).resolve(),
        mode=args.mode,
        docker=DockerOptions(image=args.image, gpu=not args.no_gpu, code_mounts=mounts),
        device=args.device,
        fault={"kind": args.fault, "at_step": args.fault_at_step} if args.fault else None,
    )
    asyncio.run(Agent(config).run_forever())


if __name__ == "__main__":
    main()

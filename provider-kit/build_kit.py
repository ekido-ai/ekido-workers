"""Builds the provider kit the website serves: apps/web-combined/public/gpuchain-provider-kit.zip.

Only the agent, runtime and core sources go in, without tests, caches or anything key-like, plus the
scripts in this folder. Writes the SHA-256 next to the zip so the download page can show it.
"""
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "apps" / "web-combined" / "public" / "gpuchain-provider-kit.zip"
TREES = {
    "core": ROOT / "packages" / "gpuchain_core" / "gpuchain_core",
    "agent": ROOT / "workers" / "agent" / "gpuchain_agent",
    "runtime": ROOT / "workers" / "runtime" / "gpuchain_runtime",
}
SCRIPTS = ["README.md", "requirements.txt", "setup.sh", "setup.ps1", "join.sh", "join.ps1"]
SKIP_PARTS = {"__pycache__", "tests", ".pytest_cache"}
SKIP_SUFFIX = {".pyc", ".key", ".env", ".pem"}
STAMP = (2026, 11, 9, 0, 0, 0)


def add(z: zipfile.ZipFile, name: str, data: bytes, mode: int) -> None:
    info = zipfile.ZipInfo(f"gpuchain-provider/{name}", date_time=STAMP)
    info.external_attr = mode << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    z.writestr(info, data)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in SCRIPTS:
            data = (HERE / name).read_bytes()
            # Shell scripts must keep Unix line endings even when the kit is built on Windows.
            if name.endswith(".sh"):
                data = data.replace(b"\r\n", b"\n")
            add(z, name, data, 0o755 if name.endswith(".sh") else 0o644)
        for top, src in TREES.items():
            for path in sorted(src.rglob("*")):
                rel = path.relative_to(src)
                if path.is_dir() or SKIP_PARTS & set(rel.parts) or path.suffix in SKIP_SUFFIX:
                    continue
                add(z, f"{top}/{src.name}/{rel.as_posix()}", path.read_bytes(), 0o644)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    (OUT.parent / "gpuchain-provider-kit.sha256").write_text(f"{digest}  gpuchain-provider-kit.zip\n", newline="\n")
    with zipfile.ZipFile(OUT) as z:
        print(f"{len(z.namelist())} files, {OUT.stat().st_size / 1024:.0f} KB, sha256 {digest}")


if __name__ == "__main__":
    main()

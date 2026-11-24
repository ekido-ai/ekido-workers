"""Export: a trained LoRA checkpoint becomes a servable model.

Two artefacts, per the C10 ticket:

- a **merged safetensors** model — the LoRA folded back into the base, loadable by any
  transformers runtime;
- a **quantised GGUF** — the merged model converted with llama.cpp's
  ``convert_hf_to_gguf.py`` and quantised with ``llama-quantize``, which is what the
  llama.cpp inference unit serves on CPU.

The checkpoint is read through ``ekido_core.gpck`` with the spec's hash as the expected
lineage, so an export can never silently merge a checkpoint from a different job.
llama.cpp is an external tool: point at a checkout with ``--llama-cpp`` or the
``EKIDO_LLAMA_CPP`` environment variable; without it the safetensors export still works
and the GGUF step reports exactly what is missing.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from ekido_core.canonical import document_hash
from ekido_core.gpck import load_checkpoint

from .llm import load_model, load_tokenizer
from .export_artifacts import export_lock, record_gguf
from .train import apply_lora, _trainable


def merge_checkpoint(spec: dict, checkpoint_dir: Path, out_dir: Path, *, base_model=None, tokenizer=None, device: str = "cpu") -> dict:
    """Folds the checkpoint's LoRA weights into the base model and saves safetensors."""
    base = base_model if base_model is not None else load_model(spec["model"]["id"], spec["model"]["revision"], device=device)
    model = apply_lora(base, spec)
    named = _trainable(model)
    shapes = {name: tuple(param.shape) for name, param in named.items()}
    manifest, trainable, _optimizer = load_checkpoint(
        Path(checkpoint_dir), shapes, dtype=torch.float32, expected_immutable_hash=document_hash(spec)
    )
    with torch.no_grad():
        for name, param in named.items():
            param.copy_(trainable[name].to(param.device, param.dtype))
    merged = model.merge_and_unload()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    if tokenizer is None:
        tokenizer = load_tokenizer(spec["model"]["id"], spec["model"]["revision"])
    tokenizer.save_pretrained(out_dir)
    info = {
        "merged_from": str(checkpoint_dir),
        "logical_step": manifest.get("logical_step"),
        "spec_sha256": document_hash(spec),
        "checkpoint_manifest_sha256": document_hash(manifest),
        "recipe_id": spec["recipe_id"],
        "base_model": spec["model"],
        "format": "safetensors",
    }
    (out_dir / "export.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def find_llama_cpp(explicit: str | Path | None = None) -> Path:
    """Locates a llama.cpp checkout with the converter built/installed, or raises."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("EKIDO_LLAMA_CPP"):
        candidates.append(Path(os.environ["EKIDO_LLAMA_CPP"]))
    candidates.append(Path(__file__).resolve().parents[1] / "tools" / "llama.cpp")
    candidates.append(Path.home() / "llama.cpp")
    for candidate in candidates:
        if (candidate / "convert_hf_to_gguf.py").is_file():
            return candidate
    raise FileNotFoundError(
        "llama.cpp not found (need convert_hf_to_gguf.py); pass --llama-cpp or set EKIDO_LLAMA_CPP"
    )


def _llama_binary(llama_cpp: Path, name: str) -> Path:
    names = (name, f"{name}.exe") if os.name == "nt" else (name,)
    for folder in (llama_cpp / "build" / "bin", llama_cpp / "build" / "bin" / "Release", llama_cpp):
        for filename in names:
            candidate = folder / filename
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"{name} not built in {llama_cpp} (build llama.cpp first)")


# Converting or quantising a small model takes minutes; a step still running after an hour has hung.
EXPORT_TIMEOUT_S = 3600


def _run(command: list[str]) -> None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=EXPORT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{Path(command[0]).name} did not finish within {EXPORT_TIMEOUT_S} s") from None
    if result.returncode != 0:
        tail = (result.stderr or result.stdout)[-500:]
        raise RuntimeError(f"{Path(command[0]).name} failed ({result.returncode}): {tail}")


def to_gguf(model_dir: Path, out_file: Path, *, llama_cpp: Path, outtype: str = "f16") -> Path:
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, str(Path(llama_cpp) / "convert_hf_to_gguf.py"), str(model_dir), "--outfile", str(out_file), "--outtype", outtype])
    return out_file


def quantize_gguf(gguf_in: Path, out_file: Path, quant: str, *, llama_cpp: Path) -> Path:
    binary = _llama_binary(Path(llama_cpp), "llama-quantize")
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    _run([str(binary), str(gguf_in), str(out_file), quant])
    return out_file


def llama_cli_generate(llama_cpp: Path, gguf: Path, prompt: str, *, n_predict: int = 16,
                       context_size: int = 512) -> str:
    """Generate with the raw recipe prompt, not an automatic chat template.

    Callers must keep the prompt and requested output within ``context_size``.
    Prefer llama-completion; older builds can use llama-cli only if it accepts
    the explicit no-conversation flag. Never fall back to implicit chat mode.
    """
    if context_size < 1 or n_predict < 1:
        raise ValueError("context_size and n_predict must be positive")
    try:
        binary = _llama_binary(Path(llama_cpp), "llama-completion")
    except FileNotFoundError:
        binary = _llama_binary(Path(llama_cpp), "llama-cli")
    else:
        command = [str(binary), "-m", str(gguf), "-p", prompt, "-n", str(n_predict),
                   "--temp", "0", "--no-conversation", "--no-display-prompt",
                   "--no-warmup", "--simple-io", "-c", str(context_size)]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, timeout=600, stdin=subprocess.DEVNULL)
        if result.returncode != 0:
            raise RuntimeError(f"llama-completion failed: {(result.stderr or '')[-400:]}")
        answer = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', result.stdout or '').strip()
        if not answer:
            raise RuntimeError("llama-completion produced no captured answer")
        return answer

    command = [str(binary), "-m", str(gguf), "-p", prompt, "-n", str(n_predict),
               "--temp", "0", "-no-cnv", "-c", str(context_size)]
    # Older llama-cli builds can write directly to a TTY. Never retry without
    # -no-cnv: that would silently change the model's prompt format.
    script = shutil.which('script') if sys.platform.startswith('linux') else None
    capture_command = [script, '-qec', shlex.join(command), '/dev/null'] if script else command
    result = subprocess.run(capture_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=600, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(f"llama-cli failed: {(result.stdout or '')[-400:]}")
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', result.stdout or '')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    prompt = prompt.replace('\r\n', '\n').replace('\r', '\n')
    # The prompt must be echoed so startup banners and diagnostics cannot be
    # mistaken for a successful generation.
    if prompt not in text:
        raise RuntimeError("llama-cli produced no captured answer after the prompt")
    text = text.split(prompt, 1)[1]
    lines = [line for line in text.splitlines() if "t/s" not in line and line.strip() not in {">", "Exiting..."}]
    answer = "\n".join(lines).strip()
    if not answer:
        raise RuntimeError("llama-cli exited successfully but produced no captured answer")
    return answer


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Export a trained llm-lora/v1 checkpoint.")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="GPCK/1 checkpoint directory")
    parser.add_argument("--out", type=Path, required=True, help="directory for the merged safetensors model")
    parser.add_argument("--gguf", action="store_true", help="also convert to GGUF (needs llama.cpp)")
    parser.add_argument("--quantize", nargs="*", default=[], help="GGUF quantisations to produce, e.g. Q4_K_M Q8_0")
    parser.add_argument("--llama-cpp", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="device used to merge the checkpoint")
    args = parser.parse_args(argv)

    if any(not re.fullmatch(r'[A-Za-z0-9_]{1,32}', quant) for quant in args.quantize):
        parser.error('quantization names must contain only letters, digits and underscores')

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    with export_lock(args.out):
        info = merge_checkpoint(spec, args.checkpoint, args.out, device=args.device)
        print(f"merged safetensors -> {args.out} (from step {info['logical_step']})")
        if args.gguf or args.quantize:
            llama_cpp = find_llama_cpp(args.llama_cpp)
            gguf = to_gguf(args.out, args.out / "model-f16.gguf", llama_cpp=llama_cpp)
            record_gguf(args.out, gguf, "F16")
            print(f"gguf -> {gguf}")
            for quant in args.quantize:
                out = args.out / f"model-{quant}.gguf"
                quantize_gguf(gguf, out, quant, llama_cpp=llama_cpp)
                record_gguf(args.out, out, quant)
                print(f"gguf {quant} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

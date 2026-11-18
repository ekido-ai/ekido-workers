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


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
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


def llama_cli_generate(llama_cpp: Path, gguf: Path, prompt: str, *, n_predict: int = 16) -> str:
    """One CPU generation through llama-cli: the exported model, serving.

    llama.cpp renames its display/conversation flags between releases, so this tries
    the quiet flags first and falls back to the minimal set every build accepts.
    """
    binary = _llama_binary(Path(llama_cpp), "llama-cli")
    base = [str(binary), "-m", str(gguf), "-p", prompt, "-n", str(n_predict), "--temp", "0"]
    attempts = [
        base + ["--single-turn", "--no-show-timings"],
        base + ["-no-cnv"],
        base,
    ]
    result = None
    for command in attempts:
        # Recent llama-cli releases can write generated text directly to a TTY.
        # util-linux script creates one on Linux; shlex.join quotes every argument
        # because script executes its command string through a shell.
        script = shutil.which('script') if sys.platform.startswith('linux') else None
        capture_command = [script, '-qec', shlex.join(command), '/dev/null'] if script else command
        result = subprocess.run(capture_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600, stdin=subprocess.DEVNULL)
        if result.returncode == 0:
            break
        if "invalid argument" not in (result.stdout or ""):
            break
    if result is None or result.returncode != 0:
        raise RuntimeError(f"llama-cli failed: {(result.stdout if result else '')[-400:]}")
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
    args = parser.parse_args(argv)

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    info = merge_checkpoint(spec, args.checkpoint, args.out)
    print(f"merged safetensors -> {args.out} (from step {info['logical_step']})")
    if args.gguf or args.quantize:
        llama_cpp = find_llama_cpp(args.llama_cpp)
        gguf = to_gguf(args.out, args.out / "model-f16.gguf", llama_cpp=llama_cpp)
        print(f"gguf -> {gguf}")
        for quant in args.quantize:
            out = args.out / f"model-{quant}.gguf"
            quantize_gguf(gguf, out, quant, llama_cpp=llama_cpp)
            print(f"gguf {quant} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

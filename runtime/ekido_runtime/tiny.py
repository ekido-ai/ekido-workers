"""A tiny CPU model config, so CI can run the whole runtime without downloading a model.

The tiny model is a real Qwen2 — same architecture, same tokenizer interface, same LoRA
target modules — just small enough (2 layers, hidden 64) to train in seconds on a CPU.
Tests and smoke runs inject it through the same parameters a provider run uses for the
real Qwen2.5 bases, so the code path under test is the production one.
"""

from __future__ import annotations

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

TINY_SEED = 1234
SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>"]


def build_tiny_tokenizer(texts: list[str]) -> PreTrainedTokenizerFast:
    """A small byte-level BPE tokenizer trained on the given texts.

    Byte-level (the GPT-2 family shape, like the real Qwen2 tokenizer) and deterministic
    for a given corpus. Note the tiny model still cannot be converted to GGUF locally:
    llama.cpp's converter recognises pre-tokenizers by a checksum table of known
    models, and a freshly trained tokenizer is in no table. The Qwen2.5 bases are in
    the table, so GGUF is proven on the real model (the T4 run), not on this fixture.
    """
    backend = Tokenizer(models.BPE(unk_token=None))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=512,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    backend.train_from_iterator(list(texts), trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
    )
    return tokenizer


def build_tiny_model(tokenizer) -> Qwen2ForCausalLM:
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
    )
    torch.manual_seed(TINY_SEED)
    return Qwen2ForCausalLM(config)

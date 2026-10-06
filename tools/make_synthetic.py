#!/usr/bin/env python3
"""Write a large synthetic llama-architecture GGUF without holding it in memory.

Weights are random but well-formed quantized blocks with sane scales, and the
file carries a working byte-level BPE tokenizer, so Kestrel and llama.cpp can
run it end to end. Use it to measure memory tiering on real model *shapes*
(TinyLlama 1.1B, Llama-2 7B, …) when real weights are unavailable. Output
text is meaningless; speed, memory and I/O behaviour are what is measured.

    python3 tools/make_synthetic.py out.gguf --shape 7b --type Q4_K
    python3 tools/make_synthetic.py out.gguf --layers 22 --embd 2048 --ff 5632 --heads 32 --kv-heads 4 --type Q8_0

Requires numpy and gguf-py (only for constants).
"""
import argparse
import struct

import numpy as np

SHAPES = {
    "1b": dict(layers=22, embd=2048, ff=5632, heads=32, kv_heads=4, vocab=32000),
    "3b": dict(layers=26, embd=3200, ff=8640, heads=32, kv_heads=32, vocab=32000),
    "7b": dict(layers=32, embd=4096, ff=11008, heads=32, kv_heads=32, vocab=32000),
    "8b": dict(layers=32, embd=4096, ff=14336, heads=32, kv_heads=8, vocab=128256),
    "13b": dict(layers=40, embd=5120, ff=13824, heads=40, kv_heads=40, vocab=32000),
    # Mixtral-style MoE (llama architecture, ff = expert width)
    "moe-6b": dict(layers=24, embd=2048, ff=1536, heads=16, kv_heads=4, vocab=32000, experts=32, experts_used=4),
}

# ggml type id, block elements, block bytes
TYPES = {"F32": (0, 1, 4), "F16": (1, 1, 2), "Q4_0": (2, 32, 18), "Q8_0": (8, 32, 34), "Q4_K": (12, 256, 144), "Q6_K": (14, 256, 210)}
ALIGN = 32


def f16(x):
    return np.float16(x).tobytes()


def random_blocks(t, n_blocks, scale, rng):
    """Random but well-formed blocks whose dequantized std is ~`scale`."""
    _, be, bb = TYPES[t]
    raw = rng.integers(0, 256, size=(n_blocks, bb), dtype=np.uint8)
    if t == "Q8_0":
        raw[:, 0:2] = np.frombuffer(f16(scale / 74.0) * n_blocks, dtype=np.uint8).reshape(n_blocks, 2)
    elif t == "Q4_0":
        raw[:, 0:2] = np.frombuffer(f16(scale / 4.6) * n_blocks, dtype=np.uint8).reshape(n_blocks, 2)
    elif t == "Q4_K":
        # Zero-mean weights: every 6-bit scale and min = 32 (packed layout of
        # get_scale_min_k4), w = d*32*q - dmin*32 with dmin = 7.5 d, q in 0..15.
        # Non-zero-mean blocks bias every row and freeze MoE routing.
        d = scale / (32 * 4.61)
        raw[:, 0:2] = np.frombuffer(f16(d) * n_blocks, dtype=np.uint8).reshape(n_blocks, 2)
        raw[:, 2:4] = np.frombuffer(f16(7.5 * d) * n_blocks, dtype=np.uint8).reshape(n_blocks, 2)
        raw[:, 4:16] = np.array([160] * 8 + [0] * 4, dtype=np.uint8)
    elif t == "Q6_K":
        raw[:, 192:208] = (raw[:, 192:208] % 32 + 16).astype(np.uint8)  # int8 scales 16..47
        raw[:, 208:210] = np.frombuffer(f16(scale / 600.0) * n_blocks, dtype=np.uint8).reshape(n_blocks, 2)
    return raw.tobytes()


def tensor_bytes(t, n):
    _, be, bb = TYPES[t]
    return n // be * bb


def put_str(b, s):
    s = s.encode()
    b += struct.pack("<Q", len(s)) + s


def put_kv(b, key, vt, val):
    put_str(b, key)
    b += struct.pack("<I", vt)
    if vt == 4:
        b += struct.pack("<I", val)
    elif vt == 6:
        b += struct.pack("<f", val)
    elif vt == 7:
        b += struct.pack("<B", int(val))
    elif vt == 8:
        put_str(b, val)
    elif vt == 9:
        et, items = val
        b += struct.pack("<IQ", et, len(items))
        for it in items:
            if et == 8:
                put_str(b, it)
            elif et == 5:
                b += struct.pack("<i", it)
            elif et == 6:
                b += struct.pack("<f", it)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--shape", choices=SHAPES)
    for k in ["layers", "embd", "ff", "heads", "kv_heads", "vocab"]:
        ap.add_argument("--" + k.replace("_", "-"), type=int)
    ap.add_argument("--type", default="Q4_K", choices=["Q4_0", "Q8_0", "Q4_K", "Q6_K", "F16"])
    ap.add_argument("--experts", type=int, help="routed experts per layer (MoE)")
    ap.add_argument("--experts-used", type=int, help="experts selected per token")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = dict(SHAPES[a.shape]) if a.shape else {}
    for k in ["layers", "embd", "ff", "heads", "kv_heads", "vocab"]:
        v = getattr(a, k)
        if v:
            cfg[k] = v
    L, E, F, H, KV, V = (cfg[k] for k in ["layers", "embd", "ff", "heads", "kv_heads", "vocab"])
    NE = a.experts or cfg.get("experts", 0)
    NU = a.experts_used or cfg.get("experts_used", 2 if NE else 0)
    hd = E // H
    rng = np.random.default_rng(a.seed)
    wt = a.type
    head_t = "Q6_K" if wt in ("Q4_K",) else wt

    # Tokenizer: 256 byte tokens (GPT-2 byte mapping) + filler tokens + specials.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_tokens = [chr(c) for _, c in sorted(zip(bs, cs))]
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    fillers = [f"<unused_{i}>" for i in range(V - 256 - len(specials))]
    tokens = byte_tokens + fillers + specials
    types = [1] * 256 + [4] * len(fillers) + [3] * len(specials)

    tensors = [("token_embd.weight", [E, V], wt, 1.0)]
    for l in range(L):
        p = f"blk.{l}."
        tensors += [
            (p + "attn_norm.weight", [E], "F32", None),
            (p + "attn_q.weight", [E, H * hd], wt, 1.0 / np.sqrt(E)),
            (p + "attn_k.weight", [E, KV * hd], wt, 1.0 / np.sqrt(E)),
            (p + "attn_v.weight", [E, KV * hd], wt, 1.0 / np.sqrt(E)),
            (p + "attn_output.weight", [H * hd, E], wt, 1.0 / np.sqrt(E)),
            (p + "ffn_norm.weight", [E], "F32", None),
        ]
        if NE:
            tensors += [
                (p + "ffn_gate_inp.weight", [E, NE], "F32", 4.0 / np.sqrt(E)),
                (p + "ffn_gate_exps.weight", [E, F, NE], wt, 1.0 / np.sqrt(E)),
                (p + "ffn_up_exps.weight", [E, F, NE], wt, 1.0 / np.sqrt(E)),
                (p + "ffn_down_exps.weight", [F, E, NE], wt, 1.0 / np.sqrt(F)),
            ]
        else:
            tensors += [
                (p + "ffn_gate.weight", [E, F], wt, 1.0 / np.sqrt(E)),
                (p + "ffn_up.weight", [E, F], wt, 1.0 / np.sqrt(E)),
                (p + "ffn_down.weight", [F, E], wt, 1.0 / np.sqrt(F)),
            ]
    tensors += [("output_norm.weight", [E], "F32", None), ("output.weight", [E, V], head_t, 1.0 / np.sqrt(E))]

    n_kv = 15 + (2 if NE else 0)
    hdr = bytearray(b"GGUF") + struct.pack("<IQQ", 3, len(tensors), n_kv)
    put_kv(hdr, "general.architecture", 8, "llama")
    put_kv(hdr, "general.name", 8, f"synthetic-{a.shape or 'custom'}-{wt}")
    put_kv(hdr, "llama.block_count", 4, L)
    put_kv(hdr, "llama.context_length", 4, 4096)
    put_kv(hdr, "llama.embedding_length", 4, E)
    put_kv(hdr, "llama.feed_forward_length", 4, F)
    put_kv(hdr, "llama.attention.head_count", 4, H)
    put_kv(hdr, "llama.attention.head_count_kv", 4, KV)
    put_kv(hdr, "llama.rope.freq_base", 6, 10000.0)
    put_kv(hdr, "llama.attention.layer_norm_rms_epsilon", 6, 1e-5)
    put_kv(hdr, "tokenizer.ggml.model", 8, "gpt2")
    put_kv(hdr, "tokenizer.ggml.pre", 8, "llama-bpe")
    put_kv(hdr, "tokenizer.ggml.tokens", 9, (8, tokens))
    put_kv(hdr, "tokenizer.ggml.token_type", 9, (5, types))
    put_kv(hdr, "tokenizer.ggml.merges", 9, (8, []))
    if NE:
        put_kv(hdr, "llama.expert_count", 4, NE)
        put_kv(hdr, "llama.expert_used_count", 4, NU)
    # (15 KV entries declared; pad count with eos/bos below)
    assert hdr.count(b"tokenizer.ggml.merges") == 1
    offsets, off = [], 0
    for name, dims, t, _ in tensors:
        n = int(np.prod(dims))
        offsets.append(off)
        off += tensor_bytes(t, n)
        off = (off + ALIGN - 1) // ALIGN * ALIGN
    # Fix declared KV count to what we wrote (15 above).
    for (name, dims, t, _), o in zip(tensors, offsets):
        put_str(hdr, name)
        hdr += struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        hdr += struct.pack("<IQ", TYPES[t][0], o)
    pad = (-len(hdr)) % ALIGN
    hdr += b"\0" * pad
    total = len(hdr) + off
    moe = f", {NE} experts (top {NU})" if NE else ""
    print(f"writing {a.out}: {L} layers, embd {E}, ff {F}, vocab {V}{moe}, {wt} → {total / 1e9:.2f} GB")
    with open(a.out, "wb") as f:
        f.write(hdr)
        written = 0
        for (name, dims, t, scale), o in zip(tensors, offsets):
            assert written == o, (name, written, o)
            n = int(np.prod(dims))
            if t == "F32":
                if scale is None:
                    data = (1.0 + 0.05 * rng.standard_normal(n)).astype(np.float32).tobytes()
                else:
                    data = (scale * rng.standard_normal(n)).astype(np.float32).tobytes()
            else:
                be = TYPES[t][1]
                chunks = []
                rows, cols = int(np.prod(dims[1:])), dims[0]
                step = max(1, (64 << 20) // max(1, tensor_bytes(t, cols)))
                for r0 in range(0, rows, step):
                    nb = min(step, rows - r0) * cols // be
                    f.write(random_blocks(t, nb, scale, rng))
                    written += tensor_bytes(t, nb * be)
                pad = (-written) % ALIGN
                f.write(b"\0" * pad)
                written += pad
                continue
            f.write(data)
            written += len(data)
            pad = (-written) % ALIGN
            f.write(b"\0" * pad)
            written += pad


if __name__ == "__main__":
    main()

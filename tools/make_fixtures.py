#!/usr/bin/env python3
"""Generate tiny GGUF models for Kestrel's correctness tests.

Models have random weights but real structure (GQA, RoPE, SwiGLU, biases,
Q/K norms, tied/untied heads) and real tokenizers (byte-level BPE trained on a
small corpus, and a SentencePiece-style vocabulary with byte fallback), so
the native executor can be compared against llama.cpp token for token.

    pip install numpy gguf     # or PYTHONPATH=<llama.cpp>/gguf-py
    python3 tools/make_fixtures.py OUT_DIR [--quantize /path/to/llama-quantize]

Writes OUT_DIR/{llama-bpe,qwen2,qwen3,llama-spm}-f32.gguf and, with
--quantize, llama-bpe-<TYPE>.gguf for the common quantization types.
"""
import argparse
import os
import subprocess
from collections import Counter

import numpy as np
import gguf

CORPUS = """The quick brown fox jumps over the lazy dog. Memory tiering moves weights
between VRAM, RAM and NVMe. A model is a graph of layers; each layer has attention
and a feed-forward network. Kestrel plans, streams and prefetches tensors so that
large models run on small machines. Hello world! Numbers: 0 1 2 3 42 1234 3.14.
Ciao, come stai? Grüße aus München. 日本語のテキスト。 Emojis 🙂 are bytes too.
def main():\n    print("hello")\n    return 0\n"""

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
CHATML = ("{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
          "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def train_bpe(text, n_merges):
    """Toy byte-level BPE trainer over regex-free whitespace pre-tokens."""
    b2u = bytes_to_unicode()
    words = Counter()
    for w in text.replace("\n", " \n ").split(" "):
        if w:
            words[" " + w] += 1
    seqs = {w: [b2u[b] for b in w.encode("utf-8")] for w in words}
    merges = []
    vocab = [b2u[b] for b in range(256)]
    for _ in range(n_merges):
        pairs = Counter()
        for w, s in seqs.items():
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += words[w]
        if not pairs:
            break
        (a, b), _ = pairs.most_common(1)[0]
        merges.append(f"{a} {b}")
        vocab.append(a + b)
        for w, s in seqs.items():
            i, out = 0, []
            while i < len(s):
                if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            seqs[w] = out
    return vocab, merges


def spm_vocab():
    toks, scores, types = ["<unk>", "<s>", "</s>"], [0.0, 0.0, 0.0], [2, 3, 3]
    for b in range(256):
        toks.append(f"<0x{b:02X}>")
        scores.append(0.0)
        types.append(6)
    pieces = Counter()
    for w in CORPUS.replace("\n", " ").split(" "):
        w = "▁" + w
        for i in range(len(w)):
            for j in range(i + 1, min(len(w), i + 6) + 1):
                pieces[w[i:j]] += 1
    for p, c in pieces.most_common(400):
        if p not in toks:
            toks.append(p)
            scores.append(float(np.log(c)) - len(p) * 0.1)
            types.append(1)
    return toks, scores, types


def make_model(path, arch, tokenizer, *, n_embd=256, n_head=4, n_kv=2, n_ff=768, n_layer=3,
               qkv_bias=False, qk_norm=False, tied=False, seed=0, n_expert=0, n_expert_used=0,
               n_ff_exp=256, n_ff_shexp=0):
    rng = np.random.default_rng(seed)
    w = gguf.GGUFWriter(path, arch)
    w.add_name(os.path.basename(path).split(".")[0])
    w.add_block_count(n_layer)
    w.add_context_length(2048)
    w.add_embedding_length(n_embd)
    w.add_feed_forward_length(n_ff)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_kv)
    w.add_rope_freq_base(10000.0)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_file_type(0)
    if n_expert:
        w.add_expert_count(n_expert)
        w.add_expert_used_count(n_expert_used)
        w.add_expert_feed_forward_length(n_ff_exp)
        if n_ff_shexp:
            w.add_expert_shared_feed_forward_length(n_ff_shexp)

    if tokenizer == "bpe":
        vocab, merges = train_bpe(CORPUS, 220)
        toks = vocab + SPECIALS
        types = [1] * len(vocab) + [3] * len(SPECIALS)
        w.add_tokenizer_model("gpt2")
        w.add_tokenizer_pre("qwen2" if arch.startswith("qwen") else "llama-bpe")
        w.add_token_list(toks)
        w.add_token_types(types)
        w.add_token_merges(merges)
        w.add_bos_token_id(len(vocab))
        w.add_eos_token_id(len(vocab) + 2)
        w.add_add_bos_token(arch == "llama")
        w.add_chat_template(CHATML)
    else:
        toks, scores, types = spm_vocab()
        w.add_tokenizer_model("llama")
        w.add_tokenizer_pre("default")
        w.add_token_list(toks)
        w.add_token_scores(scores)
        w.add_token_types(types)
        w.add_bos_token_id(1)
        w.add_eos_token_id(2)
        w.add_add_bos_token(True)
    n_vocab = len(toks)
    hd = n_embd // n_head

    def lin(rows, cols, scale=1.0):
        return (rng.standard_normal((rows, cols)) * scale / np.sqrt(cols)).astype(np.float32)

    def norm(n):
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    w.add_tensor("token_embd.weight", rng.standard_normal((n_vocab, n_embd)).astype(np.float32))
    for l in range(n_layer):
        p = f"blk.{l}."
        w.add_tensor(p + "attn_norm.weight", norm(n_embd))
        w.add_tensor(p + "attn_q.weight", lin(n_head * hd, n_embd, 2.0))
        w.add_tensor(p + "attn_k.weight", lin(n_kv * hd, n_embd, 2.0))
        w.add_tensor(p + "attn_v.weight", lin(n_kv * hd, n_embd))
        if qkv_bias:
            w.add_tensor(p + "attn_q.bias", (0.1 * rng.standard_normal(n_head * hd)).astype(np.float32))
            w.add_tensor(p + "attn_k.bias", (0.1 * rng.standard_normal(n_kv * hd)).astype(np.float32))
            w.add_tensor(p + "attn_v.bias", (0.1 * rng.standard_normal(n_kv * hd)).astype(np.float32))
        if qk_norm:
            w.add_tensor(p + "attn_q_norm.weight", norm(hd))
            w.add_tensor(p + "attn_k_norm.weight", norm(hd))
        w.add_tensor(p + "attn_output.weight", lin(n_embd, n_head * hd))
        w.add_tensor(p + "ffn_norm.weight", norm(n_embd))
        if n_expert:
            w.add_tensor(p + "ffn_gate_inp.weight", lin(n_expert, n_embd, 4.0))
            w.add_tensor(p + "ffn_gate_exps.weight", np.stack([lin(n_ff_exp, n_embd) for _ in range(n_expert)]))
            w.add_tensor(p + "ffn_up_exps.weight", np.stack([lin(n_ff_exp, n_embd) for _ in range(n_expert)]))
            w.add_tensor(p + "ffn_down_exps.weight", np.stack([lin(n_embd, n_ff_exp) for _ in range(n_expert)]))
            if n_ff_shexp:
                w.add_tensor(p + "ffn_gate_inp_shexp.weight", (rng.standard_normal(n_embd) / np.sqrt(n_embd)).astype(np.float32))
                w.add_tensor(p + "ffn_gate_shexp.weight", lin(n_ff_shexp, n_embd))
                w.add_tensor(p + "ffn_up_shexp.weight", lin(n_ff_shexp, n_embd))
                w.add_tensor(p + "ffn_down_shexp.weight", lin(n_embd, n_ff_shexp))
        else:
            w.add_tensor(p + "ffn_gate.weight", lin(n_ff, n_embd))
            w.add_tensor(p + "ffn_up.weight", lin(n_ff, n_embd))
            w.add_tensor(p + "ffn_down.weight", lin(n_embd, n_ff))
    w.add_tensor("output_norm.weight", norm(n_embd))
    if not tied:
        w.add_tensor("output.weight", lin(n_vocab, n_embd, 4.0))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def dequantize_to_f32(src, dst):
    """Rewrite `src` with every tensor dequantized to F32 by gguf-py's
    reference decoders. Running llama.cpp on the result isolates Kestrel's
    dequantization from llama.cpp's activation quantization."""
    r = gguf.GGUFReader(src)
    arch = r.fields["general.architecture"].contents()
    w = gguf.GGUFWriter(dst, arch)
    for name, f in r.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        if name == "general.file_type":
            w.add_file_type(0)
            continue
        vt = f.types[0]
        if vt == gguf.GGUFValueType.ARRAY:
            w.add_array(name, f.contents())
        else:
            w.add_key_value(name, f.contents(), vt)
    for t in r.tensors:
        data = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
        w.add_tensor(t.name, data.reshape([int(d) for d in reversed(t.shape)]))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


QUANTS = ["Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q4_K_M", "Q5_K_M", "Q6_K", "F16", "BF16"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--quantize", help="path to llama-quantize")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    make_model(f"{a.out}/llama-bpe-f32.gguf", "llama", "bpe", seed=1)
    make_model(f"{a.out}/qwen2-f32.gguf", "qwen2", "bpe", qkv_bias=True, tied=True, seed=2)
    make_model(f"{a.out}/qwen3-f32.gguf", "qwen3", "bpe", qk_norm=True, seed=3)
    make_model(f"{a.out}/llama-spm-f32.gguf", "llama", "spm", seed=4)
    make_model(f"{a.out}/qwen3moe-f32.gguf", "qwen3moe", "bpe", qk_norm=True, seed=5, n_expert=8, n_expert_used=2)
    make_model(f"{a.out}/qwen2moe-f32.gguf", "qwen2moe", "bpe", qkv_bias=True, seed=6, n_expert=8, n_expert_used=2, n_ff_shexp=512)
    make_model(f"{a.out}/llama-moe-f32.gguf", "llama", "bpe", seed=7, n_expert=4, n_expert_used=2, n_ff_exp=768)
    if a.quantize:
        for q in QUANTS:
            dst = f"{a.out}/llama-bpe-{q}.gguf"
            subprocess.run([a.quantize, f"{a.out}/llama-bpe-f32.gguf", dst, q], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            dequantize_to_f32(dst, f"{a.out}/llama-bpe-{q}-deq.gguf")
        dst = f"{a.out}/qwen3moe-Q4_K_M.gguf"
        subprocess.run([a.quantize, f"{a.out}/qwen3moe-f32.gguf", dst, "Q4_K_M"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dequantize_to_f32(dst, f"{a.out}/qwen3moe-Q4_K_M-deq.gguf")
    print("fixtures written to", a.out)


if __name__ == "__main__":
    main()

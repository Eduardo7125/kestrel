# Contributing to Kestrel

Thanks for helping. Kestrel is a research-driven runtime: a change earns its
place through **correctness tests and reproducible measurements**, not
through how fast it looks in a micro-benchmark.

## Ground rules

1. **Placement never changes semantics.** Moving a tensor between VRAM, RAM
   and NVMe must produce bit-identical weights and identical greedy output.
   Anything lossy (lower quantization, smaller context, KV quantization) is
   opt-in and labeled.
2. **Every performance claim comes with a reproducible benchmark.** Follow
   [docs/benchmark-plan.md](docs/benchmark-plan.md): name the machine, the cache
   state and the exact command, interleave the arms, repeat them, and publish
   negative results as well.
3. **Budgets before bytes.** No allocation of weight, KV or ring memory
   without a `Ledger` reservation.
4. **Keep the scheduler backend-agnostic.** `kestrel-memory` and
   `kestrel-planner` must not depend on `kestrel-engine` or any backend.
5. **No platform assumptions in the core.** OS-specific code lives in
   `kestrel-hw` (discovery, file I/O) behind `cfg` with a portable fallback.

## Development

```bash
cargo test --workspace                 # unit + integration tests
cargo clippy --workspace --all-targets
cargo build --release
```

### Correctness against llama.cpp

The oracle tests compare tokenization, logits and greedy continuations with
llama.cpp on generated fixtures (all supported quantization types):

```bash
git clone https://github.com/ggml-org/llama.cpp && cmake -S llama.cpp -B llama.cpp/build && \
  cmake --build llama.cpp/build -j --target llama-quantize llama-bench llama-server llama-cli
pip install numpy gguf
python3 tools/make_fixtures.py /tmp/fixtures --quantize llama.cpp/build/bin/llama-quantize
LLAMA_CPP=$PWD/llama.cpp tools/oracle/build.sh /tmp/llama_oracle
KESTREL_FIXTURES=/tmp/fixtures KESTREL_ORACLE=/tmp/llama_oracle \
  cargo test -p kestrel-engine --test oracle -- --nocapture
```

### Adding an architecture

1. Register it in `crates/kestrel-model/src/adapters.rs` (inspection and
   planning work immediately for `blk.N.*` tensor names).
2. For native execution, extend `crates/kestrel-engine/src/transformer.rs`
   and `check_support`, add a fixture to `tools/make_fixtures.py`, and add it
   to the oracle test.

### Adding a kernel

Add the dequantization to `quant.rs` (it is the reference), the int8 path to
`qdot.rs`, and optionally a fused SIMD kernel to `avx2.rs`. The tests compare
each fast path against the reference.

## Pull requests

Describe what changed, why, and how it was validated. Include benchmark
output for anything performance-related, in the minimum report format of
`benchmarks/README.md`.

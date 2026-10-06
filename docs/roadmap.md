# Roadmap

Each phase ends with tests and a benchmark that justifies the next phase.
A phase that fails to show value is reported, not skipped silently.

| Phase | Deliverable | Exit criterion | Status |
|---|---|---|---|
| 1. Research | `docs/colibri-analysis.md` and the design docs | Mechanisms documented with source citations | **Done** |
| 2. Hardware profiler | `kestrel hardware` (discovery + `--bench`) | Runs on Linux/Windows; cached profile; no-GPU path | **Done (MVP)** |
| 3. Model inspector | `kestrel inspect` (GGUF parser, arch adapters, KV math, MoE detection) | Matches llama.cpp tensor counts and sizes | **Done (MVP)** |
| 4. Static scheduler | Planner + `kestrel plan` (strategies, budgets, cost model, diagnostics, llama.cpp args) | Unit-tested decisions; plan explains itself | **Done (MVP)** |
| 5. Streaming | IoEngine (pool, O_DIRECT), WeightStore (resident + ring) | Bit-identical to resident; bounded RSS | **Done (MVP)** |
| 6. Prefetch | Deterministic depth-d prefetch, interleaved placement | Benchmark shows the stall reduction | **Done (MVP)**, see benchmarks |
| 7. Adaptive scheduler | Rebalancer (promote/demote), memory guard, autotune | Survives budget shrink; no regression | **Partial** (promote/demote + guard done; autotune pending) |
| 8. MoE | Native MoE execution, ExpertCache (LFRU + leases), usage history, router-lookahead prefetch, routing-trace analysis | Expert hit rate and tok/s vs uniform placement | Pending (planning and inspection done) |
| 9. Advanced hardware | ggml FFI with Kestrel-owned buffers; CUDA/Vulkan/Metal/HIP features; multi-GPU | GPU tier under Kestrel residency control | Pending |
| 10. Ecosystem | HF model resolution and download (explicit opt-in), safetensors, more architectures (gemma, phi, deepseek2) | — | Pending |

## Near-term issues (post-MVP)

1. **Kernel speed:** q8 activation quantization and AVX2/NEON dot kernels for
   Q4_K and Q6_K. This removes the native/llama.cpp kernel gap so scheduling
   effects dominate measurements.
2. **io_uring backend** behind `--io uring` (Linux), measured against the pool.
3. **Windows unbuffered I/O** (`FILE_FLAG_NO_BUFFERING`).
4. **Expert-granular streaming** for MoE GGUFs. The strided expert slices need
   three reads per expert. Evaluate an optional re-packed sidecar (`--disk-cache`)
   that makes each expert contiguous, as Colibrì's container does.
5. **Prefill batching for streamed layers:** with S prompt tokens, one load
   serves S rows. Plan prefill and decode separately (research question 9).

# Roadmap

Each phase ends with tests and a benchmark that justifies the next phase.
A phase that fails to show value is reported, not skipped silently.

| Phase | Deliverable | Exit criterion | Status |
|---|---|---|---|
| 1. Research | `docs/colibri-analysis.md` and the design docs | Mechanisms documented with source citations | **Done** |
| 2. Hardware profiler | `kestrel hardware` (discovery + `--bench`) | Runs on Linux/Windows; cached profile; no-GPU path | **Done (MVP)**; tested on Linux only, the Windows code paths compile but are untested |
| 3. Model inspector | `kestrel inspect` (GGUF parser, arch adapters, KV math, MoE detection) | Matches llama.cpp tensor counts and sizes | **Done (MVP)** |
| 4. Static scheduler | Planner + `kestrel plan` (strategies, budgets, cost model, diagnostics, llama.cpp args) | Unit-tested decisions; plan explains itself | **Done (MVP)** |
| 5. Streaming | IoEngine (pool, O_DIRECT), WeightStore (resident + ring) | Bit-identical to resident; bounded RSS | **Done (MVP)** |
| 6. Prefetch | Deterministic depth-d prefetch, interleaved placement | Benchmark shows the stall reduction | **Done (MVP)**: +63% decode vs no prefetch at equal RAM (benchmarks/README.md) |
| 7. Adaptive scheduler | Rebalancer (promote/demote), memory guard, autotune | Survives budget shrink; no regression | **Partial**: promote/demote and the guard are implemented and unit-tested; autotune (`--tune`) and layer-vs-tensor granularity switching are pending; migration cost (RQ 7) is not yet measured |
| 8. MoE | Native MoE execution, ExpertCache (LFRU + leases), usage history, router-lookahead prefetch, routing-trace analysis | Expert hit rate and tok/s vs uniform placement | **Done (MVP)**: qwen2moe/qwen3moe/Mixtral-style llama match llama.cpp; LFRU expert cache +31% over LRU; lookahead +11% (benchmarks §4). Pending: real-model routing traces, NUMA, multi-SSD |
| 9. Advanced hardware | ggml FFI with Kestrel-owned buffers; CUDA/Vulkan/Metal/HIP features; multi-GPU | GPU tier under Kestrel residency control | Pending |
| 10. Ecosystem | HF model resolution and download (explicit opt-in), safetensors, more architectures (gemma, phi, deepseek2) | — | Pending |

## Near-term issues (post-MVP)

1. **Kernel speed:** int8 activations with AVX2 kernels for Q8_0, Q4_0, Q4_K
   and Q6_K are done, and so is an f32 AVX2 GEMM for prefill. Native decode is
   still ~2.5-3× slower than llama.cpp CPU, and prefill ~2.8× slower
   (43.6 vs 122 tok/s at 880 tokens on the 1.1B model). Remaining work:
   AVX2 kernels for Q5_K/Q4_1/Q5_x, NEON, AVX-512 VNNI, a batched-prefill
   GEMM, and SIMD attention.
2. **io_uring backend** behind `--io uring` (Linux), measured against the pool.
3. **Windows unbuffered I/O** (`FILE_FLAG_NO_BUFFERING`).
4. **Expert-granular streaming** for MoE GGUFs. The strided expert slices need
   three reads per expert. Evaluate an optional re-packed sidecar (`--disk-cache`)
   that makes each expert contiguous, as Colibrì's container does.
5. **Prefill batching for streamed layers:** with S prompt tokens, one load
   serves S rows. Plan prefill and decode separately (research question 9).

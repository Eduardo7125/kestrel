# MVP specification

## Objective

> Make a 32B dense model run efficiently on an 8 GB VRAM GPU without requiring
> the entire model to reside in RAM, and prove with benchmarks what the
> scheduler contributes.

Two halves serve that objective:

1. **GPU half (llama.cpp adapter):** for a 32B Q4_K_M (~18.5 GB) on 8 GB VRAM +
   32 GB RAM, Kestrel computes the VRAM/RAM split (`n_gpu_layers`, KV
   placement, context) from measured free memory and launches llama.cpp.
   Weights offloaded to the GPU do not also stay resident in RAM: mmap pages
   are dropped after upload.
2. **Memory-tiering half (native executor):** for models larger than
   RAM + VRAM, Kestrel streams dense layers from NVMe through a prefetching ring
   with a fixed RAM budget. It runs correctly at any budget down to "two layers
   plus the ring", and measures the throughput curve.

## In scope

| Area | Requirement | Acceptance test |
|---|---|---|
| Model format | GGUF v2/v3 parsing: metadata, tensor table, alignment | Parses llama.cpp-generated files. Unit tests on synthetic files. `inspect` on real models matches llama.cpp's reported tensor count and size. |
| Architectures (execution) | `llama`, `qwen2`, `qwen3` dense | Logits match llama.cpp on generated tiny models (max abs diff < 1e-2, identical greedy tokens) |
| Architectures (inspection and planning) | Any GGUF, with MoE detection (`*_exps` tensors, `expert_count`) | `inspect` reports experts, top-k, and expert slice size |
| Quant types (execution) | F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K | Kernel unit tests against reference dequantization. End-to-end against llama.cpp. |
| Hardware | CPU (x86-64, aarch64), RAM, NVIDIA GPU detection via nvidia-smi, storage capacity and type | `kestrel hardware` on Linux and Windows. No GPU is a valid result. |
| Benchmarks | RAM BW, disk seq/random read (direct and buffered), CPU GEMV throughput | Cached profile JSON |
| Planner | Strategies GPU-full, hybrid, RAM, RAM+NVMe, VRAM+RAM+NVMe; budgets with safety; KV sizing; cost model; failure diagnosis | Unit tests over synthetic hardware/model pairs. The plan for "32B Q4 on 8 GB/32 GB" puts no weights on NVMe. Infeasible plans list remedies. |
| Memory | Ledger with hard limits; resident + streamed groups; slot ring; O_DIRECT; prefetch depth; metrics | Invariant tests. Streaming produces bit-identical logits to all-resident. |
| Dynamic | Rebalancer: promote and demote at safe points; RSS guard | Test with a shrinking budget mid-run, without breaking correctness |
| CLI | `models`, `inspect`, `hardware`, `plan`, `run`, `serve`, `benchmark` | Integration tests on a fixture model |
| Server | `POST /v1/chat/completions` (with stream), `POST /v1/completions`, `GET /v1/models`, `GET /health`, `GET /metrics` | The OpenAI Python client works unchanged |
| Safety | No allocation past budget; plan refuses infeasible configurations with an explanation | Tests |
| Privacy | No network access at runtime | The only network access is the model download in `kestrel setup`, after the user picks a model |

## Out of scope for the MVP

MoE *execution* (planning only), multi-GPU, native GPU kernels, a model
downloader beyond local resolution, concurrent request batching, speculative
decoding, Windows `FILE_FLAG_NO_BUFFERING` (buffered fallback only),
macOS Metal planning.

## Definition of done

* `cargo test --workspace` passes on Linux.
* The fixture models (`tools/make_fixtures.py`) run through `kestrel run` and
  match the llama.cpp oracle.
* `benchmarks/results/` contains the memory-sweep results from at least one
  machine, including negative results.
* The docs reflect the shipped behavior.

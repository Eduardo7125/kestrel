# Kestrel architecture

Kestrel is a local LLM inference runtime. It treats **VRAM, RAM, and NVMe as one
coordinated memory hierarchy** and builds a hardware-specific execution plan
before running any token.

> The user thinks: "I downloaded a model. Run it."
> Kestrel thinks: "How can I execute this model most efficiently on this hardware?"

The core of the project is the **memory orchestration layer**: profiling,
planning, placement, streaming, prefetching, caching, and budget enforcement.
Inference kernels are a replaceable backend. Kestrel ships a native Rust CPU
executor that runs under Kestrel's own memory scheduler, and an adapter that
drives llama.cpp/ggml for GPU execution.

Related documents:
[colibri-analysis](colibri-analysis.md) ·
[memory-model](memory-model.md) ·
[scheduler-design](scheduler-design.md) ·
[backend-strategy](backend-strategy.md) ·
[benchmark-plan](benchmark-plan.md) ·
[technology-decisions](technology-decisions.md) ·
[mvp-spec](mvp-spec.md) ·
[roadmap](roadmap.md)

---

## 1. Layered view

```text
 ┌───────────────────────────────────────────────────────────────────────┐
 │  kestrel CLI  (models · inspect · hardware · plan · run · serve · bench)│
 │  OpenAI-compatible server  (/v1/chat/completions, /v1/completions, …)  │
 └───────────────┬───────────────────────────────────────────────────────┘
                 │ ExecutionPlan
 ┌───────────────▼───────────────────────────────────────────────────────┐
 │ PLANNER  (kestrel-planner)                                             │
 │   ModelDesc × HardwareProfile × Overrides → candidate strategies        │
 │   cost model (measured bandwidths) → ranked strategies → ExecutionPlan  │
 │   feasibility + human-readable failure diagnosis                        │
 └───────┬─────────────────────────────┬─────────────────────────────────┘
         │                             │
 ┌───────▼──────────┐       ┌──────────▼────────────────────────────────┐
 │ MODEL            │       │ MEMORY ORCHESTRATOR  (kestrel-memory)      │
 │ (kestrel-model)  │       │   Ledger: per-tier budgets & reservations   │
 │  GGUF reader     │       │   WeightStore: tier-aware tensor access     │
 │  arch adapters   │       │   I/O engine: positional, direct, async     │
 │  layer/expert    │       │   Prefetcher: graph-ordered & predictive    │
 │  tensor groups   │       │   Caches: layer ring · expert LFRU · KV     │
 │  KV requirements │       │   Rebalancer: promote/demote at safe points │
 └──────────────────┘       │   Metrics: hits, bytes, stalls, accuracy    │
                            └──────────┬────────────────────────────────┘
                                       │ leases on weight bytes
 ┌─────────────────────────────────────▼─────────────────────────────────┐
 │ BACKENDS  (kestrel-backends)                                            │
 │   native  — Rust CPU executor (kestrel-engine) under Kestrel's scheduler│
 │   llamacpp — plan → llama.cpp placement (n_gpu_layers, tensor overrides,│
 │              mmap, KV type, ctx); CUDA/Vulkan/Metal/HIP via ggml         │
 └───────────────────────────────────────────────────────────────────────┘
                 ▲
 ┌───────────────┴───────────────────────────────────────────────────────┐
 │ HARDWARE  (kestrel-hw)  discovery (CPU/RAM/GPU/storage) + measured       │
 │           micro-benchmarks (RAM BW, disk seq/random, CPU GEMV), cached    │
 └───────────────────────────────────────────────────────────────────────┘
```

The arrows show the rule that matters: **the planner and memory orchestrator
know nothing about any particular backend's kernels**, and backends know nothing
about policies. A backend receives an `ExecutionPlan` and, for the native
backend, a `WeightStore` that hands out *leases* on weight bytes.

## 2. Life of `kestrel run model.gguf`

1. **Resolve the model.** A path, or a name looked up in the model directories
   (`KESTREL_MODELS`, `~/.cache/kestrel/models`, the current directory).
2. **Inspect the model.** The GGUF header, metadata, and tensor table are read.
   No weight data is touched. The architecture adapter turns tensors into a
   `ModelDesc`: global tensors, per-layer groups (`attn`, `ffn`, `experts`,
   `norms`), parameter count, quantization mix, and KV bytes per token.
3. **Profile the hardware.** Discovery runs on every launch because it is cheap.
   Micro-benchmarks are cached per machine in `~/.cache/kestrel/hwprofile.json`
   and re-run with `kestrel hardware --bench` or when the fingerprint changes.
4. **Plan.** The planner enumerates feasible strategies (GPU-full, GPU+RAM
   hybrid, RAM-only, RAM+NVMe streaming, VRAM+RAM+NVMe), assigns every tensor
   group to a tier under budgets with safety margins, and estimates per-token
   time from measured bandwidths. It selects the fastest feasible strategy. If
   nothing is feasible, it prints a structured explanation with remedies.
5. **Allocate.** The Ledger reserves the plan's budgets before any weight is
   loaded. Resident groups are read into owned buffers. Streamed groups get a
   bounded slot ring sized by the plan.
6. **Execute.** For each token, layer by layer, the backend acquires a lease on
   layer *i*. The prefetcher has already issued the reads for layers
   *i+1 … i+d*, so I/O overlaps compute. Metrics record hits, stalls, and bytes.
7. **Guard.** At token boundaries (safe points with no leases held), the
   Rebalancer demotes groups (frees buffers) when the RSS or
   available-memory guard trips, and restores them once memory is back.
   Promoting beyond the plan is opt-in (`--adapt`). For MoE models the expert
   cache adapts continuously, because which experts are hot depends on the
   prompt, not on the model.

Before step 1, `kestrel prepare` can rewrite the GGUF once into a container
laid out for Kestrel's I/O (see [prepared-format.md](prepared-format.md)).

## 3. Key abstractions

| Abstraction | Crate | Responsibility |
|---|---|---|
| `GgufFile` | kestrel-gguf | Parse GGUF v2/v3 (header, metadata KV, tensor infos, alignment) without reading weights. ggml type table (block sizes and bytes). |
| `ModelDesc` | kestrel-model | Architecture-neutral description: hyper-parameters, `TensorGroup`s with byte extents, KV geometry, MoE geometry (experts, top-k, expert slice size). |
| `ArchAdapter` | kestrel-model | Maps a GGUF architecture string (`llama`, `qwen2`, `qwen3`, `qwen2moe`, `qwen3moe`, …) to hyper-parameters and group roles. New architectures are added here without touching the scheduler. |
| `HardwareProfile` | kestrel-hw | Discovered capacities plus measured bandwidths and a fingerprint. Serializable JSON. |
| `ExecutionPlan` | kestrel-planner | The decision: strategy, per-group tier assignment, budgets (VRAM/RAM/KV/scratch/safety), prefetch depth, backend parameters, estimated tok/s, and the alternatives with their estimates. Serializable and printable. |
| `Ledger` | kestrel-memory | Per-tier budgets. Every byte Kestrel owns is reserved first. Reservation failure is an error value, not an OOM. |
| `WeightStore` | kestrel-memory | Tier-aware access: `lease(group)` returns a guard over contiguous bytes. Resident groups are always ready. Streamed groups are served from the slot ring and loaded by the I/O engine. |
| `IoEngine` | kestrel-memory | Positional reads (`pread` / `ReadFile` with offset) on a worker pool. `O_DIRECT` with aligned buffers where supported, falling back to buffered reads plus `POSIX_FADV_DONTNEED`. |
| `Prefetcher` | kestrel-memory | Issues loads ahead of use: deterministic graph order for dense layers, and predicted experts for MoE. Counts issued, useful, late, and wasted prefetches. |
| `ExpertCache` | kestrel-memory | LFRU cache of expert slices with leases, frequency decay, and promotion hysteresis (after Colibrì's `tier.h`). |
| `Backend` | kestrel-backends | `prepare(plan, store) → Session`; `Session::eval(tokens) → logits`. Implemented by `native` and `llamacpp`. |

## 4. Repository structure

```text
kestrel/
├── Cargo.toml                  workspace
├── crates/
│   ├── kestrel-gguf/           GGUF parser, ggml type table
│   ├── kestrel-hw/             hardware discovery + micro-benchmarks
│   ├── kestrel-model/          ModelDesc, architecture adapters, KV math
│   ├── kestrel-memory/         Ledger, IoEngine, WeightStore, caches, prefetch, metrics
│   ├── kestrel-planner/        cost model, strategies, ExecutionPlan, diagnostics
│   ├── kestrel-engine/         native CPU executor: kernels, forward pass, KV cache,
│   │                           tokenizer, sampling, chat templates
│   ├── kestrel-backends/       Backend trait; native + llama.cpp adapters
│   ├── kestrel-server/         OpenAI-compatible HTTP API (axum)
│   └── kestrel-cli/            `kestrel` binary
├── docs/                       architecture, research, methodology
├── benchmarks/                 benchmark harness scripts + recorded results
├── tools/                      fixture generators (tiny GGUF models), oracles
└── tests/                      cross-crate integration tests (in crates' tests/ dirs)
```

The prompt's suggested layout (`core/`, `memory/`, `hardware/`, `backends/`,
`inference/`, `api/`, `cli/`, `benchmarks/`, `tests/`) maps onto these crates
one to one. Splitting into crates enforces the dependency rule:
`kestrel-memory` and `kestrel-planner` do not depend on `kestrel-engine` or
any backend.

## 5. Dependency rules

```text
kestrel-gguf  ←  kestrel-model  ←  kestrel-planner  ←  kestrel-backends  ←  kestrel-cli
                       ↑                 ↑                    ↑                ↑
                 kestrel-hw  ────────────┘                    │          kestrel-server
                 kestrel-memory  ─────────────────────────────┤
                 kestrel-engine (uses kestrel-memory leases) ─┘
```

* No platform-specific code above `kestrel-hw` and `kestrel-memory::io`. Each
  platform branch lives behind a small `cfg` module with a portable fallback.
* No backend-specific types in the planner. Backend capabilities are described
  by data (`BackendCaps`: supported quant types, tiers it can execute on,
  whether it supports per-tensor placement).

## 6. What is reused, adapted, and built

| Category | Component | Decision |
|---|---|---|
| **Reuse** | GGUF format and quantization formats (Q4_0, Q8_0, Q4_K, Q6_K, F16, BF16, …) | Read natively. No conversion step, no new quant algorithms. |
| **Reuse** | ggml / llama.cpp GPU kernels (CUDA, Vulkan, Metal, HIP) | Driven through the llama.cpp adapter. Kestrel computes its placement (`n_gpu_layers`, `--override-tensor`, mmap/mlock, KV type, ctx). |
| **Reuse** | Chat templates embedded in GGUF (Jinja) | Rendered with `minijinja`. |
| **Adapt (from Colibrì)** | Bounded slot cache with leases (`ESlot`, `ColiExpertStore`) | `WeightStore` slot ring and `ExpertCache` lease guards (Rust lifetimes plus refcounts). |
| **Adapt** | Coalesced positional reads, async I/O pool, O_DIRECT | `IoEngine`. Extents are coalesced per tensor group. |
| **Adapt** | LFRU scoring with 25% + constant hysteresis, decay by halving | `ExpertCache` admission and victim policy. |
| **Adapt** | Router lookahead prefetch (PILOT) | Predictive expert prefetch (MoE phase). |
| **Adapt** | Budget from RAM, RSS guard, refuse-to-start when projected peak exceeds available | Ledger, guard, and the feasibility check. |
| **Adapt** | Persisted usage history (`.coli_usage`) | `<model>.kestrel-usage.json`, keyed by the model's content fingerprint. |
| **Build** | Universal planner with a measured cost model and strategy ranking | kestrel-planner |
| **Build** | Dense-layer tiering (static pin + streamed ring) for models larger than RAM | kestrel-memory |
| **Build** | Hardware profiler with cached micro-benchmarks | kestrel-hw |
| **Build** | Rebalancer: promote/demote between tiers at safe points from live metrics | kestrel-memory |
| **Build** | Cross-backend orchestration (same plan → native or llama.cpp) | kestrel-backends |
| **Build** | Observability (JSON metrics, Prometheus text, live CLI stats) | kestrel-memory metrics + server |

## 7. Non-goals for the MVP

* No custom GPU kernels. GPU execution is delegated to ggml through llama.cpp.
* No new quantization algorithms. Kestrel selects among existing formats.
* No multi-GPU execution. Discovery reports all GPUs and the planner data model
  is per-device, but the MVP plans for one.
* No batching across concurrent requests. The server serializes generation per
  model.

## 8. Safety properties

1. **Budgets before bytes.** No weight buffer is allocated without a Ledger
   reservation. Budgets are derived from *available* memory minus a safety
   margin (see [memory-model](memory-model.md)).
2. **Measured beats projected.** A runtime guard samples RSS and available
   memory at safe points, demotes groups, and shrinks caches when the
   projection was wrong. This is Colibrì issue #403's lesson.
3. **Placement never changes semantics.** Weights are bit-identical regardless
   of tier. Lossy levers (lower quantization, smaller context) are proposed by
   the planner as remedies. They are never applied silently.
4. **Local by default.** No network I/O except an explicit model download. No
   telemetry.

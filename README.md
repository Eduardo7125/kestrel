# Kestrel

**A local LLM runtime that plans across VRAM, RAM and NVMe.**

> *"I downloaded a model. Run it."*

```bash
kestrel run Qwen2.5-32B-Instruct-Q4_K_M
```

Kestrel inspects the machine (GPU, RAM, disks, *measured* bandwidths) and the
model (every tensor, its size, KV-cache needs). It then builds an execution
plan: which layers live in VRAM, which in RAM, which stream from NVMe every
token, how much KV cache fits, and what decode speed to expect from each
strategy. The plan is printed before anything runs, and every number in it is
either measured or labelled as an assumption.

Kestrel is not a GUI around llama.cpp. Its core is the **memory
orchestrator**:

* a per-tier **ledger**: nothing is allocated without a budget reservation;
* a **tier-aware weight store**: every weight access goes through a lease;
* an **I/O engine**: parallel positional reads with `O_DIRECT`;
* **prefetch** along the execution order;
* **Belady-optimal** eviction for cyclic layer scans;
* a **rebalancer** that promotes and demotes layers at runtime under a
  memory guard that trusts measured RSS over projections;
* for MoE models, an **expert cache**: one routed expert per unit, LFRU with
  hysteresis, batch-union loading, router-lookahead prefetch, and
  usage-history warm starts. This generalizes
  [Colibrì](docs/colibri-analysis.md)'s design.

Inference arithmetic is pluggable. The native Rust CPU executor runs every
weight access through Kestrel's scheduler, and an adapter hands Kestrel's
placement to llama.cpp/ggml for GPU execution.

Status: **MVP**. Native execution covers dense llama/qwen2/qwen3 and MoE
qwen2moe/qwen3moe/Mixtral-style GGUFs, with expert-granular streaming. Kestrel
can plan any GGUF. See [the roadmap](docs/roadmap.md).

---

## Quick start

```bash
cargo build --release
./target/release/kestrel hardware --bench         # measure this machine (~20 s, cached)
./target/release/kestrel inspect model.gguf        # architecture, tensors, memory needs
./target/release/kestrel plan model.gguf           # the execution plan and alternatives
./target/release/kestrel run model.gguf            # interactive chat
./target/release/kestrel run model.gguf -p "Hi" --stats
./target/release/kestrel serve model.gguf --port 8080
```

Models are found by path, or by (partial) name in `$KESTREL_MODELS`,
`~/.cache/kestrel/models`, `./models` and `.`. Kestrel never downloads
anything unless you ask it to.

### Override anything

```bash
kestrel run model.gguf \
  --vram-budget 7G --ram-budget 24G --kv-cache 2G \
  --backend llamacpp --ctx 8192 --prefetch-depth 3 --io direct
```

`--backend native|llamacpp|auto`, `--strategy gpu-full|hybrid|ram|ram-nvme|vram-ram-nvme`,
`--no-stream`, `--placement interleaved|contiguous`, `--policy belady|lru`,
`--io direct|buffered`, `--threads N`, `--no-adapt`, `--allow-overcommit`.

### OpenAI-compatible API

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
r = client.chat.completions.create(model="local", messages=[{"role": "user", "content": "Hello"}])
```

Endpoints: `POST /v1/chat/completions` (with `stream: true`),
`POST /v1/completions`, `GET /v1/models`, `GET /health`, `GET /metrics`
(Prometheus; `?format=json` for JSON).

---

## What a plan looks like

The MVP target is a 32B Q4_K_M model on an 8 GB GPU with 32 GB RAM. This is
the planner's output for that hardware profile (from the planner unit test):

```text
BUDGETS (available − safety − overhead = usable)
  VRAM  7.60 GB − 666.7 MB − 573.1 MB = 6.36 GB
  RAM   28.00 GB − 3.40 GB − 303.9 MB = 24.30 GB

PLAN  [Hybrid VRAM/RAM · backend llama.cpp]
  VRAM  5.91 GB weights (n_gpu_layers 20) + KV 335.5 MB
  RAM   13.95 GB weights resident + KV 738.2 MB

STRATEGIES
  A  Hybrid VRAM/RAM       llama.cpp  1.0 tok/s  selected
  B  RAM only (CPU)        llama.cpp  0.7 tok/s
  C  RAM only (CPU)        native     0.2 tok/s

LLAMA.CPP
  llama-server --model … --ctx-size 4096 --threads 8 --n-gpu-layers 20
```

The model fits in VRAM + RAM, so the planner does **not** stream from NVMe.
Streaming would only add disk reads. When the model does not fit (a 70B on
the same box), the plan streams the overflow and says plainly that decode is
then bounded by disk bandwidth. When nothing fits, Kestrel explains why and
what to change instead of crashing:

```text
Model cannot currently be executed: …
Required (minimum, with NVMe streaming):  RAM: 2.31 GB
Available:  RAM: 1.06 GB · Disk: 500 GB free
Possible solutions:
  1. Reduce the context (--ctx 2048) or quantize the KV cache (--kv-type q8_0): KV is 1.07 GB
  2. Close memory-heavy applications: 2.00 GB RAM is available of 4.00 GB
  3. Use a smaller model
```

## Measured results

Full methodology: [docs/benchmark-plan.md](docs/benchmark-plan.md). Raw data:
[benchmarks/results/](benchmarks/results/). Summary and interpretation,
including the negative results: [benchmarks/README.md](benchmarks/README.md).

## Architecture

```text
 CLI / OpenAI API
        │ ExecutionPlan
 PLANNER ── budgets · strategies · measured cost model · diagnostics
        │
 MEMORY ORCHESTRATOR ── Ledger · WeightStore (resident + streaming ring)
        │                IoEngine (O_DIRECT, parallel) · Prefetcher · Rebalancer
        │ leases
 BACKENDS ── native Rust CPU executor  |  llama.cpp/ggml (CUDA, Vulkan, Metal, HIP)
 HARDWARE ── discovery + micro-benchmarks (RAM, disk, kernels), cached
```

| Crate | Role |
|---|---|
| `kestrel-gguf` | GGUF parser (header, metadata, tensor table) without reading weights |
| `kestrel-hw` | CPU/RAM/GPU/storage discovery, `O_DIRECT` I/O, micro-benchmarks |
| `kestrel-model` | Architecture-neutral `ModelDesc`: tensor groups, KV and MoE geometry |
| `kestrel-memory` | Ledger, I/O engine, weight store, guard and rebalancer, LFRU cache |
| `kestrel-planner` | Strategies, cost model, execution plan, infeasibility diagnosis |
| `kestrel-engine` | Native executor: kernels (scalar reference + int8/AVX2), forward pass, tokenizers, sampling, chat templates |
| `kestrel-backends` | Plan → native session or llama.cpp |
| `kestrel-server` | OpenAI-compatible HTTP API |
| `kestrel-cli` | The `kestrel` binary |

Design documents:

* [Colibrì analysis](docs/colibri-analysis.md): source-level study of the
  reference project, and what Kestrel adopts and changes
* [Architecture](docs/architecture.md) · [Memory model](docs/memory-model.md) ·
  [Scheduler design](docs/scheduler-design.md) · [Backend strategy](docs/backend-strategy.md)
* [Technology decisions](docs/technology-decisions.md) · [MVP spec](docs/mvp-spec.md) ·
  [Roadmap](docs/roadmap.md) · [Benchmark plan](docs/benchmark-plan.md)

## Correctness

* Tokenization (SentencePiece and byte-level BPE, special tokens) matches
  llama.cpp exactly on the test fixtures.
* With `KESTREL_EXACT=1` (f32 activations), logits match llama.cpp run on
  gguf-py-dequantized weights to ~0.1% of logit scale, with identical greedy
  continuations. This holds for F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1,
  Q4_K, Q5_K and Q6_K, across llama, qwen2 and qwen3.
* The default fast path quantizes activations to int8, as llama.cpp does. Its
  deviation from llama.cpp is of the same size (~2-5% of logit scale on the
  random-weight fixtures) as llama.cpp's own deviation from exact arithmetic.
* Streaming every layer from disk produces **bit-identical** logits to
  running fully resident.

## Privacy

Local by default. Kestrel has no telemetry and no HTTP client. It reads your
model files and writes only its hardware profile cache
(`~/.cache/kestrel/hwprofile.json`).

## Platforms

Linux and Windows are the MVP targets. The core has no platform assumptions:
OS-specific code is isolated in `kestrel-hw` (discovery, positional and direct
I/O). macOS builds; Metal planning comes later. AVX2 kernels are selected at
runtime, with portable fallbacks everywhere else.

## License

Apache-2.0. See [LICENSE](LICENSE).

<div align="center">

<img src="assets/logo.png" alt="Kestrel logo" width="200">

# Kestrel

**Memory-tiered local LLM inference: plan across VRAM, RAM and NVMe, then run.**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust 1.87+](https://img.shields.io/badge/rust-1.87%2B-orange.svg)](https://www.rust-lang.org)
[![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20Windows-lightgrey.svg)](#platform-support)
[![API](https://img.shields.io/badge/API-OpenAI--compatible-green.svg)](#openai-compatible-api)
[![Status](https://img.shields.io/badge/status-alpha%20(MVP)-yellow.svg)](docs/roadmap.md)

[Get started](#get-started-in-one-step) ·
[Everyday use](#everyday-use) ·
[How it works](#how-it-works) ·
[Benchmarks](#benchmarks) ·
[Documentation](#documentation) ·
[Inspirations](#inspirations-and-related-work)

</div>

---

Kestrel is an open-source runtime for running large language models on
machines that cannot hold them in memory. It inspects the hardware and the
model, builds an **explicit, explainable execution plan** that places every
tensor group in VRAM, RAM or NVMe, and executes that plan with a memory
scheduler built for inference: budgeted allocation, prefetching along the
execution order, Belady-optimal residency for dense layers, and an
expert-granular cache for Mixture-of-Experts models.

> The kestrel is the falcon that hovers in place while it watches what is
> below. Kestrel the runtime holds a fixed plan and streams in what it needs.

## Get started in one step

You need a computer with **8 GB of RAM** at the very least (16 GB or more is
better), a few GB of free disk for the model, and an internet connection
for the download. A graphics card is optional.

**Linux** (Debian, Ubuntu; other distributions have the same packages under
their own names)

```bash
sudo apt install git build-essential curl
git clone https://github.com/eduardo7125/kestrel.git
cd kestrel
./start-here.sh
```

**macOS**

```bash
xcode-select --install
git clone https://github.com/eduardo7125/kestrel.git
cd kestrel
./start-here.sh
```

**Windows** (not tested yet): download the repository as a ZIP, unzip it,
and double-click **`START-HERE.bat`**.

You answer one question, which model, and Enter takes the recommendation.
Then the setup:

1. **builds Kestrel** for your machine. If Rust is missing it offers to
   install it (rustup on Linux and macOS, winget on Windows);
2. **looks at your machine**: CPU, RAM, free disk and GPU, and measures its
   memory bandwidth once (a few seconds);
3. **recommends a model**: the most capable one in the catalog that fits in
   your RAM and disk *and* is estimated to generate at least 4 tokens per
   second on this machine. Mixture-of-Experts models that do not fit can
   stream their experts from disk;
4. **downloads it** from Hugging Face with progress, resume and SHA-256
   verification. Stop it whenever you like; run the script again and it
   continues where it stopped;
5. **plans it, saves it as your default model and opens the chat.**

**Next time**, run `./start-here.sh` again, or just `kestrel`: it starts
straight away, with no build and no download.

| Option (after `./start-here.sh` or `START-HERE.bat`) | What it does |
|---|---|
| `--list` | Every catalog model against this machine, and why one does not fit |
| `--model ID` | Install that catalog model (ids are in `--list`) |
| `--repo OWNER/NAME [--quant Q4_K_M]` | Any GGUF repository on Hugging Face |
| `--model-file PATH` | Use a GGUF you already downloaded |
| `--yes` | No questions: take the recommendation |
| `--dir DIR` | Keep models on another disk (default `~/.cache/kestrel/models`) |
| `--serve` | Serve the OpenAI-compatible API instead of opening the chat |
| `--reconfigure` | Choose another model |

**If something goes wrong**

| What you see | What to do |
|---|---|
| The download stopped | Run the same command again: it resumes from the bytes already on disk |
| `access denied`, the repository may be gated | Accept the model's license on huggingface.co, then `export HF_TOKEN=<your token>` |
| `cannot reach https://huggingface.co` | Check the connection or proxy (`HTTPS_PROXY`, `NO_PROXY` are honoured), or set `HF_ENDPOINT` to a mirror |
| `needs N GB free on the disk` | `--dir` with a folder on a bigger disk |
| A C compiler or Rust is missing | The script prints the exact command for your system |
| Windows: the build mentions `link.exe` | Install the Visual Studio Build Tools with the "Desktop development with C++" workload |

## Table of contents

- [Why Kestrel](#why-kestrel)
- [Key capabilities](#key-capabilities)
- [Get started in one step](#get-started-in-one-step)
- [Platform support](#platform-support)
- [Everyday use](#everyday-use)
- [Install by hand](#install-by-hand)
- [Usage](#usage)
  - [Commands](#commands)
  - [Planning and budgets](#planning-and-budgets)
  - [Preparing a model](#preparing-a-model-kestrel-prepare)
  - [OpenAI-compatible API](#openai-compatible-api)
  - [Configuration reference](#configuration-reference)
- [How it works](#how-it-works)
- [Benchmarks](#benchmarks)
- [Correctness](#correctness)
- [Security and privacy](#security-and-privacy)
- [Project status and roadmap](#project-status-and-roadmap)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Inspirations and related work](#inspirations-and-related-work)
- [Citation](#citation)
- [License](#license)
- [Keywords](#keywords)

## Why Kestrel

Running a model locally usually comes down to one question: **does it fit?**
When it does not, the usual choices are a smaller quantization, manual
offload flags tuned by trial and error, or memory-mapping the file and
letting the operating system page it in. That last option is silent about
how much memory it really uses.

Kestrel treats memory placement as an engineering problem with measurable
inputs and outputs:

| Concern | Typical approach | Kestrel |
|---|---|---|
| Where each layer lives | Manual flags (`-ngl`, `-ot`) | Planner chooses from **measured** bandwidths and budgets, and prints why |
| Model larger than RAM | `mmap` and the OS page cache | Explicit NVMe streaming with `O_DIRECT`, prefetch and accounted buffers |
| Memory accounting | Process RSS, page cache invisible | Per-tier ledger: nothing is allocated without a reservation |
| Dense layer caching | LRU, or none | Static pin plus a streaming ring: the optimal policy for a cyclic scan |
| MoE experts | Whole tensors offloaded | One expert is the unit: LFRU cache, router lookahead, warm starts |
| Running out of memory | OOM kill | Memory guard demotes under pressure and restores once memory is back |
| "It does not fit" | Crash, or a cryptic error | A diagnosis with the numbers and concrete remedies |

## Key capabilities

**Planning**
- Hardware profiler: CPU, SIMD, RAM, GPUs, disks, plus micro-benchmarks for
  RAM bandwidth, disk bandwidth and per-quantization kernel throughput,
  cached per machine.
- Model inspector: reads the GGUF header only (no weights) and partitions
  the tensors into placement groups, including KV-cache and MoE geometry.
- Execution planner: budgets (available − safety margin − overhead), five
  strategies (GPU-full, GPU/RAM hybrid, RAM-only, RAM+NVMe, VRAM+RAM+NVMe), a
  bandwidth cost model with stated assumptions, and infeasibility diagnosis.

**Memory orchestration**
- Ledger with per-tier budgets and RAII reservations.
- Parallel I/O engine: positional reads, `O_DIRECT`, 4 MiB chunks.
- Weight store: resident groups plus a bounded streaming ring, depth-*d*
  prefetch, Belady-optimal eviction, interleaved placement of streamed layers.
- Memory guard: trusts measured RSS and free memory over projections.
  It demotes under pressure and restores afterwards. Promotion beyond the
  plan is opt-in (`--adapt`).

**Mixture of Experts**
- Expert-granular residency: an LFRU cache with hysteresis, leases,
  batch-union loading and scratch buffers for cold experts.
- Router-lookahead prefetch with an accuracy-gated speculative pool.
- Usage history persisted per model for warm starts.

**Execution**
- Native Rust CPU executor: llama, qwen2, qwen3, qwen2moe, qwen3moe and
  Mixtral-style models. AVX2 int8 kernels, an f32 GEMM for prefill, f16 KV
  cache with prefix reuse, SentencePiece and byte-level BPE tokenizers, and
  Jinja chat templates.
- llama.cpp adapter: hands Kestrel's placement to `llama-server` /
  `llama-cli` for GPU execution (CUDA, Vulkan, Metal, HIP via ggml).
- OpenAI-compatible HTTP server with streaming (SSE) and Prometheus metrics.

**Operations**
- `kestrel prepare`: a lossless, verified, one-time re-layout of a model
  file for Kestrel's I/O.
- `kestrel benchmark`: interleaved arms in isolated processes, page cache
  dropped per run, output identity checked. `--tune` autotunes and stores a
  per-machine profile.

## Platform support

| Platform | Planner and inspector | Native CPU executor | Direct I/O | llama.cpp backend |
|---|---|---|---|---|
| Linux x86-64 | Supported, tested | Supported, tested (AVX2 kernels, portable fallback) | `O_DIRECT` | Supported |
| Windows x86-64 | Compiles, untested | Compiles, untested | Buffered (unbuffered I/O on the roadmap) | Supported |
| macOS | Builds | Portable kernels | Buffered | Supported (Metal planning on the roadmap) |
| ARM64 (Linux, macOS) | Expected to build, untested | Portable kernels (NEON on the roadmap) | Platform-dependent | Supported |

The recorded benchmarks were measured on Linux only. Windows code paths
compile but have not been benchmarked yet.

## Everyday use

```bash
kestrel                  # chat with your default model
kestrel serve            # OpenAI-compatible API on http://127.0.0.1:8080/v1
kestrel status           # the configured model, and whether the server is running
kestrel setup --reconfigure   # switch to another model
```

Every command also takes a model file or name, for example
`kestrel chat ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf`. `-h` shows the
everyday options, `--help` every option.

For a closer look:

```bash
kestrel inspect          # architecture, tensors, memory requirements
kestrel plan             # the execution plan and the alternatives
kestrel hardware --bench # measure this machine (~20 s, cached)
```

## Install by hand

`start-here.sh` does all of this for you.

### Requirements

| Component | Version | Needed for |
|---|---|---|
| Rust toolchain | 1.87 or newer (`rustup`) | Building Kestrel |
| C toolchain | Any (`cc`, MSVC) | Some dependencies |
| llama.cpp | Optional, any recent build | GPU execution (`--backend llamacpp`) and the `llamacpp-*` benchmark arms |
| Python 3 + NumPy | Optional | `tools/` (synthetic models, test fixtures) |

### Build from source

```bash
git clone https://github.com/eduardo7125/kestrel.git
cd kestrel
cargo build --release
./target/release/kestrel --version
```

Install the binary onto your `PATH` (then `kestrel setup` picks and downloads a model):

```bash
cargo install --path crates/kestrel-cli
```

### Optional: GPU execution through llama.cpp

Kestrel's native executor runs on the CPU. For GPU execution it plans the
placement and launches llama.cpp with it. Build llama.cpp with your GPU
backend ([instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)),
then point Kestrel at it in one of three ways:

```bash
export KESTREL_LLAMA_CPP=/path/to/llama.cpp     # uses build/bin/llama-server, llama-cli, llama-bench
export KESTREL_LLAMA_SERVER=/path/to/llama-server   # or each binary explicitly
# or put the binaries on PATH
```

### Models

Kestrel reads **GGUF** files. It downloads only through `kestrel setup`, and
only the model you chose. Any GGUF you already have works too:

```bash
kestrel setup --model-file ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

Models are found by path, or by (partial) name in `$KESTREL_MODELS`,
`~/.cache/kestrel/models`, `./models` and the current directory.

## Usage

### Commands

| Command | Purpose |
|---|---|
| `kestrel setup [--list] [--model ID] [--repo R]` | Pick, download and configure a model (what `start-here.sh` runs) |
| `kestrel` / `kestrel chat [model]` | Chat with the default model, or answer one prompt with `-p` |
| `kestrel status` | The configured model and whether the API server is running |
| `kestrel models` | List local GGUF and prepared models |
| `kestrel inspect <model> [--tensors] [--json]` | Architecture, parameters, quantization, KV size, memory estimates, native support |
| `kestrel hardware [--bench] [--path DIR] [--json]` | Hardware discovery; `--bench` measures RAM, disk and kernel bandwidths |
| `kestrel plan <model> [options] [--json]` | Print the execution plan, the alternatives and the llama.cpp arguments |
| `kestrel run [model] [-p PROMPT] [options]` | Same as `chat`; `--stats` reports memory and speed |
| `kestrel serve [model] [--host] [--port] [options]` | OpenAI-compatible HTTP server |
| `kestrel prepare <model> [-o OUT] [--dry-run]` | Lossless re-layout for Kestrel's I/O (see below) |
| `kestrel benchmark <model> [--arms ...] [--tune]` | Reproducible strategy comparison, or autotuning |

### Planning and budgets

Every plan is printed before anything runs, and every number in it is either
measured or labelled as an assumption. This is the planner's output for a 32B
Q4_K_M model on an 8 GB GPU with 32 GB of RAM (from the planner's unit tests):

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
```

The model fits in VRAM plus RAM, so the planner does not stream from NVMe.
When a model does not fit, the plan streams the overflow and states that
decode is then bounded by disk bandwidth. When nothing fits, Kestrel explains
why and what to change:

```text
Model cannot currently be executed: …
Required (minimum, with NVMe streaming):  RAM: 2.31 GB
Available:  RAM: 1.06 GB · Disk: 500 GB free
Possible solutions:
  1. Reduce the context (--ctx 2048) or quantize the KV cache (--kv-type q8_0): KV is 1.07 GB
  2. Close memory-heavy applications: 2.00 GB RAM is available of 4.00 GB
  3. Use a smaller model
```

Override any decision:

```bash
kestrel run model.gguf \
  --vram-budget 7G --ram-budget 24G --kv-cache 2G --ctx 8192 \
  --backend llamacpp --prefetch-depth 3 --io direct
```

### Preparing a model (`kestrel prepare`)

```bash
kestrel prepare model.gguf       # writes model.kgguf next to it, verified
kestrel run model                # name resolution prefers the prepared file
```

`prepare` rewrites the GGUF once into a container laid out for Kestrel's I/O.
Every tensor keeps its type, shape and bytes, and both files are re-read and
compared tensor by tensor before the output is renamed into place.

- Each routed expert becomes **one contiguous, page-aligned read** instead of
  three scattered ones.
- Tensor groups are written in execution order, each on a 4 KiB boundary.
- The container uses the `KGUF` magic, so other GGUF readers refuse it
  instead of misreading it. Keep the source GGUF for llama.cpp.

On the test VM this cut expert read operations 3× with identical output, and
made no measurable difference to decode speed. It is aimed at storage where
each request is expensive. See
[docs/prepared-format.md](docs/prepared-format.md).

### OpenAI-compatible API

```bash
kestrel serve --port 8080
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
reply = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Explain NVMe streaming in one sentence."}],
    stream=True,
)
for chunk in reply:
    print(chunk.choices[0].delta.content or "", end="")
```

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | Chat completions; `stream: true` for server-sent events |
| `POST /v1/completions` | Text completions |
| `GET /v1/models` | The loaded model |
| `GET /health` | Liveness |
| `GET /metrics` | Prometheus metrics (`?format=json` for JSON): budgets, residency, hit rates, stall time, throughput |

### Configuration reference

**Placement and execution options** (`plan`, `run`, `serve`)

| Option | Default | Description |
|---|---|---|
| `--backend auto\|native\|llamacpp` | `auto` | Execution backend |
| `--strategy gpu-full\|hybrid\|ram\|ram-nvme\|vram-ram-nvme` | planner | Force a strategy |
| `--vram-budget`, `--ram-budget` | available − safety | Memory budgets, e.g. `7G`, `24G` |
| `--ctx N`, `--kv-cache SIZE`, `--kv-type f16\|q8_0` | 4096 or training ctx, f16 | Context length and KV-cache limits |
| `--no-stream` | off | Forbid NVMe streaming |
| `--prefetch-depth N` | 2 | Streamed groups loaded ahead (0 disables prefetch) |
| `--io direct\|buffered`, `--io-workers N` | direct, 8 | Streaming I/O mode and parallelism |
| `--placement interleaved\|contiguous` | interleaved | Position of streamed layers in the layer order |
| `--policy belady\|lru` | belady | Ring eviction policy (LRU is an ablation) |
| `--expert-policy lfru\|lru` | lfru | MoE expert-cache policy (LRU is an ablation) |
| `-t, --threads N` | physical cores | Compute threads |
| `--adapt` | off | Promote streamed layers beyond the plan when memory frees up |
| `--no-adapt` | off | Disable the memory guard |
| `--no-usage-history`, `--no-tune-profile` | off | Ignore saved expert usage or tuning profiles |
| `--allow-overcommit` | off | Allow budgets larger than available memory |

**Environment variables**

| Variable | Purpose |
|---|---|
| `KESTREL_MODELS` | Additional model directories |
| `HF_ENDPOINT` | Hugging Face mirror for `kestrel setup` (default `https://huggingface.co`) |
| `HF_TOKEN` | Hugging Face token, for gated repositories |
| `HTTPS_PROXY`, `NO_PROXY` | Proxy for downloads (`NO_PROXY` is honoured) |
| `KESTREL_CACHE` | Cache directory (default `~/.cache/kestrel`, `%LOCALAPPDATA%\kestrel` on Windows) |
| `KESTREL_LLAMA_CPP` | llama.cpp checkout; binaries are taken from `build/bin` |
| `KESTREL_LLAMA_SERVER`, `KESTREL_LLAMA_CLI`, `KESTREL_LLAMA_BENCH` | Explicit paths to llama.cpp binaries |
| `KESTREL_EXACT=1` | f32 activations instead of int8 (reference arithmetic) |
| `KESTREL_PROFILE=1` | Per-phase timing profile of the forward pass |
| `KESTREL_LOOKAHEAD=0` | Disable router-lookahead expert prefetch |
| `KESTREL_SPEC_MIN_ACCURACY` | Accuracy gate of the speculative expert pool (default 0.5) |

**Files Kestrel writes**: the setup choice (`$KESTREL_CACHE/setup.json`),
downloaded models (`$KESTREL_CACHE/models/`), the hardware profile
(`$KESTREL_CACHE/hwprofile.json`), tuning profiles
(`$KESTREL_CACHE/tuning/`), the expert usage history next to the model
(`<model>.kestrel-usage.json`), and prepared containers (`<model>.kgguf`)
when you ask for them.

## How it works

```text
 CLI  /  OpenAI-compatible API
            │  ExecutionPlan
 PLANNER ── budgets · strategies · measured cost model · diagnostics
            │
 MEMORY ORCHESTRATOR
   Ledger ── per-tier budgets, RAII reservations
   WeightStore ── resident groups + streaming ring (Belady), prefetch
   ExpertStore ── LFRU expert cache, lookahead, warm start      (MoE)
   IoEngine ── parallel positional reads, O_DIRECT
   MemoryGuard ── measured RSS → demote / restore
            │  leases
 BACKENDS ── native Rust CPU executor  │  llama.cpp / ggml (CUDA, Vulkan, Metal, HIP)
 HARDWARE ── discovery + micro-benchmarks, cached
```

1. **Profile.** Discover the machine and measure what matters for the plan:
   RAM bandwidth, disk bandwidth and latency, and kernel throughput per
   quantization type.
2. **Inspect.** Read the GGUF header and group tensors into placement units:
   embeddings, per-layer attention and FFN, routed experts, and the head.
3. **Plan.** Compute budgets, place groups by bandwidth gain per byte,
   interleave the streamed layers with resident ones, and estimate decode
   speed for every strategy.
4. **Execute.** Every weight access is a lease from the weight store. Reads
   for the next *d* streamed groups are already in flight, so I/O overlaps
   compute.
5. **Guard.** Between tokens, the memory guard compares measured RSS and
   free memory with the budget. It demotes groups under pressure and
   restores them later. The expert cache adapts continuously, because which
   experts are hot depends on the prompt.

Dense layers are read in a fixed cyclic order. An LRU cache smaller than the
model gets a 0% hit rate on that pattern, so Kestrel pins a fixed subset and
streams the rest through a minimal ring, which is the optimal policy for a
cyclic scan. Design details: [docs/scheduler-design.md](docs/scheduler-design.md).

| Crate | Responsibility |
|---|---|
| `kestrel-gguf` | GGUF parser and writer (header, metadata, tensor table) |
| `kestrel-hw` | Hardware discovery, direct I/O, micro-benchmarks |
| `kestrel-model` | Architecture-neutral model description; `prepare` |
| `kestrel-memory` | Ledger, I/O engine, weight store, expert store, guard, LFRU |
| `kestrel-planner` | Strategies, cost model, execution plan, diagnostics |
| `kestrel-engine` | Native executor: kernels, forward pass, tokenizers, sampling, chat templates |
| `kestrel-backends` | Plan → native session or llama.cpp |
| `kestrel-server` | OpenAI-compatible HTTP API |
| `kestrel-cli` | The `kestrel` binary |

## Benchmarks

All numbers below come from one cloud VM (4-core Xeon, 16.9 GB RAM, virtio
disk, **no GPU**) and synthetic models with real architecture shapes. They
show **relative** effects between strategies, not absolute speed on your
hardware. Full methodology, raw JSON and negative results are in
[benchmarks/README.md](benchmarks/README.md).

| Finding | Result |
|---|---|
| Prefetch at equal RAM | **+63%** decode (1.1B), **+68%** (7B) |
| Interleaved vs contiguous streamed layers | **+26%** decode |
| Static pin + ring vs the same RAM as a cache | **3.4×** an LRU cache, **2×** a Belady cache |
| Streaming half the layers (7B) | **−41% RSS** at 67% of resident speed |
| `mmap`-style page-cache streaming | Competitive only by holding 0.6-3.65 GB of page cache **outside** the budget |
| MoE, 21% of experts cached | **32% of the RAM** for 64% of resident speed |
| LFRU vs LRU expert cache | **+31%** decode |
| Memory-guard demotion / background promotion | 0.8 ms per group / 6 ms per 9 MB group |

**Known gap:** the native CPU kernels are still 1.6-2× slower than llama.cpp
at decode (1.1B: 13.5 vs 27 tok/s; 7B: 3.1 vs 4.8 tok/s) and about 2.8×
slower at prefill. Closing that gap is the first roadmap item. GPU execution
goes through llama.cpp.

Reproduce on your machine:

```bash
kestrel benchmark model.gguf --runs 3 --json results.json
```

## Correctness

Placement must never change the output.

- Tokenization (SentencePiece and byte-level BPE, special tokens) matches
  llama.cpp exactly on the test fixtures.
- With `KESTREL_EXACT=1`, logits match llama.cpp (on dequantized weights) to
  about 0.1% of logit scale, with identical greedy continuations. This holds
  for F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K and Q6_K, across
  dense and MoE architectures.
- The default int8 path deviates from llama.cpp by about as much as
  llama.cpp deviates from exact arithmetic.
- Streaming from disk, any cache state and prepared containers all produce
  **bit-identical** output to fully resident execution. Every benchmark arm
  checks this.

```bash
cargo test --workspace
```

## Security and privacy

- **Local by default.** No telemetry. The only network access is the model
  download in `kestrel setup`, for the model you chose, verified by
  SHA-256. Inference, `serve` and every other command never touch the
  network.
- The API server binds to `127.0.0.1` and has **no authentication**. Put it
  behind an authenticating reverse proxy before exposing it on a network.
- The GGUF parser bounds every count and length it reads from a file before
  allocating.

Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## Project status and roadmap

Kestrel is an **alpha (MVP)**. The planner, memory orchestrator, native CPU
executor, MoE expert cache and OpenAI API work end to end and are tested.
Interfaces and file formats may still change.

| Area | Status |
|---|---|
| Planner, profiler, inspector, diagnostics | Done |
| NVMe streaming, prefetch, Belady residency, memory guard | Done |
| MoE expert cache, router lookahead, warm starts | Done |
| `prepare`, `benchmark --tune` | Done |
| Native kernel parity with llama.cpp (AVX2, NEON, AVX-512) | In progress |
| GPU tier under Kestrel's own residency control (ggml FFI) | Planned |
| io_uring, Windows unbuffered I/O, multi-GPU | Planned |
| More architectures (Gemma, Phi, DeepSeek), safetensors | Planned |

Details: [docs/roadmap.md](docs/roadmap.md).

## Documentation

| Document | Contents |
|---|---|
| [Architecture](docs/architecture.md) | Components, data flow, key abstractions |
| [Memory model](docs/memory-model.md) | Tiers, groups, ledger, guards, page cache |
| [Scheduler design](docs/scheduler-design.md) | Placement, prefetch, caching, expert cache, autotuning |
| [Backend strategy](docs/backend-strategy.md) | Native executor and llama.cpp adapter |
| [Prepared format](docs/prepared-format.md) | The `kestrel prepare` container |
| [Benchmark plan](docs/benchmark-plan.md) | Methodology and benchmark arms |
| [Technology decisions](docs/technology-decisions.md) | Why Rust, GGUF, and the other choices |
| [MVP specification](docs/mvp-spec.md) | Scope of the first release |
| [Colibrì analysis](docs/colibri-analysis.md) | Source-level study of the main reference project |

## Contributing

Contributions are welcome, especially benchmark datapoints from real
hardware (NVMe drives, GPUs, ARM machines). Two rules come first:
placement never changes semantics, and every performance claim comes with a
reproducible benchmark. Read [CONTRIBUTING.md](CONTRIBUTING.md) before you
open a pull request.

## Inspirations and related work

Kestrel builds on ideas from these projects and papers. None of their code is
included; see [NOTICE](NOTICE).

**Direct inspirations**

- **[Colibrì](https://github.com/JustVugg/colibri)**: a pure-C engine that
  runs very large MoE models from disk. Kestrel's expert cache (LFRU with
  hysteresis, leases), router-lookahead prefetch, usage-history warm start,
  measured-RSS memory guard, accuracy-gated autotuning and packed expert
  layout all start from a source-level study of Colibrì
  ([analysis](docs/colibri-analysis.md)). Kestrel generalizes them to dense
  and MoE GGUF models and to an explicit planner.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp) and
  [ggml](https://github.com/ggml-org/ggml)**: the GGUF format, the
  quantization formats Kestrel's kernels implement, the reference
  implementation that every correctness test compares against, and Kestrel's
  GPU backend.
- **L. A. Belady, "A study of replacement algorithms for a virtual-storage
  computer"** (IBM Systems Journal, 1966): the optimal replacement policy
  behind dense-layer residency.

**Related work on offloaded inference**

- **FlexGen** (Sheng et al., ICML 2023): throughput-oriented offloading
  across GPU, CPU and disk with a cost-model-driven policy search.
- **LLM in a Flash** (Alizadeh et al., Apple, 2023): inference with weights
  on flash storage; reading larger contiguous chunks ("row-column
  bundling") is the same idea as Kestrel's packed experts.
- **Fast Inference of Mixture-of-Experts Language Models with Offloading**
  (Eliseev and Mazur, 2023): LRU expert caching and speculative expert
  loading for Mixtral.
- **PowerInfer** (Song et al., SOSP 2024): hot and cold neurons split
  between GPU and CPU.
- **DeepSpeed ZeRO-Inference** and **AirLLM**: CPU/NVMe offload and
  layer-by-layer loading.

## Citation

If you use Kestrel in research, please cite it:

```bibtex
@software{kestrel,
  title   = {Kestrel: Memory-Tiered Local LLM Inference across VRAM, RAM and NVMe},
  author  = {{The Kestrel Authors}},
  year    = {2026},
  url     = {https://github.com/eduardo7125/kestrel},
  license = {Apache-2.0}
}
```

A machine-readable [CITATION.cff](CITATION.cff) is included.

## License

Copyright 2026 The Kestrel Authors.

Licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE)
and [NOTICE](NOTICE). Unless you explicitly state otherwise, any contribution
you intentionally submit for inclusion in Kestrel is licensed as above,
without any additional terms or conditions.

## Keywords

local LLM inference · LLM runtime · large language models on consumer
hardware · run LLMs that don't fit in memory · NVMe offloading · SSD
streaming · CPU offloading · GPU offloading · VRAM / RAM / NVMe tiering ·
memory hierarchy · memory scheduler · KV cache · prefetching · Belady ·
LFRU cache · Mixture of Experts (MoE) · expert offloading · expert caching ·
GGUF · llama.cpp · ggml · quantization · Q4_K_M · Qwen · Llama · Mixtral ·
OpenAI-compatible API · self-hosted AI · on-premise AI · edge AI · Rust ·
O_DIRECT · direct I/O · inference engine · model serving

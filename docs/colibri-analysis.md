# Colibrì analysis

This is a source-level study of [JustVugg/colibri](https://github.com/JustVugg/colibri),
read at commit `ce370e8` (release 1.12.1, Apache-2.0). Every claim below cites the
file, and usually the function, where the mechanism lives. Benchmark numbers are
quoted from Colibrì's own docs and attributed to them. We did not reproduce them:
Colibrì's targets are 20 GB to 1.6 TB checkpoints, and this research environment
has no GPU.

Paths are relative to the Colibrì repository root. Most of the engine logic is in
`c/colibri.c` (the GLM-5.2 engine, 12.7k lines). Sibling engines (`c/qwen36.c`,
`c/deepseek_v4.c`, `c/kimi_k3.c`, …) share headers.

---

## 0. Summary

Colibrì is a **pure-C, MoE-only, safetensors-only** inference engine with one
`.c` file per model family. It runs 35B to 2.8T-parameter MoE models on
commodity machines with a single idea: **the dense trunk stays resident, and the
routed experts are streamed from disk into a bounded per-layer RAM cache**.
Optional VRAM tiers hold the hottest experts. Everything else is engineering to
hide or avoid the disk:

| Technique | Where | What it buys |
|---|---|---|
| Dense trunk resident (RAM, int4/int8), experts on disk | `c/colibri.c` model load; `README.md` "The idea" | RAM needs scale with *active* rather than *total* parameters |
| Per-layer expert LRU of fixed-size slots (`ecache`, `ecap`) | `ESlot`, `eslot_lru_victim` (`c/colibri.c:~410-442`) | Bounded RAM and O(1) reuse of slabs |
| One coalesced `pread` per expert (gate/up/down adjacent on disk) | `expert_load_impl` (`c/colibri.c:2954`) | Fewer syscalls; a sequential read per expert |
| Async I/O pool overlapping reads with resident-expert matmul | `PipePool`, `pipe_worker`, `pipe_dispatch`, `pipe_wait` (`c/colibri.c:3717-3850`) | Docs measure −18% disk service time |
| io_uring batch backend (Linux) | `c/uring.h`, `uring_load_add` / `uring_submit_batch` (`c/colibri.c:3449-3715`) | Queue depth >1 without threads |
| `O_DIRECT` twin fds, opt-in | `st_direct_fd` (`c/st.h:281`), `g_direct` (`c/colibri.c:1314`) | Drive-dependent; docs report +34% to +65% on some NVMe |
| Router-lookahead prefetch ("PILOT") | `pilot_prefetch` (`c/colibri.c:7007`), `pilot_worker`, `pilot_realload` | Docs claim 71.6% one-layer-ahead recall |
| Batch-union of experts across positions | `moe()` phase B (`c/colibri.c:5414+`) | Each unique expert is read once per layer per batch |
| Learned pin set from routing history (`.coli_usage`) | `c/route_trace.h`; `PIN=auto`, `AUTOPIN` | Persistent hot set that warms across sessions |
| LFRU live re-pinning with hysteresis | `c/tier.h` (`tier_pick_lfru`, `tier_should_promote`) | Adapts to the workload without ping-pong |
| RAM budget → cache cap, with auto-raise and an RSS guard | `cap_for_ram` region (`c/colibri.c:~10680-10750`), `rss_guard` (`c/colibri.c:~8770`) | Avoids OOM. Measured RSS, not the projection, is the final authority |
| Multi-SSD mirrors with weighted striping | `expert_route`, `mir_pread_striped` (`c/colibri.c:2744-2900`), `docs/multidisk.md` | +37.5% decode measured with two independent NVMe drives |
| Offline planner (`coli plan`) and autotuner (`coli tune`) | `c/resource_plan.py` (`build_plan`), `c/autotune.py` | Hot/warm/cold sizing plus a measured knob sweep with gates |

The non-negotiable design rule, repeated throughout the code and docs, is
**"placement only ever decides speed"**. Tiering must not change router
decisions or weight precision unless the user opts into a lossy mode
(`--topp`, `CACHE_ROUTE`, `DEGRADE_ZERO`), and each of those prints a warning.

---

## 1. Model loading

### 1.1 Format

* **Safetensors only.** `c/st.h` indexes one or more `*.safetensors` shards
  (`st_init_multi`, `c/st.h:645`) into a `shards` struct holding up to 512 fds
  (`c/st.h:~50-75`). The repository has no GGUF reader (`grep -ri gguf c/`
  returns nothing).
* Tensor lookup uses an FNV-1a open-addressing hash (`st_hash`, `hidx`). The
  comment explains why: a linear scan over ~120k tensors (256 experts × 78
  layers × 3 × 2) "cost tens of seconds per token".
* Each record is an `st_tensor { name, fd, off, nbytes, dtype, numel, rank, shape }`.
  The absolute file offset is precomputed, so a read is a single `pread(fd, …, off)`.
* **Pre-quantized containers.** A converter (`c/tools/convert_*.py`) produces
  `<name>` (packed weights) plus `<name>.qs` (scales). The in-memory format is
  inferred from byte arithmetic in `qt_resolve_fmt` (`c/colibri.c`). It can be
  confirmed by an optional `__metadata__["colibri.fmt"]` stamp
  (`st_fmt_stamp_ingest`, `c/st.h:441`). `docs/FORMATS.md` is the registry of
  format ordinals: f32, int8-row, int4-row, int2, int4-g64, int3-g64, E8/IQ3,
  MXFP4, fp8-e4m3 block, and rANS-compressed int4.

### 1.2 Tensor representation

* `QT` (`c/colibri.c:~230-270`) is the quantized-tensor handle: `fmt, O, I, gs`,
  data pointers `q8 / q4 / qf / s`, optional device mirrors (`cuda_*`, `vk`), and
  `mmap_view`.
* `qt_bytes()` (`c/colibri.c:273`) gives the exact resident size per format.
  The comments stress that this function is **load-bearing for budgeting**: a
  wrong branch once under-counted fp8 tensors and broke the RAM math. `qt_rb()`
  (`:307`) returns 0 for mmap views, because file-backed pages are not RSS.
* A newer cross-engine ABI, `ColiTensorView` (`c/tensor.h`), describes a view on
  bytes that are owned elsewhere. It is used by the `ColiExpertStore` interface
  (`c/expert_store.h`).

### 1.3 How weights are accessed: pread, not mmap (by default)

* The header comment of `c/st.h` states the policy: it **reads with `pread` (no
  mmap) plus `posix_fadvise(DONTNEED)`, so pages do not stay resident in the
  process**. It calls this "the fix for the RSS bug": peak RAM must be dense
  weights plus cache, not the whole model.
* The `drop` flag on reads (`st_read_f32(…, drop)`) issues DONTNEED after
  streaming reads. The default `DROP=0` (`g_drop`, `c/colibri.c:1305`) leaves
  expert pages in the OS page cache, which is not counted in RSS, as a "free L2".
  The comment notes this exploits routing imbalance.
* `mmap` exists as an option:
  * `COLI_MMAP` maps whole shards (`map_of_fd`, `c/colibri.c:~2700`) so experts
    can be served zero-copy (`expert_load_impl` checks `g_mmap`). Pages can be
    `mlock`ed ("wired") under a budget (`qt_wire_mmap`, `g_mmap_wired`).
  * `TRUNK_RESIDENT_LAYERS=N` (`c/colibri.c:2132-2139`, `qt_load_mmap` at
    `:2227`) keeps only the top N dense layers resident and maps the rest
    read-only. The comment calls it a "run at all vs run fast" lever: on hosts
    where the trunk does not fit, mapped layers page from disk instead of
    OOMing. This is the only place where **dense** weights are disk-backed, and
    it is CPU-only ("GPU backends refuse").
* Large-file handling: reads are chunked (`ST_PREAD_CHUNK`, `st_pread_full`,
  `c/st.h:376`) and EINTR is retried. A short read is fatal with a diagnostic.
  Header size is capped at 512 MiB (`ST_MAX_HEADER`) to reject crafted files.

### 1.4 Metadata

The model config comes from the HF `config.json` (`load_cfg`, `c/colibri.c:1716`).
The launcher picks the engine binary from `config.json`'s architecture
(`c/family_registry.py`).

---

## 2. Memory hierarchy

### 2.1 What lives where

From the README ("The idea") and `docs/cuda.md`:

| Data | Default tier | Mechanism |
|---|---|---|
| Embeddings, attention, norms, router, shared experts, dense MLP layers ("dense trunk") | **RAM, resident** (or VRAM with `CUDA_DENSE=1`) | `qt_load` → `qalloc`; optional NUMA interleave (`numa_slab_bind`) |
| Routed experts, hot | **Pinned RAM** (`PIN`, `PIN_GB`, auto-pin from `.coli_usage`) and/or the **VRAM expert tier** (`CUDA_EXPERT_GB`) | `pin_index`; `qt_cuda_upload` at startup |
| Routed experts, warm | **Per-layer LRU slots in RAM** (`m->ecache[layer][0..ecap)`) | `expert_load` into `ESlot.slab` |
| Routed experts, cold | **Disk** (safetensors), possibly also in the OS page cache | `pread`, optional `O_DIRECT` |
| KV cache | RAM; MLA-compressed (576 floats/token instead of 32,768) and persisted to `.coli_kv` | `KVState` (`c/colibri.c:~445`) |

Concrete numbers for GLM-5.2 744B (README): about 17B dense parameters, which is
9.9 GB resident at int4. There are 19,456 routed experts of about 19 MB each
(about 370 GB on disk). A cold token reads about 11 GB of experts.

### 2.2 Placement is mostly static, with an adaptive hot tier

* **Static at startup:** dense weights, the pin set (from a stats file or
  `.coli_usage` history), and the VRAM tier (uploaded at startup, "so capacity
  failures occur before inference" per `docs/cuda.md`).
* **Dynamic, per token:** the per-layer LRU (`eslot_lru_victim`) and PILOT
  prefetch.
* **Dynamic, at safe points:** `--repin N` / `REPIN` swap at most four pinned
  experts every N tokens using `tier_pick_lfru` (`c/tier.h`). The score is
  `(heat << 8) | recency`, so frequency dominates and recency breaks ties
  (`tier_lfru_score`). A swap needs `hot > cold + cold/4 + 4`. The 25% margin
  plus a constant prevents ping-pong (`tier_should_promote`). Heat decays by
  halving (`tier_decay_value`).
* The qwen36 engine applies the same logic one level up, from RAM to VRAM, on a
  background upload thread "so decode never blocks on placement"
  (`c/qwen36_tier.h` header comment).

### 2.3 Memory pressure

1. **Planning:** the cache cap (slots per layer) is derived from the RAM budget:
   `resident_bytes + ecap × row_bytes + slack ≤ RAM_GB`. `slack` covers the
   workspace, KV slots, and the KV write buffer. The engine **refuses to start**
   if the projected peak exceeds actually-available RAM (exit code 2, override
   `COLI_RAM_OVERCOMMIT=1`). It **auto-raises** the cap when the budget allows
   (issue #12: a 128 GB host ran with a 16 GB host's cache and got a 23-28% hit
   rate). See `c/colibri.c:~10680-10745`.
2. **Runtime RSS guard** (`rss_guard`, `c/colibri.c:~8770-8830`): every 16
   emitted tokens it compares *measured* RSS with the budget (2% + 300 MB
   tolerance). If over, it frees LRU slabs in place and **lowers `ecap`** so the
   cache cannot regrow. The comment cites issue #403, where the projection
   under-estimated real RSS by about 40 GB and the kernel OOM-killed the engine
   three times. Lesson: **projections are estimates, so enforce budgets on
   measurements**.
3. **VRAM:** the budget is clamped to free VRAM minus the projected dense set and
   `CUDA_RESERVE_GB` (default 2 GB headroom per device). For CUDA call failures,
   the failing batch size is recorded per tensor (`cuda_fail_s`), so a prefill
   chunk that OOMs does not disable S=1 decode on that tensor.

---

## 3. Tensor streaming

### 3.1 Unit of transfer

**One expert (its gate, up, and down matrices plus scales) is the unit.** The
container stores them adjacently, so `expert_load_impl` (`c/colibri.c:2954`)
issues **one coalesced pread into a per-slot slab**. `g/u/d` are then views
into `ESlot.slab`. The slab is reused across loads, and `slab_cap` lets a slot
grow when an expert is larger (for example the int8 MTP layer). Dense tensors
are never streamed, except as mmap pages under `TRUNK_RESIDENT_LAYERS`.

### 3.2 Synchronous vs asynchronous

* `PIPE=0`: blocking serial loads on the compute thread.
* `PIPE=1` (the default on Windows, opt-in elsewhere): a **bounded worker pool**
  (`PIPE_WORKERS`, default 8, maximum 16). `pipe_dispatch` publishes a batch of
  up to 64 expert ids through a generation-tagged atomic cursor
  (`cur = gen<<8 | index`). Workers CAS-claim indices and set `ready[i]`.
  Meanwhile the main thread **computes the already-resident experts first**,
  then `pipe_wait(i)` joins each missing one before use. Profiling reports both
  disk service time and the smaller "foreground-visible wait"
  (`docs/tuning.md`), so overlap is explicit.
* `URING=1`: the same batch is submitted as io_uring reads (`c/uring.h`, a raw
  syscall implementation with no liburing). Completion is reaped in
  `uring_finalize_load`.
* `COLI_PIPE_BLOCK=1` swaps the spin-yield wait for a condvar. Issue #159 found
  that a `sched_yield` storm fought the OpenMP team for cycles.

### 3.3 Buffering and cache reuse

* Slabs are owned by `ESlot`s in `m->ecache[layer]`. Workspace slots `m->ws[q]`
  receive pipe loads, and are then promoted into the LRU (`moe()` phase C,
  around `c/colibri.c:6551`: `promo = min(nmiss, ecap)` and victims come from
  `eslot_lru_victim`).
* Slots carry an `in_flight` refcount (`eslot_acquire/release`), so a GPU reader
  or a pilot load can never be evicted mid-use. The newer `ColiExpertStore` API
  (`c/expert_store.h`) formalizes this as a lease contract: `lookup()` returns a
  view plus a lease, `release()` exactly once, and advisory `prefetch()` "must
  not evict a slot that still has an active lease".
* `eslot_lru_victim` (issue #1034) distinguishes reusing a slab-less slot
  (growth) from evicting one. Growth is allowed only while live slabs are below
  `ecap`.

### 3.4 Prefetching

There are three generations of prefetch in the code, and their history is
instructive:

1. **`PREFETCH=1`: cross-layer `posix_fadvise(WILLNEED)`** (`expert_prefetch`,
   `c/colibri.c:3906`; `st_prefetch_rep`, `c/st.h:1038`). Now **off by default**.
   The code comment says real parallel loads made it redundant, and "under
   memory pressure the speculative readahead was evicted again".
2. **`PILOT=1`: router lookahead** (`pilot_prefetch`, `c/colibri.c:7007`).
   While layer L computes, layer L+1's router runs on layer L's post-attention
   hidden state to predict L+1's top-k experts. Each predicted expert not in the
   pin set or the LRU is pushed onto a lock-free SPMC ring (`pilot_q[4096]`) and
   consumed by `pilot_worker` threads. `PILOT_TWO=1` first approximates layer L's
   MoE output using only the resident shared expert, which adds about 3% recall.
   `docs/tuning.md` reports **71.6% recall** of the true top-8, versus 41.3% for
   "same experts as the last token".
3. **`PILOT_REAL=1`**: the pilot performs *real* loads into `ecache[L+1]`
   (`pilot_realload`), not just hints. A cross-layer barrier in `moe()` (top of
   the function) makes the main thread wait for in-flight pilot loads on its own
   layer before resolving residency. `PILOT_EVICT_GUARD` stops speculative loads
   from evicting experts that the current computation needs. Docs: "+11pp hit
   rate on a big-cache host", but **"on disk-saturated hosts hint-only PILOT can
   be net negative — measure"**.

### 3.5 Disk-path specifics

* **Batch union:** in `moe()` phase A, all S positions are routed first, then
  the *unique* experts across the batch are loaded once. Prefill and speculative
  verify therefore cost one read per distinct expert.
* **O_DIRECT** (`DIRECT=1`): a twin O_DIRECT fd is opened lazily per shard
  (`dfds[]`, `st_direct_fd`). Reads go through aligned buffers
  (`st_pread_aligned_try`). `DISKCLASS` (`expert_classify`, `c/colibri.c:2947`)
  sends cold experts to the direct fd and warm ones through the page cache.
* **Multi-SSD:** `COLI_MODEL_MIRROR` registers byte-identical replica
  directories, validated by size and header. `expert_route` assigns each expert
  to a replica, with weights measured at startup. `mir_pread_striped` can stripe
  a single expert across drives under O_DIRECT. Read errors fall back to the
  primary.

---

## 4. MoE optimization

* **Why it works** (README, "The idea"): GLM-5.2 is 744B total, about 40B
  active per token, and **only about 11 GB of that changes from token to
  token** (the routed experts). The ~17B dense parameters are touched every
  token, so they stay resident. Routed experts are 5.4% of parameters per token.
  RAM requirements therefore scale with dense + cache, not total size. Disk
  bandwidth over bytes-per-token sets the decode floor: about 1 GB/s divided by
  about 11 GB/token gives the documented 0.05-0.1 tok/s cold floor.
* **Expert residency:** pin set ∪ per-layer LRU ∪ (optional) VRAM tier.
  `expert_is_resident()` (`c/colibri.c:5356`) is the single residency predicate.
* **Frequency tracking:** `m->eusage` (persistent, written to `.coli_usage` each
  turn by `c/route_trace.h` as sparse `layer expert count` text with a versioned
  header), `m->eheat` (session heat, decayed), and `m->elast` / `eaccess_clock`
  (recency).
* **Expert caching policies:** LRU for the warm tier. The hot tier uses a
  learned pin (top-N by `.coli_usage`), optionally adapted live by LFRU repin.
  `PIN_FILL=1` fills spare VRAM with zero-heat experts.
* **Expert streaming:** a one-pread slab per expert, the async pool, PILOT
  prefetch, and batch union.
* **Routing-side levers (lossy, opt-in):** `CACHE_ROUTE=1` keeps the true top-J
  and fills the remaining slots with *resident* experts ranked inside the top-M
  (`docs/CACHE_ROUTE.md`, arXiv:2412.00099). `--topp` drops low-mass experts.
  `DEGRADE_ZERO` zero-fills low-gate misses when a prefetch deadline is missed.
  Each reports agreement/KL or drop counts so the quality cost stays visible.
* **CPU vs GPU for streamed experts:** `docs/cuda.md` is explicit:
  "Streaming experts deliberately remain on the original CPU path: copying an
  expert from NVMe to the GPU on every use would only replace the disk
  bottleneck with a PCIe bottleneck." The GPU only computes experts that are
  **resident** in VRAM. Several community datapoints show a GPU expert tier
  adding ≈0-6% when the CPU is fast (AVX-512) and the box is disk-bound
  (`docs/benchmarks.md`, rows #101 and the i9-12900K row).

---

## 5. Supported architectures

From `README.md` ("Other supported models") and the per-engine files:

| Architecture | Supported | Engine file | Backend(s) | Dense/MoE | Memory strategy | Special handling |
|---|---|---|---|---|---|---|
| GLM-5.2 / 5.3 (744B/40B active) | Yes (reference) | `c/colibri.c` | CPU, CUDA, HIP, Metal, Vulkan | MoE (256 routed + shared, top-8) | Dense int4 in RAM; experts streamed; pin, LRU, VRAM tier | MLA compressed KV (57× smaller) persisted to disk; DSA sparse attention; int8 MTP speculative head |
| GLM-5.3-Flash (321B) | Yes | `c/glm53.c` | CPU (+Metal) | MoE | Same | Vision tower; BF16 dense with load-time precision choice |
| Inkling (975B/41B) | Yes | `c/inkling.c` | CPU, CUDA (`backend_cuda_ink.cu`) | MoE | Same; bf16 dense is 49 GB, with an int4-dense tool down to 15 GB | Own IKU1 usage format |
| Kimi K3 (2.8T/104B) | Yes | `c/kimi_k3.c` | CPU, Metal, Vulkan | MoE (MXFP4 experts) | Streams native MXFP4 from the original HF shards | KDA + MLA; recurrent-state checkpoints |
| DeepSeek V4 Flash (284B/13B) | Yes | `c/deepseek_v4.c` | CPU, CUDA (`backend_cuda_dsv4.cu`) | MoE (native fp4 experts, fp8 dense) | Whole-process `RAM_GB` ceiling | MLA + DSA; DSpark/MTP drafts (off by default); prefix checkpoints |
| DeepSeek V4.1 Flash (552B/16B) | Yes | `c/deepseek_v41.c` | CPU (+CUDA) | MoE + a 203 GB n-gram memory | n-gram table read from disk a few hundred bytes at a time | Vision, tool calling |
| Qwen3.8-Flash-Next (125B + 51B n-gram) | Yes | `c/qwen38.c` | CPU, CUDA VRAM tier | MoE | PLE pageable; block-FP8 experts | Vision |
| Qwen3.6 35B-A3B | Yes | `c/qwen36.c`, `c/qwen36_tier.c` | CPU, CUDA (multi-GPU tier), Vulkan | MoE (hybrid Gated Attention + Gated DeltaNet) | **Full RAM residency required**; VRAM hot tier on top | Docs: 1.44 → 10.05 tok/s with two 8 GB GPUs |
| OLMoE 7B/1B | Yes | `c/olmoe.c` | CPU | MoE | int8 container (~7 GB) | Smallest supported model |
| Dense transformers (Llama, Qwen dense, Mistral, …) | **No** | — | — | Dense | — | No dense family has an engine. The streaming design assumes routed experts |

**Architecture-specific code:** essentially everything outside the shared
headers. Each engine owns its config parsing, attention variant (MLA, DSA,
DeltaNet, KDA), forward pass, and checkpoint naming
(`model.layers.%d.mlp.experts.%d.gate_proj.weight` is hard-coded in
`expert_load_impl`). The shared, generalizable parts are `st.h` (tensor I/O),
`quant.h` (decoders), `expert_ffn.h`, `expert_store.h` (cache ABI), `tier.h`
(placement policy), `route_trace.h` (usage history), `uring.h`, and `kv_prefix.h`.
The README states the rule: "One `.c` per model family, over shared single
headers."

---

## 6. Backend architecture

* **CPU:** hand-written kernels in C with OpenMP (`qgemv.h`, `idot.h`,
  `fused_simd.h`, `sse41_kernels.h`, AVX2/AVX-512/VNNI/NEON variants). Thread
  count is capped to physical cores (`omp_tune.h`), because SMT siblings regress
  memory-bound int4 kernels (`docs/tuning.md`). **No BLAS, ggml, or llama.cpp
  dependency.**
* **CUDA / HIP:** a single `c/backend_cuda.cu` compiled for either vendor
  through `c/backend_gpu_compat.h` (`GPU_BACKENDS.md`). It is used for resident
  dense tensors (`CUDA_DENSE`), the VRAM expert tier (`CUDA_EXPERT_GB`), and a
  "GPU-resident pipeline" (`COLI_CUDA_PIPE=2`) that keeps the residual stream
  on-device across layers. On Windows the backend ships as a DLL loaded at
  runtime (`c/backend_loader.c`).
* **Metal:** `c/backend_metal.mm`, using unified memory. It borrows llama.cpp's
  `newBufferWithBytesNoCopy` residency trick (README acknowledgements).
* **Vulkan:** `c/backend_vulkan.c` plus `c/shaders/`, for any Vulkan 1.2 GPU
  including AMD via RADV.
* **Existing inference engines:** none are linked. vLLM, transformers, and
  llama.cpp are used only as *references*: llama.cpp for the GBNF grammar subset
  and the Metal trick, transformers as the correctness oracle (CI
  teacher-forcing), and vLLM/SGLang as benchmark baselines
  (`docs/benchmarking.md`).
* **Front end:** Python. `c/coli` (the CLI), `c/openai_server.py` (a
  `ThreadingHTTPServer` gateway exposing `/v1/chat/completions`,
  `/v1/completions`, `/v1/models`, `/v1/messages`, `/health`, `/metrics`, plus
  Colibrì-specific `/v1/brio`), `c/resource_plan.py` (`coli plan`), and
  `c/autotune.py` (`coli tune`). The C engine speaks a line protocol to the
  gateway (`docs/serve_protocol.md`).

**What can be abstracted into a generic runtime layer:**
`ColiExpertStore` (lookup/lease/prefetch/stats), `tier.h`'s admission and
victim policies, the pipe/uring loader, the usage history, the planner's
budget arithmetic, and the RSS guard. They are already written to be
engine-agnostic and are the closest thing in the repository to the "memory
orchestrator" this project builds.

---

## 7. Performance characteristics

Quoted from `docs/benchmarks.md` and `docs/cuda.md`. These are Colibrì's
measurements, not ours.

| Model | Hardware | RAM / VRAM | Storage | Result |
|---|---|---|---|---|
| GLM-5.2 744B int4 (370 GB) | WSL2, 12 cores | 25 GB / — | VHDX on NVMe, ~1 GB/s | 0.05-0.1 tok/s cold; RSS ~20 GB; ~11 GB read/token |
| GLM-5.2 | Ryzen 9 9950X | 123 GB | QLC Gen3 1.5 GB/s → Samsung 9100 Gen5 8.8 GB/s | 0.10 → 0.28 tok/s; profile shifts from 66% disk to 57% matmul |
| GLM-5.2 | EPYC 7443, 430 GB RAM | 77.5 GB pin, hit 98% | ~1 GB/s | 1.00 tok/s, RAM-bandwidth and matmul bound |
| GLM-5.2 | Threadripper 7965WX | 123 GB | 1 vs 2 independent NVMe | 0.80 → 1.10 tok/s (+37.5%) |
| GLM-5.2 | 6× RTX 5090, 251 GB host | 176.7 GB VRAM tier + 191.3 GB RAM tier (full residency) | NVMe (out of the decode path) | 5.8-6.8 tok/s decode, TTFT ~13 s |
| GLM-5.2 | i9-12900K + RTX 3090 | 64 GB / 24 GB | 990 Pro 6.4 GB/s | 0.34 tok/s (MTP off); the GPU tier adds only 0-6% |
| Qwen3.6 35B-A3B | 2× 8 GB GPUs | full RAM residency | — | 1.44 → 10.05 tok/s with the VRAM expert tier |
| OLMoE 7B int8 | Apple M3, 16 GB | 1.5-1.8 GB RSS | 3.2 GB/s | 3.69 cold → 4.18 warm tok/s |
| DeepSeek V4 Flash | Ryzen 7 5800X | 32 GB | 980 PRO | 0.93 tok/s, hit 52.5% |

**Why it is achievable:**
1. **Bytes per token, not model size, sets speed.** Decode time per token is
   roughly `max(compute, Σ_miss bytes / disk_BW)`. MoE sparsity makes the bytes
   small: about 11 GB/token for 744B, and only on misses.
2. **Hit rate is the dominant variable.** Routing has exploitable structure:
   one-layer-ahead predictability (71.6%), and a skewed, workload-specific
   popularity measured as an "expert atlas" (issue #175). The learned pin set
   plus the LRU turns a 3% cold hit rate into 55-98%.
3. **Overlap converts serial latency into parallel bandwidth.** The pipe/uring
   loaders and batch union keep queue depth high, which NVMe needs to reach its
   rated bandwidth. The profiles separate "disk service" from "visible wait".
4. **Once disk is hidden, the bottleneck moves** to RAM bandwidth and CPU
   kernels (the 9950X and EPYC rows). Faster kernels then matter more than any
   tiering trick.

---

## 8. Lessons Kestrel adopts

1. **Budget on bytes moved per token.** The planner's cost model must be
   `bytes_from_tier / tier_bandwidth` per token for each tier, not "does it fit".
2. **Measured RSS beats projected RSS.** Kestrel enforces budgets at runtime
   (guard plus eviction), not only at plan time (issue #403).
3. **Keep the dense, always-touched set in the fastest tier that holds it.
   Stream only what is either sparse (experts) or unavoidable (the dense layers
   of an oversized dense model).**
4. **Do not stream through the GPU by default.** PCIe is a bottleneck too.
   Stream into RAM and compute there unless the planner measures that a GPU
   upload-plus-compute beats CPU compute for that tensor.
5. **The unit of transfer should match the access pattern.** That is one expert
   (coalesced) for MoE and one layer block for dense models. Kestrel lays out
   reads as coalesced extents.
6. **Prefetch must be measurable and switchable.** Colibrì found that WILLNEED
   hints became counter-productive under memory pressure and that PILOT can
   lose on disk-saturated hosts. Every Kestrel prefetch policy has an off switch
   and accuracy metrics (prefetch hits ÷ prefetches issued, and wasted bytes).
7. **Leases on cache entries.** Eviction must not race in-flight compute or I/O
   (`ColiExpertStore` contract, `ESlot.in_flight`).
8. **Hysteresis in promotion** (25% + constant), plus frequency-dominant LFRU
   scoring.
9. **Persist the learned hot set** across sessions, keyed to the model identity.
10. **Never silently change semantics.** Lossy levers are opt-in, labeled, and
    metered.

## 9. Where Kestrel departs from Colibrì

| Colibrì choice | Kestrel choice | Reason |
|---|---|---|
| One hand-written engine per model family | One generic graph/adapter layer; architectures are data-described adapters | Generality across 7B → 1T and dense ↔ MoE is the research question |
| Safetensors-only, custom converted containers | **GGUF first** (no conversion), safetensors later | GGUF is the dominant local format and already quantized |
| MoE-only streaming; dense weights must be resident (except the CPU-only trunk mmap knob) | Dense layers are first-class tier citizens (layer-granular streaming) | The MVP target is a 32B *dense* model on 8 GB VRAM without it all in RAM |
| Python planner and gateway, C engine | Rust for the planner, scheduler, server, and CLI; C/C++ only behind backend FFI | One process, typed budgets, memory safety in the orchestrator |
| Custom kernels for every backend | Reuse ggml/llama.cpp kernels via a backend adapter, plus a small native Rust CPU executor to prove the scheduler | Avoid re-implementing GPU kernels |
| Knobs are mostly environment variables | Typed execution plan; CLI flags override plan fields | Explainability (`kestrel plan`), reproducibility |

---

## 10. Capability comparison

Every Colibrì cell is backed by the evidence column.

| Capability | Colibrì | Evidence | Kestrel (target) | Kestrel MVP status |
|---|---|---|---|---|
| Small models (≤7B) | **Partial.** OLMoE 7B (MoE) only | README roster; `c/olmoe.c` | Yes | Yes (any llama-family GGUF) |
| Medium models (8-70B) | **Partial.** Qwen3.6 35B-A3B (MoE) only | README; `c/qwen36.c` | Yes | Dense llama/qwen2/qwen3/mistral GGUF |
| Large models (100B+) | **Yes** (MoE) | GLM, DeepSeek, Qwen3.8 | Yes | Planner yes; executor limited by kernels |
| Huge MoE (400B-3T) | **Yes** | GLM-5.2 744B, Inkling 975B, Kimi K3 2.8T | Yes | Expert-aware planning; MoE execution is phase 8 |
| Dense models | **No** | No dense engine in `c/`; streaming code is expert-keyed (`expert_load_impl`) | Yes | **Yes, the primary MVP target** |
| MoE | **Yes** | All nine families | Yes | Detection and planning yes; native execution later |
| VRAM tiering | **Yes** (experts + dense via `CUDA_DENSE`) | `docs/cuda.md` | Yes | Planned and delegated to the llama.cpp backend; no native CUDA in MVP |
| RAM tiering | **Yes** | `ecache`, pin | Yes | Yes |
| NVMe tiering | **Yes** (experts; dense only via `TRUNK_RESIDENT_LAYERS` mmap) | `c/st.h`, `c/colibri.c:2132` | Yes | Yes (layer-granular streaming of dense weights) |
| Dynamic scheduling | **Partial.** Per-token LRU and PILOT; LFRU repin at safe points; tier sizes fixed at start (RSS guard can only shrink) | `c/tier.h`, `rss_guard` | Yes | Adaptive LFRU cache with hysteresis, pressure-driven shrink |
| Automatic hardware optimization | **Partial.** `coli plan` sizes tiers from capacity; `coli tune` sweeps knobs with gates | `c/resource_plan.py`, `c/autotune.py`, `docs/tuning.md` | Yes | Measured micro-benchmarks feed a cost model that ranks strategies |
| Multiple inference backends | **Yes, own kernels** (CPU, CUDA, HIP, Metal, Vulkan); no third-party engine | `GPU_BACKENDS.md` | Yes (native and llama.cpp) | Native CPU executor plus llama.cpp adapter |
| OpenAI API | **Yes** | `c/openai_server.py` | Yes | Yes |
| Cross-platform | **Yes** (Linux, macOS, Windows CI matrix) | `.github/workflows/ci.yml` | Yes | Linux and Windows (no platform-specific code in the core) |
| GGUF | **No** | No reader in the tree | Yes | Yes |

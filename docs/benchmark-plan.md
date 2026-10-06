# Benchmark plan

The project must show that the scheduler adds measurable value, and that every
performance claim can be reproduced. This document is the methodology.
Results live in `benchmarks/results/` as raw JSON plus a summary.

## 1. Rules

1. **Report the machine.** CPU model, cores/threads, RAM size and type, GPU(s)
   and driver, storage device and file system, OS and kernel. Include the
   commit, model file (name, size, sha256 prefix), and exact command.
   `kestrel hardware --json` emits all of this.
2. **Control the cache state.** Name the state for every row: `cold` (page
   cache dropped, or O_DIRECT reads) or `warm`. Buffered-read numbers on a
   machine with spare RAM measure the page cache, not the disk. Colibrì learned
   this the hard way (their caveat #86).
3. **Change one variable at a time.** Interleave arms (A B A B …), never run
   blocks (A A A B B B), so thermal and background drift cancel out.
4. **Repeat.** At least 3 runs for each arm. Report the median and min-max.
5. **Check correctness alongside speed.** For every arm, greedy output must be
   token-identical to the reference arm on the same prompt. A fast wrong answer
   is a failure.
6. **Publish negative results.** If an arm loses, say so in the results file.

## 2. Metrics

| Metric | Definition | Source |
|---|---|---|
| decode tok/s | generated tokens ÷ decode wall time (excludes prefill) | engine metrics |
| TTFT | request start → first token | engine / server |
| prompt tok/s | prompt tokens ÷ prefill wall time | engine metrics |
| peak RSS | max VmHWM | `/proc/self/status` |
| VRAM used | NVML / nvidia-smi delta | backend |
| disk bytes read | bytes requested from the cold tier | IoEngine |
| disk bandwidth | bytes read ÷ disk service time | IoEngine |
| stall time | compute-thread time blocked waiting for weights | WeightStore |
| hit rate | bytes served from resident or ready slots ÷ bytes leased | WeightStore |
| prefetch accuracy | prefetched groups used before eviction ÷ prefetches issued | Prefetcher |
| prefetch lateness | leases that waited on an in-flight prefetch ÷ all leases | Prefetcher |
| transfer volume | bytes moved per token per tier pair | IoEngine |
| energy | RAPL `energy_uj` delta when readable (Linux, Intel/AMD) | optional |

## 3. Arms (from the project brief §19)

| # | Arm | How |
|---|---|---|
| 1 | Standard llama.cpp configuration | `llama-bench` / `llama-cli` with defaults (mmap, `-ngl` default) |
| 2 | CPU-only | llama.cpp `-ngl 0`, and Kestrel native `--strategy ram` |
| 3 | Normal GPU offload | llama.cpp with a hand-chosen `-ngl` (the user's usual practice) |
| 4 | Kestrel | `kestrel run` (automatic plan) |
| 5 | Kestrel without predictive prefetch | `--prefetch-depth 0` |
| 6 | Kestrel without NVMe tiering | `--no-stream` (resident only. Fails or OOMs when the model does not fit, which is the point) |
| 7 | Kestrel with page-cache (mmap-like) streaming | `--io buffered --prefetch-depth 0` |
| 8 | Kestrel with contiguous vs interleaved placement | `--placement contiguous` |
| 9 | Kestrel with LRU instead of static pin | `--policy lru` (shows the cyclic-scan cliff) |

Arms 1 and 3 need llama.cpp binaries (`KESTREL_LLAMA_BENCH`). Arms 3 and 4 need
a GPU for the GPU portion.

## 4. Workloads

* **Decode:** 128-token generation from a 32-token prompt, greedy.
* **Prefill:** 512- and 2048-token prompts, first-token latency.
* **Context sweep:** n_ctx ∈ {2k, 8k, 32k} at fixed model, for KV placement
  (research question 10).
* **Memory sweep:** RAM budget from "everything resident" down to "two layers
  resident", which traces the throughput curve from RAM-only to NVMe-streaming.
  This is the central experiment for research questions 1, 2, 8, and 12.

## 5. Mapping to research questions

| RQ | Question | Experiment |
|---|---|---|
| 1 | When does NVMe streaming beat CPU-only inference? | Memory sweep: compare streaming at budget B against the next-smaller model that fits in B, and against CPU-only with swap |
| 2 | At what size does RAM+VRAM beat pure CPU? | Arms 2 vs 3/4 across 7B/14B/32B (GPU host) |
| 3 | How much does prefetching help? | Arm 4 vs 5 across the memory sweep |
| 4 | How large should the VRAM safety margin be? | Margin sweep {256 MiB … 1.5 GiB} × context; record OOM or fragmentation failures (GPU host) |
| 5 | Which tensors benefit most from VRAM? | Attention-only vs FFN-only vs experts-on-CPU placements (llama.cpp `-ot`) at fixed VRAM |
| 6 | Layer- vs tensor-level scheduling? | `--granularity layer` vs `attn-ffn` in the memory sweep |
| 7 | Cost of dynamic migration? | Time to promote or demote one layer; tok/s dip during migration |
| 8 | How does NVMe latency affect generation? | O_DIRECT vs buffered; I/O worker count sweep; chunk-size sweep |
| 9 | How does batch size change placement? | Prefill (S=512) vs decode (S=1) tok/s across the memory sweep. Streaming amortizes over S |
| 10 | How does context length change KV placement? | Context sweep with KV in VRAM vs RAM (GPU host) |
| 11 | How predictable are MoE expert accesses? | Routing trace: one-layer-ahead recall, reuse distance, popularity skew (MoE phase) |
| 12 | When does NVMe streaming become counterproductive? | Memory sweep tail: tok/s vs `disk_bw / streamed_bytes_per_token`; compare against a smaller quant that fits |

## 6. Expected outcomes and falsification

These outcomes follow from bandwidth arithmetic. They are **hypotheses, not
results**:

* **Dense streaming is bandwidth-bound by construction.** For dense models,
  `tok/s ≤ disk_bw / streamed_bytes_per_token`. A 70B Q4 with 10 GB streamed
  per token on a 7 GB/s NVMe is capped at 0.7 tok/s. Prefetch cannot exceed
  that ceiling. It can only approach it, by hiding compute behind I/O. The
  hypothesis that "Kestrel streaming ≈ min(compute, disk) bound" is falsified
  if measured tok/s is far below `1 / max(t_compute, t_disk)`.
* **Explicit streaming vs mmap paging.** We expect explicit prefetch with
  O_DIRECT to beat demand paging (page faults at 4 KiB-128 KiB readahead
  granularity with no lookahead), with the gap largest at low queue depth. This
  is falsified if `--io buffered --prefetch-depth 0` matches arm 4.
* **MoE is where tiering pays.** The hit rate × bytes argument (Colibrì §7)
  predicts large wins for MoE, small ones for dense.

## 7. Harness

```bash
kestrel benchmark model.gguf --arms auto --runs 3 --tokens 128 --json out.json
benchmarks/run_memory_sweep.sh model.gguf     # budget sweep, interleaved arms
benchmarks/compare_llamacpp.sh model.gguf     # arms 1-3 via llama-bench
```

Each run writes a self-describing JSON record (hardware, plan, metrics, output
hash) to `benchmarks/results/`. `benchmarks/README.md` explains how to add
community datapoints.

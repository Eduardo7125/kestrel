# Benchmarks

Methodology: [../docs/benchmark-plan.md](../docs/benchmark-plan.md). Each
native arm runs in a fresh process (so peak RSS is per arm). Arms are
interleaved across repetitions, the model's page-cache pages are dropped
before every run, and every arm's greedy output is compared token by token
with the first arm's. All arms below produced **identical output**.

```bash
kestrel benchmark model.gguf --runs 3 --tokens 32 --stream-fraction 0.5 --json out.json
benchmarks/run_memory_sweep.sh model.gguf 2 24
```

## Environment of the recorded runs (read this first)

| | |
|---|---|
| Machine | cloud VM: Intel Xeon @ 2.80 GHz, 4 cores (AVX2, AVX-512 VNNI), 16.9 GB RAM, **no GPU** |
| RAM bandwidth (measured) | 26.1 GB/s (4 threads), 7.8 GB/s (1 thread) |
| Disk (measured) | virtio block device, ext4. **1.33 GB/s O_DIRECT** (4 MiB random reads, QD8), 58 µs 4K latency. Not a local NVMe: the host may cache, and the effective streaming bandwidth in-run reached 1-12 GB/s (see the `disk BW` column). Absolute streaming numbers are therefore **not** representative of a real NVMe drive. The relative comparisons between arms are. |
| Models | **Synthetic** GGUFs from `tools/make_synthetic.py`: real llama shapes (TinyLlama-1.1B: 22 layers, d=2048, ff=5632, GQA 32/4; Llama-2-7B: 32 layers, d=4096, ff=11008), Q4_K weights with a Q6_K head, random values. Hugging Face was unreachable from this environment. Output text is meaningless; speed, memory and I/O behaviour are what is measured. |
| Commit | `99b2939` (benchmark harness), llama.cpp `43fe9c6` (CPU build) |

Treat every number here as one datapoint from one virtual machine.
Contributions from real hardware are what this directory is for.

## 1. Strategy comparison: 1.1B Q4_K, half of the layer bytes streamed

3 interleaved runs × 32 generated tokens, 32-token prompt, 4 threads. The
streaming arms share a RAM budget of 710 MB (weights are 636 MB, including
545 MB of layer weights). Raw data: `results/syn-1b-q4k-arms.json`.

| Arm | Decode tok/s (median [min-max]) | Peak RSS | Streamed / token | Hit rate | Stall / token |
|---|---|---|---|---|---|
| `resident` (everything in RAM) | **9.79** [8.55-10.46] | 671 MB | 0 | 100% | 0 |
| `kestrel` (interleaved pin + depth-2 prefetch, O_DIRECT) | **7.52** [6.74-7.59] | **455 MB** | 273 MB | 83% | 0.030 s |
| `no-prefetch` (same, depth 0) | 4.60 [4.39-4.67] | 456 MB | 234 MB | 40% | 0.109 s |
| `contiguous` (streamed layers back to back) | 5.99 [5.33-6.20] | 456 MB | 273 MB | 74% | 0.055 s |
| `belady-cache` (same RAM as a Belady cache, no pin) | 3.84 [3.63-4.00] | 456 MB | 545 MB | 62% | 0.138 s |
| `lru-cache` (same RAM as an LRU cache) | 2.20 [2.14-2.31] | 456 MB | 545 MB | **6%** | 0.342 s |
| `page-cache` (buffered, no prefetch: ≈ mmap) | 6.92 [6.44-6.92] | 456 MB **+ 605 MB page cache** | 234 MB | 40% | 0.079 s |
| `llamacpp-cpu` (llama-bench, mmap, own kernels) | 26.98 [26.56-27.80] | not measured | — | — | — |

### What this shows

1. **Streaming trades speed for memory, predictably.** Streaming half of the
   layers cut peak RSS by 32% (671 → 455 MB) for a 23% decode slowdown, with
   identical output.
2. **Prefetch matters: +63%** decode at equal RAM (7.52 vs 4.60 tok/s). Stall
   per token fell from 109 ms to 30 ms. Prefetch accuracy is 100% by
   construction (the dense layer order is known). The remaining stall comes
   from late prefetches (22% of streamed leases waited), meaning the disk is
   slower than the compute it hides behind.
3. **Interleaving matters: +26%** over contiguous placement at the same
   streamed bytes (7.52 vs 5.99). Each load gets resident compute to hide
   behind, as predicted in `docs/scheduler-design.md` §3.3.
4. **Static pinning beats caching for dense layers.** The same RAM used as an
   LRU cache gives a **6% hit rate** (the cyclic-scan pathology) and 2.20
   tok/s. As a Belady cache it gives 62% hits and 3.84 tok/s. Kestrel's
   static pin plus a minimal ring reaches 7.52 tok/s: 3.4× LRU and 2× Belady
   at equal memory. The Belady cache streams *all* layers through slots and
   hides less I/O behind compute. Fixed residency plus prefetch wins.
5. **The OS page cache hides memory: a negative result for "just use mmap".**
   `page-cache` looks competitive (6.92 tok/s) only because the OS cached
   **605 MB** of the model during the run, outside Kestrel's budget and
   invisible to process RSS. On a machine without that spare RAM, the page
   cache cannot help and this arm degrades to `no-prefetch` or worse. This is
   why Kestrel streams with `O_DIRECT` and accounts for what it holds.
6. **Kernel gap (a negative result for the native executor).** llama.cpp's
   CPU kernels are 2.8× faster at decode (26.98 vs 9.79 tok/s fully resident)
   and about 8× faster at prefill. The scheduling effects above are measured
   *within* Kestrel's executor. Closing the kernel gap is roadmap item 1, and
   it will make streaming relatively *more* disk-bound.

## 2. Memory sweep: 1.1B Q4_K

2 interleaved runs × 24 tokens per point (the median of 2 is the faster run).
Raw data: `results/syn-1b-q4k-sweep-*.json`.

| Layer bytes streamed | RAM budget | Peak RSS | `kestrel` tok/s | `no-prefetch` tok/s | Prefetch gain | Hit rate (kestrel) |
|---|---|---|---|---|---|---|
| 0% (resident) | — | 671 MB | 9.79 | — | — | 100% |
| 25% | 847 MB* | 591 MB | 10.50 | 6.51 | +61% | 100% |
| 50% | 710 MB | 456 MB | 6.92 | 3.47 | +99% | 84% |
| 75% | 574 MB | 319 MB | 3.47 | 2.14 | +62% | 72% |
| 90% | 492 MB | 222 MB | 1.98 | 1.68 | +18% | 41% |
| 100% | 438 MB | 183 MB | 1.90 | 1.61 | +18% | 23% |

\*The budget includes the runtime overhead estimate (256 MB + activations);
RSS stays below it in every arm.

* **At 25%, streaming was free.** Prefetch hid all I/O (100% hit, zero stall)
  and speed matched resident within noise, for 12% less RSS.
* **Prefetch helps most in the middle of the curve.** When compute covers
  most of the I/O, prefetch hides it. When almost everything streams, the
  disk is the bottleneck and prefetch can only overlap the remainder (+18%).
* **Fully streamed, the model runs in 27% of its resident memory** (183 vs
  671 MB) at 19% of the speed. Falsification check from the benchmark plan:
  at 100% the bound is `1 / max(t_compute, t_disk)`, with
  `t_compute = 0.10 s` (resident decode) and `t_disk = 545 MB / 1.33 GB/s =
  0.41 s`. That gives 2.44 tok/s. Kestrel reached **1.90 tok/s, 78% of the
  bandwidth bound**, so the hypothesis "Kestrel streaming ≈ bandwidth bound"
  holds on this machine with ~22% overhead (first-layer latency, KV and head
  work not overlapped).

## 3. 7B Q4_K (Llama-2-7B shape), half of the layer bytes streamed

2 interleaved runs × 12 tokens. Weights are 3.83 GB; the streaming arms share
a 2.78 GB budget. Raw data: `results/syn-7b-q4k-arms.json`.

| Arm | Decode tok/s | Peak RSS | Streamed / token | Hit | Stall / token |
|---|---|---|---|---|---|
| `resident` | 2.18 [2.07-2.18] | 4.12 GB | 0 | 100% | 0 |
| `kestrel` | **1.46** [1.46-1.46] | **2.45 GB** | 1.89 GB | 77% | 0.174 s |
| `no-prefetch` | 0.87 [0.79-0.87] | 2.45 GB | 1.74 GB | 42% | 0.769 s |
| `page-cache` | 1.47 [1.46-1.47] | 2.45 GB **+ 3.65 GB page cache** | 1.74 GB | 42% | 0.582 s |
| `llamacpp-cpu` | 4.81 [4.68-4.81] | not measured | — | — | — |

The same pattern holds at 7B:

* Streaming saves 41% of RSS (4.12 → 2.45 GB) at 67% of resident speed.
* Prefetch is worth **+68%** at equal RAM.
* `page-cache` matches Kestrel only by consuming 3.65 GB of page cache (the
  whole model) outside the budget.
* The kernel gap to llama.cpp is 2.2× at decode. Native prefill (2.9 tok/s)
  is ~8× slower than llama.cpp; the native prefill path is the weakest part
  of the executor.

## Research questions: status after these runs

| RQ | Status |
|---|---|
| 1. When does NVMe streaming beat CPU-only inference? | Partially answered. Streaming lets a model run in 27-68% of its resident RAM at 19-100% of resident speed, depending on the streamed fraction. Comparing against a *smaller quant that fits* needs real models |
| 3. How much does prefetching improve throughput? | **+18% to +99%** at equal RAM, largest when compute can cover the I/O |
| 6. Layer- vs tensor-level scheduling? | Not yet: groups are attention/FFN halves; a granularity switch is pending |
| 8. How does NVMe latency affect generation? | Indirectly: late prefetches explain the remaining stall. Needs real NVMe and the I/O-worker sweep |
| 12. When does streaming become counter-productive? | On this disk, beyond ~75% streamed, decode falls below a fifth of resident speed. The planner reports the disk bound in every plan that streams |
| 2, 4, 5, 9, 10 | Need a GPU host |
| 7 | Migration cost: rebalancer implemented; not yet measured |
| 11 | MoE predictability: phase 8 |

## Minimum report for community datapoints

`kestrel hardware --bench --json` output, the model file (name, size, quant),
the exact `kestrel benchmark` command, and the generated JSON. Report negative
results too.

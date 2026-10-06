# Scheduler design

The scheduler is the set of policies that decide **where** each tensor group
lives, **when** it moves, and **what is evicted**. It is built in the order the
project's engineering principle prescribes:

```text
correctness → memory accounting → static tiering → prefetching → caching
            → dynamic scheduling → autotuning → MoE optimization → multi-GPU
```

Each stage is a separate, switchable policy with its own metrics, so every
stage can be benchmarked against the one before it.

## 1. Access-pattern classes

The design rests on one observation: **dense layers and MoE experts have
different access patterns, so they need different cache policies.**

| Class | Example | Pattern | Right policy |
|---|---|---|---|
| **Cyclic-deterministic** | Dense layer `i` during decode | Every token touches layers 0…L-1 in order. The next access to layer `i` is exactly one token later. | **Static pin + streaming ring.** Never LRU. |
| **Stochastic-skewed** | MoE expert `(layer, e)` | Data-dependent, with skewed popularity and partial predictability. | **LFRU cache + predictive prefetch.** |
| **Always-hot** | Norms, router, shared expert, output head | Every token | Resident in the fastest tier that fits |
| **Sparse-gather** | `token_embd` rows | One row per token | Resident in RAM. A row is cheap to fetch, and the GPU rarely benefits. |

### Why LRU fails for dense layers

With a cyclic scan over L layers and a cache of C < L layers, LRU evicts exactly
the layer that will be needed soonest. **The hit rate is 0%** no matter how
close C is to L. Belady's optimal policy for a loop keeps a fixed subset of C
layers and streams the other L−C every token, giving a C/L hit rate. Kestrel
implements that optimum directly:

* The planner **pins** C layers resident (RAM or VRAM).
* The other L−C layers flow through a **ring of `d+1` slots** (d = prefetch
  depth). The ring does not try to cache. It exists only to overlap I/O with
  compute.

Which layers to pin does not matter for the hit rate, since all dense layers
are touched once per token. It does matter for overlap: spreading the streamed
layers evenly between resident ones gives the I/O engine more compute time to
hide each load behind. The planner interleaves them (see §3.3). The
`kestrel benchmark --policy lru` ablation exists to demonstrate the 0% hit-rate
cliff.

## 2. Static placement (Phase 4)

Input: `ModelDesc`, `HardwareProfile`, budgets. Output: a tier for each group.

```text
1. reserve safety margins and fixed overheads (KV, compute buffers, scratch)
2. compute for every group g:
      bytes(g), touch_prob(g)            # 1.0 dense, p(e) for experts (uniform k/E,
                                         #   or from usage history if present)
      gain_vram(g) = touch_prob × bytes × (1/bw_ram_compute − 1/bw_vram)
      gain_ram(g)  = touch_prob × bytes × (1/bw_disk − 1/bw_ram_compute)
3. VRAM: if the backend can execute on GPU, greedily take groups by
         gain_vram / bytes until usable VRAM is exhausted, respecting
         backend granularity (llama.cpp: whole layers via n_gpu_layers,
         then per-tensor overrides for experts)
4. RAM:  greedily take the remaining groups by gain_ram / bytes until
         usable RAM minus ring/cache reservations is exhausted
5. rest → NVMe (streamed)
```

For dense models, `gain/bytes` is equal for every layer, so the result is
"first N layers on GPU, next M resident in RAM, rest streamed". The ordering
inside the RAM tier is chosen for overlap (§3.3). For MoE, `touch_prob` makes
shared and attention tensors outrank experts, and hot experts outrank cold ones.
This is the generalization of llama.cpp users' manual `-ot exps=CPU` recipe and
of Colibrì's pin set.

## 3. Streaming and prefetch (Phases 5-6)

### 3.1 The I/O engine

* A worker pool (default `min(8, 2 × physical cores)`) serves `ReadRequest
  { extents, dst slot, ticket }`. NVMe needs queue depth to reach rated
  bandwidth, and a single synchronous reader gets a fraction of it.
* Each group's extents are split into ≤ 8 MiB chunks so one large layer is read
  in parallel by several workers.
* Completion is a ticket: `Ticket::wait()` blocks the compute thread only if the
  data is not yet there, and the wait time is recorded as **stall time**. Disk
  service time and visible stall are reported separately, as Colibrì does.

### 3.2 Deterministic prefetch (dense)

Before computing layer `i`, the executor calls `store.prefetch_after(i, d)`.
That issues loads for the next `d` *streamed* groups in execution order,
wrapping around to layer 0 of the next token. The ring has `d+1` slots: one
being computed, `d` in flight or ready.

* `d = 0` gives synchronous streaming. This is the "no prefetch" benchmark arm.
* `d ≥ 1` overlaps. The planner chooses `d` as the smallest depth for which
  `d × compute_time_between_streamed_layers ≥ load_time(layer)`, capped by the
  RAM left for the ring.

Prefetch accuracy is 100% by construction for dense layers. The relevant
metric is **lateness**: the fraction of leases that had to wait, and how long.

### 3.3 Interleaving

If R layers are resident and S are streamed, with R ≥ S, placing the streamed
layers evenly (for example `R S R R S R …`) gives each load the compute time of
the resident layers between them. Placing them contiguously (`R R R … S S S`)
gives back-to-back loads with nothing to hide behind except the streamed
layers' own compute. Per-token time:

```text
t_token ≈ Σ compute(all layers) + max(0, Σ load(S) − Σ compute(overlappable))
```

Interleaving maximizes the overlappable compute for each load. Benchmark arm:
`--placement contiguous|interleaved`.

### 3.4 Predictive prefetch (MoE, Phase 8): implemented

* **Usage-history prior:** per-(layer, expert) counts are saved after every
  request to `<model>.kestrel-usage.json`, keyed by the model fingerprint. At
  the next start the hottest experts are preloaded up to cache capacity, and
  their heat is seeded (warm start).
* **Router lookahead** (Colibrì's PILOT): after layer `l`'s attention, layer
  `l+1`'s `ffn_norm` and router are applied to the current residual stream,
  and the predicted top-k experts are prefetched (decode only, `S ≤ 4`).
  Predictions the cache would admit load into the cache. The others load into
  a small **speculative pool** that never displaces a cached expert, which is
  what makes lookahead useful when the cache is full. Recall is measured on
  every MoE layer (`lookahead_recall`). `KESTREL_LOOKAHEAD=0` disables it.
* **Batch union:** a layer's routed experts are requested together
  (`ExpertStore::request`), so all misses load in parallel. Cached experts are
  computed first while the missing ones arrive. Each expert runs once for all
  rows routed to it.

## 4. Caching (Phase 7)

### 4.1 Layer cache

There is no LRU for dense layers (see §1). The "layer cache" is the resident
set plus the ring, sized by the planner and adjusted by the Rebalancer.

### 4.2 Expert cache (LFRU with leases): implemented in `kestrel-memory/src/expert.rs`

The unit is one routed expert: its slices of the stacked `ffn_{gate,up,down}_exps`
tensors, read as three extents into one aligned buffer.

```text
score(e) = (heat(e) << 8) | recency(e)          recency = max(0, 255 − age)
victim   = argmin score among unleased, fully loaded entries
admit x over victim v only if  score(x) > score(v) + score(v)/4 + 4·256   (hysteresis)
heat decays by halving after every request
```

* Capacity (in experts) comes from the plan: RAM left after the dense part,
  the KV cache, scratch and the speculative pool.
* Leases are `Arc`s. An entry is evictable only when its `Arc` is unshared and
  its load has completed.
* Experts that are not admitted are served through scratch buffers. A
  one-off expert never evicts a hot one.
* When the whole expert set fits, it is preloaded at startup (resident).
* `--expert-policy lru` is the ablation.

### 4.3 KV cache

KV is allocated once for `n_ctx` at plan time, in the tier of its layer, and is
not evicted. Context-length planning happens in the planner (§2).

## 5. Dynamic scheduling (Phase 7)

The **Rebalancer** runs at safe points every `rebalance_interval` tokens
(default 16):

| Signal | Action |
|---|---|
| RSS over limit, or available memory below half the safety margin | Demote the resident layer with the lowest interleave value, free its buffer, and shrink the RAM limit (with hysteresis) |
| Sustained headroom ≥ one layer's bytes plus the safety margin (k consecutive checks) | Promote the streamed layer whose removal most reduces stall time (the one with the least compute before it) |
| Stall fraction > threshold and RAM ring has headroom | Increase prefetch depth `d` |
| Expert heat shift (MoE) | LFRU swap (≤ 4 per interval, with hysteresis) |

Promotion happens without stopping inference. The layer is read into a new
buffer by the I/O engine in the background, and the placement table entry
flips at the next safe point. Demotion is immediate, because the GGUF file
still has the bytes.

## 6. Autotuning (Phase 7+)

`kestrel benchmark --tune` runs short decode trials over a bounded candidate
set: prefetch depth, I/O mode (direct/buffered), I/O workers, compute threads,
and placement order. Each candidate must produce **bit-identical greedy tokens**
to the baseline, and is kept only if median tok/s improves by ≥ 3% (Colibrì's
`coli tune` gate). The search is coordinate descent over threads, prefetch
depth and I/O workers. The streaming axes are skipped when the plan keeps
every weight resident. A winner is rerun before the baseline (reverse order)
and kept only if it still wins. Profiles live in
`~/.cache/kestrel/tuning/<hardware>-<model>.json`. `run`, `serve` and `plan`
apply them unless a flag sets the knob explicitly; `--no-tune-profile`
ignores them.

## 7. Multi-GPU (Phase 9)

The data model is already per-device: `Tier::Vram(device)`. Strategies to add:
layer split (pipeline), expert distribution (each expert has one home device,
`e mod n` as in Colibrì's qwen36 tier, or least-loaded), and tensor-parallel
through the llama.cpp adapter's `--split-mode row`.

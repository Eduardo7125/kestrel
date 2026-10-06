# Memory model

This document defines how Kestrel accounts for memory, how it sizes each tier,
and how the plan stays safe when the projection turns out to be wrong.

## 1. Tiers

| Tier | Name | Holds | Access cost model |
|---|---|---|---|
| T0 | **VRAM** (hot) | Weights the GPU computes on, the KV cache of GPU layers, compute buffers | `bytes / vram_bw` (no transfer if resident) |
| T1 | **RAM** (warm) | CPU-computed resident weights, the streaming slot ring, the expert cache, the CPU KV cache, scratch | `bytes / ram_bw` |
| T2 | **NVMe** (cold) | The model file itself (GGUF is read in place) | `bytes / disk_bw` (+ latency × requests ÷ queue depth) |

Model files are never copied into a private disk cache in the MVP. The GGUF
file *is* the cold tier. A disk cache (`--disk-cache`) is reserved for future
format conversions and re-packed expert layouts.

## 2. Units of placement

Kestrel places **tensor groups**, not single tensors and not whole models:

| Group | Contents | Typical size (32B Q4_K_M) | Access pattern |
|---|---|---|---|
| `global.embed` | `token_embd` | ~0.4 GB | Row gather per token (sparse) |
| `global.head` | `output_norm`, `output` | ~0.4 GB | Dense, every token |
| `layer[i].attn` | `attn_norm`, `attn_q/k/v/o` (+bias, q/k norms) | ~0.1 GB | Dense, every token, cyclic |
| `layer[i].ffn` | `ffn_norm`, `ffn_gate/up/down` | ~0.3 GB | Dense, every token, cyclic |
| `layer[i].expert[e]` | Expert *e*'s slices of `ffn_{gate,up,down}_exps` | MB-scale | Sparse, data-dependent |
| `layer[i].shared` | Shared expert and router (`ffn_gate_inp`) | small | Dense, every token |

Each group is a list of **extents** `(file_offset, length)`. Extents that are
adjacent in the file are coalesced into one read. In GGUF a layer's tensors are
usually contiguous, so a layer group is 1-3 reads. Expert slices are strided
(GGUF stacks experts per tensor), so one expert is 3 reads, one per stacked
tensor.

Why groups: a per-tensor policy gives the scheduler more freedom than it can
use. A dense layer is consumed as a unit, and attention and FFN have different
compute/byte ratios and different GPU benefits. Experts are the natural unit
for MoE (Colibrì's choice, adopted here). Research question 6 ("layer-level vs
tensor-level") is benchmarked by toggling between `layer` and `attn/ffn`
granularity.

## 3. Budget derivation

```text
usable_T = available_T − safety_T − overhead_T
```

| Tier | `available` | `safety` (default) | `overhead` |
|---|---|---|---|
| VRAM | free VRAM reported by the driver (NVML / nvidia-smi) | `max(512 MiB, 8% of total)` | backend context + compute buffers (estimated from `n_ctx`, `n_embd`, `n_vocab`, batch) |
| RAM | `MemAvailable` (Linux), `ullAvailPhys` (Windows), `vm_stat` reclaimable (macOS) | `max(1.5 GiB, 10% of total)` | runtime + tokenizer + scratch (activations, logits) |
| Disk | free space on the cache volume | 5% | — |

All three are overridable: `--vram-budget 7G --ram-budget 24G --disk-cache 200G`.
An explicit budget is clamped to the measured available memory unless
`--allow-overcommit` is passed. This mirrors Colibrì's refuse-to-start rule and
its `COLI_RAM_OVERCOMMIT` escape hatch.

### Example: the MVP target

```text
GPU  RTX 4060 8 GB, free 7.6 GB      RAM  32 GB, available 27 GB      NVMe 7 GB/s
model Qwen2.5-32B Q4_K_M: 18.5 GB weights, KV 4096 ctx f16 = 1.0 GB

VRAM  usable = 7.6 − 0.66 (safety) − 0.55 (compute bufs) = 6.4 GB
      KV for GPU layers + attention/FFN of the first N layers
RAM   usable = 27 − 3.2 (safety) − 0.4 (runtime) = 23.4 GB
      remaining layers resident: fits → no NVMe streaming needed
```

When a model fits in VRAM+RAM, NVMe streaming is pure overhead. The planner
only selects it when `weights > usable_vram + usable_ram`, for example a 70B
Q4 (40 GB) on the same machine. Streaming does not reduce compute. It trades
bandwidth for capacity, and the plan says so.

## 4. KV cache

```text
kv_bytes = 2 × n_layer × n_ctx × n_head_kv × head_dim × bytes(kv_type)
```

* The default type is f16. q8_0 is offered as a remedy (it halves KV) and is
  only applied when requested (`--kv-type q8_0`) or when the backend supports
  it and the user selected a lossy policy.
* Placement follows the attention computation: KV for GPU layers goes in VRAM,
  KV for CPU layers in RAM. The KV cache is never streamed to NVMe in the MVP.
  Streaming KV per token costs more than recomputation for short contexts.
* `--kv-cache 2G` caps KV bytes. The planner derives the maximum `n_ctx` from
  it and reports the result.

## 5. The Ledger

```rust
pub struct Ledger { tiers: [TierAccount; 3] }
pub struct TierAccount { limit: u64, reserved: AtomicU64, peak: AtomicU64 }
pub struct Reservation { tier: Tier, bytes: u64, ledger: Arc<Ledger> } // Drop → release
```

* `Ledger::reserve(tier, bytes, purpose) -> Result<Reservation, BudgetError>`
  returns an error rather than allocating past the limit. Every owned weight
  buffer, slot, expert slab, and KV block carries a `Reservation`, and dropping
  the buffer releases it.
* `purpose` is a label (`weights`, `stream-ring`, `kv`, `expert-cache`,
  `scratch`) used in metrics and in OOM diagnostics ("RAM budget exhausted:
  23.1/23.4 GB reserved; weights 18.2, kv 1.0, ring 3.6, scratch 0.3").

## 6. Guards (measured beats projected)

At safe points (between tokens, with no leases held), the **MemoryGuard**:

1. Samples process RSS (`/proc/self/statm`, or `GetProcessMemoryInfo` on
   Windows) and system available memory.
2. If `RSS > ram_limit × 1.02 + 256 MiB` or `available < safety / 2`, it asks
   the Rebalancer to **demote**: drop the least valuable resident groups to the
   streamed tier, free their buffers, and lower the ring or cache ceiling so
   they cannot regrow.
3. If headroom has been above `promote_threshold` for *k* consecutive checks,
   it **restores** the groups it demoted, in the background. By default that
   is all: the plan is static, and the guard only protects it. With `--adapt`
   it also promotes streamed groups the plan never made resident (most
   compute-to-hide-behind first).

Demotion is always possible because the cold tier, the model file, still holds
every byte. Kestrel never writes weights back to disk.

## 7. Page cache

Buffered reads put streamed bytes into the OS page cache. That is a second,
invisible copy, and it can push the system into swap while Kestrel's own
accounting looks fine. Kestrel therefore:

* Uses `O_DIRECT` (Linux) for streamed groups when the file system supports it.
  Buffers are 4 KiB-aligned, and extents are widened to alignment with the
  slack discarded. On Windows, `FILE_FLAG_NO_BUFFERING` is the equivalent
  (planned; the MVP uses buffered reads there).
* Falls back to buffered reads followed by `posix_fadvise(POSIX_FADV_DONTNEED)`
  when `O_DIRECT` is refused.
* Exposes `--io buffered|direct|auto` so the trade-off is measurable. Colibrì
  found `O_DIRECT` drive-dependent: +34-65% on some NVMe and neutral or
  negative on DRAM-less, QLC, and virtualized disks.

Resident groups are loaded once with the same reader, so the page cache does
not retain them either.

## 8. Accounting invariants (tested)

1. `Σ reservations(T) ≤ limit(T)` at all times, for every tier.
2. Every lease holds a reference to a buffer that cannot be freed or reused
   until the lease drops. Eviction skips leased slots.
3. Bytes served = bytes read from disk + bytes hit in RAM, per group, so hit
   rates are derived rather than estimated.
4. The plan's predicted resident bytes equal the Ledger's reserved weight bytes
   after load, within alignment slack.

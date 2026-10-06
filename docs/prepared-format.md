# Prepared models (`kestrel prepare`)

```bash
kestrel prepare model.gguf            # writes model.kgguf next to it, verified
kestrel run model                     # name resolution prefers model.kgguf
```

`kestrel prepare` rewrites a GGUF **once** into a container laid out for
Kestrel's I/O. It does work that would otherwise be paid on every load or
every token, and that no runtime policy can do: it changes where the bytes
sit in the file.

## What changes, and what does not

The transformation is **lossless**. Every tensor keeps its name, type, shape
and bytes. With verification on (the default), both files are re-read after
writing and every tensor is compared slice by slice. The prepared file is only
renamed into place if all of them match.

| | Plain GGUF | Prepared container |
|---|---|---|
| Routed experts | Three stacked tensors per layer (`ffn_{gate,up,down}_exps`): one expert is **3 scattered reads** | Expert `e` is `[gate_e │ up_e │ down_e]`, padded to 4 KiB: **1 contiguous, page-aligned read** |
| Tensor groups | Usually in layer order; may be split into several extents | Execution order; every group starts on a 4 KiB boundary and is **one extent** |
| Data section | 32-byte aligned | 4 KiB aligned (direct I/O with no partial pages) |
| Readable by | Every GGUF reader | Kestrel's native backend only |

Quantization does not change. Choosing a different quantization per tensor
for the hardware would change the model's output, so it is not part of
`prepare`. If it is ever added, it will be an explicit, separately measured
option with a quality check (perplexity against the source).

## Format

A prepared container is GGUF v3 with three differences:

1. **Magic `KGUF`** instead of `GGUF`. Other GGUF readers refuse the file
   instead of misreading it. llama.cpp keeps using the source GGUF, and the
   planner marks llama.cpp strategies infeasible for a prepared file.
2. **Strided tensors.** `kestrel.stride.<tensor name>` = `S` (u64) means the
   tensor is split along its outermost dimension into `dims[-1]` equal slices,
   and slice `i` starts at `offset + i·S` instead of `offset + i·slice_bytes`.
   The three expert tensors of a layer share one stride and start at different
   offsets inside the first expert's block, so they interleave. A stride key
   in a plain `GGUF` file is rejected as corrupt.
3. **Provenance.** `kestrel.prepared.version`, `kestrel.prepared.source` (file
   name) and `kestrel.prepared.source_fingerprint`. A `kestrel.pad` string
   entry pads the header so the data section starts on a page.

The model fingerprint changes with the layout, so tuning profiles
(`benchmark --tune`) are per file. `prepare` copies the source's expert
usage history to the new file, so warm starts carry over.

## Cost

* **Disk:** a second copy of the model while both exist (4.3 GB for the
  synthetic 7.6B MoE). Keep the source for llama.cpp, or delete it.
* **Time:** one sequential read and write, plus a re-read of both files for
  verification. That was 60 s for 4.3 GB on the test VM, bounded by its disk.
* **Machine binding:** none in version 1. The layout does not depend on the
  hardware, so a prepared file works on any machine. A plan-specific order
  (resident groups first, streamed groups together) would bind the file to
  one RAM size. It was left out, because interleaved streaming already hides
  dense loads.

## Measured effect

See [benchmarks §7](../benchmarks/README.md#7-prepared-models-packed-experts-kestrel-prepare).
On two MoE shapes (7.6B Mixtral-style, and Qwen3-30B-A3B's expert geometry)
packing cut expert read operations 3× with byte-identical output. Decode
speed on the test VM did not change measurably (−3% to +3%; one +23% session
did not reproduce). Its per-request cost is low at 0.9-1.8 MB reads. Disks
with expensive requests (IOPS-capped volumes, SATA, HDD) and smaller experts
are where it should matter, and they are not measured yet.

For dense models the source GGUF is already one extent per group, so
`prepare` changes nothing measurable there, and the tool says so.

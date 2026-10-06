# Backend strategy

## 1. Principle

Kestrel owns **placement, movement, and policy**. Backends own **arithmetic**.
Kestrel does not write GPU kernels when mature ones exist (ggml provides CUDA,
Vulkan, Metal, HIP, SYCL, and CPU). It does write the decisions those engines
do not make on their own.

## 2. The two MVP backends

### 2.1 `native`: Rust CPU executor under Kestrel's scheduler

**Why it exists:** a backend whose weight access goes *entirely* through
Kestrel's `WeightStore` is the only way to implement and measure streaming,
prefetching, eviction, and rebalancing truthfully. llama.cpp's weight access is
internal. With mmap it is governed by the OS page cache, and without mmap
everything is loaded up front.

Scope:

* Architectures: `llama` (also Mistral and Llama-3 GGUFs), `qwen2`, `qwen3`.
  MoE (`qwen2moe`, `qwen3moe`, `mixtral` as llama + experts) is phase 8.
* Weight types: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K.
  These cover the common `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, and `F16` files.
* Kernels: fused dequantize-dot GEMV for decode. GEMM for prefill is a GEMV
  per row, parallelized over output rows with rayon. Activations are f32.
* Tokenizers: GGUF `llama` (SentencePiece-style scores) and `gpt2` (byte-level
  BPE with merges and pre-tokenizer variants). Chat templates are rendered from
  the GGUF's Jinja template.

**It is not a performance competitor to llama.cpp's CPU path** in the MVP.
llama.cpp quantizes activations to q8 and uses hand-tuned SIMD dot kernels.
Every native-backend benchmark is reported next to the llama.cpp CPU number on
the same machine, so the kernel gap and the scheduling effects stay separated.

### 2.2 `llamacpp`: plan → llama.cpp/ggml

Kestrel computes the placement and translates it into llama.cpp parameters:

| Plan field | llama.cpp parameter |
|---|---|
| GPU layer count (dense prefix) | `--n-gpu-layers N` |
| Group-level CPU overrides (for example experts) | `--override-tensor "<regex>=CPU"` |
| Weights not resident in RAM (page in on demand) | default mmap (`--no-mmap` only when everything is resident) |
| Lock resident RAM tier | `--mlock` (only when the RAM budget allows it) |
| KV type | `--cache-type-k/-v` |
| KV placement | `--no-kv-offload` when KV must stay in RAM |
| Context | `--ctx-size` (planned from the KV budget) |
| Threads | `--threads` (physical cores) |
| Main GPU / split | `--main-gpu`, `--tensor-split` (multi-GPU, later) |

`kestrel plan model.gguf --backend llamacpp` prints the exact command line.
`kestrel serve --backend llamacpp` launches `llama-server` with it, found via
`KESTREL_LLAMA_SERVER` or `PATH`, and passes the OpenAI-compatible API through
on the same port.

**Decisions llama.cpp does not make automatically**, which this adapter
provides:

1. Choosing `n_gpu_layers` from *measured* free VRAM, KV size, and compute
   buffers, with a safety margin, instead of a fixed number or a trial-and-error
   fit.
2. For MoE models, moving experts (not layers) to the CPU so attention and
   shared tensors fill VRAM, ranked by usage history when it is available.
3. Choosing the context size that fits the KV budget, and reporting the
   trade-off.
4. Deciding mmap vs load-all vs mlock from the RAM budget, and refusing to
   start with a diagnosis when even the RAM+disk plan is infeasible.

What the adapter **cannot** do in the MVP is explicit prefetching or eviction
inside llama.cpp. For the NVMe tier it relies on mmap and the page cache. The
benchmark plan compares this "OS-managed" NVMe tier against the native
backend's explicit streaming.

## 3. Backend interface

The planner produces an `ExecutionPlan`. Backends consume it:

```rust
// kestrel-backends
pub trait InferenceSession: Send {
    fn model_name(&self) -> &str;
    fn render_chat(&self, messages: &[ChatMessage]) -> Result<String>;
    fn tokenize(&self, text: &str) -> Vec<u32>;
    fn generate(&mut self, prompt: &[u32], params: &GenParams,
                on_text: &mut dyn FnMut(&str) -> bool) -> Result<GenStats>;
    fn status(&self) -> serde_json::Value;   // metrics for /metrics
}
```

`NativeSession::load(plan, gguf, model, opts)` builds the ledger, the
`WeightStore` (resident mask, ring, I/O mode, prefetch depth from the plan) and
the executor, and runs the rebalancer at safe points between tokens. The
llama.cpp backend is a process: `llamacpp::spawn_server(plan.llamacpp_args, …)`.
Backend capabilities are currently encoded in the planner (the native
executor is CPU-only, and llama.cpp gets GPU candidates). A data-driven
`BackendCaps` is the next step, once a second GPU path exists.

## 4. Path to native GPU execution (post-MVP)

1. Link ggml directly through FFI and build ggml graphs per layer. The weight
   tensors' `data` pointers come from Kestrel's `WeightStore` (host-pinned
   buffers for streamed groups, device buffers for VRAM-resident groups). This
   gives the native backend GPU compute while Kestrel keeps ownership of
   residency.
2. Async host→device uploads on a dedicated stream, gated by the planner's PCIe
   cost model. Colibrì's finding applies: streaming NVMe → GPU per token only
   wins when GPU compute minus PCIe time beats CPU compute.
3. Backends are added behind cargo features (`cuda`, `vulkan`, `metal`, `hip`)
   so the default build stays dependency-free and portable.

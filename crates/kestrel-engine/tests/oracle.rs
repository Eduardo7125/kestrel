//! Correctness against llama.cpp.
//!
//! Requires fixtures from `tools/make_fixtures.py` and the oracle from
//! `tools/oracle/build.sh`:
//!
//!     KESTREL_FIXTURES=/path/fixtures KESTREL_ORACLE=/path/llama_oracle cargo test -p kestrel-engine --test oracle
//!
//! Skipped (passes trivially, with a note) when either is unset.

use kestrel_engine::Engine;
use kestrel_gguf::GgufFile;
use kestrel_hw::fileio::IoMode;
use kestrel_memory::{ExpertPolicy, ExpertStore, Ledger, RingPolicy, StoreConfig, WeightStore};
use kestrel_model::ModelDesc;
use std::path::{Path, PathBuf};
use std::sync::Arc;

struct Oracle {
    tokens: Vec<u32>,
    logits: Vec<f32>,
    greedy: Vec<u32>,
}

fn env() -> Option<(PathBuf, PathBuf)> {
    let f = std::env::var_os("KESTREL_FIXTURES")?;
    let o = std::env::var_os("KESTREL_ORACLE")?;
    Some((PathBuf::from(f), PathBuf::from(o)))
}

fn oracle(bin: &Path, model: &Path, prompt: &str, n: usize) -> Oracle {
    let out = std::process::Command::new(bin).arg(model).arg(prompt).arg(n.to_string()).output().expect("run oracle");
    assert!(out.status.success(), "oracle failed: {}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let ints = |k: &str| v[k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect::<Vec<_>>();
    Oracle { tokens: ints("tokens"), logits: v["logits"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect(), greedy: ints("greedy") }
}

fn engine(path: &Path, streamed: bool) -> Engine {
    let g = GgufFile::open(path).unwrap();
    let m = Arc::new(ModelDesc::from_gguf(&g).unwrap());
    let n = m.groups.len();
    let ledger = Ledger::new(0, 1 << 34, 0);
    let resident: Vec<bool> = (0..n).map(|_| !streamed).collect();
    let cfg = StoreConfig { io_mode: IoMode::Direct, prefetch_depth: 2, ring_slots: 3, io_workers: 4, policy: RingPolicy::Belady, drop_page_cache: true, external_experts: false };
    let mut cfg = cfg;
    cfg.external_experts = m.moe.is_some();
    let store = WeightStore::new(m.clone(), &resident, (0..n).collect(), cfg, ledger.clone()).unwrap();
    // MoE: everything cached when resident; a 2-expert cache (most experts
    // streamed through scratch buffers) when streamed.
    let experts = m.moe.as_ref().map(|moe| {
        let cap = if streamed { 2 } else { (moe.n_expert * moe.moe_layers) as usize };
        Arc::new(ExpertStore::new(m.clone(), cap, 2 * moe.n_expert_used as usize, IoMode::Direct, 4, ExpertPolicy::Lfru, ledger.clone()).unwrap())
    });
    Engine::new(&g, m, store, experts, 256, 4, &ledger).unwrap()
}

fn compare(path: &Path, bin: &Path, prompt: &str, tol: f32) {
    compare_with(path, path, bin, prompt, tol)
}

/// Run Kestrel on `path` and llama.cpp on `reference` (the same weights,
/// possibly pre-dequantized to F32).
fn compare_with(path: &Path, reference: &Path, bin: &Path, prompt: &str, tol: f32) {
    compare_mode(path, reference, bin, prompt, tol, true, true)
}

fn compare_mode(path: &Path, reference: &Path, bin: &Path, prompt: &str, tol: f32, exact: bool, check_greedy: bool) {
    kestrel_engine::quant::set_exact(exact);
    let o = oracle(bin, reference, prompt, 8);
    let mut e = engine(path, false);
    let toks = e.tokenizer.encode(prompt, true, true);
    assert_eq!(toks, o.tokens, "{}: tokenization differs from llama.cpp", path.display());
    let logits = e.tf.forward(&toks).unwrap();
    assert_eq!(logits.len(), o.logits.len());
    let scale = o.logits.iter().fold(0f32, |a, b| a.max(b.abs()));
    let maxdiff = logits.iter().zip(&o.logits).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("{} [{}]: max|Δlogit| = {maxdiff:.5} (scale {scale:.2})", path.file_name().unwrap().to_string_lossy(), if exact { "exact" } else { "int8" });
    assert!(maxdiff <= tol * scale.max(1.0), "{}: logits differ by {maxdiff} (scale {scale})", path.display());

    // Greedy continuation: identical tokens.
    let mut got = Vec::new();
    let mut l = logits.clone();
    for _ in 0..o.greedy.len() {
        let t = kestrel_engine::sampler::argmax(&l);
        got.push(t);
        l = e.tf.forward(&[t]).unwrap();
    }
    if check_greedy {
        assert_eq!(got, o.greedy, "{}: greedy continuation differs", path.display());
    } else if got != o.greedy {
        eprintln!("  note: greedy continuation differs from llama.cpp (near-tie under activation quantization): {got:?} vs {:?}", o.greedy);
    }

    // Streaming everything from disk must be bit-identical to resident.
    let mut s = engine(path, true);
    let ls = s.tf.forward(&toks).unwrap();
    assert_eq!(ls, logits, "streamed execution must match resident bit for bit");
    let m = s.tf.store.metrics();
    assert!(m.stream_bytes > 0 && m.resident_hits == 0, "{m:?}");
    if let Some(ex) = &s.tf.experts {
        let em = ex.metrics();
        assert!(em.scratch_loads > 0 && em.cached <= 2, "{em:?}");
    }
}

#[test]
fn matches_llama_cpp() {
    let Some((dir, bin)) = env() else {
        eprintln!("KESTREL_FIXTURES/KESTREL_ORACLE not set: skipping oracle comparison");
        return;
    };
    let prompt = "The quick brown fox jumps over the lazy dog. Hello world! Numbers: 42 1234";
    for name in ["llama-bpe-f32", "qwen2-f32", "qwen3-f32", "llama-spm-f32"] {
        compare(&dir.join(format!("{name}.gguf")), &bin, prompt, 2e-3);
    }
    // Quantized weights. llama.cpp also quantizes *activations* (to q8 or to
    // the weight's float type) inside its dot products, while Kestrel keeps
    // activations in f32. To test Kestrel's dequantization in isolation,
    // llama.cpp runs on a copy whose tensors were dequantized to F32 by
    // gguf-py's reference decoders (`*-deq.gguf`, from make_fixtures.py).
    for q in ["F16", "BF16", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q4_K_M", "Q5_K_M", "Q6_K"] {
        let p = dir.join(format!("llama-bpe-{q}.gguf"));
        let r = dir.join(format!("llama-bpe-{q}-deq.gguf"));
        if p.exists() && r.exists() {
            compare_with(&p, &r, &bin, prompt, 2e-3);
            // Fast path: int8 activations, like llama.cpp; compare against
            // llama.cpp on the quantized file itself.
            if !matches!(q, "F16" | "BF16") {
                compare_mode(&p, &p, &bin, prompt, 6e-2, false, false);
            }
        } else {
            eprintln!("missing fixture {}", p.display());
        }
    }
}

#[test]
fn chat_prompt_special_tokens() {
    let Some((dir, bin)) = env() else { return };
    let p = dir.join("qwen3-f32.gguf");
    let prompt = "<|im_start|>user\nCiao, come stai? 🙂<|im_end|>\n<|im_start|>assistant\n";
    compare(&p, &bin, prompt, 2e-3);
}

#[test]
fn moe_matches_llama_cpp() {
    let Some((dir, bin)) = env() else { return };
    let prompt = "The quick brown fox jumps over the lazy dog. Hello world! Numbers: 42 1234";
    for name in ["qwen3moe-f32", "qwen2moe-f32", "llama-moe-f32"] {
        compare(&dir.join(format!("{name}.gguf")), &bin, prompt, 2e-3);
    }
    let p = dir.join("qwen3moe-Q4_K_M.gguf");
    let r = dir.join("qwen3moe-Q4_K_M-deq.gguf");
    if p.exists() && r.exists() {
        compare_with(&p, &r, &bin, prompt, 2e-3);
    }
}

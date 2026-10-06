//! `kestrel prepare`: rewrite a GGUF once into a container laid out for
//! Kestrel's I/O.
//!
//! The transformation is **lossless**: every tensor keeps its type, shape and
//! bytes. Only the placement of the bytes in the file changes:
//!
//! * **Packed experts.** A plain GGUF stores the routed experts of a layer as
//!   three stacked tensors (`ffn_{gate,up,down}_exps`), so loading one expert
//!   means three scattered reads. The prepared container interleaves them:
//!   expert `e` of a layer is `[gate_e | up_e | down_e]`, padded to 4 KiB, and
//!   the three tensors become *strided* ([`kestrel_gguf::TensorInfo::stride`]).
//!   One expert is one contiguous, page-aligned read.
//! * **Execution order, page-aligned groups.** Tensor groups are written in
//!   execution order and each group starts on a 4 KiB boundary, so every dense
//!   group is a single direct-I/O extent with no partial pages.
//!
//! The container uses the `KGUF` magic. Other GGUF readers refuse it instead of
//! misreading the strided expert tensors; the source GGUF stays the file to
//! use with llama.cpp.

use crate::{coalesce, GroupKind, ModelDesc};
use kestrel_gguf::writer::{encode_header, HeaderTensor};
use kestrel_gguf::{align_up, GgufFile, TensorInfo, Value, KGUF_MAGIC, STRIDE_KEY_PREFIX};
use serde::Serialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const PREPARED_VERSION: u64 = 1;
/// Page size used for group and expert alignment (direct I/O granularity).
pub const PAGE: u64 = 4096;
const COPY_CHUNK: usize = 8 << 20;

#[derive(Debug, thiserror::Error)]
pub enum PrepareError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Gguf(#[from] kestrel_gguf::GgufError),
    #[error("verification failed: tensor '{0}' differs from the source")]
    Mismatch(String),
    #[error("{0}")]
    Invalid(String),
}

/// What `prepare` will write, computed from the header alone.
pub struct Layout {
    /// Output tensor table (same order as the source table).
    tensors: Vec<HeaderTensor>,
    strides: Vec<(String, u64)>,
    /// (output rel offset, source tensor index, slice index or None, len), sorted by output offset.
    pieces: Vec<(u64, usize, Option<u64>, u64)>,
    data_len: u64,
    pub report: PrepareReport,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct PrepareReport {
    pub source: PathBuf,
    pub output: PathBuf,
    pub source_bytes: u64,
    pub output_bytes: u64,
    /// Bytes added by alignment padding.
    pub padding_bytes: u64,
    pub groups: usize,
    /// Dense groups whose file extents were split before / after.
    pub split_groups_before: usize,
    pub split_groups_after: usize,
    pub moe_layers_packed: usize,
    pub experts_packed: u64,
    /// File reads (extents) needed to load one expert, before and after.
    pub reads_per_expert_before: f64,
    pub reads_per_expert_after: f64,
    pub verified: bool,
    pub seconds: f64,
}

/// The default output path: `<dir>/<stem>.kgguf`.
pub fn default_output(source: &Path) -> PathBuf {
    let stem = source.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "model".into());
    source.with_file_name(format!("{stem}.kgguf"))
}

/// Indices of the stacked (gate, up, down) expert tensors of `layer`.
fn expert_triplet(g: &GgufFile, layer: u32) -> Option<[usize; 3]> {
    let idx = |p: &str| g.tensors.iter().position(|t| t.name == format!("blk.{layer}.ffn_{p}_exps.weight"));
    let t = [idx("gate")?, idx("up")?, idx("down")?];
    let n = g.tensors[t[0]].n_slices();
    (n > 1 && t.iter().all(|&i| g.tensors[i].n_slices() == n && g.tensors[i].dims.len() == 3)).then_some(t)
}

/// Extents (after coalescing) needed to read expert `e` of the triplet.
fn expert_reads(g: &GgufFile, t: [usize; 3], e: u64) -> usize {
    let mut ext: Vec<crate::Extent> = t.iter().map(|&i| crate::Extent { offset: g.tensors[i].slice_offset(e), len: g.tensors[i].slice_bytes() }).collect();
    ext.sort_by_key(|x| x.offset);
    coalesce(&ext, g.alignment).len()
}

pub fn plan_layout(g: &GgufFile, model: &ModelDesc, output: &Path) -> Layout {
    let align = g.alignment;
    let mut rel: HashMap<usize, u64> = HashMap::new();
    let mut strides = Vec::new();
    let mut pieces = Vec::new();
    let mut cursor = 0u64;
    let mut report = PrepareReport { source: g.path.clone(), output: output.to_path_buf(), source_bytes: g.file_size, groups: model.groups.len(), ..Default::default() };
    let (mut reads_before, mut reads_after, mut n_exp) = (0usize, 0usize, 0u64);

    for grp in &model.groups {
        if grp.kind != GroupKind::Experts && grp.extents.len() > 1 {
            report.split_groups_before += 1;
        }
        let start = align_up(cursor, PAGE);
        report.padding_bytes += start - cursor;
        cursor = start;
        let triplet = (grp.kind == GroupKind::Experts).then(|| grp.layer.and_then(|l| expert_triplet(g, l))).flatten();
        if let Some(t) = triplet {
            let s: Vec<u64> = t.iter().map(|&i| g.tensors[i].slice_bytes()).collect();
            let o = [0, align_up(s[0], align), align_up(align_up(s[0], align) + s[1], align)];
            let stride = align_up(o[2] + s[2], PAGE);
            let n = g.tensors[t[0]].n_slices();
            for k in 0..3 {
                rel.insert(t[k], cursor + o[k]);
                strides.push((g.tensors[t[k]].name.clone(), stride));
            }
            for e in 0..n {
                reads_before += expert_reads(g, t, e);
                for k in 0..3 {
                    pieces.push((cursor + e * stride + o[k], t[k], Some(e), s[k]));
                }
            }
            reads_after += n as usize;
            n_exp += n;
            report.moe_layers_packed += 1;
            report.padding_bytes += n * stride - s.iter().sum::<u64>() * n;
            cursor += n * stride;
        }
        for &ti in &grp.tensors {
            if rel.contains_key(&ti) {
                continue;
            }
            let t = &g.tensors[ti];
            let at = align_up(cursor, align);
            report.padding_bytes += at - cursor;
            rel.insert(ti, at);
            match t.stride {
                // Re-preparing a prepared file: gather the slices back.
                Some(_) => {
                    for e in 0..t.n_slices() {
                        pieces.push((at + e * t.slice_bytes(), ti, Some(e), t.slice_bytes()));
                    }
                }
                None => pieces.push((at, ti, None, t.size)),
            }
            cursor = at + t.size;
        }
    }
    pieces.sort_by_key(|p| p.0);
    let tensors = g
        .tensors
        .iter()
        .enumerate()
        .map(|(i, t)| HeaderTensor { name: t.name.clone(), dims: t.dims.clone(), ggml_type: t.ggml_type, rel_offset: rel[&i] })
        .collect();
    report.experts_packed = n_exp;
    if n_exp > 0 {
        report.reads_per_expert_before = reads_before as f64 / n_exp as f64;
        report.reads_per_expert_after = reads_after as f64 / n_exp as f64;
    }
    Layout { tensors, strides, pieces, data_len: align_up(cursor, align), report }
}

/// FNV-1a over a byte stream, per tensor (slices in index order).
#[derive(Clone, Copy)]
struct Fnv(u64);
impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf29ce484222325)
    }
    fn feed(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= x as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
}

fn read_range(f: &mut File, offset: u64, len: u64, buf: &mut Vec<u8>, mut sink: impl FnMut(&[u8]) -> std::io::Result<()>) -> std::io::Result<()> {
    f.seek(SeekFrom::Start(offset))?;
    let mut left = len;
    while left > 0 {
        let n = left.min(COPY_CHUNK as u64) as usize;
        buf.resize(n, 0);
        f.read_exact(&mut buf[..n])?;
        sink(&buf[..n])?;
        left -= n as u64;
    }
    Ok(())
}

/// Hash every tensor of `g` slice by slice (layout independent).
fn tensor_hashes(g: &GgufFile, mut progress: impl FnMut(u64)) -> std::io::Result<Vec<u64>> {
    let mut f = File::open(&g.path)?;
    let mut buf = Vec::new();
    let mut out = Vec::with_capacity(g.tensors.len());
    for t in &g.tensors {
        let mut h = Fnv::new();
        let mut hash_range = |off: u64, len: u64| read_range(&mut f, off, len, &mut buf, |b| {
            h.feed(b);
            Ok(())
        });
        if t.stride.is_some() {
            for e in 0..t.n_slices() {
                hash_range(t.slice_offset(e), t.slice_bytes())?;
            }
        } else {
            hash_range(t.offset, t.size)?;
        }
        progress(t.size);
        out.push(h.0);
    }
    Ok(out)
}

/// Write the prepared container. `progress` receives bytes copied (then
/// bytes verified). Writes to `<output>.partial` and renames on success.
pub fn prepare(g: &GgufFile, model: &ModelDesc, output: &Path, verify: bool, mut progress: impl FnMut(&str, u64)) -> Result<PrepareReport, PrepareError> {
    let t0 = Instant::now();
    if output == g.path {
        return Err(PrepareError::Invalid("output would overwrite the source".into()));
    }
    let layout = plan_layout(g, model, output);
    let mut kv: Vec<(String, Value)> = g
        .metadata
        .iter()
        .filter(|(k, _)| !k.starts_with("kestrel."))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    kv.push(("kestrel.prepared.version".into(), Value::U64(PREPARED_VERSION)));
    kv.push(("kestrel.prepared.source".into(), Value::String(g.path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default())));
    kv.push(("kestrel.prepared.source_fingerprint".into(), Value::String(g.fingerprint())));
    for (name, st) in &layout.strides {
        kv.push((format!("{STRIDE_KEY_PREFIX}{name}"), Value::U64(*st)));
    }
    let header = encode_header(KGUF_MAGIC, &kv, &layout.tensors, g.alignment, PAGE);

    let partial = {
        let mut p = output.as_os_str().to_owned();
        p.push(".partial");
        PathBuf::from(p)
    };
    let res = (|| -> Result<(), PrepareError> {
        let mut out = std::io::BufWriter::with_capacity(COPY_CHUNK, File::create(&partial)?);
        out.write_all(&header)?;
        let mut src = File::open(&g.path)?;
        let mut buf = Vec::new();
        let mut pos = 0u64;
        let zeros = vec![0u8; PAGE as usize * 4];
        let pad_to = |out: &mut std::io::BufWriter<File>, pos: &mut u64, to: u64| -> std::io::Result<()> {
            while *pos < to {
                let n = (to - *pos).min(zeros.len() as u64) as usize;
                out.write_all(&zeros[..n])?;
                *pos += n as u64;
            }
            Ok(())
        };
        for &(at, ti, slice, len) in &layout.pieces {
            pad_to(&mut out, &mut pos, at)?;
            let t: &TensorInfo = &g.tensors[ti];
            let off = slice.map(|e| t.slice_offset(e)).unwrap_or(t.offset);
            read_range(&mut src, off, len, &mut buf, |b| out.write_all(b))?;
            pos += len;
            progress("copy", len);
        }
        pad_to(&mut out, &mut pos, layout.data_len)?;
        out.flush()?;
        out.get_ref().sync_all()?;
        Ok(())
    })();
    if let Err(e) = res {
        let _ = std::fs::remove_file(&partial);
        return Err(e);
    }

    let mut report = layout.report;
    if verify {
        let check = (|| -> Result<(), PrepareError> {
            let new = GgufFile::open(&partial)?;
            let a = tensor_hashes(g, |n| progress("verify", n / 2))?;
            let b = tensor_hashes(&new, |n| progress("verify", n - n / 2))?;
            if let Some(i) = (0..a.len()).find(|&i| a[i] != b[i] || g.tensors[i].name != new.tensors[i].name) {
                return Err(PrepareError::Mismatch(g.tensors[i].name.clone()));
            }
            let nm = ModelDesc::from_gguf(&new).map_err(|e| PrepareError::Invalid(e.to_string()))?;
            report.split_groups_after = nm.groups.iter().filter(|gr| gr.kind != GroupKind::Experts && gr.extents.len() > 1).count();
            Ok(())
        })();
        if let Err(e) = check {
            let _ = std::fs::remove_file(&partial);
            return Err(e);
        }
        report.verified = true;
    }
    std::fs::rename(&partial, output)?;
    report.output_bytes = std::fs::metadata(output)?.len();
    report.seconds = t0.elapsed().as_secs_f64();
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kestrel_gguf::writer::{GgufWriter, TensorData};

    fn moe_file(dir: &Path) -> PathBuf {
        let p = dir.join("m.gguf");
        let mut w = GgufWriter::new();
        let arch = "qwen3moe";
        w.kv("general.architecture", Value::String(arch.into()));
        w.kv(&format!("{arch}.block_count"), Value::U32(2));
        w.kv(&format!("{arch}.embedding_length"), Value::U32(64));
        w.kv(&format!("{arch}.attention.head_count"), Value::U32(4));
        w.kv(&format!("{arch}.feed_forward_length"), Value::U32(128));
        w.kv(&format!("{arch}.expert_count"), Value::U32(4));
        w.kv(&format!("{arch}.expert_used_count"), Value::U32(2));
        let ramp = |n: usize, k: f32| TensorData::F32((0..n).map(|i| i as f32 + k).collect());
        w.tensor("token_embd.weight", &[64, 10], ramp(640, 0.5));
        for l in 0..2u32 {
            let k = l as f32 * 1000.0;
            w.tensor(&format!("blk.{l}.attn_norm.weight"), &[64], ramp(64, k));
            w.tensor(&format!("blk.{l}.ffn_gate_inp.weight"), &[64, 4], ramp(256, k + 1.0));
            w.tensor(&format!("blk.{l}.ffn_gate_exps.weight"), &[64, 24, 4], ramp(64 * 24 * 4, k + 2.0));
            w.tensor(&format!("blk.{l}.ffn_up_exps.weight"), &[64, 24, 4], ramp(64 * 24 * 4, k + 3.0));
            w.tensor(&format!("blk.{l}.ffn_down_exps.weight"), &[24, 64, 4], ramp(64 * 24 * 4, k + 4.0));
        }
        w.tensor("output_norm.weight", &[64], ramp(64, 7.0));
        w.write(&p).unwrap();
        p
    }

    fn slice(g: &GgufFile, name: &str, e: u64) -> Vec<u8> {
        let t = g.tensor(name).unwrap();
        let bytes = std::fs::read(&g.path).unwrap();
        let o = t.slice_offset(e) as usize;
        bytes[o..o + t.slice_bytes() as usize].to_vec()
    }

    #[test]
    fn packs_experts_losslessly() {
        let d = tempfile::tempdir().unwrap();
        let src_path = moe_file(d.path());
        let g = GgufFile::open(&src_path).unwrap();
        let m = ModelDesc::from_gguf(&g).unwrap();
        let out = default_output(&src_path);
        let r = prepare(&g, &m, &out, true, |_, _| {}).unwrap();
        assert!(r.verified);
        assert_eq!(r.experts_packed, 8);
        assert_eq!(r.reads_per_expert_before, 3.0);
        assert_eq!(r.reads_per_expert_after, 1.0);

        let p = GgufFile::open(&out).unwrap();
        assert!(p.prepared);
        assert_eq!(p.data_offset % PAGE, 0);
        let pm = ModelDesc::from_gguf(&p).unwrap();
        assert!(pm.prepared.as_ref().unwrap().packed_experts);
        assert_eq!(pm.prepared.as_ref().unwrap().source_fingerprint, g.fingerprint());
        for l in 0..2 {
            for part in ["gate", "up", "down"] {
                let name = format!("blk.{l}.ffn_{part}_exps.weight");
                for e in 0..4 {
                    assert_eq!(slice(&g, &name, e), slice(&p, &name, e), "{name}[{e}]");
                }
            }
            // One expert = one read; each group starts on a page.
            let t = p.tensor(&format!("blk.{l}.ffn_gate_exps.weight")).unwrap();
            assert_eq!(t.offset % PAGE, 0);
            assert_eq!(expert_reads(&p, expert_triplet(&p, l).unwrap(), 1), 1);
        }
        // Every group is one extent, page aligned.
        for gr in &pm.groups {
            assert_eq!(gr.extents.len(), 1, "{}", gr.label());
            assert_eq!(gr.extents[0].offset % PAGE, 0, "{}", gr.label());
        }
        assert_eq!(pm.weight_bytes(), m.weight_bytes());

        // Re-preparing a prepared file round-trips the data.
        let again = d.path().join("again.kgguf");
        prepare(&p, &pm, &again, true, |_, _| {}).unwrap();
    }

    #[test]
    fn plain_readers_reject_strides() {
        // A stride key in a plain GGUF is corrupt, not silently honored.
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("x.gguf");
        let mut w = GgufWriter::new();
        w.kv("general.architecture", Value::String("llama".into()));
        w.kv("kestrel.stride.a.weight", Value::U64(4096));
        w.tensor("a.weight", &[32, 2, 2], TensorData::F32(vec![0.0; 128]));
        w.write(&p).unwrap();
        assert!(GgufFile::open(&p).unwrap_err().to_string().contains("strided"));
    }
}

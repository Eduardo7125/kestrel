//! Measured micro-benchmarks. The planner prefers these over spec-sheet
//! numbers: a VHDX-backed "NVMe" or a QLC drive past its SLC cache can be an
//! order of magnitude slower than its label.

use crate::fileio::{aligned_span, read_range, AlignedBuf, IoMode, ReadFile, DIRECT_ALIGN};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Measurements {
    /// Multi-threaded sequential read bandwidth of RAM, bytes/s.
    pub ram_read_bw: f64,
    /// Single-thread RAM read bandwidth, bytes/s.
    pub ram_read_bw_1t: f64,
    pub disks: Vec<DiskBench>,
    /// Native CPU kernel throughput: weight bytes consumed per second by a
    /// decode GEMV, per ggml type name. Filled by `kestrel-engine`.
    pub cpu_gemv_bytes_per_s: BTreeMap<String, f64>,
    /// Unix seconds when measured.
    pub when: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiskBench {
    pub path: PathBuf,
    pub mode: IoMode,
    /// Bytes/s for large (chunk-sized) reads at queue depth `threads`.
    pub read_bw: f64,
    /// Bytes/s for the same reads issued by a single thread (QD1).
    pub read_bw_qd1: f64,
    /// Mean latency of 4 KiB random reads at QD1, microseconds.
    pub latency_4k_us: f64,
    pub chunk: usize,
    pub threads: usize,
    /// True if the measured file is small enough that a cache could have
    /// served the reads (results are then an upper bound).
    pub cache_suspect: bool,
}

#[derive(Serialize, Deserialize)]
pub struct CachedMeasurements {
    pub fingerprint: String,
    pub measurements: Measurements,
}

impl Measurements {
    /// Best (fastest direct, else buffered) disk measurement for a path.
    pub fn disk_for(&self, path: &Path) -> Option<&DiskBench> {
        let p = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let candidates: Vec<&DiskBench> = self.disks.iter().filter(|d| d.path == p || p.starts_with(d.path.parent().unwrap_or(Path::new("/")))).collect();
        let pool = if candidates.is_empty() { self.disks.iter().collect() } else { candidates };
        pool.into_iter().max_by(|a, b| {
            (a.mode == IoMode::Direct, a.read_bw).partial_cmp(&(b.mode == IoMode::Direct, b.read_bw)).unwrap()
        })
    }
}

/// Measure RAM read bandwidth with `threads` workers over `bytes` bytes.
pub fn ram_bandwidth(bytes: usize, threads: usize) -> f64 {
    let words = bytes / 8;
    let buf: Vec<u64> = (0..words as u64).collect();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build().unwrap();
    let chunk = words.div_ceil(threads.max(1));
    let run = || {
        pool.install(|| {
            buf.par_chunks(chunk)
                .map(|c| {
                    // 4 independent accumulators keep the loop load-bound.
                    let (mut a, mut b, mut cc, mut d) = (0u64, 0u64, 0u64, 0u64);
                    for q in c.chunks_exact(4) {
                        a = a.wrapping_add(q[0]);
                        b = b.wrapping_add(q[1]);
                        cc = cc.wrapping_add(q[2]);
                        d = d.wrapping_add(q[3]);
                    }
                    a ^ b ^ cc ^ d
                })
                .reduce(|| 0, |x, y| x ^ y)
        })
    };
    std::hint::black_box(run()); // warm up, fault pages in
    let mut best = 0.0f64;
    for _ in 0..3 {
        let t = Instant::now();
        std::hint::black_box(run());
        best = best.max(bytes as f64 / t.elapsed().as_secs_f64());
    }
    best
}

/// Measure read throughput of `file` with random chunk-aligned reads.
pub fn disk_bandwidth(file: &Path, mode: IoMode, chunk: usize, threads: usize, budget: Duration) -> std::io::Result<DiskBench> {
    let f = ReadFile::open(file, mode)?;
    let len = f.len()?;
    if len < (chunk as u64) * 2 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "file too small to benchmark"));
    }
    let nchunks = len / chunk as u64;
    let mode = f.mode();
    let measure = |threads: usize, budget: Duration| -> f64 {
        let total = AtomicU64::new(0);
        let start = Instant::now();
        std::thread::scope(|s| {
            for t in 0..threads {
                let f = &f;
                let total = &total;
                s.spawn(move || {
                    let mut buf = AlignedBuf::new(aligned_span(0, chunk) + DIRECT_ALIGN, DIRECT_ALIGN).unwrap();
                    let mut rng = 0x9e3779b97f4a7c15u64 ^ (t as u64 * 0x632be59bd9b4e019);
                    while start.elapsed() < budget {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        let off = (rng % nchunks) * chunk as u64;
                        if read_range(f, buf.as_mut_slice(), off, chunk).is_ok() {
                            total.fetch_add(chunk as u64, Ordering::Relaxed);
                        }
                        if mode == IoMode::Buffered {
                            f.drop_cache(off, chunk as u64);
                        }
                    }
                });
            }
        });
        total.load(Ordering::Relaxed) as f64 / start.elapsed().as_secs_f64()
    };
    let read_bw = measure(threads, budget);
    let read_bw_qd1 = measure(1, budget / 2);

    // 4 KiB latency at QD1.
    let mut buf = AlignedBuf::new(2 * DIRECT_ALIGN, DIRECT_ALIGN)?;
    let n4k = len / 4096;
    let mut rng = 0x2545f4914f6cdd1du64;
    let t = Instant::now();
    let mut n = 0u32;
    while t.elapsed() < Duration::from_millis(300) && n < 2000 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let off = (rng % n4k) * 4096;
        read_range(&f, buf.as_mut_slice(), off, 4096)?;
        n += 1;
    }
    let latency_4k_us = t.elapsed().as_secs_f64() * 1e6 / n.max(1) as f64;

    let ram = crate::mem::discover();
    let cache_suspect = mode == IoMode::Buffered && len < ram.available;
    Ok(DiskBench {
        path: std::fs::canonicalize(file).unwrap_or_else(|_| file.to_path_buf()),
        mode,
        read_bw,
        read_bw_qd1,
        latency_4k_us,
        chunk,
        threads,
        cache_suspect,
    })
}

/// Create (or reuse) a scratch file of `bytes` in `dir` for disk benchmarks
/// when no model file is available to read.
pub fn scratch_file(dir: &Path, bytes: u64) -> std::io::Result<PathBuf> {
    use std::io::Write;
    std::fs::create_dir_all(dir)?;
    let p = dir.join("kestrel-diskbench.bin");
    if std::fs::metadata(&p).map(|m| m.len() == bytes).unwrap_or(false) {
        return Ok(p);
    }
    let mut f = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(&p)?);
    let mut block = vec![0u8; 8 << 20];
    let mut x = 0x12345678u32;
    for b in block.iter_mut() {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *b = x as u8;
    }
    let mut written = 0u64;
    while written < bytes {
        let n = ((bytes - written) as usize).min(block.len());
        f.write_all(&block[..n])?;
        written += n as u64;
    }
    f.flush()?;
    f.get_ref().sync_all()?;
    Ok(p)
}

pub fn now_unix() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ram_bw_positive() {
        let bw = ram_bandwidth(16 << 20, 2);
        assert!(bw > 1e8, "{bw}");
    }

    #[test]
    fn disk_bw_runs() {
        let dir = tempfile::tempdir().unwrap();
        let p = scratch_file(dir.path(), 16 << 20).unwrap();
        let d = disk_bandwidth(&p, IoMode::Direct, 1 << 20, 2, Duration::from_millis(100)).unwrap();
        assert!(d.read_bw > 0.0 && d.latency_4k_us > 0.0);
    }
}

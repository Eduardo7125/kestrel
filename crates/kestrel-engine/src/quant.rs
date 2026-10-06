//! Dequantization and matrix-vector kernels for the ggml block formats.
//!
//! Layouts and arithmetic follow `ggml/src/ggml-quants.c` (`dequantize_row_*`)
//! exactly, so results match llama.cpp up to float summation order. Kernels
//! dequantize one weight row into a small f32 buffer and dot it with every
//! activation row: for decode (S = 1) that is a fused dequant-dot; for prefill
//! the dequantization is amortized over S rows.

use kestrel_gguf::GgmlType;
use rayon::prelude::*;
use std::sync::OnceLock;

fn f16_lut() -> &'static [f32] {
    static LUT: OnceLock<Vec<f32>> = OnceLock::new();
    LUT.get_or_init(|| (0..=u16::MAX).map(|h| half::f16::from_bits(h).to_f32()).collect())
}

#[inline(always)]
pub fn f16(lo: u8, hi: u8) -> f32 {
    // SAFETY: the LUT has 65536 entries.
    unsafe { *f16_lut().get_unchecked(u16::from_le_bytes([lo, hi]) as usize) }
}

#[inline(always)]
fn f16_at(b: &[u8], i: usize) -> f32 {
    f16(b[i], b[i + 1])
}

static EXACT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);

/// Use f32 activations everywhere (exact w.r.t. the dequantized weights)
/// instead of the int8 fast path. Also enabled by `KESTREL_EXACT=1`.
pub fn set_exact(on: bool) {
    EXACT.store(on as u8, std::sync::atomic::Ordering::Relaxed);
}

pub fn exact() -> bool {
    match EXACT.load(std::sync::atomic::Ordering::Relaxed) {
        2 => {
            let on = std::env::var("KESTREL_EXACT").map(|v| v == "1").unwrap_or(false);
            set_exact(on);
            on
        }
        v => v == 1,
    }
}

pub fn is_supported(t: GgmlType) -> bool {
    use GgmlType::*;
    matches!(t, F32 | F16 | BF16 | Q8_0 | Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q4_K | Q5_K | Q6_K)
}

#[inline(always)]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Dequantize one row of `n` elements.
pub fn dequant_row(t: GgmlType, src: &[u8], out: &mut [f32]) {
    use GgmlType::*;
    let n = out.len();
    match t {
        F32 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        F16 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(2)) {
                *o = f16(c[0], c[1]);
            }
        }
        BF16 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(2)) {
                *o = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
            }
        }
        Q8_0 => {
            for (b, y) in src.chunks_exact(34).zip(out.chunks_exact_mut(32)) {
                let d = f16_at(b, 0);
                for j in 0..32 {
                    y[j] = (b[2 + j] as i8) as f32 * d;
                }
            }
        }
        Q4_0 => {
            for (b, y) in src.chunks_exact(18).zip(out.chunks_exact_mut(32)) {
                let d = f16_at(b, 0);
                let qs = &b[2..18];
                for j in 0..16 {
                    y[j] = ((qs[j] & 0xF) as i32 - 8) as f32 * d;
                    y[j + 16] = ((qs[j] >> 4) as i32 - 8) as f32 * d;
                }
            }
        }
        Q4_1 => {
            for (b, y) in src.chunks_exact(20).zip(out.chunks_exact_mut(32)) {
                let d = f16_at(b, 0);
                let m = f16_at(b, 2);
                let qs = &b[4..20];
                for j in 0..16 {
                    y[j] = (qs[j] & 0xF) as f32 * d + m;
                    y[j + 16] = (qs[j] >> 4) as f32 * d + m;
                }
            }
        }
        Q5_0 => {
            for (b, y) in src.chunks_exact(22).zip(out.chunks_exact_mut(32)) {
                let d = f16_at(b, 0);
                let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                let qs = &b[6..22];
                for j in 0..16 {
                    let h0 = (((qh >> j) << 4) & 0x10) as u8;
                    let h1 = ((qh >> (j + 12)) & 0x10) as u8;
                    y[j] = (((qs[j] & 0xF) | h0) as i32 - 16) as f32 * d;
                    y[j + 16] = (((qs[j] >> 4) | h1) as i32 - 16) as f32 * d;
                }
            }
        }
        Q5_1 => {
            for (b, y) in src.chunks_exact(24).zip(out.chunks_exact_mut(32)) {
                let d = f16_at(b, 0);
                let m = f16_at(b, 2);
                let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                let qs = &b[8..24];
                for j in 0..16 {
                    let h0 = (((qh >> j) << 4) & 0x10) as u8;
                    let h1 = ((qh >> (j + 12)) & 0x10) as u8;
                    y[j] = ((qs[j] & 0xF) | h0) as f32 * d + m;
                    y[j + 16] = ((qs[j] >> 4) | h1) as f32 * d + m;
                }
            }
        }
        Q4_K => {
            for (b, y) in src.chunks_exact(144).zip(out.chunks_exact_mut(256)) {
                let d = f16_at(b, 0);
                let dmin = f16_at(b, 2);
                let scales = &b[4..16];
                let qs = &b[16..144];
                for (k, (q, yy)) in qs.chunks_exact(32).zip(y.chunks_exact_mut(64)).enumerate() {
                    let (s1, m1) = scale_min_k4(2 * k, scales);
                    let (s2, m2) = scale_min_k4(2 * k + 1, scales);
                    let (d1, m1) = (d * s1 as f32, dmin * m1 as f32);
                    let (d2, m2) = (d * s2 as f32, dmin * m2 as f32);
                    for l in 0..32 {
                        yy[l] = d1 * (q[l] & 0xF) as f32 - m1;
                        yy[l + 32] = d2 * (q[l] >> 4) as f32 - m2;
                    }
                }
            }
        }
        Q5_K => {
            for (b, y) in src.chunks_exact(176).zip(out.chunks_exact_mut(256)) {
                let d = f16_at(b, 0);
                let dmin = f16_at(b, 2);
                let scales = &b[4..16];
                let qh = &b[16..48];
                let ql = &b[48..176];
                let (mut u1, mut u2) = (1u8, 2u8);
                for (k, (q, yy)) in ql.chunks_exact(32).zip(y.chunks_exact_mut(64)).enumerate() {
                    let (s1, m1) = scale_min_k4(2 * k, scales);
                    let (s2, m2) = scale_min_k4(2 * k + 1, scales);
                    let (d1, m1) = (d * s1 as f32, dmin * m1 as f32);
                    let (d2, m2) = (d * s2 as f32, dmin * m2 as f32);
                    for l in 0..32 {
                        let hi1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                        let hi2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                        yy[l] = d1 * ((q[l] & 0xF) + hi1) as f32 - m1;
                        yy[l + 32] = d2 * ((q[l] >> 4) + hi2) as f32 - m2;
                    }
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }
        Q6_K => {
            for (b, y) in src.chunks_exact(210).zip(out.chunks_exact_mut(256)) {
                let ql = &b[0..128];
                let qh = &b[128..192];
                let sc = &b[192..208];
                let d = f16_at(b, 208);
                for h in 0..2 {
                    let (ql, qh, sc, y) = (&ql[h * 64..], &qh[h * 32..], &sc[h * 8..], &mut y[h * 128..]);
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                        y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                }
            }
        }
        other => panic!("dequantization of {other} is not implemented in the native executor"),
    }
    debug_assert_eq!(out.len(), n);
}

#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Eight independent accumulators so the compiler vectorizes the loop.
    let mut acc = [0f32; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// `out[s][r] = Σ_c W[r][c] · x[s][c]` for a quantized weight `W` of shape
/// `[rows, cols]` (ggml order: `cols` contiguous), activations `x` of shape
/// `[S, cols]`, output `[S, rows]`.
pub fn matmul(t: GgmlType, w: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    let s_rows = x.len() / cols;
    assert_eq!(x.len(), s_rows * cols);
    assert_eq!(out.len(), s_rows * rows);
    let row_bytes = t.bytes_for(cols as u64).expect("cols multiple of block size") as usize;
    assert!(w.len() >= row_bytes * rows, "weight buffer {} < {}×{}", w.len(), rows, row_bytes);

    // Work in tiles of output rows; each tile writes a disjoint column range
    // of `out` for every activation row, so collect per tile then scatter.
    const TILE: usize = 16;
    if !exact() && crate::qdot::supports(t) && cols.is_multiple_of(32) {
        let qx: Vec<crate::qdot::Q8Row> = (0..s_rows).map(|s| crate::qdot::Q8Row::quantize(&x[s * cols..(s + 1) * cols])).collect();
        let tiles: Vec<(usize, Vec<f32>)> = (0..rows.div_ceil(TILE))
            .into_par_iter()
            .map(|ti| {
                let r0 = ti * TILE;
                let r1 = (r0 + TILE).min(rows);
                let mut res = vec![0f32; (r1 - r0) * s_rows];
                let mut u = crate::qdot::Unpacked::default();
                let fused = crate::qdot::has_fused(t);
                for r in r0..r1 {
                    let wr = &w[r * row_bytes..(r + 1) * row_bytes];
                    if fused {
                        for (s, q) in qx.iter().enumerate() {
                            res[(r - r0) * s_rows + s] = crate::qdot::dot_fused(t, wr, q);
                        }
                    } else {
                        crate::qdot::unpack(t, wr, cols, &mut u);
                        for (s, q) in qx.iter().enumerate() {
                            res[(r - r0) * s_rows + s] = crate::qdot::dot(&u, q);
                        }
                    }
                }
                (r0, res)
            })
            .collect();
        scatter(tiles, s_rows, rows, out);
        return;
    }
    let tiles: Vec<(usize, Vec<f32>)> = (0..rows.div_ceil(TILE))
        .into_par_iter()
        .map(|ti| {
            let r0 = ti * TILE;
            let r1 = (r0 + TILE).min(rows);
            let mut buf = vec![0f32; cols];
            let mut res = vec![0f32; (r1 - r0) * s_rows];
            for r in r0..r1 {
                dequant_row(t, &w[r * row_bytes..(r + 1) * row_bytes], &mut buf);
                for s in 0..s_rows {
                    res[(r - r0) * s_rows + s] = dot(&buf, &x[s * cols..(s + 1) * cols]);
                }
            }
            (r0, res)
        })
        .collect();
    scatter(tiles, s_rows, rows, out);
}

fn scatter(tiles: Vec<(usize, Vec<f32>)>, s_rows: usize, rows: usize, out: &mut [f32]) {
    for (r0, res) in tiles {
        let n = res.len() / s_rows;
        for i in 0..n {
            for s in 0..s_rows {
                out[s * rows + r0 + i] = res[i * s_rows + s];
            }
        }
    }
}

/// Copy one row of a (possibly quantized) matrix into f32 (embedding lookup).
pub fn get_row(t: GgmlType, w: &[u8], cols: usize, row: usize, out: &mut [f32]) {
    let row_bytes = t.bytes_for(cols as u64).unwrap() as usize;
    dequant_row(t, &w[row * row_bytes..(row + 1) * row_bytes], &mut out[..cols]);
}

/// Decode GEMV throughput of this machine for a type: weight bytes per second.
pub fn bench_gemv(t: GgmlType, rows: usize, cols: usize, iters: usize) -> f64 {
    let row_bytes = t.bytes_for(cols as u64).unwrap() as usize;
    // Plausible block contents: small scales so values stay finite.
    let mut w = vec![0u8; rows * row_bytes];
    let mut x = 0x1234_5678u32;
    for b in w.iter_mut() {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *b = (x & 0x3f) as u8;
    }
    let act: Vec<f32> = (0..cols).map(|i| (i % 7) as f32 * 0.01).collect();
    let mut out = vec![0f32; rows];
    matmul(t, &w, rows, cols, &act, &mut out);
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        matmul(t, &w, rows, cols, &act, &mut out);
        std::hint::black_box(&out);
    }
    (w.len() * iters) as f64 / t0.elapsed().as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_and_q4_0_known_values() {
        // Q8_0 block: d = 0.5, qs = 0..32 as i8 (with negatives).
        let mut b = vec![0u8; 34];
        b[..2].copy_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());
        for j in 0..32 {
            b[2 + j] = (j as i8 - 16) as u8;
        }
        let mut y = vec![0f32; 32];
        dequant_row(GgmlType::Q8_0, &b, &mut y);
        assert_eq!(y[0], -8.0);
        assert_eq!(y[31], 7.5);

        let mut b = vec![0u8; 18];
        b[..2].copy_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes());
        b[2] = 0x9F; // low nibble 15 -> +7, high nibble 9 -> +1
        dequant_row(GgmlType::Q4_0, &b, &mut y);
        assert_eq!(y[0], 14.0);
        assert_eq!(y[16], 2.0);
        assert_eq!(y[1], -16.0);
    }

    #[test]
    fn matmul_matches_naive() {
        let (rows, cols, s) = (37, 64, 3);
        let wf: Vec<f32> = (0..rows * cols).map(|i| ((i * 31 % 17) as f32 - 8.0) * 0.1).collect();
        let w: Vec<u8> = wf.iter().flat_map(|v| v.to_le_bytes()).collect();
        let x: Vec<f32> = (0..s * cols).map(|i| (i % 5) as f32 - 2.0).collect();
        let mut out = vec![0f32; s * rows];
        matmul(GgmlType::F32, &w, rows, cols, &x, &mut out);
        for si in 0..s {
            for r in 0..rows {
                let e: f32 = (0..cols).map(|c| wf[r * cols + c] * x[si * cols + c]).sum();
                assert!((out[si * rows + r] - e).abs() < 1e-3);
            }
        }
    }
}

//! Hand-written AVX2 kernels for the hottest decode formats.
//!
//! Weight bytes are consumed in place (no unpack): 4-bit and 6-bit values
//! are unsigned, so `_mm256_maddubs_epi16(weights_u8, activations_i8)` gives
//! pairwise products directly, `_mm256_madd_epi16(·, 1)` widens them to i32
//! per four elements, and per-block float scales are applied with one FMA.
//! In a 256-bit vector, elements 0–15 of a 32-element run land in the low
//! 128-bit half and 16–31 in the high half, which is how per-16 scales (Q6_K)
//! are applied with `_mm256_set_m128`.
//!
//! Every kernel is checked against the scalar reference in tests.

#![cfg(target_arch = "x86_64")]

use crate::qdot::{Q8Row, Unpacked};
use crate::quant::f16;
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn hsum(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

/// Σ u8·i8 over 32 bytes → 8 × i32 (lanes 0–3: elements 0–15, 4–7: 16–31).
#[inline(always)]
unsafe fn mul_u8_i8(w: __m256i, x: __m256i) -> __m256i {
    _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), _mm256_set1_epi16(1))
}

/// Σ i8·i8 via the sign trick.
#[inline(always)]
unsafe fn mul_i8_i8(w: __m256i, x: __m256i) -> __m256i {
    let aw = _mm256_sign_epi8(w, w);
    let sx = _mm256_sign_epi8(x, w);
    mul_u8_i8(aw, sx)
}

#[inline(always)]
unsafe fn load(p: *const u8) -> __m256i {
    _mm256_loadu_si256(p as *const __m256i)
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q8_0(w: &[u8], a: &Q8Row) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for (b, blk) in w.chunks_exact(34).enumerate() {
        let qw = load(blk.as_ptr().add(2));
        let qx = load(a.q.as_ptr().add(b * 32) as *const u8);
        let p = _mm256_cvtepi32_ps(mul_i8_i8(qw, qx));
        acc = _mm256_fmadd_ps(p, _mm256_set1_ps(f16(blk[0], blk[1]) * a.d[b]), acc);
    }
    hsum(acc)
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q4_0(w: &[u8], a: &Q8Row) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let m4 = _mm256_set1_epi8(0x0F);
    let eight = _mm256_set1_epi8(8);
    for (b, blk) in w.chunks_exact(18).enumerate() {
        let q = _mm_loadu_si128(blk.as_ptr().add(2) as *const __m128i);
        // low nibbles → elements 0..16, high → 16..32
        let both = _mm256_set_m128i(_mm_srli_epi16(q, 4), q);
        let qw = _mm256_sub_epi8(_mm256_and_si256(both, m4), eight);
        let qx = load(a.q.as_ptr().add(b * 32) as *const u8);
        let p = _mm256_cvtepi32_ps(mul_i8_i8(qw, qx));
        acc = _mm256_fmadd_ps(p, _mm256_set1_ps(f16(blk[0], blk[1]) * a.d[b]), acc);
    }
    hsum(acc)
}

#[inline(always)]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q4_k(w: &[u8], a: &Q8Row) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut mins = 0f32;
    let m4 = _mm256_set1_epi8(0x0F);
    for (sb, blk) in w.chunks_exact(144).enumerate() {
        let d = f16(blk[0], blk[1]);
        let dmin = f16(blk[2], blk[3]);
        let scales = &blk[4..16];
        let qs = blk.as_ptr().add(16);
        for k in 0..4 {
            let q = load(qs.add(k * 32));
            let lo = _mm256_and_si256(q, m4);
            let hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            let b0 = sb * 8 + 2 * k;
            let b1 = b0 + 1;
            let x0 = load(a.q.as_ptr().add(b0 * 32) as *const u8);
            let x1 = load(a.q.as_ptr().add(b1 * 32) as *const u8);
            let (s0, m0) = scale_min_k4(2 * k, scales);
            let (s1, m1) = scale_min_k4(2 * k + 1, scales);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(mul_u8_i8(lo, x0)), _mm256_set1_ps(d * s0 as f32 * a.d[b0]), acc);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(mul_u8_i8(hi, x1)), _mm256_set1_ps(d * s1 as f32 * a.d[b1]), acc);
            mins += dmin * (m0 as f32 * a.d[b0] * (a.s16[2 * b0] + a.s16[2 * b0 + 1]) as f32 + m1 as f32 * a.d[b1] * (a.s16[2 * b1] + a.s16[2 * b1 + 1]) as f32);
        }
    }
    hsum(acc) - mins
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q6_k(w: &[u8], a: &Q8Row) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut offs = 0f32;
    let m4 = _mm256_set1_epi8(0x0F);
    let m3 = _mm256_set1_epi8(0x03);
    for (sb, blk) in w.chunks_exact(210).enumerate() {
        let d = f16(blk[208], blk[209]);
        for h in 0..2 {
            let ql = blk.as_ptr().add(h * 64);
            let qh = load(blk.as_ptr().add(128 + h * 32));
            let s = &blk[192 + h * 8..192 + h * 8 + 8];
            let qa = load(ql);
            let qb = load(ql.add(32));
            let r = [
                _mm256_or_si256(_mm256_and_si256(qa, m4), _mm256_slli_epi16(_mm256_and_si256(qh, m3), 4)),
                _mm256_or_si256(_mm256_and_si256(qb, m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2), m3), 4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qa, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 4), m3), 4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 6), m3), 4)),
            ];
            for (ri, rv) in r.iter().enumerate() {
                let b = sb * 8 + h * 4 + ri;
                let x = load(a.q.as_ptr().add(b * 32) as *const u8);
                let lo = d * (s[2 * ri] as i8) as f32 * a.d[b];
                let hi = d * (s[2 * ri + 1] as i8) as f32 * a.d[b];
                let sc = _mm256_set_m128(_mm_set1_ps(hi), _mm_set1_ps(lo));
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(mul_u8_i8(*rv, x)), sc, acc);
                // values are stored +32: subtract 32 · Σx per 16
                offs += 32.0 * (lo * a.s16[2 * b] as f32 + hi * a.s16[2 * b + 1] as f32);
            }
        }
    }
    hsum(acc) - offs
}

/// Dot of an unpacked row (any format) with a quantized activation row.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_unpacked(u: &Unpacked, a: &Q8Row) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut mins = 0f32;
    let nb = u.q.len() / 32;
    for b in 0..nb {
        let qw = load(u.q.as_ptr().add(b * 32) as *const u8);
        let qx = load(a.q.as_ptr().add(b * 32) as *const u8);
        let p = _mm256_cvtepi32_ps(mul_i8_i8(qw, qx));
        let sc = _mm256_set_m128(_mm_set1_ps(u.scale[2 * b + 1] * a.d[b]), _mm_set1_ps(u.scale[2 * b] * a.d[b]));
        acc = _mm256_fmadd_ps(p, sc, acc);
        mins += a.d[b] * (u.min[2 * b] * a.s16[2 * b] as f32 + u.min[2 * b + 1] * a.s16[2 * b + 1] as f32);
    }
    hsum(acc) - mins
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qdot::{unpack, Q8Row, Unpacked};
    use kestrel_gguf::GgmlType;

    fn rnd(n: usize, seed: u32) -> Vec<u8> {
        let mut x = seed.max(1);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x as u8
            })
            .collect()
    }

    #[test]
    fn fused_kernels_match_unpacked_reference() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        let n = 2048;
        let x: Vec<f32> = (0..n).map(|i| ((i * 53 % 211) as f32 - 105.0) / 40.0).collect();
        let a = Q8Row::quantize(&x);
        for (t, f) in [
            (GgmlType::Q8_0, dot_q8_0 as unsafe fn(&[u8], &Q8Row) -> f32),
            (GgmlType::Q4_0, dot_q4_0),
            (GgmlType::Q4_K, dot_q4_k),
            (GgmlType::Q6_K, dot_q6_k),
        ] {
            let mut w = rnd(t.bytes_for(n as u64).unwrap() as usize, 9);
            let (bs, offs): (usize, &[usize]) = match t {
                GgmlType::Q8_0 | GgmlType::Q4_0 => (t.type_size(), &[0]),
                GgmlType::Q4_K => (144, &[0, 2]),
                _ => (210, &[208]),
            };
            for blk in w.chunks_exact_mut(bs) {
                for &o in offs {
                    blk[o..o + 2].copy_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
                }
            }
            let mut u = Unpacked::default();
            unpack(t, &w, n, &mut u);
            // Scalar reference over the unpacked representation.
            let mut r = 0f32;
            for k in 0..n / 16 {
                let p: i32 = (0..16).map(|j| u.q[k * 16 + j] as i32 * a.q[k * 16 + j] as i32).sum();
                r += a.d[k / 2] * (u.scale[k] * p as f32 - u.min[k] * a.s16[k] as f32);
            }
            let fused = unsafe { f(&w, &a) };
            let generic = unsafe { dot_unpacked(&u, &a) };
            let tol = 1e-4 * r.abs().max(1.0);
            assert!((fused - r).abs() <= tol, "{t}: fused {fused} vs ref {r}");
            assert!((generic - r).abs() <= tol, "{t}: generic {generic} vs ref {r}");
        }
    }
}

/// `res[r * s_rows + s] = w[r] · x[s]` for f32 rows of length `cols`
/// (a multiple of 8). Register-blocked 3 weight rows × 4 activation rows so
/// each loaded vector feeds several FMAs; remainders fall back to 1×1.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_f32(w: &[f32], rows: usize, x: &[f32], s_rows: usize, cols: usize, res: &mut [f32]) {
    debug_assert_eq!(cols % 8, 0);
    let wp = w.as_ptr();
    let xp = x.as_ptr();
    let rows3 = rows / 3 * 3;
    // Outer loop over activation blocks (kept in L1), inner over the weight
    // tile (kept in L2).
    let mut s = 0;
    while s + 4 <= s_rows {
        let (x0, x1, x2, x3) = (xp.add(s * cols), xp.add((s + 1) * cols), xp.add((s + 2) * cols), xp.add((s + 3) * cols));
        let mut r = 0;
        while r < rows3 {
            let mut acc = [_mm256_setzero_ps(); 12];
            let (w0, w1, w2) = (wp.add(r * cols), wp.add((r + 1) * cols), wp.add((r + 2) * cols));
            let mut c = 0;
            while c < cols {
                let a0 = _mm256_loadu_ps(w0.add(c));
                let a1 = _mm256_loadu_ps(w1.add(c));
                let a2 = _mm256_loadu_ps(w2.add(c));
                let b = _mm256_loadu_ps(x0.add(c));
                acc[0] = _mm256_fmadd_ps(a0, b, acc[0]);
                acc[1] = _mm256_fmadd_ps(a1, b, acc[1]);
                acc[2] = _mm256_fmadd_ps(a2, b, acc[2]);
                let b = _mm256_loadu_ps(x1.add(c));
                acc[3] = _mm256_fmadd_ps(a0, b, acc[3]);
                acc[4] = _mm256_fmadd_ps(a1, b, acc[4]);
                acc[5] = _mm256_fmadd_ps(a2, b, acc[5]);
                let b = _mm256_loadu_ps(x2.add(c));
                acc[6] = _mm256_fmadd_ps(a0, b, acc[6]);
                acc[7] = _mm256_fmadd_ps(a1, b, acc[7]);
                acc[8] = _mm256_fmadd_ps(a2, b, acc[8]);
                let b = _mm256_loadu_ps(x3.add(c));
                acc[9] = _mm256_fmadd_ps(a0, b, acc[9]);
                acc[10] = _mm256_fmadd_ps(a1, b, acc[10]);
                acc[11] = _mm256_fmadd_ps(a2, b, acc[11]);
                c += 8;
            }
            for si in 0..4 {
                for ri in 0..3 {
                    res[(r + ri) * s_rows + s + si] = hsum(acc[si * 3 + ri]);
                }
            }
            r += 3;
        }
        for r in rows3..rows {
            for si in 0..4 {
                res[r * s_rows + s + si] = dot_f32(wp.add(r * cols), xp.add((s + si) * cols), cols);
            }
        }
        s += 4;
    }
    for s in s..s_rows {
        for r in 0..rows {
            res[r * s_rows + s] = dot_f32(wp.add(r * cols), xp.add(s * cols), cols);
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_f32(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut c = 0;
    while c + 16 <= n {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(c)), _mm256_loadu_ps(b.add(c)), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(c + 8)), _mm256_loadu_ps(b.add(c + 8)), acc1);
        c += 16;
    }
    while c + 8 <= n {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(c)), _mm256_loadu_ps(b.add(c)), acc0);
        c += 8;
    }
    let mut s = hsum(_mm256_add_ps(acc0, acc1));
    while c < n {
        s += *a.add(c) * *b.add(c);
        c += 1;
    }
    s
}

#[cfg(test)]
mod gemm_tests {
    #[test]
    fn gemm_matches_naive() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        for (rows, s_rows, cols) in [(7, 9, 64), (3, 4, 8), (1, 1, 24), (10, 3, 256)] {
            let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1).collect();
            let x: Vec<f32> = (0..s_rows * cols).map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.2).collect();
            let mut res = vec![0f32; rows * s_rows];
            unsafe { super::gemm_f32(&w, rows, &x, s_rows, cols, &mut res) };
            for r in 0..rows {
                for s in 0..s_rows {
                    let e: f32 = (0..cols).map(|c| w[r * cols + c] * x[s * cols + c]).sum();
                    assert!((res[r * s_rows + s] - e).abs() < 1e-3, "{rows}x{s_rows}x{cols} r{r} s{s}");
                }
            }
        }
    }
}

/// f16 → f32 conversion, 8 values per instruction (F16C).
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn f16_to_f32(src: &[u16], dst: &mut [f32]) {
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i + 8 <= n {
        let h = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_cvtph_ps(h));
        i += 8;
    }
    while i < n {
        dst[i] = half::f16::from_bits(src[i]).to_f32();
        i += 1;
    }
}

/// Q4_K · Q8_K: sub-block scales applied with `madd` in integer arithmetic,
/// one float FMA per 256 weights.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q4_k_8k(w: &[u8], a: &crate::qdot::Q8KRow) -> f32 {
    let m4 = _mm256_set1_epi8(0x0F);
    let mut acc = _mm256_setzero_ps();
    let mut summs = 0f32;
    for (b, blk) in w.chunks_exact(144).enumerate() {
        let d = f16(blk[0], blk[1]) * a.d[b];
        let dmin = f16(blk[2], blk[3]) * a.d[b];
        let scales = &blk[4..16];
        let qs = blk.as_ptr().add(16);
        let xq = a.q.as_ptr().add(b * 256) as *const u8;
        let mut sumi = _mm256_setzero_si256();
        let mut mins = 0i32;
        for k in 0..4 {
            let (s0, m0) = scale_min_k4(2 * k, scales);
            let (s1, m1) = scale_min_k4(2 * k + 1, scales);
            let q = load(qs.add(k * 32));
            let lo = _mm256_and_si256(q, m4);
            let hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            let p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, load(xq.add(64 * k))), _mm256_set1_epi16(s0 as i16));
            let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, load(xq.add(64 * k + 32))), _mm256_set1_epi16(s1 as i16));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p1));
            let bs = &a.bsums[b * 16 + 4 * k..b * 16 + 4 * k + 4];
            mins += m0 as i32 * (bs[0] + bs[1]) + m1 as i32 * (bs[2] + bs[3]);
        }
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi), _mm256_set1_ps(d), acc);
        summs += dmin * mins as f32;
    }
    hsum(acc) - summs
}

/// Q6_K · Q8_K: 6-bit values as unsigned bytes (offset 32 corrected with
/// the activation block sums), int8 sub-block scales applied with `madd`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q6_k_8k(w: &[u8], a: &crate::qdot::Q8KRow) -> f32 {
    let m4 = _mm256_set1_epi8(0x0F);
    let m3 = _mm256_set1_epi8(0x03);
    let mut acc = _mm256_setzero_ps();
    for (b, blk) in w.chunks_exact(210).enumerate() {
        let d = f16(blk[208], blk[209]) * a.d[b];
        let xq = a.q.as_ptr().add(b * 256) as *const u8;
        let mut sumi = _mm256_setzero_si256();
        let mut offs = 0i32;
        for h in 0..2 {
            let ql = blk.as_ptr().add(h * 64);
            let qh = load(blk.as_ptr().add(128 + h * 32));
            let sc = &blk[192 + h * 8..192 + h * 8 + 8];
            let qa = load(ql);
            let qb = load(ql.add(32));
            let r = [
                _mm256_or_si256(_mm256_and_si256(qa, m4), _mm256_slli_epi16(_mm256_and_si256(qh, m3), 4)),
                _mm256_or_si256(_mm256_and_si256(qb, m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2), m3), 4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qa, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 4), m3), 4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 6), m3), 4)),
            ];
            for (ri, rv) in r.iter().enumerate() {
                let (s0, s1) = (sc[2 * ri] as i8 as i16, sc[2 * ri + 1] as i8 as i16);
                let x = load(xq.add(h * 128 + ri * 32));
                let p = _mm256_madd_epi16(_mm256_maddubs_epi16(*rv, x), _mm256_set_m128i(_mm_set1_epi16(s1), _mm_set1_epi16(s0)));
                sumi = _mm256_add_epi32(sumi, p);
                let k16 = b * 16 + h * 8 + ri * 2;
                offs += s0 as i32 * a.bsums[k16] + s1 as i32 * a.bsums[k16 + 1];
            }
        }
        // Σ sc·(q−32)·x = Σ sc·q·x − 32·Σ sc·Σx
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi), _mm256_set1_ps(d), acc);
        acc = _mm256_add_ps(acc, _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -32.0 * d * offs as f32));
    }
    hsum(acc)
}

#[cfg(test)]
mod q8k_tests {
    use crate::qdot::{Q8KRow, Unpacked};

    #[test]
    fn q8k_kernels_match_dequantized_math() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        let n = 2048;
        let x: Vec<f32> = (0..n).map(|i| ((i * 53 % 211) as f32 - 105.0) / 40.0).collect();
        let a = Q8KRow::quantize(&x);
        for t in [kestrel_gguf::GgmlType::Q4_K, kestrel_gguf::GgmlType::Q6_K] {
            let mut st = 7u32;
            let mut w: Vec<u8> = (0..t.bytes_for(n as u64).unwrap() as usize)
                .map(|_| {
                    st ^= st << 13;
                    st ^= st >> 17;
                    st ^= st << 5;
                    st as u8
                })
                .collect();
            let offs: &[usize] = if t == kestrel_gguf::GgmlType::Q4_K { &[0, 2] } else { &[208] };
            for blk in w.chunks_exact_mut(t.type_size()) {
                for &o in offs {
                    blk[o..o + 2].copy_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
                }
            }
            // Reference: exactly dequantized weights · dequantized activations.
            let mut u = Unpacked::default();
            crate::qdot::unpack(t, &w, n, &mut u);
            let mut r = 0f64;
            for i in 0..n {
                let wv = (u.scale[i / 16] * u.q[i] as f32 - u.min[i / 16]) as f64;
                r += wv * a.q[i] as f64 * a.d[i / 256] as f64;
            }
            let got = unsafe { if t == kestrel_gguf::GgmlType::Q4_K { super::dot_q4_k_8k(&w, &a) } else { super::dot_q6_k_8k(&w, &a) } } as f64;
            assert!((got - r).abs() <= 1e-3 * r.abs().max(1.0), "{t}: {got} vs {r}");
        }
    }
}

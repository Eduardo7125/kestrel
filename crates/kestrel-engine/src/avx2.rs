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

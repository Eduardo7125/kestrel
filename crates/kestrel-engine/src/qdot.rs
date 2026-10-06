//! Integer dot products against int8-quantized activations.
//!
//! Like llama.cpp, the fast path quantizes each activation row to int8 in
//! blocks of 32 (`d = max|x| / 127`) once. Each weight row is *unpacked* once
//! into a uniform representation: int8 values plus, per 16 elements, a float
//! scale and min, so `w = scale·q − min`. That covers every legacy and
//! K-quant block format. The dot product against each activation row is then
//! a tight int8 multiply-accumulate that LLVM vectorizes:
//!
//! ```text
//! w·x = Σ_16-blocks  dx · (scale · Σ q·qx  −  min · Σ qx)
//! ```
//!
//! Prefill amortizes the unpack over all S activation rows. Kernels are
//! compiled twice (baseline and AVX2+FMA) and selected at runtime.

use crate::quant::f16;
use kestrel_gguf::GgmlType;

/// An activation row quantized to int8 in blocks of 32.
pub struct Q8Row {
    pub q: Vec<i8>,
    /// Scale per 32-block.
    pub d: Vec<f32>,
    /// Σq per 16 elements.
    pub s16: Vec<i32>,
}

impl Q8Row {
    pub fn quantize(x: &[f32]) -> Self {
        assert_eq!(x.len() % 32, 0);
        let nb = x.len() / 32;
        let mut q = vec![0i8; x.len()];
        let mut d = vec![0f32; nb];
        let mut s16 = vec![0i32; nb * 2];
        for b in 0..nb {
            let xs = &x[b * 32..(b + 1) * 32];
            let amax = xs.iter().fold(0f32, |a, v| a.max(v.abs()));
            let scale = amax / 127.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            d[b] = scale;
            for j in 0..32 {
                q[b * 32 + j] = ((xs[j] * inv).round() as i32).clamp(-127, 127) as i8;
            }
            s16[2 * b] = q[b * 32..b * 32 + 16].iter().map(|&v| v as i32).sum();
            s16[2 * b + 1] = q[b * 32 + 16..b * 32 + 32].iter().map(|&v| v as i32).sum();
        }
        Q8Row { q, d, s16 }
    }
}

/// One weight row as int8 values with per-16 scale and min.
#[derive(Default)]
pub struct Unpacked {
    pub q: Vec<i8>,
    pub scale: Vec<f32>,
    pub min: Vec<f32>,
}

pub fn supports(t: GgmlType) -> bool {
    use GgmlType::*;
    matches!(t, Q8_0 | Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q4_K | Q5_K | Q6_K)
}

#[inline(always)]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

#[inline(always)]
fn unpack_impl(t: GgmlType, w: &[u8], n: usize, u: &mut Unpacked) {
    u.q.resize(n, 0);
    u.scale.resize(n / 16, 0.0);
    u.min.resize(n / 16, 0.0);
    let (q, sc, mn) = (&mut u.q[..], &mut u.scale[..], &mut u.min[..]);
    match t {
        GgmlType::Q8_0 => {
            for (b, blk) in w.chunks_exact(34).enumerate() {
                let d = f16(blk[0], blk[1]);
                for j in 0..32 {
                    q[b * 32 + j] = blk[2 + j] as i8;
                }
                sc[2 * b] = d;
                sc[2 * b + 1] = d;
                mn[2 * b] = 0.0;
                mn[2 * b + 1] = 0.0;
            }
        }
        GgmlType::Q4_0 | GgmlType::Q4_1 => {
            let (bs, off) = if t == GgmlType::Q4_0 { (18, 2) } else { (20, 4) };
            for (b, blk) in w.chunks_exact(bs).enumerate() {
                let d = f16(blk[0], blk[1]);
                let (bias, m) = if t == GgmlType::Q4_0 { (8i8, 0.0) } else { (0i8, -f16(blk[2], blk[3])) };
                let qs = &blk[off..off + 16];
                for j in 0..16 {
                    q[b * 32 + j] = (qs[j] & 0xF) as i8 - bias;
                    q[b * 32 + j + 16] = (qs[j] >> 4) as i8 - bias;
                }
                sc[2 * b] = d;
                sc[2 * b + 1] = d;
                mn[2 * b] = m;
                mn[2 * b + 1] = m;
            }
        }
        GgmlType::Q5_0 | GgmlType::Q5_1 => {
            let (bs, off) = if t == GgmlType::Q5_0 { (22, 2) } else { (24, 4) };
            for (b, blk) in w.chunks_exact(bs).enumerate() {
                let d = f16(blk[0], blk[1]);
                let (bias, m) = if t == GgmlType::Q5_0 { (16i8, 0.0) } else { (0i8, -f16(blk[2], blk[3])) };
                let qh = u32::from_le_bytes([blk[off], blk[off + 1], blk[off + 2], blk[off + 3]]);
                let qs = &blk[off + 4..off + 20];
                for j in 0..16 {
                    let h0 = (((qh >> j) << 4) & 0x10) as u8;
                    let h1 = ((qh >> (j + 12)) & 0x10) as u8;
                    q[b * 32 + j] = ((qs[j] & 0xF) | h0) as i8 - bias;
                    q[b * 32 + j + 16] = ((qs[j] >> 4) | h1) as i8 - bias;
                }
                sc[2 * b] = d;
                sc[2 * b + 1] = d;
                mn[2 * b] = m;
                mn[2 * b + 1] = m;
            }
        }
        GgmlType::Q4_K | GgmlType::Q5_K => {
            let k5 = t == GgmlType::Q5_K;
            for (sb, blk) in w.chunks_exact(t.type_size()).enumerate() {
                let d = f16(blk[0], blk[1]);
                let dmin = f16(blk[2], blk[3]);
                let scales = &blk[4..16];
                let (qh, qs) = if k5 { (&blk[16..48], &blk[48..176]) } else { (&blk[0..0], &blk[16..144]) };
                for k in 0..4 {
                    let src = &qs[k * 32..k * 32 + 32];
                    let base = sb * 256 + k * 64;
                    for j in 0..32 {
                        let (mut lo, mut hi) = (src[j] & 0xF, src[j] >> 4);
                        if k5 {
                            lo |= ((qh[j] >> (2 * k)) & 1) << 4;
                            hi |= ((qh[j] >> (2 * k + 1)) & 1) << 4;
                        }
                        q[base + j] = lo as i8;
                        q[base + 32 + j] = hi as i8;
                    }
                    let (s0, m0) = scale_min_k4(2 * k, scales);
                    let (s1, m1) = scale_min_k4(2 * k + 1, scales);
                    let i16b = (sb * 256 + k * 64) / 16;
                    for h in 0..2 {
                        sc[i16b + h] = d * s0 as f32;
                        mn[i16b + h] = dmin * m0 as f32;
                        sc[i16b + 2 + h] = d * s1 as f32;
                        mn[i16b + 2 + h] = dmin * m1 as f32;
                    }
                }
            }
        }
        GgmlType::Q6_K => {
            for (sb, blk) in w.chunks_exact(210).enumerate() {
                let d = f16(blk[208], blk[209]);
                for h in 0..2 {
                    let (ql, qh, s) = (&blk[h * 64..], &blk[128 + h * 32..], &blk[192 + h * 8..]);
                    let base = sb * 256 + h * 128;
                    for l in 0..32 {
                        q[base + l] = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                        q[base + l + 32] = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                        q[base + l + 64] = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                        q[base + l + 96] = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                    }
                    // y[l + 32r] uses scale s[2r + l/16].
                    for r in 0..4 {
                        for half in 0..2 {
                            let i16b = (base + 32 * r + 16 * half) / 16;
                            sc[i16b] = d * (s[2 * r + half] as i8) as f32;
                            mn[i16b] = 0.0;
                        }
                    }
                }
            }
        }
        _ => unreachable!("qdot does not support {t}"),
    }
}

#[inline(always)]
fn dot_impl(u: &Unpacked, a: &Q8Row) -> f32 {
    let mut sum = 0f32;
    let nb16 = u.scale.len();
    for k in 0..nb16 {
        let wq = &u.q[k * 16..k * 16 + 16];
        let xq = &a.q[k * 16..k * 16 + 16];
        let mut p = 0i32;
        for j in 0..16 {
            p += wq[j] as i32 * xq[j] as i32;
        }
        sum += a.d[k / 2] * (u.scale[k] * p as f32 - u.min[k] * a.s16[k] as f32);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn unpack_avx2(t: GgmlType, w: &[u8], n: usize, u: &mut Unpacked) {
    unpack_impl(t, w, n, u)
}

fn use_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *F.get_or_init(|| std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Unpack one weight row of `n` elements.
#[inline]
pub fn unpack(t: GgmlType, w: &[u8], n: usize, u: &mut Unpacked) {
    #[cfg(target_arch = "x86_64")]
    if use_avx2() {
        return unsafe { unpack_avx2(t, w, n, u) };
    }
    unpack_impl(t, w, n, u)
}

/// Dot product of an unpacked weight row with a quantized activation row.
#[inline]
pub fn dot(u: &Unpacked, a: &Q8Row) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if use_avx2() {
        return unsafe { crate::avx2::dot_unpacked(u, a) };
    }
    dot_impl(u, a)
}

/// Whether `t` has a fused kernel that reads packed weight bytes directly.
pub fn has_fused(t: GgmlType) -> bool {
    #[cfg(target_arch = "x86_64")]
    if use_avx2() {
        return matches!(t, GgmlType::Q8_0 | GgmlType::Q4_0 | GgmlType::Q4_K | GgmlType::Q6_K);
    }
    let _ = t;
    false
}

/// Fused dot product on packed weight bytes (requires [`has_fused`]).
#[inline]
pub fn dot_fused(t: GgmlType, w: &[u8], a: &Q8Row) -> f32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        return match t {
            GgmlType::Q8_0 => crate::avx2::dot_q8_0(w, a),
            GgmlType::Q4_0 => crate::avx2::dot_q4_0(w, a),
            GgmlType::Q4_K => crate::avx2::dot_q4_k(w, a),
            GgmlType::Q6_K => crate::avx2::dot_q6_k(w, a),
            _ => unreachable!(),
        };
    }
    #[allow(unreachable_code)]
    {
        let _ = (w, a);
        unreachable!("no fused kernel for {t}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::dequant_row;

    fn blocks(t: GgmlType, n: usize, seed: u32) -> Vec<u8> {
        let bytes = t.bytes_for(n as u64).unwrap() as usize;
        let mut x = seed.max(1);
        let mut v: Vec<u8> = (0..bytes)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x as u8
            })
            .collect();
        let bs = t.type_size();
        let offs: &[usize] = match t {
            GgmlType::Q8_0 | GgmlType::Q4_0 | GgmlType::Q5_0 => &[0],
            GgmlType::Q4_1 | GgmlType::Q5_1 | GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
            GgmlType::Q6_K => &[208],
            _ => unreachable!(),
        };
        for b in v.chunks_exact_mut(bs) {
            for &o in offs {
                b[o..o + 2].copy_from_slice(&half::f16::from_f32(0.01).to_bits().to_le_bytes());
            }
        }
        v
    }

    #[test]
    fn unpack_matches_dequantization_exactly() {
        let n = 1024;
        for t in [GgmlType::Q8_0, GgmlType::Q4_0, GgmlType::Q4_1, GgmlType::Q5_0, GgmlType::Q5_1, GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K] {
            let w = blocks(t, n, 7);
            let mut wf = vec![0f32; n];
            dequant_row(t, &w, &mut wf);
            let mut u = Unpacked::default();
            unpack(t, &w, n, &mut u);
            for i in 0..n {
                let v = u.scale[i / 16] * u.q[i] as f32 - u.min[i / 16];
                assert!((v - wf[i]).abs() <= 1e-5 * wf[i].abs().max(1.0), "{t} element {i}: {v} vs {}", wf[i]);
            }
        }
    }

    #[test]
    fn dot_close_to_float() {
        let n = 1024;
        let x: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
        let a = Q8Row::quantize(&x);
        for t in [GgmlType::Q8_0, GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::Q5_1] {
            let w = blocks(t, n, 3);
            let mut wf = vec![0f32; n];
            dequant_row(t, &w, &mut wf);
            let exact = crate::quant::dot(&wf, &x);
            let mut u = Unpacked::default();
            unpack(t, &w, n, &mut u);
            let fast = dot(&u, &a);
            let tol = wf.iter().map(|v| v.abs()).sum::<f32>() * 2.0 / 127.0 * 0.6;
            assert!((exact - fast).abs() <= tol.max(1e-3), "{t}: {exact} vs {fast}");
        }
    }
}

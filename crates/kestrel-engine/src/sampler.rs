//! Token sampling: greedy, temperature, top-k, top-p, min-p and repetition
//! penalty, with a seeded generator for reproducibility.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplerConfig {
    /// 0 = greedy.
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig { temperature: 0.8, top_k: 40, top_p: 0.95, min_p: 0.05, repeat_penalty: 1.0, repeat_last_n: 64, seed: 0x5eed }
    }
}

impl SamplerConfig {
    pub fn greedy() -> Self {
        SamplerConfig { temperature: 0.0, ..Default::default() }
    }
}

pub struct Sampler {
    cfg: SamplerConfig,
    rng: u64,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig) -> Self {
        let rng = cfg.seed ^ 0x9E37_79B9_7F4A_7C15;
        Sampler { cfg, rng: rng.max(1) }
    }

    fn next_f32(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> u32 {
        let mut l = logits.to_vec();
        if self.cfg.repeat_penalty != 1.0 {
            let start = history.len().saturating_sub(self.cfg.repeat_last_n);
            for &t in &history[start..] {
                if let Some(v) = l.get_mut(t as usize) {
                    *v = if *v > 0.0 { *v / self.cfg.repeat_penalty } else { *v * self.cfg.repeat_penalty };
                }
            }
        }
        if self.cfg.temperature <= 0.0 {
            return argmax(&l);
        }
        let mut idx: Vec<usize> = (0..l.len()).collect();
        idx.sort_unstable_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap_or(std::cmp::Ordering::Equal));
        if self.cfg.top_k > 0 {
            idx.truncate(self.cfg.top_k.max(1));
        }
        let t = self.cfg.temperature;
        let maxl = l[idx[0]];
        let mut p: Vec<f32> = idx.iter().map(|&i| ((l[i] - maxl) / t).exp()).collect();
        let sum: f32 = p.iter().sum();
        p.iter_mut().for_each(|x| *x /= sum);
        // min-p: drop tokens below min_p × p_max.
        if self.cfg.min_p > 0.0 {
            let thr = p[0] * self.cfg.min_p;
            let keep = p.iter().take_while(|&&x| x >= thr).count().max(1);
            p.truncate(keep);
            idx.truncate(keep);
        }
        // top-p.
        if self.cfg.top_p < 1.0 {
            let mut c = 0.0;
            let mut keep = p.len();
            for (i, &x) in p.iter().enumerate() {
                c += x;
                if c >= self.cfg.top_p {
                    keep = i + 1;
                    break;
                }
            }
            p.truncate(keep);
            idx.truncate(keep);
        }
        let total: f32 = p.iter().sum();
        let mut r = self.next_f32() * total;
        for (i, &x) in p.iter().enumerate() {
            r -= x;
            if r <= 0.0 {
                return idx[i] as u32;
            }
        }
        idx[p.len() - 1] as u32
    }
}

pub fn argmax(l: &[f32]) -> u32 {
    let mut best = 0;
    for i in 1..l.len() {
        if l[i] > l[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_seeded() {
        let logits = vec![0.1, 2.0, 1.5, -1.0];
        assert_eq!(Sampler::new(SamplerConfig::greedy()).sample(&logits, &[]), 1);
        let cfg = SamplerConfig { temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.0, ..Default::default() };
        let a: Vec<u32> = { let mut s = Sampler::new(cfg.clone()); (0..20).map(|_| s.sample(&logits, &[])).collect() };
        let b: Vec<u32> = { let mut s = Sampler::new(cfg); (0..20).map(|_| s.sample(&logits, &[])).collect() };
        assert_eq!(a, b);
        assert!(a.iter().any(|&t| t != 1), "sampling explores");
    }
}

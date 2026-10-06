//! Tokenizers stored in GGUF metadata: `llama` (SentencePiece-style, score
//! merges with byte fallback) and `gpt2` (byte-level BPE with ranked merges
//! and a pre-tokenizer regex). Behaviour follows `src/llama-vocab.cpp`.

use kestrel_gguf::GgufFile;
use std::collections::HashMap;

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("tokenizer metadata missing: {0}")]
    Missing(&'static str),
    #[error("unsupported tokenizer model '{0}' (supported: llama, gpt2)")]
    Unsupported(String),
    #[error("bad pre-tokenizer regex: {0}")]
    Regex(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Spm,
    Bpe,
}

// llama_token_type
const TT_NORMAL: i32 = 1;
const TT_UNKNOWN: i32 = 2;
const TT_CONTROL: i32 = 3;
const TT_USER_DEFINED: i32 = 4;
const TT_BYTE: i32 = 6;

pub struct Tokenizer {
    kind: Kind,
    tokens: Vec<String>,
    scores: Vec<f32>,
    types: Vec<i32>,
    index: HashMap<String, u32>,
    merges: HashMap<(String, String), usize>,
    /// Special tokens (control/user-defined), longest first, for splitting.
    specials: Vec<(String, u32)>,
    pre: Option<fancy_regex::Regex>,
    ignore_merges: bool,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub add_bos: bool,
    add_space_prefix: bool,
    eog: Vec<u32>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
}

const LLAMA3_RE: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN2_RE: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const GPT2_RE: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

const EOG_TEXTS: &[&str] = &[
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "<|eom_id|>",
    "<EOT>",
    "<｜end▁of▁sentence｜>",
];

impl Tokenizer {
    pub fn from_gguf(g: &GgufFile) -> Result<Self, TokenizerError> {
        let model = g.get_str("tokenizer.ggml.model").ok_or(TokenizerError::Missing("tokenizer.ggml.model"))?;
        let kind = match model {
            "llama" => Kind::Spm,
            "gpt2" => Kind::Bpe,
            other => return Err(TokenizerError::Unsupported(other.to_string())),
        };
        let tokens: Vec<String> = g
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .ok_or(TokenizerError::Missing("tokenizer.ggml.tokens"))?
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();
        let n = tokens.len();
        let scores: Vec<f32> = g
            .get("tokenizer.ggml.scores")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect())
            .unwrap_or_else(|| vec![0.0; n]);
        let types: Vec<i32> = g
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(1.0) as i32).collect())
            .unwrap_or_else(|| vec![TT_NORMAL; n]);
        let mut merges = HashMap::new();
        if let Some(a) = g.get("tokenizer.ggml.merges").and_then(|v| v.as_array()) {
            for (rank, m) in a.iter().enumerate() {
                if let Some((l, r)) = m.as_str().and_then(|s| s.split_once(' ')) {
                    merges.insert((l.to_string(), r.to_string()), rank);
                }
            }
        }
        let mut index = HashMap::with_capacity(n);
        for (i, t) in tokens.iter().enumerate() {
            index.entry(t.clone()).or_insert(i as u32);
        }
        let mut specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, t)| matches!(types.get(*i), Some(&TT_CONTROL) | Some(&TT_USER_DEFINED)) && !t.is_empty())
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let pre_name = g.get_str("tokenizer.ggml.pre").unwrap_or("default");
        let (re, ignore_merges) = match pre_name {
            "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "pixtral" => (LLAMA3_RE, true),
            "qwen2" | "deepseek-r1-qwen" => (QWEN2_RE, false),
            _ => (GPT2_RE, false),
        };
        let pre = if kind == Kind::Bpe { Some(fancy_regex::Regex::new(re).map_err(|e| TokenizerError::Regex(e.to_string()))?) } else { None };

        let bos = g.get_u64("tokenizer.ggml.bos_token_id").map(|v| v as u32);
        let eos = g.get_u64("tokenizer.ggml.eos_token_id").map(|v| v as u32);
        let add_bos = g.get("tokenizer.ggml.add_bos_token").and_then(|v| v.as_bool()).unwrap_or(kind == Kind::Spm);
        let add_space_prefix = g.get("tokenizer.ggml.add_space_prefix").and_then(|v| v.as_bool()).unwrap_or(kind == Kind::Spm);

        let mut eog: Vec<u32> = eos.into_iter().collect();
        for key in ["tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"] {
            if let Some(id) = g.get_u64(key) {
                eog.push(id as u32);
            }
        }
        for t in EOG_TEXTS {
            if let Some(&id) = index.get(*t) {
                eog.push(id);
            }
        }
        eog.sort_unstable();
        eog.dedup();

        let (byte_to_char, char_to_byte) = bytes_to_unicode();
        Ok(Tokenizer {
            kind,
            tokens,
            scores,
            types,
            index,
            merges,
            specials,
            pre,
            ignore_merges,
            bos,
            eos,
            add_bos,
            add_space_prefix,
            eog,
            byte_to_char,
            char_to_byte,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_eog(&self, t: u32) -> bool {
        self.eog.contains(&t)
    }

    pub fn token_text(&self, t: u32) -> &str {
        self.tokens.get(t as usize).map(String::as_str).unwrap_or("")
    }

    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.index.get(text).copied()
    }

    /// Tokenize `text`. With `parse_special`, special-token strings in the
    /// text (e.g. `<|im_start|>`) map to their ids.
    pub fn encode(&self, text: &str, add_bos: bool, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_bos && self.add_bos {
            if let Some(b) = self.bos {
                out.push(b);
            }
        }
        let mut first_text = true;
        for frag in self.split_specials(text, parse_special) {
            match frag {
                Frag::Special(id) => out.push(id),
                Frag::Text(s) => {
                    match self.kind {
                        Kind::Spm => self.encode_spm(s, first_text, &mut out),
                        Kind::Bpe => self.encode_bpe(s, &mut out),
                    }
                    first_text = false;
                }
            }
        }
        out
    }

    fn split_specials<'a>(&self, text: &'a str, parse_special: bool) -> Vec<Frag<'a>> {
        let mut frags = Vec::new();
        if !parse_special || self.specials.is_empty() {
            if !text.is_empty() {
                frags.push(Frag::Text(text));
            }
            return frags;
        }
        let mut rest = text;
        'outer: while !rest.is_empty() {
            // Find the earliest special occurrence (longest wins at a position).
            let mut best: Option<(usize, usize, u32)> = None;
            for (s, id) in &self.specials {
                if let Some(p) = rest.find(s.as_str()) {
                    if best.is_none_or(|(bp, bl, _)| p < bp || (p == bp && s.len() > bl)) {
                        best = Some((p, s.len(), *id));
                    }
                }
            }
            match best {
                Some((p, l, id)) => {
                    if p > 0 {
                        frags.push(Frag::Text(&rest[..p]));
                    }
                    frags.push(Frag::Special(id));
                    rest = &rest[p + l..];
                }
                None => {
                    frags.push(Frag::Text(rest));
                    break 'outer;
                }
            }
        }
        frags
    }

    fn encode_spm(&self, text: &str, first: bool, out: &mut Vec<u32>) {
        let mut s = String::with_capacity(text.len() + 3);
        if self.add_space_prefix && first {
            s.push('▁');
        }
        for c in text.chars() {
            s.push(if c == ' ' { '▁' } else { c });
        }
        // Symbols: byte ranges of UTF-8 chars, merged greedily by score.
        let mut syms: Vec<(usize, usize)> = s.char_indices().map(|(i, c)| (i, i + c.len_utf8())).collect();
        loop {
            let mut best: Option<(f32, usize, u32)> = None;
            for i in 0..syms.len().saturating_sub(1) {
                let cand = &s[syms[i].0..syms[i + 1].1];
                if let Some(&id) = self.index.get(cand) {
                    let sc = self.scores[id as usize];
                    if best.is_none_or(|(bs, _, _)| sc > bs) {
                        best = Some((sc, i, id));
                    }
                }
            }
            let Some((_, i, _)) = best else { break };
            syms[i].1 = syms[i + 1].1;
            syms.remove(i + 1);
        }
        for (a, b) in syms {
            let piece = &s[a..b];
            match self.index.get(piece) {
                Some(&id) if self.types.get(id as usize) != Some(&TT_UNKNOWN) => out.push(id),
                _ => {
                    for byte in piece.bytes() {
                        let key = format!("<0x{byte:02X}>");
                        if let Some(&id) = self.index.get(&key) {
                            out.push(id);
                        } else if let Some(&unk) = self.index.get("<unk>") {
                            out.push(unk);
                        }
                    }
                }
            }
        }
    }

    fn encode_bpe(&self, text: &str, out: &mut Vec<u32>) {
        let re = self.pre.as_ref().expect("bpe has a pre-tokenizer");
        for m in re.find_iter(text) {
            let Ok(m) = m else { continue };
            let word: String = m.as_str().bytes().map(|b| self.byte_to_char[b as usize]).collect();
            if self.ignore_merges {
                if let Some(&id) = self.index.get(&word) {
                    out.push(id);
                    continue;
                }
            }
            let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            loop {
                let mut best: Option<(usize, usize)> = None;
                for i in 0..parts.len().saturating_sub(1) {
                    if let Some(&r) = self.merges.get(&(parts[i].clone(), parts[i + 1].clone())) {
                        if best.is_none_or(|(br, _)| r < br) {
                            best = Some((r, i));
                        }
                    }
                }
                let Some((_, i)) = best else { break };
                let right = parts.remove(i + 1);
                parts[i].push_str(&right);
            }
            for p in parts {
                match self.index.get(&p) {
                    Some(&id) => out.push(id),
                    None => {
                        // Should not happen with a complete byte-level vocab.
                        for c in p.chars() {
                            if let Some(&id) = self.index.get(&c.to_string()) {
                                out.push(id);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Bytes of one token as it appears in generated text. Control tokens
    /// render as empty unless `show_special`.
    pub fn token_bytes(&self, t: u32, show_special: bool) -> Vec<u8> {
        let Some(text) = self.tokens.get(t as usize) else { return Vec::new() };
        let ty = self.types.get(t as usize).copied().unwrap_or(TT_NORMAL);
        if ty == TT_CONTROL {
            return if show_special { text.as_bytes().to_vec() } else { Vec::new() };
        }
        match self.kind {
            Kind::Spm => {
                if ty == TT_BYTE {
                    if let Some(h) = text.strip_prefix("<0x").and_then(|s| s.strip_suffix('>')) {
                        if let Ok(b) = u8::from_str_radix(h, 16) {
                            return vec![b];
                        }
                    }
                }
                text.replace('▁', " ").into_bytes()
            }
            Kind::Bpe => {
                if ty == TT_USER_DEFINED {
                    return text.as_bytes().to_vec();
                }
                text.chars().map(|c| self.char_to_byte.get(&c).copied().unwrap_or(b'?')).collect()
            }
        }
    }

    /// Decode a token sequence to text (lossy for incomplete UTF-8).
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();
        for (i, &t) in tokens.iter().enumerate() {
            let mut b = self.token_bytes(t, false);
            if i == 0 && self.kind == Kind::Spm && self.add_space_prefix && b.first() == Some(&b' ') {
                b.remove(0);
            }
            bytes.extend(b);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

enum Frag<'a> {
    Text(&'a str),
    Special(u32),
}

/// GPT-2's reversible byte → printable-unicode map.
fn bytes_to_unicode() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut fwd = ['\0'; 256];
    let mut rev = HashMap::new();
    for (b, c) in bs.into_iter().zip(cs) {
        let ch = char::from_u32(c).unwrap();
        fwd[b as usize] = ch;
        rev.insert(ch, b as u8);
    }
    (fwd, rev)
}

/// Incremental UTF-8 assembly for streaming output: holds back bytes of an
/// incomplete multi-byte character until it is complete.
#[derive(Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let s = s.to_string();
                self.pending.clear();
                s
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if e.error_len().is_some() {
                    // Invalid sequence: emit lossily and reset.
                    let s = String::from_utf8_lossy(&self.pending).into_owned();
                    self.pending.clear();
                    return s;
                }
                let s = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                self.pending.drain(..valid);
                s
            }
        }
    }
    pub fn flush(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

//! A minimal GGUF v3 writer, used by tests and fixture generators to build
//! synthetic models. Tensors are written as F32, F16 or as already-encoded
//! raw blocks of any ggml type.

use crate::{align_up, GgmlType, Value, DEFAULT_ALIGNMENT};
use std::io::Write;
use std::path::Path;

pub enum TensorData {
    F32(Vec<f32>),
    F16(Vec<f32>),
    /// Pre-encoded bytes of the given type.
    Raw(GgmlType, Vec<u8>),
}

impl TensorData {
    fn ggml_type(&self) -> GgmlType {
        match self {
            TensorData::F32(_) => GgmlType::F32,
            TensorData::F16(_) => GgmlType::F16,
            TensorData::Raw(t, _) => *t,
        }
    }
    fn bytes(&self) -> Vec<u8> {
        match self {
            TensorData::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::F16(v) => v.iter().flat_map(|x| f32_to_f16_bits(*x).to_le_bytes()).collect(),
            TensorData::Raw(_, b) => b.clone(),
        }
    }
}

#[derive(Default)]
pub struct GgufWriter {
    kv: Vec<(String, Value)>,
    tensors: Vec<(String, Vec<u64>, TensorData)>,
}

impl GgufWriter {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kv(&mut self, key: &str, value: Value) -> &mut Self {
        self.kv.push((key.to_string(), value));
        self
    }
    /// `dims` in GGUF order (`dims[0]` is the contiguous dimension).
    pub fn tensor(&mut self, name: &str, dims: &[u64], data: TensorData) -> &mut Self {
        self.tensors.push((name.to_string(), dims.to_vec(), data));
        self
    }

    pub fn write(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut offset = 0u64;
        let mut infos = Vec::new();
        let mut payloads = Vec::new();
        for (name, dims, data) in &self.tensors {
            let bytes = data.bytes();
            let n: u64 = dims.iter().product();
            assert_eq!(
                Some(bytes.len() as u64),
                data.ggml_type().bytes_for(n),
                "tensor {name}: payload size does not match dims/type"
            );
            infos.push(HeaderTensor { name: name.clone(), dims: dims.clone(), ggml_type: data.ggml_type(), rel_offset: offset });
            offset = align_up(offset + bytes.len() as u64, DEFAULT_ALIGNMENT);
            payloads.push(bytes);
        }
        out.write_all(&encode_header(*b"GGUF", &self.kv, &infos, DEFAULT_ALIGNMENT, 1))?;
        let mut written = 0u64;
        for p in payloads {
            out.write_all(&p)?;
            written += p.len() as u64;
            let padded = align_up(written, DEFAULT_ALIGNMENT);
            out.write_all(&vec![0u8; (padded - written) as usize])?;
            written = padded;
        }
        out.flush()
    }
}

/// One entry of the tensor info table.
pub struct HeaderTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub rel_offset: u64,
}

/// Encode a GGUF v3 header (magic, metadata, tensor table) padded so the data
/// section starts at a multiple of `data_align` (itself a multiple of the
/// tensor `alignment`). Alignment beyond `alignment` is reached with a
/// `kestrel.pad` string entry, which readers ignore.
pub fn encode_header(magic: [u8; 4], kv: &[(String, Value)], tensors: &[HeaderTensor], alignment: u64, data_align: u64) -> Vec<u8> {
    let build = |pad: Option<usize>| {
        let mut buf = Vec::new();
        buf.extend_from_slice(&magic);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&((kv.len() + pad.is_some() as usize) as u64).to_le_bytes());
        for (k, v) in kv {
            put_str(&mut buf, k);
            put_value(&mut buf, v, true);
        }
        if let Some(n) = pad {
            put_str(&mut buf, "kestrel.pad");
            put_value(&mut buf, &Value::String(" ".repeat(n)), true);
        }
        for t in tensors {
            put_str(&mut buf, &t.name);
            buf.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&t.ggml_type.id().to_le_bytes());
            buf.extend_from_slice(&t.rel_offset.to_le_bytes());
        }
        buf
    };
    let mut buf = if data_align > alignment {
        // The pad entry costs a fixed 35 bytes plus its length.
        let base = build(Some(0)).len() as u64;
        let target = align_up(base, data_align);
        let b = build(Some((target - base) as usize));
        debug_assert_eq!(b.len() as u64 % data_align, 0);
        b
    } else {
        build(None)
    };
    let header_end = align_up(buf.len() as u64, alignment);
    buf.resize(header_end as usize, 0);
    buf
}

fn put_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn type_id(v: &Value) -> u32 {
    match v {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::String(_) => 8,
        Value::Array(_) => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

fn put_value(buf: &mut Vec<u8>, v: &Value, with_type: bool) {
    if with_type {
        buf.extend_from_slice(&type_id(v).to_le_bytes());
    }
    match v {
        Value::U8(x) => buf.push(*x),
        Value::I8(x) => buf.push(*x as u8),
        Value::U16(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::I16(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::U32(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::I32(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::F32(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::Bool(x) => buf.push(*x as u8),
        Value::String(s) => put_str(buf, s),
        Value::Array(a) => {
            let et = a.first().map(type_id).unwrap_or(4);
            buf.extend_from_slice(&et.to_le_bytes());
            buf.extend_from_slice(&(a.len() as u64).to_le_bytes());
            for e in a {
                put_value(buf, e, false);
            }
        }
        Value::U64(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::I64(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::F64(x) => buf.extend_from_slice(&x.to_le_bytes()),
    }
}

/// Round-to-nearest-even f32 → IEEE half conversion.
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && (r & 1) == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) {
        r += 1;
    }
    sign | r as u16
}

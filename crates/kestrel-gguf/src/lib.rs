//! A GGUF reader that never touches weight data.
//!
//! Parsing reads the header, the metadata key/value table and the tensor info
//! table, and computes for every tensor the absolute file offset and byte size
//! of its data. Weight bytes are read later, by whoever owns placement
//! (`kestrel-memory`), with positional reads against [`GgufFile::path`].
//!
//! Format reference: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>.

mod types;
pub mod writer;

pub use types::{file_type_name, GgmlType};

use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";
/// Magic of a Kestrel-prepared container (`kestrel prepare`). The layout is
/// GGUF v3, but routed-expert tensors may be *strided* (see
/// [`TensorInfo::stride`]), which other GGUF readers would misread. The
/// distinct magic makes them refuse the file instead.
pub const KGUF_MAGIC: [u8; 4] = *b"KGUF";
/// Metadata key prefix for per-tensor strides in a prepared container.
pub const STRIDE_KEY_PREFIX: &str = "kestrel.stride.";
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// Defensive limits against corrupt or hostile files: a crafted header could
/// otherwise make us allocate gigabytes before reading a single tensor.
const MAX_STRING_LEN: u64 = 64 << 20;
const MAX_ARRAY_LEN: u64 = 1 << 28;
const MAX_TENSORS: u64 = 1 << 24;
const MAX_KV: u64 = 1 << 20;
const MAX_DIMS: u32 = 8;

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a GGUF file (bad magic {0:?})")]
    BadMagic([u8; 4]),
    #[error("unsupported GGUF version {0} (supported: 2, 3)")]
    Version(u32),
    #[error("corrupt GGUF: {0}")]
    Corrupt(String),
    #[error("unknown ggml tensor type id {id} for tensor '{tensor}'")]
    UnknownType { id: u32, tensor: String },
}

pub type Result<T> = std::result::Result<T, GgufError>;

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64,
            Value::U16(v) => v as u64,
            Value::U32(v) => v as u64,
            Value::U64(v) => v,
            Value::I8(v) if v >= 0 => v as u64,
            Value::I16(v) if v >= 0 => v as u64,
            Value::I32(v) if v >= 0 => v as u64,
            Value::I64(v) if v >= 0 => v as u64,
            Value::Bool(b) => b as u64,
            _ => return None,
        })
    }
    pub fn as_f64(&self) -> Option<f64> {
        Some(match *self {
            Value::F32(v) => v as f64,
            Value::F64(v) => v,
            _ => return self.as_u64().map(|v| v as f64).or(match *self {
                Value::I8(v) => Some(v as f64),
                Value::I16(v) => Some(v as f64),
                Value::I32(v) => Some(v as f64),
                Value::I64(v) => Some(v as f64),
                _ => None,
            }),
        })
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => self.as_u64().map(|v| v != 0),
        }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
    /// A short rendering for reports: arrays are summarized by length.
    pub fn summary(&self) -> String {
        match self {
            Value::String(s) if s.len() > 80 => format!("{:?}… ({} chars)", &s[..s.floor_char_boundary_compat(60)], s.len()),
            Value::String(s) => format!("{s:?}"),
            Value::Array(a) => format!("[array of {}]", a.len()),
            Value::F32(v) => format!("{v}"),
            Value::F64(v) => format!("{v}"),
            Value::Bool(b) => format!("{b}"),
            other => other.as_u64().map(|v| v.to_string()).unwrap_or_else(|| format!("{other:?}")),
        }
    }
}

trait FloorCharBoundary {
    fn floor_char_boundary_compat(&self, i: usize) -> usize;
}
impl FloorCharBoundary for str {
    fn floor_char_boundary_compat(&self, mut i: usize) -> usize {
        if i >= self.len() {
            return self.len();
        }
        while !self.is_char_boundary(i) {
            i -= 1;
        }
        i
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TensorInfo {
    pub name: String,
    /// GGUF/ggml order: `dims[0]` is the contiguous (row) dimension.
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Offset relative to the start of the data section.
    pub rel_offset: u64,
    /// Absolute offset in the file.
    pub offset: u64,
    pub size: u64,
    /// Prepared containers only: the tensor is split along its outermost
    /// dimension into `dims.last()` equal slices, and slice `i` starts at
    /// `offset + i * stride` instead of being contiguous. Used to interleave
    /// the gate/up/down slices of each routed expert so one expert is one
    /// contiguous read.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stride: Option<u64>,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
    /// Number of rows when viewed as a 2-D matrix `[rows, dims[0]]`.
    pub fn rows(&self) -> u64 {
        self.dims.iter().skip(1).product()
    }
    pub fn row_bytes(&self) -> u64 {
        self.size / self.rows().max(1)
    }
    /// Number of outermost-dimension slices (experts, for stacked tensors).
    pub fn n_slices(&self) -> u64 {
        *self.dims.last().unwrap_or(&1)
    }
    pub fn slice_bytes(&self) -> u64 {
        self.size / self.n_slices().max(1)
    }
    /// File offset of outermost slice `i`.
    pub fn slice_offset(&self, i: u64) -> u64 {
        self.offset + i * self.stride.unwrap_or_else(|| self.slice_bytes())
    }
    /// Bytes from `offset` to the end of the last slice.
    pub fn span(&self) -> u64 {
        match self.stride {
            Some(st) => (self.n_slices() - 1) * st + self.slice_bytes(),
            None => self.size,
        }
    }
}

#[derive(Debug)]
pub struct GgufFile {
    pub path: PathBuf,
    pub version: u32,
    /// A Kestrel-prepared container (`KGUF` magic) rather than a plain GGUF.
    pub prepared: bool,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    pub data_offset: u64,
    pub file_size: u64,
    index: std::collections::HashMap<String, usize>,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let file_size = file.metadata()?.len();
        let mut r = Reader { inner: BufReader::with_capacity(1 << 20, file), pos: 0 };

        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        let prepared = magic == KGUF_MAGIC;
        if magic != GGUF_MAGIC && !prepared {
            return Err(GgufError::BadMagic(magic));
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(GgufError::Version(version));
        }
        let n_tensors = r.u64()?;
        let n_kv = r.u64()?;
        if n_tensors > MAX_TENSORS || n_kv > MAX_KV {
            return Err(GgufError::Corrupt(format!("implausible counts: {n_tensors} tensors, {n_kv} kv")));
        }

        let mut metadata = BTreeMap::new();
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            let value = r.value(ty)?;
            metadata.insert(key, value);
        }

        let alignment = match metadata.get("general.alignment") {
            Some(v) => v.as_u64().filter(|a| *a > 0 && a.is_power_of_two()).ok_or_else(|| {
                GgufError::Corrupt("general.alignment must be a power of two".into())
            })?,
            None => DEFAULT_ALIGNMENT,
        };

        let mut raw = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32()?;
            if n_dims == 0 || n_dims > MAX_DIMS {
                return Err(GgufError::Corrupt(format!("tensor '{name}' has {n_dims} dims")));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let type_id = r.u32()?;
            let rel_offset = r.u64()?;
            let ggml_type = GgmlType::from_id(type_id)
                .ok_or(GgufError::UnknownType { id: type_id, tensor: name.clone() })?;
            raw.push((name, dims, ggml_type, rel_offset));
        }

        let data_offset = align_up(r.pos, alignment);
        let mut tensors = Vec::with_capacity(raw.len());
        let mut index = std::collections::HashMap::with_capacity(raw.len());
        for (name, dims, ggml_type, rel_offset) in raw {
            let n: u64 = dims.iter().try_fold(1u64, |a, &d| a.checked_mul(d)).ok_or_else(|| {
                GgufError::Corrupt(format!("tensor '{name}' element count overflows"))
            })?;
            if dims[0] % ggml_type.block_size() as u64 != 0 {
                return Err(GgufError::Corrupt(format!(
                    "tensor '{name}': row length {} not a multiple of {} block size {}",
                    dims[0],
                    ggml_type,
                    ggml_type.block_size()
                )));
            }
            let size = ggml_type.bytes_for(n).unwrap();
            if rel_offset % alignment != 0 {
                return Err(GgufError::Corrupt(format!("tensor '{name}' offset {rel_offset} not aligned")));
            }
            let offset = data_offset + rel_offset;
            let stride = match metadata.get(&format!("{STRIDE_KEY_PREFIX}{name}")) {
                None => None,
                Some(_) if !prepared => {
                    return Err(GgufError::Corrupt(format!("tensor '{name}' is strided in a plain GGUF file")));
                }
                Some(v) => {
                    let st = v.as_u64().ok_or_else(|| GgufError::Corrupt(format!("stride of '{name}' is not an integer")))?;
                    let slices = *dims.last().unwrap();
                    let slice = size / slices.max(1);
                    if slices < 2 || size % slices != 0 || st < slice || st % alignment != 0 {
                        return Err(GgufError::Corrupt(format!("tensor '{name}': invalid stride {st} for {slices} slices of {slice} bytes")));
                    }
                    Some(st)
                }
            };
            let span = match stride {
                Some(st) => (dims.last().unwrap() - 1).checked_mul(st).and_then(|x| x.checked_add(size / dims.last().unwrap())),
                None => Some(size),
            };
            if span.and_then(|sp| offset.checked_add(sp)).is_none_or(|end| end > file_size) {
                return Err(GgufError::Corrupt(format!(
                    "tensor '{name}' [{offset}, +{size}) extends past end of file ({file_size} bytes) — truncated download?"
                )));
            }
            index.insert(name.clone(), tensors.len());
            tensors.push(TensorInfo { name, dims, ggml_type, rel_offset, offset, size, stride });
        }

        Ok(GgufFile { path, version, prepared, metadata, tensors, alignment, data_offset, file_size, index })
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(Value::as_u64)
    }
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(Value::as_f64)
    }
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }
    pub fn architecture(&self) -> Option<&str> {
        self.get_str("general.architecture")
    }
    /// Arch-scoped key, e.g. `arch_key("block_count")` → `llama.block_count`.
    pub fn arch_u64(&self, suffix: &str) -> Option<u64> {
        let arch = self.architecture()?;
        self.get_u64(&format!("{arch}.{suffix}"))
    }
    pub fn arch_f64(&self, suffix: &str) -> Option<f64> {
        let arch = self.architecture()?;
        self.get_f64(&format!("{arch}.{suffix}"))
    }
    /// Arch-scoped value that may be a scalar or a per-layer array.
    pub fn arch_value(&self, suffix: &str) -> Option<&Value> {
        let arch = self.architecture()?;
        self.get(&format!("{arch}.{suffix}"))
    }

    pub fn total_tensor_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.size).sum()
    }

    /// A cheap identity for caches keyed by model: path-independent, derived
    /// from the header (tensor table + size), not from hashing gigabytes.
    pub fn fingerprint(&self) -> String {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut feed = |b: &[u8]| {
            for &x in b {
                h ^= x as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        };
        feed(&self.file_size.to_le_bytes());
        for t in &self.tensors {
            feed(t.name.as_bytes());
            feed(&t.offset.to_le_bytes());
            feed(&t.ggml_type.id().to_le_bytes());
        }
        format!("{h:016x}")
    }
}

pub fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

struct Reader<R> {
    inner: R,
    pos: u64,
}

impl<R: Read + Seek> Reader<R> {
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        self.inner.read_exact(buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                GgufError::Corrupt("unexpected end of file in header".into())
            } else {
                e.into()
            }
        })?;
        self.pos += buf.len() as u64;
        Ok(())
    }
    fn arr<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut b = [0u8; N];
        self.read_exact(&mut b)?;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.arr()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.arr()?))
    }
    fn string(&mut self) -> Result<String> {
        let len = self.u64()?;
        if len > MAX_STRING_LEN {
            return Err(GgufError::Corrupt(format!("string of {len} bytes")));
        }
        let mut b = vec![0u8; len as usize];
        self.read_exact(&mut b)?;
        // Tokenizer vocabularies occasionally hold invalid UTF-8 byte tokens.
        Ok(String::from_utf8(b).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned()))
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(u8::from_le_bytes(self.arr()?)),
            1 => Value::I8(i8::from_le_bytes(self.arr()?)),
            2 => Value::U16(u16::from_le_bytes(self.arr()?)),
            3 => Value::I16(i16::from_le_bytes(self.arr()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.arr()?)),
            6 => Value::F32(f32::from_le_bytes(self.arr()?)),
            7 => Value::Bool(self.arr::<1>()?[0] != 0),
            8 => Value::String(self.string()?),
            9 => {
                let elem_ty = self.u32()?;
                let len = self.u64()?;
                if len > MAX_ARRAY_LEN || elem_ty == 9 {
                    return Err(GgufError::Corrupt(format!("array of {len} elements of type {elem_ty}")));
                }
                let mut v = Vec::with_capacity(len.min(1 << 20) as usize);
                for _ in 0..len {
                    v.push(self.value(elem_ty)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.arr()?)),
            12 => Value::F64(f64::from_le_bytes(self.arr()?)),
            other => return Err(GgufError::Corrupt(format!("unknown metadata value type {other}"))),
        })
    }
    #[allow(dead_code)]
    fn seek(&mut self, pos: u64) -> Result<()> {
        self.inner.seek(SeekFrom::Start(pos))?;
        self.pos = pos;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::{GgufWriter, TensorData};

    #[test]
    fn write_then_read() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.gguf");
        let mut w = GgufWriter::new();
        w.kv("general.architecture", Value::String("llama".into()));
        w.kv("llama.block_count", Value::U32(2));
        w.kv("tokenizer.ggml.tokens", Value::Array(vec![Value::String("a".into()), Value::String("b".into())]));
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        w.tensor("a.weight", &[32, 2], TensorData::F32(data.clone()));
        w.tensor("b.weight", &[8], TensorData::F32(vec![1.0; 8]));
        w.write(&p).unwrap();

        let g = GgufFile::open(&p).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.architecture(), Some("llama"));
        assert_eq!(g.arch_u64("block_count"), Some(2));
        assert_eq!(g.tensors.len(), 2);
        let a = g.tensor("a.weight").unwrap();
        assert_eq!(a.size, 64 * 4);
        assert_eq!(a.offset % 32, 0);
        assert_eq!(a.rows(), 2);
        let bytes = std::fs::read(&p).unwrap();
        let first = f32::from_le_bytes(bytes[a.offset as usize + 4..a.offset as usize + 8].try_into().unwrap());
        assert_eq!(first, 1.0);
    }

    #[test]
    fn rejects_truncated() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.gguf");
        let mut w = GgufWriter::new();
        w.kv("general.architecture", Value::String("llama".into()));
        w.tensor("a.weight", &[1024], TensorData::F32(vec![0.0; 1024]));
        w.write(&p).unwrap();
        let len = std::fs::metadata(&p).unwrap().len();
        let f = std::fs::OpenOptions::new().write(true).open(&p).unwrap();
        f.set_len(len - 100).unwrap();
        let err = GgufFile::open(&p).unwrap_err();
        assert!(err.to_string().contains("past end of file"), "{err}");
    }

    #[test]
    fn rejects_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.gguf");
        std::fs::write(&p, b"GGMLxxxxxxxxxxxxxxxxxxxxxxxx").unwrap();
        assert!(matches!(GgufFile::open(&p), Err(GgufError::BadMagic(_))));
    }
}

//! Positional file reads, optionally bypassing the OS page cache.
//!
//! Streaming weights through the page cache creates a second, invisible copy
//! of every streamed byte: RSS looks fine while the system drifts into swap.
//! Kestrel therefore reads streamed weights with `O_DIRECT` on Linux and falls
//! back to buffered reads followed by `POSIX_FADV_DONTNEED` elsewhere.

use std::fs::File;
use std::io;
use std::path::Path;

/// Alignment required for direct I/O buffers, offsets and lengths. 4 KiB covers
/// every logical block size we expect on NVMe/SATA devices.
pub const DIRECT_ALIGN: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IoMode {
    /// Page-cache reads (what mmap and plain `read` do).
    Buffered,
    /// Bypass the page cache (`O_DIRECT`); requires aligned buffers.
    Direct,
}

/// An owned, aligned, zero-initialized byte buffer.
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
    align: usize,
}

unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    pub fn new(len: usize, align: usize) -> io::Result<Self> {
        let size = len.max(1).div_ceil(align) * align;
        let layout = std::alloc::Layout::from_size_align(size, align)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        // SAFETY: layout has non-zero size.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(io::Error::new(io::ErrorKind::OutOfMemory, format!("allocating {size} bytes")));
        }
        Ok(Self { ptr, len: size, align })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for len bytes for the lifetime of self.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and &mut self guarantees exclusivity.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.len, self.align).unwrap();
        // SAFETY: allocated with the same layout in `new`.
        unsafe { std::alloc::dealloc(self.ptr, layout) }
    }
}

/// A file opened for positional reads in a given mode.
pub struct ReadFile {
    file: File,
    mode: IoMode,
}

impl ReadFile {
    /// Open for reading. `Direct` silently degrades to `Buffered` if the
    /// platform or file system refuses it; check [`ReadFile::mode`].
    pub fn open(path: &Path, want: IoMode) -> io::Result<Self> {
        if want == IoMode::Direct {
            if let Some(file) = open_direct(path) {
                return Ok(Self { file, mode: IoMode::Direct });
            }
        }
        Ok(Self { file: File::open(path)?, mode: IoMode::Buffered })
    }
    pub fn mode(&self) -> IoMode {
        self.mode
    }
    pub fn len(&self) -> io::Result<u64> {
        Ok(self.file.metadata()?.len())
    }

    /// Read exactly `buf.len()` bytes at `offset`. In direct mode `buf`'s
    /// address, `offset` and `buf.len()` must be multiples of [`DIRECT_ALIGN`]
    /// except that the final read may stop at end of file (short tail allowed:
    /// returns the number of bytes actually read).
    pub fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        let mut done = 0usize;
        while done < buf.len() {
            let n = read_at_once(&self.file, &mut buf[done..], offset + done as u64)?;
            if n == 0 {
                break; // EOF
            }
            done += n;
        }
        Ok(done)
    }

    /// Advise the OS that cached pages for this range are no longer needed.
    pub fn drop_cache(&self, offset: u64, len: u64) {
        #[cfg(target_os = "linux")]
        unsafe {
            use std::os::fd::AsRawFd;
            libc::posix_fadvise(self.file.as_raw_fd(), offset as i64, len as i64, libc::POSIX_FADV_DONTNEED);
        }
        #[cfg(not(target_os = "linux"))]
        let _ = (offset, len);
    }
}

#[cfg(target_os = "linux")]
fn open_direct(path: &Path) -> Option<File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(path).ok()
}

#[cfg(target_os = "macos")]
fn open_direct(path: &Path) -> Option<File> {
    use std::os::fd::AsRawFd;
    let f = File::open(path).ok()?;
    // F_NOCACHE stops new caching; it cannot evict pages already resident.
    unsafe { libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1) };
    Some(f)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn open_direct(_path: &Path) -> Option<File> {
    // Windows FILE_FLAG_NO_BUFFERING is on the roadmap; buffered fallback.
    None
}

#[cfg(unix)]
fn read_at_once(f: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    loop {
        match f.read_at(buf, off) {
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            r => return r,
        }
    }
}

#[cfg(windows)]
fn read_at_once(f: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    f.seek_read(buf, off)
}

/// Read the byte range `[offset, offset+len)` of `file` into `dst` (which must
/// be at least `len` bytes). In direct mode the read is widened to aligned
/// boundaries through `scratch`-free in-place staging: `dst` must then be an
/// aligned buffer of at least `aligned_span(offset, len)` bytes, and the data
/// starts at `offset % DIRECT_ALIGN` inside it. Returns that start index.
pub fn read_range(file: &ReadFile, dst: &mut [u8], offset: u64, len: usize) -> io::Result<usize> {
    match file.mode() {
        IoMode::Buffered => {
            let n = file.read_at(&mut dst[..len], offset)?;
            if n != len {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, format!("short read: {n} of {len} at {offset}")));
            }
            Ok(0)
        }
        IoMode::Direct => {
            let a = DIRECT_ALIGN as u64;
            let start = offset / a * a;
            let head = (offset - start) as usize;
            let span = aligned_span(offset, len);
            assert!(dst.len() >= span, "direct read needs {span} byte buffer, got {}", dst.len());
            assert_eq!(dst.as_ptr() as usize % DIRECT_ALIGN, 0, "direct read buffer must be aligned");
            let n = file.read_at(&mut dst[..span], start)?;
            if n < head + len {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, format!("short direct read at {offset}")));
            }
            Ok(head)
        }
    }
}

/// Bytes an aligned read covering `[offset, offset+len)` spans.
pub fn aligned_span(offset: u64, len: usize) -> usize {
    let a = DIRECT_ALIGN as u64;
    let start = offset / a * a;
    let end = (offset + len as u64).div_ceil(a) * a;
    (end - start) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_and_buffered_agree() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f.bin");
        let data: Vec<u8> = (0..300_000u32).map(|i| (i * 7 % 251) as u8).collect();
        std::fs::write(&p, &data).unwrap();
        for mode in [IoMode::Buffered, IoMode::Direct] {
            let f = ReadFile::open(&p, mode).unwrap();
            for &(off, len) in &[(0u64, 10usize), (4095, 4097), (12345, 100_000), (299_000, 1000)] {
                let mut buf = AlignedBuf::new(aligned_span(off, len), DIRECT_ALIGN).unwrap();
                let start = read_range(&f, buf.as_mut_slice(), off, len).unwrap();
                assert_eq!(&buf.as_slice()[start..start + len], &data[off as usize..off as usize + len], "{mode:?} {off}");
            }
        }
    }
}

//! The I/O engine: a pool of workers serving positional reads.
//!
//! NVMe needs queue depth to reach its rated bandwidth, so one group load is
//! split into chunks that several workers read in parallel. Completion is
//! signalled through a [`Ticket`]; the compute thread blocks only if it needs
//! the data before the ticket completes, and that wait is accounted as stall.

use kestrel_hw::fileio::ReadFile;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Largest single read. Large enough for sequential throughput, small enough
/// that one layer spreads over several workers.
pub const CHUNK: usize = 4 << 20;

/// Raw destination pointer; the store guarantees chunks of one load are
/// disjoint and that the buffer outlives the load (it is not leased or
/// reallocated until the ticket completes).
#[derive(Clone, Copy)]
pub(crate) struct DstPtr(pub *mut u8);
unsafe impl Send for DstPtr {}

pub(crate) struct ChunkJob {
    pub file: Arc<ReadFile>,
    pub file_offset: u64,
    pub dst: DstPtr,
    pub len: usize,
    /// Minimum bytes that must be read (the rest may lie past EOF).
    pub required: usize,
    pub ticket: Arc<Ticket>,
}

#[derive(Default)]
pub struct Ticket {
    remaining: AtomicUsize,
    state: Mutex<TicketState>,
    cv: Condvar,
    issued: Mutex<Option<Instant>>,
    /// Sum of per-chunk read durations (disk service time), nanoseconds.
    pub service_ns: AtomicU64,
    pub bytes: AtomicU64,
}

#[derive(Default)]
struct TicketState {
    done: bool,
    error: Option<String>,
    finished: Option<Instant>,
}

impl Ticket {
    pub(crate) fn new(chunks: usize) -> Arc<Self> {
        let t = Ticket::default();
        t.remaining.store(chunks, Ordering::Relaxed);
        *t.issued.lock().unwrap() = Some(Instant::now());
        if chunks == 0 {
            t.state.lock().unwrap().done = true;
        }
        Arc::new(t)
    }

    fn complete_chunk(&self, err: Option<String>) {
        if let Some(e) = err {
            self.state.lock().unwrap().error.get_or_insert(e);
        }
        if self.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            let mut s = self.state.lock().unwrap();
            s.done = true;
            s.finished = Some(Instant::now());
            self.cv.notify_all();
        }
    }

    pub fn is_done(&self) -> bool {
        self.state.lock().unwrap().done
    }

    /// Block until complete. Returns the time spent waiting.
    pub fn wait(&self) -> Result<Duration, String> {
        let t0 = Instant::now();
        let mut s = self.state.lock().unwrap();
        while !s.done {
            s = self.cv.wait(s).unwrap();
        }
        match &s.error {
            Some(e) => Err(e.clone()),
            None => Ok(t0.elapsed()),
        }
    }

    /// Wall time from issue to completion, if complete.
    pub fn latency(&self) -> Option<Duration> {
        let s = self.state.lock().unwrap();
        let issued = (*self.issued.lock().unwrap())?;
        s.finished.map(|f| f - issued)
    }
}

/// Counters shared by the workers. Kept apart from the engine so a worker
/// never holds a strong reference to it: the engine must always be dropped
/// by its owner, whose drop joins the workers *before* the owner frees the
/// buffers that queued reads still target.
#[derive(Default)]
pub struct IoStats {
    pub bytes_read: AtomicU64,
    pub service_ns: AtomicU64,
}

pub struct IoEngine {
    tx: Mutex<Option<Sender<ChunkJob>>>,
    workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
    pub n_workers: usize,
    pub stats: Arc<IoStats>,
}

impl IoEngine {
    pub fn new(n_workers: usize) -> Arc<Self> {
        let n_workers = n_workers.max(1);
        let (tx, rx) = channel::<ChunkJob>();
        let rx = Arc::new(Mutex::new(rx));
        let eng = Arc::new(IoEngine { tx: Mutex::new(Some(tx)), workers: Mutex::new(Vec::new()), n_workers, stats: Arc::new(IoStats::default()) });
        let mut ws = Vec::new();
        for i in 0..n_workers {
            let rx: Arc<Mutex<Receiver<ChunkJob>>> = rx.clone();
            let stats = eng.stats.clone();
            ws.push(
                std::thread::Builder::new()
                    .name(format!("kestrel-io-{i}"))
                    .spawn(move || loop {
                        let job = { rx.lock().unwrap().recv() };
                        let Ok(job) = job else { return };
                        let t0 = Instant::now();
                        // SAFETY: see DstPtr.
                        let dst = unsafe { std::slice::from_raw_parts_mut(job.dst.0, job.len) };
                        let res = job.file.read_at(dst, job.file_offset);
                        let ns = t0.elapsed().as_nanos() as u64;
                        let err = match res {
                            Ok(n) if n >= job.required => None,
                            Ok(n) => Some(format!("short read: {n} of {} bytes at offset {}", job.required, job.file_offset)),
                            Err(e) => Some(format!("read at offset {}: {e}", job.file_offset)),
                        };
                        job.ticket.service_ns.fetch_add(ns, Ordering::Relaxed);
                        job.ticket.bytes.fetch_add(job.required as u64, Ordering::Relaxed);
                        stats.bytes_read.fetch_add(job.required as u64, Ordering::Relaxed);
                        stats.service_ns.fetch_add(ns, Ordering::Relaxed);
                        job.ticket.complete_chunk(err);
                    })
                    .expect("spawn io worker"),
            );
        }
        *eng.workers.lock().unwrap() = ws;
        eng
    }

    pub(crate) fn submit(&self, jobs: Vec<ChunkJob>) {
        let tx = self.tx.lock().unwrap();
        let tx = tx.as_ref().expect("io engine shut down");
        for j in jobs {
            tx.send(j).expect("io workers alive");
        }
    }
}

impl Drop for IoEngine {
    fn drop(&mut self) {
        self.tx.lock().unwrap().take();
        // Workers finish every queued read, then exit when the channel closes.
        // Joining them here is what makes it safe for the owner to free the
        // destination buffers after dropping the engine.
        for w in self.workers.lock().unwrap().drain(..) {
            let _ = w.join();
        }
    }
}

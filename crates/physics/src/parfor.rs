//! The atomic block-claim parallel-for, ported from box3d's `parallel_for.c` (`b3ParallelForShared` /
//! `b3ParallelForTrampoline` / `b3ParallelFor`). The outer phases — convex narrowphase dispatch and
//! contact recycle — are flat sweeps of independent records, so they need none of the staged solver's
//! machinery (`stages.rs`): no stage list, no barrier, no serial spill. Every thread races on one
//! counter for the next block and runs it; a thread that finishes early steals the next block, so a
//! slow chunk can't strand the others.
//!
//! **Determinism.** The partition (`block_size`, `block_count`) depends on the worker count, so eight
//! threads sweep different ranges than two. That cannot change the result: each record's work reads only
//! its own inputs and writes only its own outputs (a contact's manifold + cache slots),
//! so the sweep order is free. The same property is what lets box3d run collide on any worker count and
//! promise the same bits — and what physics.md already states for the convex/recycle partition.
//!
//! The claim loop ends when blocks run out; the scheduler finishes the trampoline handles.
//!
//! `core::sync::atomic`, not `std` — this compiles into the wasm artifact unchanged.

use core::sync::atomic::{AtomicU32, Ordering};

/// Target blocks per worker, so a worker that finishes early can steal (box3d `blocksPerWorker`). The
/// block size grows once the item count passes `min_range * BLOCKS_PER_WORKER * workers`, which keeps the
/// block count — and so the per-block claim overhead — bounded.
const BLOCKS_PER_WORKER: usize = 32;

/// Minimum items per collide block (box3d `physics_world.c`: "task should take at least 40us on a 4GHz
/// CPU"). Both outer collide phases — recycle and convex dispatch — are per-contact sweeps.
pub const COLLIDE_MIN_RANGE: usize = 20;

/// One parallel-for invocation: the block partition, plus the counter every thread claims from.
pub struct ParFor {
    next_block: AtomicU32,
    block_count: usize,
    block_size: usize,
    item_count: usize,
}

impl ParFor {
    /// Partition `[0, item_count)` into blocks of at least `min_range` items, at most
    /// `BLOCKS_PER_WORKER * worker_count` of them (b3ParallelFor's sizing).
    pub fn new(item_count: usize, min_range: usize, worker_count: usize) -> ParFor {
        debug_assert!(min_range > 0);
        debug_assert!(worker_count >= 1);

        let max_block_count = BLOCKS_PER_WORKER * worker_count;
        let (block_size, block_count) = if item_count == 0 {
            (min_range, 0)
        } else if item_count <= min_range * max_block_count {
            (min_range, item_count.div_ceil(min_range))
        } else {
            let size = item_count.div_ceil(max_block_count);
            (size, item_count.div_ceil(size))
        };

        ParFor {
            next_block: AtomicU32::new(0),
            block_count,
            block_size,
            item_count,
        }
    }

    /// Box3D caps the number of enqueued trampolines at the block count.
    pub fn block_count(&self) -> usize {
        self.block_count
    }

    /// One thread's claim loop: take the next block until they run out, running `f(start, end)` on each.
    /// Every thread of the pool — the orchestrator included — calls this exactly once per invocation, and
    /// the last one to leave has run every block.
    pub fn run(&self, f: impl Fn(usize, usize)) {
        loop {
            let index = self.next_block.fetch_add(1, Ordering::SeqCst) as usize;
            if index >= self.block_count {
                return;
            }
            let start = index * self.block_size;
            let end = (start + self.block_size).min(self.item_count);
            f(start, end);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The blocks must tile `[0, item_count)` exactly — no gap, no overlap, no dropped tail — at every
    /// worker count, or a sweep silently skips records. Also the block-count cap the sizing promises.
    #[test]
    fn blocks_tile_the_range() {
        for workers in 1..9usize {
            for item_count in [0usize, 1, 15, 16, 20, 21, 79, 80, 81, 640, 641, 4095] {
                let p = ParFor::new(item_count, 20, workers);
                assert!(p.block_count <= BLOCKS_PER_WORKER * workers);

                let covered: Vec<AtomicU32> = (0..item_count).map(|_| AtomicU32::new(0)).collect();
                p.run(|s, e| {
                    for c in covered.iter().take(e).skip(s) {
                        c.fetch_add(1, Ordering::SeqCst);
                    }
                });
                assert!(
                    covered.iter().all(|c| c.load(Ordering::SeqCst) == 1),
                    "workers={workers} item_count={item_count} not tiled exactly"
                );
            }
        }
    }

    /// The claim loop under real contention: every block runs exactly once across the pool however the
    /// threads interleave, and every thread leaves. This is the invariant the outer phases rest on — a
    /// doubly-claimed block would run a contact's narrowphase twice, and a dropped one not at all.
    ///
    /// Every thread rendezvouses inside its first block, so the concurrency is forced rather than hoped
    /// for: without it the first thread scheduled drains every block before the others start, and the
    /// test passes vacuously (observed — it is why the rendezvous is here).
    #[test]
    fn blocks_are_claimed_exactly_once_under_contention() {
        for workers in [2usize, 8] {
            let items = 4095usize;
            let par = ParFor::new(items, COLLIDE_MIN_RANGE, workers);
            assert!(par.block_count() >= workers); // else the rendezvous below cannot be met

            let covered: Vec<AtomicU32> = (0..items).map(|_| AtomicU32::new(0)).collect();
            let claimed: Vec<AtomicU32> = (0..workers).map(|_| AtomicU32::new(0)).collect();
            let entered = AtomicU32::new(0);

            std::thread::scope(|scope| {
                for w in 0..workers {
                    let (par, covered, claimed, entered) = (&par, &covered, &claimed, &entered);
                    scope.spawn(move || {
                        let first = core::cell::Cell::new(true);
                        par.run(|s, e| {
                            if first.replace(false) {
                                entered.fetch_add(1, Ordering::SeqCst);
                                while entered.load(Ordering::SeqCst) < workers as u32 {
                                    core::hint::spin_loop();
                                }
                            }
                            claimed[w].fetch_add(1, Ordering::SeqCst);
                            for c in covered.iter().take(e).skip(s) {
                                c.fetch_add(1, Ordering::SeqCst);
                            }
                        });
                    });
                }
            });

            assert!(covered.iter().all(|c| c.load(Ordering::SeqCst) == 1));
            let ran: u32 = claimed.iter().map(|c| c.load(Ordering::SeqCst)).sum();
            assert_eq!(ran as usize, par.block_count());
            assert!(claimed.iter().all(|c| c.load(Ordering::SeqCst) > 0));
        }
    }
}

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
//! promise the same bits — and what tumble.md already states for the convex/recycle partition.
//!
//! **No fault flag, no join.** Unlike `stages::run`, nothing here spins: a worker's loop ends when the
//! blocks run out, and the orchestrator's does too. So a dead worker cannot hang a live one inside wasm —
//! the pool's JS ack is the whole join, and its fault check (`src/pool.ts`) is the whole fault path.
//!
//! `core::sync::atomic`, not `std` — this compiles into the wasm artifact unchanged.

use core::sync::atomic::{AtomicU32, Ordering};

/// Target blocks per worker, so a worker that finishes early can steal (box3d `blocksPerWorker`). The
/// block size grows once the item count passes `min_range * BLOCKS_PER_WORKER * workers`, which keeps the
/// block count — and so the per-block claim overhead — bounded.
const BLOCKS_PER_WORKER: usize = 4;

/// Minimum items per collide block (box3d `physics_world.c`: "task should take at least 40us on a 4GHz
/// CPU"). Both outer collide phases — recycle and convex dispatch — are per-contact sweeps.
pub const COLLIDE_MIN_RANGE: usize = 20;

// The fork floor. box3d's `min_range` is the whole gate there, because its fork is an `enqueueTaskFcn`
// push onto a live task system — nanoseconds. Ours is an `Atomics.notify`, N worker wakeups off
// `Atomics.wait`, and a spin join (`src/pool.ts`), and its cost scales with the workers woken. So a phase
// must carry enough work to beat its own wake, and the floor has to scale the same way: **items per
// woken worker**, not items.
//
// Derived, not tuned. `large_pyramid` on 8 threads (7 workers, Ryzen 5900X, bun): forking the outer
// phases *loses* at 210 bodies (0.72 → 1.05 ms/step) and at 1035 (2.41 → 2.80), breaks even at 1830
// (4.21 → 4.23), and wins above it (2850: 7.80 → 7.28; 4095: 15.3 → 12.7). At that break-even scene the
// phases sweep 5310 recycle records per step, so the break-even floor is ≈ 5310/7 ≈ 758 items per
// worker. (The pose finalize once carried its own floor here; it now rides the staged solve as its
// terminal stage — `stages.rs` — where the workers are already awake and no floor applies.)

/// Collide-sweep items per woken worker (recycle, and the convex dispatch with it — a dispatch record is
/// strictly more work than a recycle record, so the recycle floor is a conservative bound for it, and it
/// is not perf-load-bearing anyway: post-settle almost every contact recycles, so the convex sweep fires
/// on the settle-in steps and then essentially never).
pub const COLLIDE_FORK_MIN: usize = 768;

/// Is `item_count` enough work to be worth waking `worker_count` workers for? A sweep under the floor
/// loses to its own wake, and runs inline instead.
pub fn worth_forking(item_count: usize, worker_count: usize, fork_min: usize) -> bool {
    worker_count >= 1 && item_count >= fork_min * worker_count
}

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

    /// How many blocks the work split into. One block means there is nothing to steal — the caller may as
    /// well run it inline rather than pay a fork (box3d caps its task count at the block count for the
    /// same reason).
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

    /// The fork floor scales with the workers woken, and a pool with no workers never forks.
    #[test]
    fn fork_floor_scales_with_the_workers_woken() {
        assert!(!worth_forking(1 << 20, 0, COLLIDE_FORK_MIN));
        assert!(!worth_forking(COLLIDE_FORK_MIN - 1, 1, COLLIDE_FORK_MIN));
        assert!(worth_forking(COLLIDE_FORK_MIN, 1, COLLIDE_FORK_MIN));
        assert!(!worth_forking(COLLIDE_FORK_MIN * 6, 7, COLLIDE_FORK_MIN));
        assert!(worth_forking(COLLIDE_FORK_MIN * 7, 7, COLLIDE_FORK_MIN));
        // large_pyramid at 1035 bodies on 8 threads: the measured loss — the collide sweep must not fork.
        assert!(!worth_forking(2970, 7, COLLIDE_FORK_MIN));
        // …and at 4095 bodies, the measured win — it must.
        assert!(worth_forking(12015, 7, COLLIDE_FORK_MIN));
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

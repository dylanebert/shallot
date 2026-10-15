//! Box3D parallel_for.c: b3ParallelForShared, b3ParallelForTrampoline and b3ParallelFor.
//! Work must be independent of partition and execution order, which vary with
//! worker count. The scheduler must finish every trampoline handle before reuse.

use core::sync::atomic::{AtomicU32, Ordering};

/// Box3D blocksPerWorker: bounds claim overhead while allowing work stealing.
const BLOCKS_PER_WORKER: usize = 32;

/// Box3D physics_world.c's minimum per-contact collide task range.
pub const COLLIDE_MIN_RANGE: usize = 20;

pub struct ParFor {
    next_block: AtomicU32,
    block_count: usize,
    block_size: usize,
    item_count: usize,
}

impl ParFor {
    /// b3ParallelFor sizing: at most BLOCKS_PER_WORKER * worker_count blocks.
    /// The final block may contain fewer than min_range items.
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

    /// Caps enqueued trampolines at the block count so no task is queued without work.
    pub fn block_count(&self) -> usize {
        self.block_count
    }

    /// Calls f with half-open ranges. All blocks have completed only after every
    /// concurrent run call returns; one caller's return is not a barrier.
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

//! Independently owned dynamic trees. The public bridge borrows the report and input words.
use crate::math::Vec3;
use crate::tree::{self, Rebuild, STACK_SIZE, STRIDE};

#[link(wasm_import_module = "env")]
extern "C" {
    fn queryCallback(kind: u32, proxy: u32, low: u32, high: u32) -> f32;
}

struct Tree {
    pool: Vec<u32>,
    root: i32,
    count: usize,
    free: i32,
    proxies: usize,
    indices: Vec<i32>,
    centers: Vec<f32>,
    // Bounds / ray / point occupy 0..6; report occupies 8..16.
    words: [u32; 32],
}

impl Tree {
    fn reserve(&mut self, cap: usize) {
        let old = self.pool.len() / STRIDE;
        if cap <= old {
            return;
        }
        self.pool
            .resize(cap.checked_mul(STRIDE).expect("tree capacity overflow"), 0);
        for i in old..cap {
            self.pool[i * STRIDE + 10] = if i + 1 == cap {
                u32::MAX
            } else {
                (i + 1) as u32
            };
        }
        if self.free == -1 {
            self.free = old as i32;
        } else {
            let mut i = self.free as usize;
            while self.pool[i * STRIDE + 10] != u32::MAX {
                i = self.pool[i * STRIDE + 10] as usize;
            }
            self.pool[i * STRIDE + 10] = old as u32;
        }
    }
    fn metadata(&mut self) {
        self.words[8] = self.root as u32;
        self.words[9] = self.count as u32;
        self.words[10] = (self.pool.len() / STRIDE) as u32;
        self.words[11] = self.proxies as u32;
        self.words[12] = if self.root < 0 {
            0
        } else {
            self.pool[self.root as usize * STRIDE + 11] >> 16
        };
        self.words[17] = self.free as u32;
    }
    fn report(&mut self) {
        self.metadata();
        let mut total = 0.0;
        if self.root >= 0 {
            for i in 0..self.pool.len() / STRIDE {
                if self.pool[i * STRIDE + 11] & 5 == 1 && i != self.root as usize {
                    total += tree::perimeter(&self.pool, i as i32);
                }
            }
            total /= tree::perimeter(&self.pool, self.root);
        }
        self.words[13] = total.to_bits();
    }
    fn valid(&self) -> bool {
        let mut stack = Vec::new();
        if self.root == -1 {
            if self.count != 0 || self.proxies != 0 {
                return false;
            }
        } else {
            if self.root < 0 || self.root as usize >= self.pool.len() / STRIDE {
                return false;
            }
            stack.push((self.root, -1));
        }
        let mut seen = vec![false; self.pool.len() / STRIDE];
        let mut leaves = 0;
        while let Some((id, parent)) = stack.pop() {
            if id < 0 || id as usize >= seen.len() || seen[id as usize] {
                return false;
            }
            seen[id as usize] = true;
            let n = id as usize * STRIDE;
            let (lo, hi) = tree::node_aabb(&self.pool, id);
            for k in 0..3 {
                if !lo[k].is_finite() || !hi[k].is_finite() || lo[k] > hi[k] {
                    return false;
                }
            }
            if self.pool[n + 11] & 1 == 0 || self.pool[n + 10] as i32 != parent {
                return false;
            }
            if self.pool[n + 11] & 4 != 0 {
                if self.pool[n + 11] >> 16 != 0 {
                    return false;
                }
                leaves += 1;
            } else {
                let a = self.pool[n + 8] as usize;
                let b = self.pool[n + 9] as usize;
                if a >= seen.len() || b >= seen.len() {
                    return false;
                }
                if self.pool[n + 11] >> 16
                    != 1 + (self.pool[a * STRIDE + 11] >> 16).max(self.pool[b * STRIDE + 11] >> 16)
                {
                    return false;
                }
                let (lo, hi) = tree::node_aabb(&self.pool, id);
                for child in [a, b] {
                    let (cl, ch) = tree::node_aabb(&self.pool, child as i32);
                    for k in 0..3 {
                        if lo[k] > cl[k] || hi[k] < ch[k] {
                            return false;
                        }
                    }
                    for k in 6..8 {
                        if self.pool[n + k] & self.pool[child * STRIDE + k]
                            != self.pool[child * STRIDE + k]
                        {
                            return false;
                        }
                    }
                }
                stack.push((a as i32, id));
                stack.push((b as i32, id));
            }
        }
        if leaves != self.proxies || seen.iter().filter(|v| **v).count() != self.count {
            return false;
        }
        let mut free = self.free;
        while free != -1 {
            if free < 0 || free as usize >= seen.len() || seen[free as usize] {
                return false;
            }
            seen[free as usize] = true;
            if self.pool[free as usize * STRIDE + 11] & 1 != 0 {
                return false;
            }
            free = self.pool[free as usize * STRIDE + 10] as i32;
        }
        seen.iter().all(|v| *v)
    }
}

#[export_name = "standaloneTreeCreate"]
pub extern "C" fn create(capacity: usize) -> *mut u8 {
    let mut t = Box::new(Tree {
        pool: Vec::new(),
        root: -1,
        count: 0,
        free: -1,
        proxies: 0,
        indices: Vec::new(),
        centers: Vec::new(),
        words: [0; 32],
    });
    t.reserve(
        capacity
            .max(16)
            .checked_mul(2)
            .expect("tree capacity overflow")
            - 1,
    );
    t.report();
    Box::into_raw(t) as *mut u8
}
#[export_name = "standaloneTreeDestroy"]
pub unsafe extern "C" fn destroy(ptr: *mut u8) {
    drop(Box::from_raw(ptr as *mut Tree));
}
#[export_name = "standaloneTreeWords"]
pub unsafe extern "C" fn words(ptr: *mut u8) -> *mut u32 {
    (&mut *(ptr as *mut Tree)).words.as_mut_ptr()
}
#[export_name = "standaloneTreePool"]
pub unsafe extern "C" fn pool(ptr: *mut u8) -> *mut u32 {
    (&mut *(ptr as *mut Tree)).pool.as_mut_ptr()
}

// Mutation uses the six input bounds words. Query callbacks borrow the current limit at word 6.
#[export_name = "standaloneTreeCall"]
pub unsafe extern "C" fn call(
    ptr: *mut u8,
    op: u32,
    id: i32,
    mh: u32,
    ml: u32,
    all: u32,
    low: u32,
    high: u32,
    limit: f32,
) -> i32 {
    let t = &mut *(ptr as *mut Tree);
    let lo = std::array::from_fn(|i| f32::from_bits(t.words[i]));
    let hi = std::array::from_fn(|i| f32::from_bits(t.words[i + 3]));
    let mut result = id;
    match op {
        0 => {
            let cap = t.pool.len() / STRIDE;
            if cap - t.count < 2 {
                t.reserve(cap + (cap >> 1));
            }
            result = tree::create_proxy(
                &mut t.pool,
                &mut t.root,
                &mut t.count,
                &mut t.free,
                lo,
                hi,
                mh,
                ml,
                low as u64 | ((high as u64) << 32),
            );
            t.proxies += 1;
        }
        1 => tree::move_proxy(
            &mut t.pool,
            &mut t.root,
            &mut t.count,
            &mut t.free,
            id,
            lo,
            hi,
        ),
        2 => {
            tree::destroy_proxy(&mut t.pool, &mut t.root, &mut t.count, &mut t.free, id);
            t.proxies -= 1;
        }
        3 => {
            let n = t.proxies.max(1);
            t.indices.resize(n, 0);
            t.centers.resize(n * 3, 0.0);
            let mut gather = [0; STACK_SIZE];
            let mut build = [0; STACK_SIZE * 5];
            let mut rb = Rebuild {
                node_count: &mut t.count,
                free_list: &mut t.free,
                leaf_indices: &mut t.indices,
                leaf_centers: &mut t.centers,
                gather_stack: &mut gather,
                build_stack: &mut build,
            };
            if t.proxies > 0 {
                t.root = tree::rebuild(&mut t.pool, t.root, t.proxies, id != 0, &mut rb);
            }
            result = t.proxies as i32;
        }
        4..=6 => {
            let p = &t.pool;
            let limit_ptr = &mut t.words[6] as *mut u32;
            let cb = |value: f32, proxy: i32, user: u32| {
                *limit_ptr = value.to_bits();
                queryCallback(9, proxy as u32, user, p[proxy as usize * STRIDE + 9])
            };
            let mut distance = limit;
            let stats = match op {
                4 => {
                    let mut stack = [0; STACK_SIZE];
                    tree::query(
                        p,
                        t.root,
                        t.count,
                        lo,
                        hi,
                        mh,
                        ml,
                        all != 0,
                        &mut stack,
                        |proxy, user| cb(0.0, proxy, user) != 0.0,
                    )
                }
                5 => tree::ray_cast(
                    p,
                    t.root,
                    t.count,
                    Vec3::new(lo[0], lo[1], lo[2]),
                    Vec3::new(hi[0], hi[1], hi[2]),
                    limit,
                    mh,
                    ml,
                    all != 0,
                    cb,
                ),
                _ => tree::query_closest(
                    p,
                    t.root,
                    t.count,
                    Vec3::new(lo[0], lo[1], lo[2]),
                    mh,
                    ml,
                    all != 0,
                    &mut distance,
                    cb,
                ),
            };
            t.words[14] = stats.0;
            t.words[15] = stats.1;
            t.words[16] = distance.to_bits();
            return 0;
        }
        7 => return i32::from(t.valid()),
        8 => {
            t.root = t.words[8] as i32;
            t.count = t.words[9] as usize;
            t.proxies = t.words[11] as usize;
            t.free = t.words[17] as i32;
            if !t.valid() {
                return 0;
            }
            for node in t.pool.as_chunks_mut::<STRIDE>().0 {
                for value in &mut node[..6] {
                    *value = (f32::from_bits(*value) * limit).to_bits();
                }
            }
            if !t.valid() {
                return 0;
            }
            result = 1;
        }
        9 => t.reserve(id as usize),
        10 => {
            t.report();
            return 0;
        }
        _ => unreachable!(),
    }
    t.metadata();
    result
}

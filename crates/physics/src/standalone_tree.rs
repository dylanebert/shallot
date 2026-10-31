//! Independently owned dynamic trees. The bridge borrows packed reports, never the tree's storage.
use crate::dynamic_tree::{Node, Proxy, Tree, EMPTY, MOVED};
use crate::math::Vec3;

#[link(wasm_import_module = "env")]
extern "C" {
    fn queryCallback(kind: u32, proxy: u32, low: u32, high: u32) -> f32;
}
// Node report (12 words), proxy record (6), leaf proxy ID, padding.
const STRIDE: usize = 20;
struct Standalone {
    tree: Tree,
    pool: Vec<u32>,
    words: [u32; 32],
}
impl Standalone {
    fn report(&mut self) {
        let t = &self.tree;
        self.words[8] = 0;
        self.words[9] = t.end as u32;
        self.words[10] = t.nodes.len() as u32;
        self.words[11] = t.count as u32;
        self.words[12] = t.nodes[0].height() as u32;
        self.words[13] = t.area_ratio().to_bits();
        self.words[17] = t.pair_free as u32;
        self.words[18] = t.proxies.len() as u32;
        self.words[19] = t.proxy_free as u32;
        self.words[20] = u32::from(t.ordered);
        self.pool
            .resize(t.nodes.len().max(t.proxies.len()) * STRIDE, 0);
        self.pool.fill(0);
        for i in 0..t.nodes.len().max(t.proxies.len()) {
            let n = t.nodes.get(i).copied().unwrap_or_else(Node::empty);
            let p = i * STRIDE;
            self.pool[p + 10] = t.parents.get(i).copied().unwrap_or(-1) as u32;
            if i < t.end && n.flags != EMPTY {
                self.pool[p..p + 6].copy_from_slice(&[
                    n.lo.x.to_bits(),
                    n.lo.y.to_bits(),
                    n.lo.z.to_bits(),
                    n.hi.x.to_bits(),
                    n.hi.y.to_bits(),
                    n.hi.z.to_bits(),
                ]);
                self.pool[p + 11] = 1
                    | if n.flags & MOVED != 0 { 2 } else { 0 }
                    | if n.leaf() { 4 } else { 0 }
                    | ((n.height() as u32) << 16);
                if n.leaf() {
                    let proxy = t.proxies[n.index()];
                    self.pool[p + 6] = (proxy.category >> 32) as u32;
                    self.pool[p + 7] = proxy.category as u32;
                    self.pool[p + 8] = proxy.user as u32;
                    self.pool[p + 9] = (proxy.user >> 32) as u32;
                    self.pool[p + 18] = n.index() as u32;
                } else {
                    self.pool[p + 8] = n.index() as u32;
                    self.pool[p + 9] = (n.index() + 1) as u32;
                }
            }
            if i < t.proxies.len() {
                let proxy = t.proxies[i];
                self.pool[p + 12] = proxy.user as u32;
                self.pool[p + 13] = (proxy.user >> 32) as u32;
                self.pool[p + 14] = proxy.category as u32;
                self.pool[p + 15] = (proxy.category >> 32) as u32;
                self.pool[p + 16] = proxy.node as u32;
                self.pool[p + 17] = proxy.next as u32;
            }
        }
    }
    fn restore(&mut self, scale: f32) -> bool {
        let cap = self.words[10] as usize;
        let end = self.words[9] as usize;
        let pc = self.words[18] as usize;
        if self.words[8] != 0
            || cap != self.tree.nodes.len()
            || end < 2
            || end > cap
            || end & 1 != 0
            || pc == 0
        {
            return false;
        }
        self.tree.end = end;
        self.tree.count = self.words[11] as usize;
        self.tree.pair_free = self.words[17] as i32;
        self.tree.proxy_free = self.words[19] as i32;
        self.tree.ordered = self.words[20] != 0;
        self.tree.proxies.resize(
            pc,
            Proxy {
                user: 0,
                category: 0,
                node: -1,
                next: -1,
            },
        );
        for i in 0..cap.max(pc) {
            let p = i * STRIDE;
            let flags = self.pool[p + 11];
            if i < cap {
                self.tree.parents[i] = self.pool[p + 10] as i32;
                self.tree.nodes[i] = if flags & 1 == 0 {
                    Node::empty()
                } else {
                    let leaf = flags & 4 != 0;
                    Node {
                        lo: Vec3::new(
                            f32::from_bits(self.pool[p]),
                            f32::from_bits(self.pool[p + 1]),
                            f32::from_bits(self.pool[p + 2]),
                        ),
                        hi: Vec3::new(
                            f32::from_bits(self.pool[p + 3]),
                            f32::from_bits(self.pool[p + 4]),
                            f32::from_bits(self.pool[p + 5]),
                        ),
                        flags: if leaf {
                            crate::dynamic_tree::LEAF | self.pool[p + 18]
                        } else {
                            self.pool[p + 8]
                        } | if flags & 2 != 0 { MOVED } else { 0 },
                        height: if leaf {
                            self.pool[p + 8] as i32
                        } else {
                            (flags >> 16) as i32
                        },
                    }
                };
            }
            if i < pc {
                self.tree.proxies[i] = Proxy {
                    user: self.pool[p + 12] as u64 | ((self.pool[p + 13] as u64) << 32),
                    category: self.pool[p + 14] as u64 | ((self.pool[p + 15] as u64) << 32),
                    node: self.pool[p + 16] as i32,
                    next: self.pool[p + 17] as i32,
                };
            }
        }
        if !self.tree.valid() {
            return false;
        }
        for n in &mut self.tree.nodes[..end] {
            if n.flags != EMPTY {
                n.lo = n.lo.scale(scale);
                n.hi = n.hi.scale(scale);
            }
        }
        self.tree.valid()
    }
}
#[export_name = "standaloneTreeCreate"]
pub extern "C" fn create(capacity: usize) -> *mut u8 {
    let mut t = Box::new(Standalone {
        tree: Tree::new(capacity),
        pool: Vec::new(),
        words: [0; 32],
    });
    t.report();
    Box::into_raw(t) as *mut u8
}
#[export_name = "standaloneTreeDestroy"]
pub unsafe extern "C" fn destroy(ptr: *mut u8) {
    drop(Box::from_raw(ptr as *mut Standalone));
}
#[export_name = "standaloneTreeWords"]
pub unsafe extern "C" fn words(ptr: *mut u8) -> *mut u32 {
    (&mut *(ptr as *mut Standalone)).words.as_mut_ptr()
}
#[export_name = "standaloneTreePool"]
pub unsafe extern "C" fn pool(ptr: *mut u8) -> *mut u32 {
    (&mut *(ptr as *mut Standalone)).pool.as_mut_ptr()
}
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
    let t = &mut *(ptr as *mut Standalone);
    let lo = Vec3::new(
        f32::from_bits(t.words[0]),
        f32::from_bits(t.words[1]),
        f32::from_bits(t.words[2]),
    );
    let hi = Vec3::new(
        f32::from_bits(t.words[3]),
        f32::from_bits(t.words[4]),
        f32::from_bits(t.words[5]),
    );
    let mask = ml as u64 | ((mh as u64) << 32);
    let mut result = id;
    match op {
        0 => {
            result = t
                .tree
                .create(lo, hi, mask, low as u64 | ((high as u64) << 32), false)
                as i32
        }
        3 => result = t.tree.rebuild(id != 0) as i32,
        4..=6 => {
            let limit_ptr = &mut t.words[6] as *mut u32;
            let cb = |value: f32, proxy: usize, user: u64| {
                *limit_ptr = value.to_bits();
                queryCallback(9, proxy as u32, user as u32, (user >> 32) as u32)
            };
            let mut distance = limit;
            let stats = match op {
                4 => t
                    .tree
                    .query(lo, hi, mask, all != 0, |id, user| cb(0.0, id, user) != 0.0),
                5 => t.tree.cast(lo, lo, hi, limit, mask, all != 0, false, cb),
                _ => t.tree.closest(lo, mask, all != 0, &mut distance, cb),
            };
            t.words[14] = stats.0;
            t.words[15] = stats.1;
            t.words[16] = distance.to_bits();
            return 0;
        }
        7 => return i32::from(t.tree.valid()),
        8 => {
            if !t.restore(limit) {
                return 0;
            }
            result = 1;
        }
        9 => {
            let cap = id as usize;
            t.tree.nodes.resize(cap, Node::empty());
            t.tree.parents.resize(cap, -1);
            t.tree.proxies.resize(
                t.words[18] as usize,
                Proxy {
                    user: 0,
                    category: 0,
                    node: -1,
                    next: -1,
                },
            );
        }
        10 => {}
        _ => unreachable!(),
    }
    t.report();
    result
}

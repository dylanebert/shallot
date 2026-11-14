//! Box3D broad_phase.c: moved sibling cross traversals and globally sorted shape-pair keys.
use crate::broad;
use crate::dynamic_tree::{Node, MOVED, STACK_SIZE};
use crate::math::Vec3;
use crate::mesh_query::aabb_overlaps;
use crate::shapes::{col_slice as shape_col, SHAPE_STRIDE, S_TYPE};
use crate::table;

#[derive(Clone, Copy)]
#[repr(C)]
struct Job {
    a: Node,
    b: Node,
    tree_b: usize,
}
#[derive(Clone, Copy)]
#[repr(C)]
struct Candidate {
    child: u32,
    a: u32,
    b: u32,
}
/// One block's candidates in its worker's array: `start` is the block's first job.
#[derive(Clone, Copy)]
struct Segment {
    start: usize,
    begin: usize,
    end: usize,
}
/// Box3D's taskContexts[worker].pairKeys: each worker's candidates, cleared each step and kept
/// across steps. Only its own task grows them; the serial point reads them in block order.
struct Worker {
    candidates: Vec<Candidate>,
    segments: Vec<Segment>,
}
struct Pairs {
    workers: [Worker; crate::solve::MAX_THREADS],
    order: Vec<u64>,
}
impl Pairs {
    const EMPTY: Self = Self {
        workers: [const {
            Worker {
                candidates: Vec::new(),
                segments: Vec::new(),
            }
        }; crate::solve::MAX_THREADS],
        order: Vec::new(),
    };
}
// Heap-allocated so the 128 empty worlds stay out of the data segment: an empty Vec is not all
// zero bytes. Raw, so a task projects its own worker with mutable provenance.
static mut WORLDS: [*mut Pairs; crate::regions::MAX_WORLDS] =
    [core::ptr::null_mut(); crate::regions::MAX_WORLDS];
/// Serial callers only; no reference it returns may live while tasks run.
unsafe fn state(world: usize) -> &'static mut Pairs {
    let slot = &mut (*(&raw mut WORLDS))[world];
    if slot.is_null() {
        *slot = Box::into_raw(Box::new(Pairs::EMPTY));
    }
    &mut **slot
}
/// A task's own worker, reached without a reference to the enclosing `Pairs`.
unsafe fn worker_state(world: usize, index: usize) -> &'static mut Worker {
    let pairs = (*(&raw const WORLDS))[world];
    &mut *(&raw mut (*pairs).workers[index])
}
pub unsafe fn reset(world: usize) {
    let slot = &mut (*(&raw mut WORLDS))[world];
    if !slot.is_null() {
        drop(Box::from_raw(*slot));
        *slot = core::ptr::null_mut();
    }
}
static mut JOB_PTR: usize = 0;
static mut JOB_COUNT: usize = 0;
static mut KEY_PTR: usize = 0;
static mut KEY_COUNT: usize = 0;
static mut REBUILD_PENDING: bool = false;
pub unsafe fn schedule_rebuild(world: usize) {
    for i in [1, 2] {
        broad::tree_mut(world, i).reserve_rebuild();
        // The rebuild also refreshes the exported tree state. Reserve its mirror before
        // enqueueing: a pending rebuild can execute while the scheduler joins solver tasks.
        broad::sync_tree(world, i);
    }
    REBUILD_PENDING = true;
}
pub unsafe fn rebuild_pending() -> bool {
    REBUILD_PENDING
}
pub unsafe fn job_count() -> usize {
    JOB_COUNT
}
pub unsafe fn key_count() -> usize {
    KEY_COUNT
}
pub unsafe fn keys() -> *const u64 {
    KEY_PTR as *const u64
}
fn test(a: Node, b: Node) -> bool {
    (a.flags | b.flags) & MOVED != 0 && aabb_overlaps(a.lo, a.hi, b.lo, b.hi)
}
unsafe fn seeds(world: usize, tree_b: usize, out: &mut [Job]) -> usize {
    let a = &broad::tree(world, 2).nodes;
    let b = &broad::tree(world, tree_b).nodes;
    let empty = Job {
        a: Node::empty(),
        b: Node::empty(),
        tree_b,
    };
    let mut queue = [empty; 128];
    let mut head = 0;
    let mut tail = 0;
    if test(a[0], b[0]) {
        queue[0] = Job {
            a: a[0],
            b: b[0],
            tree_b,
        };
        tail = 1;
    }
    let mut count = 0;
    while head < tail && count + (tail - head) + 3 < 64 {
        let pair = queue[head & 127];
        head += 1;
        if pair.a.leaf() || pair.b.leaf() {
            out[count] = pair;
            count += 1;
            continue;
        }
        for i in 0..2 {
            for j in 0..2 {
                let na = a[pair.a.index() + i];
                let nb = b[pair.b.index() + j];
                if test(na, nb) {
                    queue[tail & 127] = Job {
                        a: na,
                        b: nb,
                        tree_b,
                    };
                    tail += 1;
                }
            }
        }
    }
    while head < tail {
        out[count] = queue[head & 127];
        count += 1;
        head += 1;
    }
    count
}
pub unsafe fn needs_update(world: usize) -> bool {
    (0..3).any(|i| {
        let t = broad::tree(world, i);
        t.nodes[0].flags & MOVED != 0 || i != 0 && !t.ordered
    })
}
pub extern "C" fn reserve_pairs_in_world(world: usize) {
    unsafe {
        let empty = Job {
            a: Node::empty(),
            b: Node::empty(),
            tree_b: 0,
        };
        let mut cross = [empty; 128];
        let s = seeds(world, 0, &mut cross);
        let k = seeds(world, 1, &mut cross[s..]);
        let t = broad::tree(world, 2);
        let self_count = (2..t.end)
            .step_by(2)
            .filter(|&p| (t.nodes[p].flags | t.nodes[p + 1].flags) & MOVED != 0)
            .count();
        JOB_COUNT = s + k + self_count;
        KEY_COUNT = 0;
        for w in &mut state(world).workers {
            w.candidates.clear();
            w.segments.clear();
        }
        JOB_PTR = crate::arena::reserve_scratch(
            world,
            (JOB_COUNT * core::mem::size_of::<Job>()).next_multiple_of(16),
        );
        let jobs = core::slice::from_raw_parts_mut(JOB_PTR as *mut Job, JOB_COUNT);
        jobs[..s + k].copy_from_slice(&cross[..s + k]);
        let mut n = s + k;
        for p in (2..t.end).step_by(2) {
            if (t.nodes[p].flags | t.nodes[p + 1].flags) & MOVED != 0 {
                jobs[n] = Job {
                    a: t.nodes[p],
                    b: t.nodes[p + 1],
                    tree_b: 2,
                };
                n += 1;
            }
        }
    }
}
unsafe fn accepts(world: usize, a: u32, b: u32) -> bool {
    let shapes = shape_col(world);
    let sa = &shapes[a as usize * SHAPE_STRIDE..][..SHAPE_STRIDE];
    let sb = &shapes[b as usize * SHAPE_STRIDE..][..SHAPE_STRIDE];
    sa[1] != sb[1]
        && sa[4] == u32::MAX
        && sb[4] == u32::MAX
        && crate::shapes::should_shapes_collide(
            crate::shapes::Filter::read(sa),
            crate::shapes::Filter::read(sb),
        )
        && crate::bodies::should_collide_in_world(world, sa[1], sb[1])
}
struct Emitter<'a> {
    world: usize,
    output: &'a mut Vec<Candidate>,
}
impl Emitter<'_> {
    unsafe fn emit(&mut self, child: u32, a: u32, b: u32) {
        self.output.push(Candidate { child, a, b });
        if !crate::callbacks::filter_needed(self.world, a as usize, b as usize) {
            crate::callback_work::note(self.world, crate::callback_work::Work::PairParallel);
        }
    }
    unsafe fn candidate(&mut self, a: u32, b: u32) {
        let (a, b) = (a.min(b), a.max(b));
        let shapes = shape_col(self.world);
        let compound = if shapes[a as usize * SHAPE_STRIDE + S_TYPE] == crate::finalize::TY_COMPOUND
        {
            Some((a, b))
        } else if shapes[b as usize * SHAPE_STRIDE + S_TYPE] == crate::finalize::TY_COMPOUND {
            Some((b, a))
        } else {
            None
        };
        let items =
            core::slice::from_raw_parts(broad::set_items(self.world), broad::set_cap(self.world));
        if let Some((c, other)) = compound {
            if !accepts(self.world, c, other) {
                return;
            }
            let (geometry, _) = crate::query_abi::active_shape(self.world, c as usize);
            let crate::query::Shape::Compound(compound) = geometry else {
                unreachable!()
            };
            let key = shapes[other as usize * SHAPE_STRIDE + crate::shapes::S_PROXY_KEY];
            let t = broad::tree(self.world, (key & 3) as usize);
            let node = t.nodes[t.proxies[(key >> 2) as usize].node as usize];
            let transform = crate::world_query::pose(self.world, c as usize, Vec3::ZERO).invert();
            let center = transform.point(node.lo.add(node.hi).scale(0.5));
            let extent = crate::math::Mat3::from_quat(transform.q)
                .abs()
                .mul_v(node.hi.sub(node.lo).scale(0.5));
            crate::compound_query::query(
                compound,
                center.sub(extent),
                center.add(extent),
                |_, child| {
                    if !table::contains_item(items, c, other, child) {
                        self.emit(child, c, other);
                    }
                    true
                },
            );
        } else if !table::contains_item(items, a, b, 0) && accepts(self.world, a, b) {
            self.emit(0, a, b);
        }
    }
    unsafe fn proxy_subtree(
        &mut self,
        proxy: Node,
        nodes: &[Node],
        pair: usize,
        stack: &mut [usize; STACK_SIZE],
    ) {
        stack[0] = pair;
        let mut count = 1;
        while count > 0 {
            count -= 1;
            let p = stack[count];
            for n in &nodes[p..p + 2] {
                if !test(proxy, *n) {
                    continue;
                }
                if n.leaf() {
                    self.candidate(proxy.height as u32, n.height as u32);
                } else {
                    assert!(count < STACK_SIZE);
                    stack[count] = n.index();
                    count += 1;
                }
            }
        }
    }
    unsafe fn visit(
        &mut self,
        array_a: &[Node],
        array_b: &[Node],
        a: Node,
        b: Node,
        stack: &mut [(usize, usize); STACK_SIZE],
        count: &mut usize,
        proxy_stack: &mut [usize; STACK_SIZE],
    ) {
        if !test(a, b) {
            return;
        }
        match (a.leaf(), b.leaf()) {
            (true, true) => self.candidate(a.height as u32, b.height as u32),
            (true, false) => self.proxy_subtree(a, array_b, b.index(), proxy_stack),
            (false, true) => self.proxy_subtree(b, array_a, a.index(), proxy_stack),
            (false, false) => {
                assert!(*count < STACK_SIZE);
                stack[*count] = (a.index(), b.index());
                *count += 1;
            }
        }
    }
    unsafe fn cross(
        &mut self,
        array_a: &[Node],
        array_b: &[Node],
        a: Node,
        b: Node,
        stacks: &mut Stacks,
    ) {
        let mut count = 0;
        self.visit(
            array_a,
            array_b,
            a,
            b,
            &mut stacks.pairs,
            &mut count,
            &mut stacks.proxy,
        );
        while count > 0 {
            count -= 1;
            let (a, b) = stacks.pairs[count];
            for i in 0..2 {
                for j in 0..2 {
                    self.visit(
                        array_a,
                        array_b,
                        array_a[a + i],
                        array_b[b + j],
                        &mut stacks.pairs,
                        &mut count,
                        &mut stacks.proxy,
                    );
                }
            }
        }
    }
}
/// Traversal stacks, made once per block. Rust zero-fills an array at each call, where Box3D's
/// broad_phase.c leaves its stack arrays uninitialized, so stacks local to each call cost a fill per job.
struct Stacks {
    pairs: [(usize, usize); STACK_SIZE],
    proxy: [usize; STACK_SIZE],
}
pub unsafe fn query_block(world: usize, worker: usize, start: usize, end: usize) {
    let jobs = core::slice::from_raw_parts(JOB_PTR as *const Job, JOB_COUNT);
    let nodes = &broad::tree(world, 2).nodes;
    let mut stacks = Stacks {
        pairs: [(0, 0); STACK_SIZE],
        proxy: [0; STACK_SIZE],
    };
    let w = worker_state(world, worker);
    let begin = w.candidates.len();
    let mut emitter = Emitter {
        world,
        output: &mut w.candidates,
    };
    for job in &jobs[start..end] {
        emitter.cross(
            nodes,
            &broad::tree(world, job.tree_b).nodes,
            job.a,
            job.b,
            &mut stacks,
        );
    }
    let end = w.candidates.len();
    w.segments.push(Segment { start, begin, end });
}
pub(crate) unsafe fn finish_queries(world: usize) {
    let pairs = state(world);
    pairs.order.clear();
    let mut count = 0;
    let mut bytes = 0;
    for (i, w) in pairs.workers.iter().enumerate() {
        // First job, worker, segment: packed so the sort shares the key sort's instantiation.
        pairs.order.extend(
            w.segments
                .iter()
                .enumerate()
                .map(|(j, s)| (s.start as u64) << 32 | (i as u64) << 24 | j as u64),
        );
        count += w.candidates.len();
        bytes += w.candidates.capacity() * core::mem::size_of::<Candidate>()
            + w.segments.capacity() * core::mem::size_of::<Segment>();
    }
    bytes += pairs.order.capacity() * core::mem::size_of::<u64>();
    // Blocks tile the jobs in order, so sorting by first job restores job order at every worker count.
    pairs.order.sort_unstable();
    let job_bytes = (JOB_COUNT * core::mem::size_of::<Job>()).next_multiple_of(16);
    KEY_PTR = crate::arena::extend_scratch(world, count * 8);
    crate::callback_work::pair_scratch(world, job_bytes + (count * 8).next_multiple_of(16) + bytes);
    KEY_COUNT = 0;
    for &order in &pairs.order {
        let w = &pairs.workers[(order >> 24 & 0xff) as usize];
        let segment = w.segments[(order & 0xff_ffff) as usize];
        let mut previous = None;
        let mut accepted = false;
        for c in &w.candidates[segment.begin..segment.end] {
            let pair = (c.a, c.b);
            if previous != Some(pair) {
                previous = Some(pair);
                accepted = if crate::callbacks::filter_needed(world, c.a as usize, c.b as usize) {
                    crate::callback_work::note(world, crate::callback_work::Work::PairSerial);
                    crate::callbacks::filter(world, c.a as usize, c.b as usize)
                } else {
                    true
                };
            }
            if accepted {
                *(KEY_PTR as *mut u64).add(KEY_COUNT) = ((table::pair_key_hi(c.a, c.b) as u64)
                    << 32)
                    | table::pair_key_lo(c.a, c.b, c.child) as u64;
                KEY_COUNT += 1;
            }
        }
    }
    let keys = core::slice::from_raw_parts_mut(KEY_PTR as *mut u64, KEY_COUNT);
    keys.sort_unstable();
    for i in 1..keys.len() {
        debug_assert_ne!(keys[i - 1], keys[i], "physics: duplicate broad-phase pair");
    }
    broad::tree_mut(world, 0).clear_moved();
}
pub extern "C" fn rebuild_trees_in_world(world: usize) {
    unsafe {
        for i in [2, 1] {
            broad::tree_mut(world, i).rebuild(false);
            broad::sync_tree(world, i);
        }
        REBUILD_PENDING = false;
    }
}

//! Cross-worker-count determinism of the staged parallel solver (`stages.rs`).
//!
//! The contract the whole MT design rests on: a staged solve is bit-identical at any worker count,
//! and identical to the serial path the engine ships today. The mechanism is structural — within a
//! graph color no two constraints share a body, so a color's blocks are write-disjoint; the flat
//! prepare/store stages touch only their own record; and the overflow color (whose constraints *do*
//! share bodies) stays serial, in creation order, on the orchestrator.
//!
//! This drives the machinery natively with `std::thread` over plain slice-backed columns. The scene
//! is synthetic — column-level, not a simulated world — because what is under test is the block
//! partitioning, the claim loop, the barriers, and the stage order, not the arithmetic (the gold
//! vectors pin that). It carries what the invariant needs to be falsifiable: three colors, wide and
//! mesh contacts in each, a partial (2-lane) wide record, static-body lanes, and a non-empty
//! overflow color whose contacts deliberately share bodies with the colored ones.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::ThreadId;

use tumble_kernel::body::flags::DYNAMIC;
use tumble_kernel::body::{SIM_STRIDE, STATE_STRIDE};
use tumble_kernel::col::Col;
use tumble_kernel::contact::{
    self, Columns, Softness, CC_META_STRIDE, CC_STRIDE, MCP_STRIDE, MC_META_STRIDE, MC_STRIDE,
    NULL_INDEX,
};
use tumble_kernel::contact_wide::{self, LANES, WIDE_IDX_STRIDE, WIDE_META_STRIDE, WIDE_STRIDE};
use tumble_kernel::integrate;
use tumble_kernel::manifold_abi::{
    DIR_STRIDE, MANIFOLD_STRIDE, M_NORMAL, M_POINTS, M_POINT_COUNT, POOL_POINT_STRIDE, P_ANCHOR_A,
    P_ANCHOR_B, P_BASE_SEPARATION, P_FEATURE_ID, P_NORMAL_IMPULSE, P_SEPARATION, SLOT_STRIDE,
};
use tumble_kernel::math::Vec3;
use tumble_kernel::stages::{self, Block, BlockType, ColorSpan, Plan, Stage, StageWork, SyncBlock};

/// SAFETY: the staged solve's blocks are write-disjoint (`stages.rs`'s header states why), which is
/// exactly `Col`'s promise; the serial run is single-threaded. Both are what this test asserts.
fn col<T: Copy>(v: &mut [T]) -> Col<'_, T> {
    unsafe { Col::of(v) }
}

const BODY_COUNT: usize = 200;
const SUB_STEPS: usize = 4;

const DIR_FRICTION: usize = 0;
const DIR_RESTITUTION: usize = 1;
const DIR_ROLLING: usize = 2;
const DIR_FLAGS: usize = 6;
const DIR_MANIFOLD_COUNT: usize = 7;
const DIR_MANIFOLD_BASE: usize = 8;
const DIR_INDEX_A: usize = 9;
const DIR_INDEX_B: usize = 10;
/// b3_simEnableHitEvent — makes `store` write the per-contact hit flag, so the hash covers it.
const ENABLE_HIT_EVENT: u32 = 0x0010_0000;

/// One contact of the synthetic scene: its bodies (B may be static) and the color that owns it.
struct Contact {
    a: usize,
    b: Option<usize>,
}

/// The scene's contacts, grouped the way the columns lay them out.
struct Layout {
    /// Per active color: the color's convex contacts, then its mesh contacts.
    colors: Vec<(u8, Vec<Contact>, Vec<Contact>)>,
    overflow: Vec<Contact>,
}

/// Three colors, each a body-disjoint matching (the coloring invariant), plus an overflow color whose
/// six contacts all share bodies 0..3 — with each other and with the colored contacts.
fn layout() -> Layout {
    // color 0: convex over bodies 0..79 (pairs), mesh over 80..99
    let c0_wide = (0..40).map(|i| Contact { a: 2 * i, b: Some(2 * i + 1) }).collect();
    let c0_mesh = (0..10).map(|i| Contact { a: 80 + 2 * i, b: Some(81 + 2 * i) }).collect();

    // color 1: convex 0..39 vs 100..139, mesh 40..49 vs 140..149
    let c1_wide = (0..40).map(|i| Contact { a: i, b: Some(100 + i) }).collect();
    let c1_mesh = (0..10).map(|i| Contact { a: 40 + i, b: Some(140 + i) }).collect();

    // color 2: 18 convex against static bodies (null lane B; not a multiple of 4, so the last wide
    // record is a partial 2-lane one), mesh 70..74 vs 170..174.
    let c2_wide = (0..18).map(|i| Contact { a: 50 + i, b: None }).collect();
    let c2_mesh = (0..5).map(|i| Contact { a: 70 + i, b: Some(170 + i) }).collect();

    let overflow = vec![
        Contact { a: 0, b: Some(1) },
        Contact { a: 0, b: Some(2) },
        Contact { a: 0, b: Some(3) },
        Contact { a: 1, b: Some(2) },
        Contact { a: 1, b: Some(3) },
        Contact { a: 2, b: Some(3) },
    ];

    Layout {
        colors: vec![
            (0, c0_wide, c0_mesh),
            (3, c1_wide, c1_mesh),
            (7, c2_wide, c2_mesh),
        ],
        overflow,
    }
}

/// Every column the solve reads or writes, plus the per-step scalars.
struct Scene {
    state: Vec<f32>,
    flags: Vec<u32>,
    sim: Vec<f32>,
    slot: Vec<u32>,
    dir: Vec<u32>,
    pool: Vec<f32>,
    cc: Vec<f32>,
    cc_meta: Vec<u32>,
    mc: Vec<f32>,
    mc_meta: Vec<u32>,
    mcp: Vec<f32>,
    wide: Vec<f32>,
    wide_idx: Vec<u32>,
    wide_meta: Vec<u32>,

    spans: Vec<ColorSpan>,
    wide_total: usize,
    mesh_start: usize,
    mesh_total: usize,
    overflow_start: usize,
    overflow_count: usize,
}

/// A deterministic, non-degenerate float from an integer seed — real enough that a lost or
/// duplicated block moves the hash, without pretending to be a simulated world.
fn jitter(seed: usize, k: usize) -> f32 {
    let n = (seed * 2654435761 + k * 40503) % 1000;
    (n as f32) / 1000.0 - 0.5
}

fn build_scene() -> Scene {
    let l = layout();

    // Contact ids are handed out in column order: each color's convex run, then its mesh run, then
    // the overflow. Every contact owns exactly one manifold (2 points) at pool slot = its id.
    let mut contacts: Vec<&Contact> = Vec::new();
    let mut spans = Vec::new();
    let mut wide_lanes: Vec<usize> = Vec::new(); // contact id per active wide lane, in record order
    let mut mesh_ids: Vec<usize> = Vec::new(); // contact id per scalar record, in record order

    for (color, wide, mesh) in &l.colors {
        let wide_start = wide_lanes.len().div_ceil(LANES);
        for c in wide {
            wide_lanes.push(contacts.len());
            contacts.push(c);
        }
        let wide_count = wide.len().div_ceil(LANES);
        // Each color's wide records start on a fresh record (a record's four lanes must share no
        // body, which only holds within a color).
        assert_eq!(wide_lanes.len().div_ceil(LANES), wide_start + wide_count);
        while wide_lanes.len() % LANES != 0 {
            wide_lanes.push(usize::MAX);
        }

        let mesh_start = mesh_ids.len();
        for c in mesh {
            mesh_ids.push(contacts.len());
            contacts.push(c);
        }

        spans.push(ColorSpan {
            color: *color,
            wide_start,
            wide_count,
            mesh_start,
            mesh_count: mesh.len(),
            joint_start: 0,
            joint_count: 0,
        });
    }

    let mesh_total = mesh_ids.len();
    let overflow_start = mesh_ids.len();
    for c in &l.overflow {
        mesh_ids.push(contacts.len());
        contacts.push(c);
    }
    let overflow_count = l.overflow.len();

    let wide_total = wide_lanes.len() / LANES;
    let contact_count = contacts.len();
    let record_count = mesh_ids.len();

    let mut s = Scene {
        state: vec![0.0; BODY_COUNT * STATE_STRIDE],
        flags: vec![DYNAMIC; BODY_COUNT],
        sim: vec![0.0; BODY_COUNT * SIM_STRIDE],
        slot: vec![0; record_count * SLOT_STRIDE],
        dir: vec![0; contact_count * DIR_STRIDE],
        pool: vec![0.0; contact_count * MANIFOLD_STRIDE],
        cc: vec![0.0; record_count * CC_STRIDE],
        cc_meta: vec![0; record_count * CC_META_STRIDE],
        mc: vec![0.0; record_count * MC_STRIDE],
        mc_meta: vec![0; record_count * MC_META_STRIDE],
        mcp: vec![0.0; 2 * record_count * MCP_STRIDE],
        wide: vec![0.0; wide_total * WIDE_STRIDE],
        wide_idx: vec![0; wide_total * WIDE_IDX_STRIDE],
        wide_meta: vec![0; wide_total * WIDE_META_STRIDE],
        spans,
        wide_total,
        mesh_start: 0,
        mesh_total,
        overflow_start,
        overflow_count,
    };

    for i in 0..BODY_COUNT {
        let o = i * STATE_STRIDE;
        s.state[o] = jitter(i, 1); // linear velocity
        s.state[o + 1] = jitter(i, 2);
        s.state[o + 2] = jitter(i, 3);
        s.state[o + 3] = 0.25 * jitter(i, 4); // angular velocity
        s.state[o + 4] = 0.25 * jitter(i, 5);
        s.state[o + 5] = 0.25 * jitter(i, 6);
        s.state[o + 12] = 1.0; // delta rotation = identity

        let o = i * SIM_STRIDE;
        s.sim[o] = 1.0; // inv mass
        s.sim[o + 1] = 1.0; // gravity scale
        s.sim[o + 2] = 0.1; // linear damping
        s.sim[o + 3] = 0.1; // angular damping
        for k in 0..3 {
            // inv inertia (local == world; the bodies start unrotated)
            s.sim[o + 10 + 4 * k] = 5.0;
            s.sim[o + 19 + 4 * k] = 5.0;
        }
        s.sim[o + 31] = 1.0; // rotation = identity quat
    }

    for (id, c) in contacts.iter().enumerate() {
        let o = id * DIR_STRIDE;
        s.dir[o + DIR_FRICTION] = 0.6f32.to_bits();
        s.dir[o + DIR_RESTITUTION] = 0.4f32.to_bits();
        s.dir[o + DIR_ROLLING] = 0.0f32.to_bits();
        s.dir[o + DIR_FLAGS] = ENABLE_HIT_EVENT;
        s.dir[o + DIR_MANIFOLD_COUNT] = 1;
        s.dir[o + DIR_MANIFOLD_BASE] = id as u32;
        s.dir[o + DIR_INDEX_A] = c.a as u32;
        s.dir[o + DIR_INDEX_B] = c.b.map_or(NULL_INDEX, |b| b as u32);

        let m = id * MANIFOLD_STRIDE;
        // A unit normal, tilted per contact so the tangent basis isn't degenerate.
        let n = Vec3::new(0.1 * jitter(id, 7), 1.0, 0.1 * jitter(id, 8)).normalize();
        s.pool[m + M_NORMAL] = n.x;
        s.pool[m + M_NORMAL + 1] = n.y;
        s.pool[m + M_NORMAL + 2] = n.z;
        s.pool[m + M_POINT_COUNT] = f32::from_bits(2);
        for p in 0..2 {
            let po = m + M_POINTS + p * POOL_POINT_STRIDE;
            let sep = -0.01 + 0.02 * jitter(id, 9 + p);
            s.pool[po + P_ANCHOR_A] = 0.5 * jitter(id, 11 + p);
            s.pool[po + P_ANCHOR_A + 1] = -0.5;
            s.pool[po + P_ANCHOR_A + 2] = 0.5 * jitter(id, 13 + p);
            s.pool[po + P_ANCHOR_B] = 0.5 * jitter(id, 15 + p);
            s.pool[po + P_ANCHOR_B + 1] = 0.5;
            s.pool[po + P_ANCHOR_B + 2] = 0.5 * jitter(id, 17 + p);
            s.pool[po + P_SEPARATION] = sep;
            s.pool[po + P_BASE_SEPARATION] = sep;
            // A warm-start impulse from the "previous step", so the warm start does real work.
            s.pool[po + P_NORMAL_IMPULSE] = 0.2 + 0.1 * jitter(id, 19 + p);
            s.pool[po + P_FEATURE_ID] = f32::from_bits(p as u32 + 1);
        }
    }

    for (r, id) in mesh_ids.iter().enumerate() {
        let o = r * SLOT_STRIDE;
        s.slot[o] = *id as u32; // contact id
        s.slot[o + 1] = r as u32; // transient manifold base (1 manifold per contact)
        s.slot[o + 2] = 2 * r as u32; // transient point base (2 points per manifold)
    }

    for r in 0..wide_total {
        let mo = r * WIDE_META_STRIDE;
        let mut lanes = 0;
        for lane in 0..LANES {
            let id = wide_lanes[r * LANES + lane];
            if id == usize::MAX {
                break;
            }
            s.wide_meta[mo + lane] = id as u32;
            lanes += 1;
        }
        s.wide_meta[mo + LANES] = lanes;
    }

    s
}

// --- the work seam ----------------------------------------------------------------------------

/// The columns, as the workers see them: one [`Col`] over each, shared by every block of a stage. The
/// blocks are write-disjoint (the module header on `stages.rs` states why), which is exactly the
/// contract `Col` exists to express — and the same shape the wasm arena hands the phases.
struct Work<'a> {
    state: Col<'a, f32>,
    flags: Col<'a, u32>,
    sim: Col<'a, f32>,
    slot: Col<'a, u32>,
    dir: Col<'a, u32>,
    pool: Col<'a, f32>,
    cc: Col<'a, f32>,
    cc_meta: Col<'a, u32>,
    mc: Col<'a, f32>,
    mc_meta: Col<'a, u32>,
    mcp: Col<'a, f32>,
    wide: Col<'a, f32>,
    wide_idx: Col<'a, u32>,
    wide_meta: Col<'a, u32>,

    overflow_start: usize,
    overflow_count: usize,

    /// Blocks run by a thread other than the orchestrator. Zero here would make the whole test
    /// vacuous — the workers would have exited before claiming anything and only the serial path
    /// would have been exercised — so the test asserts it moved.
    stolen: AtomicUsize,
    orchestrator: ThreadId,
    /// Make the next block a worker claims panic (`fault_releases_the_orchestrator`).
    die: AtomicBool,

    contact_softness: Softness,
    static_softness: Softness,
    warm_start_scale: f32,
    gravity: Vec3,
    h: f32,
    inv_h: f32,
    inv_dt: f32,
    contact_speed: f32,
    max_linear_velocity: f32,
    restitution_threshold: f32,
    hit_threshold: f32,
}

impl<'a> Work<'a> {
    fn new(s: &'a mut Scene) -> Work<'a> {
        let overflow_start = s.overflow_start;
        let overflow_count = s.overflow_count;
        Work {
            state: col(&mut s.state),
            flags: col(&mut s.flags),
            sim: col(&mut s.sim),
            slot: col(&mut s.slot),
            dir: col(&mut s.dir),
            pool: col(&mut s.pool),
            cc: col(&mut s.cc),
            cc_meta: col(&mut s.cc_meta),
            mc: col(&mut s.mc),
            mc_meta: col(&mut s.mc_meta),
            mcp: col(&mut s.mcp),
            wide: col(&mut s.wide),
            wide_idx: col(&mut s.wide_idx),
            wide_meta: col(&mut s.wide_meta),
            overflow_start,
            overflow_count,
            stolen: AtomicUsize::new(0),
            orchestrator: std::thread::current().id(),
            die: AtomicBool::new(false),
            contact_softness: Softness {
                bias_rate: 30.0,
                mass_scale: 0.35,
                impulse_scale: 0.65,
            },
            static_softness: Softness {
                bias_rate: 60.0,
                mass_scale: 0.7,
                impulse_scale: 0.3,
            },
            warm_start_scale: 1.0,
            gravity: Vec3::new(0.0, -10.0, 0.0),
            h: 1.0 / 240.0,
            inv_h: 240.0,
            inv_dt: 60.0,
            contact_speed: 3.0,
            max_linear_velocity: 400.0,
            restitution_threshold: 1.0,
            hit_threshold: 1.0,
        }
    }

    fn columns(&self) -> Columns<'a> {
        Columns {
            state: self.state,
            flags: self.flags,
            sim: self.sim,
            slot: self.slot,
            dir: self.dir,
            pool: self.pool,
            cc: self.cc,
            cc_meta: self.cc_meta,
            mc: self.mc,
            mc_meta: self.mc_meta,
            mcp: self.mcp,
        }
    }

    fn tick(&self) {
        if std::thread::current().id() != self.orchestrator {
            self.stolen.fetch_add(1, Ordering::Relaxed);
            if self.die.load(Ordering::Relaxed) {
                // A worker dying *inside a block it has already claimed*: the claimed block's
                // completion is never counted, which is what makes the orchestrator's stage barrier a
                // deadlock. The wasm shape is a trap unwinding into the JS round body, which faults the
                // context and rethrows — here, the `catch_unwind` in `fault_releases_the_orchestrator`.
                panic!("worker trapped mid-block");
            }
        }
    }

    fn overflow(&self) -> Block {
        Block {
            start: self.overflow_start,
            count: self.overflow_count,
            block_type: BlockType::Contact,
            color: u8::MAX,
        }
    }
}

impl StageWork for Work<'_> {
    fn prepare_wide(&self, b: Block) {
        self.tick();
        contact_wide::prepare(
            self.wide,
            self.wide_idx,
            self.wide_meta,
            self.state,
            self.sim,
            self.dir,
            self.pool,
            b.start,
            b.count,
            self.contact_softness,
            self.static_softness,
            self.warm_start_scale,
        );
    }

    fn prepare_mesh(&self, b: Block) {
        self.tick();
        contact::prepare(
            &self.columns(),
            b.start,
            b.count,
            self.contact_softness,
            self.static_softness,
            self.warm_start_scale,
        );
    }

    fn integrate_velocities(&self, b: Block) {
        self.tick();
        integrate::integrate_velocities(
            self.state,
            self.sim,
            b.start,
            b.count,
            self.gravity,
            self.h,
        );
    }

    fn integrate_positions(&self, b: Block) {
        self.tick();
        integrate::integrate_positions(
            self.state,
            self.flags,
            b.start,
            b.count,
            self.h,
            self.max_linear_velocity,
            self.inv_dt,
        );
    }

    fn warm_start_wide(&self, b: Block, worker: usize) {
        self.tick();
        contact_wide::warm_start(
            self.wide,
            self.wide_idx,
            self.state,
            self.flags,
            b.start,
            b.count,
            worker,
        );
    }

    fn warm_start_mesh(&self, b: Block) {
        self.tick();
        contact::warm_start(&self.columns(), b.start, b.count);
    }

    fn solve_wide(&self, b: Block, use_bias: bool, worker: usize) {
        self.tick();
        contact_wide::solve(
            self.wide,
            self.wide_idx,
            self.state,
            self.flags,
            b.start,
            b.count,
            use_bias,
            self.inv_h,
            self.contact_speed,
            worker,
        );
    }

    fn solve_mesh(&self, b: Block, use_bias: bool) {
        self.tick();
        contact::solve(
            &self.columns(),
            b.start,
            b.count,
            use_bias,
            self.inv_h,
            self.contact_speed,
        );
    }

    fn restitution_wide(&self, b: Block, worker: usize) {
        self.tick();
        contact_wide::restitution(
            self.wide,
            self.wide_idx,
            self.state,
            self.flags,
            b.start,
            b.count,
            self.restitution_threshold,
            worker,
        );
    }

    fn restitution_mesh(&self, b: Block) {
        self.tick();
        contact::restitution(&self.columns(), b.start, b.count, self.restitution_threshold);
    }

    fn store_wide(&self, b: Block, _worker_index: usize) {
        self.tick();
        contact_wide::store(
            self.wide,
            self.wide_meta,
            self.dir,
            self.pool,
            b.start,
            b.count,
            self.hit_threshold,
        );
    }

    fn store_mesh(&self, b: Block, _worker_index: usize) {
        self.tick();
        contact::store(&self.columns(), b.start, b.count, self.hit_threshold);
    }

    fn finalize(&self, b: Block) {
        self.tick();
        // A stand-in pose advance (the real finalize arithmetic is pinned by `finalize_gold.rs` and
        // the fixture suite; what is under test here is the terminal stage's claim + barrier). Same
        // dataflow shape as `finalize::finalize`: reads the solved velocities and deltas, folds the
        // deltas into the velocity slots, and resets them — per-body write-disjoint, hash-visible,
        // and dependent on every velocity-writing stage before it, so a broken solve→finalize
        // barrier or a doubled/dropped block diverges the worker-count hashes.
        for i in b.start..b.start + b.count {
            let o = i * STATE_STRIDE;
            for k in 0..6 {
                let v = self.state.get(o + k);
                let d = self.state.get(o + 6 + k);
                self.state.set(o + k, v + d * self.inv_dt);
                self.state.set(o + 6 + k, 0.0);
            }
        }
    }

    fn prepare_overflow(&self) {
        self.prepare_mesh(self.overflow());
    }

    fn warm_start_overflow(&self) {
        self.warm_start_mesh(self.overflow());
    }

    fn solve_overflow(&self, use_bias: bool) {
        self.solve_mesh(self.overflow(), use_bias);
    }

    fn restitution_overflow(&self) {
        self.restitution_mesh(self.overflow());
    }

    fn store_overflow(&self) {
        self.store_mesh(self.overflow(), 0);
    }
}

// --- the two runs -----------------------------------------------------------------------------

fn plan<'a>(s: &'a Scene, worker_count: usize) -> Plan<'a> {
    Plan {
        body_count: BODY_COUNT,
        wide_total: s.wide_total,
        mesh_start: s.mesh_start,
        mesh_total: s.mesh_total,
        joint_total: 0,
        colors: &s.spans,
        sub_step_count: SUB_STEPS,
        worker_count,
    }
}

fn whole(start: usize, count: usize, block_type: BlockType) -> Block {
    Block {
        start,
        count,
        block_type,
        color: u8::MAX,
    }
}

/// The serial path the engine ships (`src/solver.ts`), phase for phase: the flat prepares, then the
/// sub-step loop with the overflow ahead of each color group, then restitution, store, and the
/// finalize sweep. The independent oracle the staged run must reproduce bit for bit.
fn solve_serial(work: &Work, spans: &[ColorSpan], wide_total: usize, mesh_start: usize, mesh_total: usize) {
    let bodies = whole(0, BODY_COUNT, BlockType::Body);
    let all_wide = whole(0, wide_total, BlockType::WideContact);
    let all_mesh = whole(mesh_start, mesh_total, BlockType::Contact);
    let color_wide =
        |s: &ColorSpan| whole(s.wide_start, s.wide_count, BlockType::GraphWideContact);
    let color_mesh = |s: &ColorSpan| whole(s.mesh_start, s.mesh_count, BlockType::GraphContact);

    work.prepare_wide(all_wide);
    work.prepare_mesh(all_mesh);
    work.prepare_overflow();

    for _ in 0..SUB_STEPS {
        work.integrate_velocities(bodies);

        work.warm_start_overflow();
        for s in spans {
            work.warm_start_wide(color_wide(s), 0);
            work.warm_start_mesh(color_mesh(s));
        }

        work.solve_overflow(true);
        for s in spans {
            work.solve_wide(color_wide(s), true, 0);
            work.solve_mesh(color_mesh(s), true);
        }

        work.integrate_positions(bodies);

        work.solve_overflow(false);
        for s in spans {
            work.solve_wide(color_wide(s), false, 0);
            work.solve_mesh(color_mesh(s), false);
        }
    }

    work.restitution_overflow();
    for s in spans {
        work.restitution_wide(color_wide(s), 0);
        work.restitution_mesh(color_mesh(s));
    }

    work.store_overflow();
    work.store_wide(all_wide, 0);
    work.store_mesh(all_mesh, 0);

    work.finalize(bodies);
}

fn solve_staged(work: &Work, p: &Plan, worker_count: usize) {
    let need = stages::sizes(p);
    let mut stage_buf: Vec<Stage> = (0..need.stages).map(|_| Stage::EMPTY).collect();
    let mut block_buf: Vec<SyncBlock> = (0..need.blocks).map(|_| SyncBlock::EMPTY).collect();
    let ctx = stages::build(p, &mut stage_buf, &mut block_buf);

    let ctx = &ctx;
    std::thread::scope(|scope| {
        for w in 1..worker_count {
            scope.spawn(move || stages::run(ctx, work, w));
        }
        stages::run(ctx, work, 0);
    });
}

/// FNV-1a over the persistent outputs of a solve: body state, the manifold pool (the impulses the
/// next step warm-starts from), and the directory (the hit flags `store` writes).
fn hash(work: &Work) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut byte = |b: u8| {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    };
    for i in 0..work.state.len() {
        for b in work.state.get(i).to_bits().to_le_bytes() {
            byte(b);
        }
    }
    for i in 0..work.pool.len() {
        for b in work.pool.get(i).to_bits().to_le_bytes() {
            byte(b);
        }
    }
    for i in 0..work.dir.len() {
        for b in work.dir.get(i).to_le_bytes() {
            byte(b);
        }
    }
    for i in 0..work.flags.len() {
        for b in work.flags.get(i).to_le_bytes() {
            byte(b);
        }
    }
    h
}

#[test]
fn staged_solve_is_worker_count_independent() {
    let mut serial_scene = build_scene();
    let serial = Work::new(&mut serial_scene);
    let scene = build_scene();
    solve_serial(
        &serial,
        &scene.spans,
        scene.wide_total,
        scene.mesh_start,
        scene.mesh_total,
    );
    let want = hash(&serial);

    // The solve must have moved the world: an all-zero or unchanged state would make every hash
    // trivially agree.
    let mut base_scene = build_scene();
    let baseline = hash(&Work::new(&mut base_scene));
    assert_ne!(want, baseline, "serial solve changed nothing");
    assert!(
        (0..serial.state.len()).all(|i| serial.state.get(i).is_finite()),
        "serial solve produced a non-finite body state"
    );

    // The scene must actually exercise the machinery: several blocks per stage, and a real overflow.
    let s = build_scene();
    let sizes = stages::sizes(&plan(&s, 8));
    assert!(sizes.blocks >= 20, "too few blocks to expose a claim race: {sizes:?}");
    assert_eq!(s.spans.len(), 3);
    assert_eq!(s.overflow_count, 6);

    for workers in [1, 2, 8] {
        let mut stolen = 0;
        // Repeat: whether a worker wins a CAS is a genuine race, so one run could finish before the
        // spawned threads ever claim a block. The hash must hold on every run; over three runs at
        // least one block must have been stolen, or the multi-worker path proved nothing.
        for _ in 0..3 {
            let scene = build_scene();
            let mut work_scene = build_scene();
            let work = Work::new(&mut work_scene);
            solve_staged(&work, &plan(&scene, workers), workers);
            assert_eq!(
                hash(&work),
                want,
                "staged solve with {workers} worker(s) diverged from the serial path"
            );
            stolen += work.stolen.load(Ordering::Relaxed);
        }

        if workers > 1 {
            assert!(stolen > 0, "no block was stolen at {workers} workers: the test is vacuous");
        }
    }
}

/// The orchestrator's `run` must not return while a worker is still inside the solve. Without that
/// join the caller could rebuild the context — zeroing every block's sync index — while a late worker
/// sits in a stale `execute_stage`, whose CAS would then win against a freshly reset block and run it
/// a second time over the next step's columns.
///
/// The worker here is deliberately slow to arrive: the orchestrator finishes the whole stage list
/// alone in well under a millisecond, so if it returned without waiting it would return immediately.
#[test]
fn orchestrator_waits_for_late_workers() {
    let scene = build_scene();
    let mut work_scene = build_scene();
    let work = Work::new(&mut work_scene);
    let p = plan(&scene, 2);

    let need = stages::sizes(&p);
    let mut stage_buf: Vec<Stage> = (0..need.stages).map(|_| Stage::EMPTY).collect();
    let mut block_buf: Vec<SyncBlock> = (0..need.blocks).map(|_| SyncBlock::EMPTY).collect();
    let ctx = stages::build(&p, &mut stage_buf, &mut block_buf);
    let ctx = &ctx;

    const LATE: std::time::Duration = std::time::Duration::from_millis(50);

    let elapsed = std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(LATE);
            stages::run(ctx, &work, 1);
        });

        let start = std::time::Instant::now();
        stages::run(ctx, &work, 0);
        start.elapsed()
    });

    assert!(
        elapsed >= LATE,
        "orchestrator returned after {elapsed:?}, before the worker had left the solve"
    );
}

/// A worker that dies mid-solve must release the orchestrator, not hang it.
///
/// The orchestrator's spins — the stage barrier and the exit join — live inside the kernel, and on wasm
/// a dead worker is a trap that unwound into JS. No JS event can reach a thread spinning in wasm, so
/// the JS fault protocol (`src/pool.ts`) only gets its turn once the orchestrator has *returned*: the
/// context's fault flag is the only thing that can get it there. Without it this test hangs forever —
/// the worker panics inside a block it has already CAS-claimed, so that block's completion is never
/// counted and the barrier can never be satisfied.
///
/// The watchdog is what turns that hang into a failure instead of a stuck run.
#[test]
fn fault_releases_the_orchestrator() {
    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        let scene = build_scene();
        let mut work_scene = build_scene();
        let work = Work::new(&mut work_scene);
        let p = plan(&scene, 2);

        let need = stages::sizes(&p);
        let mut stage_buf: Vec<Stage> = (0..need.stages).map(|_| Stage::EMPTY).collect();
        let mut block_buf: Vec<SyncBlock> = (0..need.blocks).map(|_| SyncBlock::EMPTY).collect();
        let ctx = stages::build(&p, &mut stage_buf, &mut block_buf);
        let ctx = &ctx;
        let work = &work;
        work.die.store(true, Ordering::Relaxed);

        // Silence the panic the worker is *supposed* to raise; a test that prints a backtrace on its
        // happy path trains the reader to ignore backtraces.
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        std::thread::scope(|scope| {
            scope.spawn(move || {
                // The worker's round body: a trap unwinds, the fault is published, the failure is
                // reported. Exactly `src/pool.ts`'s catch, and in that order.
                let died =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| stages::run(ctx, work, 1)));
                if died.is_err() {
                    ctx.fault();
                }
            });
            stages::run(ctx, work, 0);
        });

        std::panic::set_hook(hook);
        tx.send(()).ok();
    });

    rx.recv_timeout(std::time::Duration::from_secs(10))
        .expect("the orchestrator never returned: a dead worker deadlocked the solve");
}

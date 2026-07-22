//! The staged parallel solver machinery, ported from box3d's `solver.c` (`b3SolverStage` /
//! `b3SolverBlock` / `b3SyncBlock`, `b3ExecuteBlock` / `b3ExecuteStage` / `b3ExecuteMainStage`, and
//! the `b3SolverTask` orchestrator). Structure only — every block's arithmetic is the same phase
//! function the serial path already calls, over the same columns.
//!
//! The model: the solve is a fixed list of **stages**, each an array of **blocks** (an index range
//! of bodies / contacts / wide records). Worker 0 is the orchestrator: it walks the stage list, and
//! for each stage publishes `syncBits = (syncIndex << 16) | stageIndex` into one atomic. Workers
//! spin on that atomic, decode the stage, and CAS-claim blocks by bumping each block's `syncIndex`
//! from `syncIndex - 1` to `syncIndex`. The orchestrator works the same stage itself, then spins
//! until the stage's completion count reaches its block count — that spin is the barrier. The sync
//! index grows monotonically so a delayed worker can never re-run a completed block (ABA).
//!
//! **Determinism across worker counts.** Within a graph color no two constraints share a body, so
//! the blocks of a color stage are write-disjoint and their execution order cannot change the
//! result. Prepare/store blocks write only their own record's transient row (and their own contact's
//! persistent manifold), so the flat stages are order-free too. The overflow color — where
//! constraints *do* share bodies — never becomes blocks: the orchestrator runs it serially, in
//! creation order, between stages while every worker is parked in its spin. No reduction depends on
//! worker identity. Two threads give the same bits as eight.
//!
//! Deviations from the C, all mechanical:
//!   - a stage names its blocks by `[start, start+count)` into one shared block array instead of
//!     holding a pointer (a raw pointer isn't `Sync`; an index is);
//!   - box3d's per-color constraint arrays are tumble's flat columns plus a per-color base, so a
//!     graph block's `start` is already a flat column index (box3d adds the color base inside the
//!     task);
//!   - box3d's `mainClaimed` race is gone. It exists so *some* thread orchestrates when the user's
//!     task system schedules worker 0 late; our pool has no external task system, so the thread that
//!     drives the step is always worker 0;
//!   - the pose finalize is fused in as the terminal stage, over the body blocks the integrate stages
//!     already use. box3d runs `b3FinalizeBodiesTask` as a separate task after the solver task — a
//!     free hand-off on its live task system, but a whole extra park/wake round on our pool. The
//!     ordinary stage barrier (the orchestrator publishes a stage only after the previous stage's
//!     completion count reached its block count) is what keeps finalize from overlapping the solve,
//!     and finalize's per-body work is write-disjoint like every body stage, so the fusion cannot
//!     change a bit.
//!
//! Joints are present as stage/block *slots* only (`PrepareJoints`, `GraphJoint` blocks, the
//! `*_joints` trait methods) with no work behind them — a jointed scene keeps the TS per-color
//! interleave until the joints-in-kernel unit fills them in.
//!
//! `core::sync::atomic` (not `std`) so this module compiles into the wasm artifact unchanged. On the
//! single-thread artifact (no `+atomics`) these lower to plain loads and stores, which is correct:
//! `worker_count == 1` means the orchestrator runs every block itself and nothing is ever contended.

use core::sync::atomic::{AtomicI32, AtomicU32, Ordering};

/// Solve iterations per sub-step (box3d `ITERATIONS`).
pub const ITERATIONS: usize = 1;
/// Relax iterations per sub-step (box3d `RELAX_ITERATIONS`).
pub const RELAX_ITERATIONS: usize = 1;

/// Graph colors (`GRAPH_COLOR_COUNT` in `src/core.ts`); the last is the serial overflow color, so at
/// most `MAX_COLORS - 1` colors can be active.
pub const MAX_COLORS: usize = 24;

/// A block that belongs to no color (box3d's `UINT8_MAX` color tag).
const NULL_COLOR: u8 = u8::MAX;

/// Every block of one stage is claimed and run before the next stage starts (b3SolverStageType).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StageType {
    PrepareJoints,
    PrepareWideContacts,
    PrepareContacts,
    IntegrateVelocities,
    WarmStart,
    Solve,
    IntegratePositions,
    Relax,
    Restitution,
    StoreWideImpulses,
    StoreImpulses,
    Finalize,
}

/// What a block's index range indexes (b3SolverBlockType). The tag rides the block, not the stage,
/// so a color stage can mix joint, wide-contact, and mesh-contact blocks and run them concurrently.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlockType {
    Body,
    Joint,
    WideContact,
    Contact,
    GraphJoint,
    GraphWideContact,
    GraphContact,
}

/// One unit of work: a `[start, start + count)` range of the column the block's type names
/// (b3SolverBlock).
#[derive(Clone, Copy, Debug)]
pub struct Block {
    pub start: usize,
    pub count: usize,
    pub block_type: BlockType,
    pub color: u8,
}

impl Block {
    const EMPTY: Block = Block {
        start: 0,
        count: 0,
        block_type: BlockType::Body,
        color: NULL_COLOR,
    };
}

/// A block plus the atomic workers CAS to claim it (b3SyncBlock). The descriptor is written once at
/// build and only ever read after; the atomic is the only mutable field, so the claim winner copies
/// the descriptor out by value and never aliases what other workers are CAS-writing.
pub struct SyncBlock {
    block: Block,
    sync_index: AtomicI32,
}

impl SyncBlock {
    /// The initializer for a caller's block buffer — a `const` so wasm can put that buffer in a
    /// `static`. Each use constructs a fresh, unclaimed block; that is the point, not a hazard.
    #[allow(clippy::declare_interior_mutable_const)]
    pub const EMPTY: SyncBlock = SyncBlock {
        block: Block::EMPTY,
        sync_index: AtomicI32::new(0),
    };
}

/// A stage: a slice of the shared block array, plus the counter its barrier spins on (b3SolverStage).
/// Iterative stages (warm start / solve / relax / restitution of one color) re-use one block slice
/// every sub-step; the blocks' sync indices grow monotonically across those re-uses.
///
/// One cache line each. `completion` is `fetch_add`ed by every worker that ran a block of the stage and
/// spun on by the orchestrator, so two stages sharing a line put one stage's barrier under the other's
/// write traffic. Unaligned, the data-section layout decides how much of that a build gets, and any
/// static added near this array re-rolls it (measured: a layout shift alone cost the 8-thread solve 34%).
#[repr(align(64))]
pub struct Stage {
    ty: StageType,
    block_start: usize,
    block_count: usize,
    color: u8,
    completion: AtomicI32,
}

impl Stage {
    /// As [`SyncBlock::EMPTY`].
    #[allow(clippy::declare_interior_mutable_const)]
    pub const EMPTY: Stage = Stage {
        ty: StageType::PrepareJoints,
        block_start: 0,
        block_count: 0,
        color: NULL_COLOR,
        completion: AtomicI32::new(0),
    };
}

/// One active graph color's constraints, as flat column ranges (the `colorSpan` column the batched
/// serial path already writes, plus the joint range the joints unit will fill).
#[derive(Clone, Copy, Debug, Default)]
pub struct ColorSpan {
    pub color: u8,
    pub wide_start: usize,
    pub wide_count: usize,
    pub mesh_start: usize,
    pub mesh_count: usize,
    pub joint_start: usize,
    pub joint_count: usize,
}

impl ColorSpan {
    /// The initializer for a caller's span buffer (`Default` isn't `const`).
    pub const EMPTY: ColorSpan = ColorSpan {
        color: NULL_COLOR,
        wide_start: 0,
        wide_count: 0,
        mesh_start: 0,
        mesh_count: 0,
        joint_start: 0,
        joint_count: 0,
    };
}

/// What the solve is over: the awake body count, the flat constraint ranges the order-free
/// prepare/store stages sweep, and the active colors. The overflow color is deliberately absent —
/// it never becomes blocks; the work impl owns its range and the orchestrator calls it serially.
#[derive(Clone, Copy, Debug)]
pub struct Plan<'a> {
    pub body_count: usize,
    /// Flat wide (convex) record range: prepare + store sweep `[0, wide_total)`.
    pub wide_total: usize,
    /// Flat colored scalar (mesh) record range: prepare + store sweep `[mesh_start, +mesh_total)`.
    pub mesh_start: usize,
    pub mesh_total: usize,
    /// Flat joint range for `PrepareJoints`. Zero until the joints unit lands.
    pub joint_total: usize,
    pub colors: &'a [ColorSpan],
    pub sub_step_count: usize,
    pub worker_count: usize,
}

/// The per-block work, behind a seam so the same machinery drives owned columns natively and the
/// linear-memory columns on wasm. Each method runs one phase over one block's index range; the impl
/// owns the columns and the per-step scalars (softness, `h`, thresholds).
///
/// Blocks of a stage run **concurrently on different threads** — an impl must treat its columns as
/// shared and rely on the write-disjointness this module's header states, exactly as box3d does.
/// The `_overflow` methods are the exception: the orchestrator calls them alone, between stages.
///
/// The columns are [`Col`](crate::col::Col)s, not `&mut [f32]`: every block of a stage holds the whole
/// column and indexes it by body / record id, so a `&mut` would be live aliasing `&mut` on each worker
/// even though the writes never overlap. `col.rs` carries the argument.
pub trait StageWork: Sync {
    fn prepare_wide(&self, block: Block);
    fn prepare_mesh(&self, block: Block);
    fn integrate_velocities(&self, block: Block);
    fn warm_start_mesh(&self, block: Block);
    fn solve_mesh(&self, block: Block, use_bias: bool);
    fn integrate_positions(&self, block: Block);
    fn restitution_mesh(&self, block: Block);
    // `worker_index` on the three wide phases: their gather/scatter remaps null/static lanes onto a
    // per-worker identity record (`contact_wide::ident_rec`), which is what keeps the state column
    // write-disjoint across blocks. The store phases and the joint solve take it because box3d's do —
    // there it selects the worker's `b3TaskContext` event bit sets; this port's store writes a whole
    // `u32` hit slot per contact instead, so nothing needs the lane yet, and the joints unit inherits
    // the parameter. A parallel phase that ever *appends* must take a per-worker lane merged in worker
    // order — never a shared cursor, which would make the output depend on claim order.
    fn warm_start_wide(&self, block: Block, worker_index: usize);
    fn solve_wide(&self, block: Block, use_bias: bool, worker_index: usize);
    fn restitution_wide(&self, block: Block, worker_index: usize);
    fn store_wide(&self, block: Block, worker_index: usize);
    fn store_mesh(&self, block: Block, worker_index: usize);

    /// The pose finalize, fused as the solve's terminal stage over the body blocks (header deviations).
    /// Per-body write-disjoint, like every body stage.
    fn finalize(&self, block: Block);

    fn prepare_overflow(&self);
    fn warm_start_overflow(&self);
    fn solve_overflow(&self, use_bias: bool);
    fn restitution_overflow(&self);
    fn store_overflow(&self);

    // Colored joint phases: the `PrepareJoints` flat stage sweeps every colored joint, then each
    // color's `GraphJoint` block warm-starts + solves its joints (write-disjoint by coloring). A
    // jointless plan builds no joint block, so these never run for it.
    fn prepare_joints(&self, _block: Block) {}
    fn warm_start_joints(&self, _block: Block) {}
    fn solve_joints(&self, _block: Block, _use_bias: bool, _worker_index: usize) {}

    // Overflow joint phases: joints the graph coloring could not separate (they share bodies), run
    // serially on the orchestrator between stages, in creation order — box3d's `b3*Joints_Overflow`.
    fn prepare_overflow_joints(&self) {}
    fn warm_start_overflow_joints(&self) {}
    fn solve_overflow_joints(&self, _use_bias: bool) {}
}

// --- block sizing -----------------------------------------------------------------------------

/// Blocks per worker, so a worker that finishes early can steal (box3d: "target 4 blocks per
/// worker"). Small enough to keep the per-block sync overhead low.
const BLOCKS_PER_WORKER: usize = 4;
/// Minimum bodies per body block (box3d `minBodiesPerBlock`).
const MIN_BODIES_PER_BLOCK: usize = 32;
/// Minimum constraints per contact/joint block (box3d `minContactsPerBlock` / `minJointsPerBlock`).
const MIN_CONSTRAINTS_PER_BLOCK: usize = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BlockDim {
    size: usize,
    count: usize,
}

/// b3ComputeBlockCount. Each worker gets at most `max_block_count / worker_count` blocks; below that
/// the block size is the minimum, above it the size grows so the block count stays bounded.
fn compute_block_count(item_count: usize, min_size: usize, max_block_count: usize) -> BlockDim {
    if item_count == 0 {
        return BlockDim::default();
    }

    let size = if item_count <= min_size * max_block_count {
        min_size
    } else {
        item_count.div_ceil(max_block_count)
    };

    BlockDim {
        size,
        count: item_count.div_ceil(size),
    }
}

/// b3InitBlocks. `base` is the block range's first flat column index (box3d's per-color arrays are
/// our flat columns plus a color base). The claim counter is zeroed so the first stage over these
/// blocks CASes 0 → 1.
fn init_blocks(
    blocks: &mut [SyncBlock],
    dim: BlockDim,
    base: usize,
    item_count: usize,
    block_type: BlockType,
    color: u8,
) {
    for (i, sb) in blocks.iter_mut().enumerate().take(dim.count) {
        sb.block = Block {
            start: base + i * dim.size,
            count: dim.size,
            block_type,
            color,
        };
        sb.sync_index.store(0, Ordering::SeqCst);
    }
    if dim.count > 0 {
        blocks[dim.count - 1].block.count = item_count - (dim.count - 1) * dim.size;
    }
}

/// The block dims of one plan, in the order they are laid out in the shared block array.
struct Dims {
    body: BlockDim,
    wide: BlockDim,
    mesh: BlockDim,
    joint: BlockDim,
    color_joint: [BlockDim; MAX_COLORS],
    color_wide: [BlockDim; MAX_COLORS],
    color_mesh: [BlockDim; MAX_COLORS],
    graph_block_count: usize,
}

fn dims(plan: &Plan) -> Dims {
    let max_block_count = BLOCKS_PER_WORKER * plan.worker_count;
    let contacts = |n| compute_block_count(n, MIN_CONSTRAINTS_PER_BLOCK, max_block_count);

    let mut d = Dims {
        body: compute_block_count(plan.body_count, MIN_BODIES_PER_BLOCK, max_block_count),
        wide: contacts(plan.wide_total),
        mesh: contacts(plan.mesh_total),
        joint: contacts(plan.joint_total),
        color_joint: [BlockDim::default(); MAX_COLORS],
        color_wide: [BlockDim::default(); MAX_COLORS],
        color_mesh: [BlockDim::default(); MAX_COLORS],
        graph_block_count: 0,
    };

    for (i, span) in plan.colors.iter().enumerate() {
        d.color_joint[i] = contacts(span.joint_count);
        d.color_wide[i] = contacts(span.wide_count);
        d.color_mesh[i] = contacts(span.mesh_count);
        d.graph_block_count +=
            d.color_joint[i].count + d.color_wide[i].count + d.color_mesh[i].count;
    }

    d
}

/// Storage a plan needs. The caller owns the buffers (no allocator in the live wasm path).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sizes {
    pub stages: usize,
    pub blocks: usize,
}

/// The storage any plan can ever need at `max_workers` threads, for a caller sizing a fixed buffer (the
/// wasm path has no allocator — kex `tumble.md`). Every dim's block count is capped at
/// `BLOCKS_PER_WORKER * worker_count` by [`compute_block_count`], and a plan has 4 flat dims (body,
/// wide, mesh, joint) plus 3 per color.
pub const fn max_sizes(max_workers: usize) -> Sizes {
    let colors = MAX_COLORS - 1; // the last color is the serial overflow, which never becomes blocks
    Sizes {
        stages: 8 + colors * (2 + ITERATIONS + RELAX_ITERATIONS),
        blocks: (4 + 3 * colors) * BLOCKS_PER_WORKER * max_workers,
    }
}

pub fn sizes(plan: &Plan) -> Sizes {
    let d = dims(plan);
    let colors = plan.colors.len();
    Sizes {
        // prepare joints/wide/mesh + integrate velocities + integrate positions + store wide/mesh +
        // finalize, plus one stage per color for warm start, each solve iteration, each relax
        // iteration, and restitution. Finalize re-uses the body blocks, so it costs no block storage.
        stages: 8 + colors * (2 + ITERATIONS + RELAX_ITERATIONS),
        blocks: d.body.count + d.wide.count + d.mesh.count + d.joint.count + d.graph_block_count,
    }
}

// --- build ------------------------------------------------------------------------------------

/// The solve's stage list and its blocks, plus the one atomic the orchestrator publishes to.
/// Everything but `sync_bits` is read-only once built.
pub struct Context<'a> {
    stages: &'a [Stage],
    blocks: &'a [SyncBlock],
    colors: &'a [ColorSpan],
    sub_step_count: usize,
    worker_count: usize,
    /// `(syncIndex << 16) | stageIndex`, monotone within a step. `u32::MAX` is the finish sentinel.
    /// Its own cache line: every worker read-spins on it while the orchestrator writes it, and it must
    /// not share a line with the counters the workers *write* (`exited`) — see [`Stage`].
    sync_bits: Line<AtomicU32>,
    /// Workers that have seen the finish sentinel and left [`run`]. The orchestrator waits on this
    /// before returning — see [`run`].
    exited: AtomicI32,
    /// Set when a worker dies inside the solve — see [`Context::fault`].
    fault: AtomicU32,
}

/// One cache line to itself.
#[repr(align(64))]
struct Line<T>(T);

impl Context<'_> {
    /// Abandon this solve: a worker died (on wasm, a trap that unwound into its JS round body, which
    /// calls this before it acks — `src/pool.ts`).
    ///
    /// Every spin in this module tests the flag, because a dead worker claims no further block and
    /// never acks the finish sentinel — so without it the orchestrator spins forever, either at the
    /// stage barrier of a block the worker claimed and never completed, or in the exit join. Both spins
    /// are *inside* wasm, where no JS error event can reach it: the flag is the only way out, and the
    /// JS fault protocol only gets its turn once the orchestrator has returned.
    ///
    /// It lives in the context, not in a global: a context is rebuilt per solve, so the flag is
    /// per-solve by construction and a faulted step cannot poison the next.
    pub fn fault(&self) {
        self.fault.store(1, Ordering::SeqCst);
    }

    #[inline]
    fn faulted(&self) -> bool {
        self.fault.load(Ordering::SeqCst) != 0
    }
}

/// b3InitStage.
fn init_stage(
    stages: &mut [Stage],
    s: &mut usize,
    ty: StageType,
    block_start: usize,
    block_count: usize,
    color: u8,
) {
    stages[*s] = Stage {
        ty,
        block_start,
        block_count,
        color,
        completion: AtomicI32::new(0),
    };
    *s += 1;
}

/// b3InitColorStages: one stage per color per iteration. Every iteration of a color re-uses that
/// color's one block slice, so its blocks' sync indices grow monotonically across the iterations.
fn init_color_stages(
    stages: &mut [Stage],
    s: &mut usize,
    ty: StageType,
    iterations: usize,
    plan: &Plan,
    color_at: &[usize],
    color_blocks: &[usize],
) {
    for _ in 0..iterations {
        for (i, span) in plan.colors.iter().enumerate() {
            init_stage(stages, s, ty, color_at[i], color_blocks[i], span.color);
        }
    }
}

/// Lay out the blocks and the stage list for `plan` in caller-owned storage. `stages` and `blocks`
/// must be at least [`sizes`] long.
pub fn build<'a>(plan: &Plan<'a>, stages: &'a mut [Stage], blocks: &'a mut [SyncBlock]) -> Context<'a> {
    assert!(plan.colors.len() < MAX_COLORS);
    assert!(plan.worker_count >= 1);

    let need = sizes(plan);
    assert!(stages.len() >= need.stages && blocks.len() >= need.blocks);
    // The sync bits pack the sync index and the stage index into 16 bits each. The graph blocks are
    // re-swept once per warm-start / solve / relax group, so their sync index is the one that grows.
    assert!(need.stages < 0x10000);
    assert!(1 + (1 + ITERATIONS + RELAX_ITERATIONS) * plan.sub_step_count < 0x10000);

    let d = dims(plan);

    // Block array: body | wide | mesh | joint | per-color (joint, wide, mesh).
    let body_at = 0;
    let wide_at = body_at + d.body.count;
    let mesh_at = wide_at + d.wide.count;
    let joint_at = mesh_at + d.mesh.count;
    let graph_at = joint_at + d.joint.count;

    init_blocks(&mut blocks[body_at..], d.body, 0, plan.body_count, BlockType::Body, NULL_COLOR);
    init_blocks(&mut blocks[wide_at..], d.wide, 0, plan.wide_total, BlockType::WideContact, NULL_COLOR);
    init_blocks(
        &mut blocks[mesh_at..],
        d.mesh,
        plan.mesh_start,
        plan.mesh_total,
        BlockType::Contact,
        NULL_COLOR,
    );
    init_blocks(&mut blocks[joint_at..], d.joint, 0, plan.joint_total, BlockType::Joint, NULL_COLOR);

    // Each color's blocks are contiguous: joints, then wide contacts, then mesh contacts (box3d's
    // order). A color stage points at the whole run, so all three kinds run concurrently.
    let mut color_at = [0usize; MAX_COLORS];
    let mut color_blocks = [0usize; MAX_COLORS];
    let mut at = graph_at;
    for (i, span) in plan.colors.iter().enumerate() {
        color_at[i] = at;

        init_blocks(
            &mut blocks[at..],
            d.color_joint[i],
            span.joint_start,
            span.joint_count,
            BlockType::GraphJoint,
            span.color,
        );
        at += d.color_joint[i].count;

        init_blocks(
            &mut blocks[at..],
            d.color_wide[i],
            span.wide_start,
            span.wide_count,
            BlockType::GraphWideContact,
            span.color,
        );
        at += d.color_wide[i].count;

        init_blocks(
            &mut blocks[at..],
            d.color_mesh[i],
            span.mesh_start,
            span.mesh_count,
            BlockType::GraphContact,
            span.color,
        );
        at += d.color_mesh[i].count;

        color_blocks[i] = at - color_at[i];
    }
    debug_assert_eq!(at - graph_at, d.graph_block_count);

    let mut s = 0;
    init_stage(stages, &mut s, StageType::PrepareJoints, joint_at, d.joint.count, NULL_COLOR);
    init_stage(stages, &mut s, StageType::PrepareWideContacts, wide_at, d.wide.count, NULL_COLOR);
    init_stage(stages, &mut s, StageType::PrepareContacts, mesh_at, d.mesh.count, NULL_COLOR);
    init_stage(stages, &mut s, StageType::IntegrateVelocities, body_at, d.body.count, NULL_COLOR);
    init_color_stages(stages, &mut s, StageType::WarmStart, 1, plan, &color_at, &color_blocks);
    init_color_stages(stages, &mut s, StageType::Solve, ITERATIONS, plan, &color_at, &color_blocks);
    init_stage(stages, &mut s, StageType::IntegratePositions, body_at, d.body.count, NULL_COLOR);
    init_color_stages(stages, &mut s, StageType::Relax, RELAX_ITERATIONS, plan, &color_at, &color_blocks);
    init_color_stages(stages, &mut s, StageType::Restitution, 1, plan, &color_at, &color_blocks);
    init_stage(stages, &mut s, StageType::StoreWideImpulses, wide_at, d.wide.count, NULL_COLOR);
    init_stage(stages, &mut s, StageType::StoreImpulses, mesh_at, d.mesh.count, NULL_COLOR);
    init_stage(stages, &mut s, StageType::Finalize, body_at, d.body.count, NULL_COLOR);

    debug_assert_eq!(s, need.stages);

    Context {
        stages: &stages[..need.stages],
        blocks: &blocks[..need.blocks],
        colors: plan.colors,
        sub_step_count: plan.sub_step_count,
        worker_count: plan.worker_count,
        sync_bits: Line(AtomicU32::new(0)),
        exited: AtomicI32::new(0),
        fault: AtomicU32::new(0),
    }
}

// --- execution --------------------------------------------------------------------------------

/// b3ExecuteBlock: stage type + block type → phase.
fn execute_block<W: StageWork>(work: &W, ty: StageType, block: Block, worker_index: usize) {
    match ty {
        StageType::PrepareJoints => work.prepare_joints(block),
        StageType::PrepareWideContacts => work.prepare_wide(block),
        StageType::PrepareContacts => work.prepare_mesh(block),
        StageType::IntegrateVelocities => work.integrate_velocities(block),
        StageType::WarmStart => match block.block_type {
            BlockType::GraphJoint => work.warm_start_joints(block),
            BlockType::GraphWideContact => work.warm_start_wide(block, worker_index),
            _ => work.warm_start_mesh(block),
        },
        StageType::Solve => match block.block_type {
            BlockType::GraphJoint => work.solve_joints(block, true, worker_index),
            BlockType::GraphWideContact => work.solve_wide(block, true, worker_index),
            _ => work.solve_mesh(block, true),
        },
        StageType::IntegratePositions => work.integrate_positions(block),
        StageType::Relax => match block.block_type {
            BlockType::GraphJoint => work.solve_joints(block, false, worker_index),
            BlockType::GraphWideContact => work.solve_wide(block, false, worker_index),
            _ => work.solve_mesh(block, false),
        },
        StageType::Restitution => match block.block_type {
            BlockType::GraphWideContact => work.restitution_wide(block, worker_index),
            BlockType::GraphContact => work.restitution_mesh(block),
            // Joints have no restitution pass.
            _ => {}
        },
        StageType::StoreWideImpulses => work.store_wide(block, worker_index),
        StageType::StoreImpulses => work.store_mesh(block, worker_index),
        StageType::Finalize => work.finalize(block),
    }
}

/// GetWorkerStartIndex: stagger the workers' home blocks so they don't all start on block 0. `None`
/// when there are more workers than blocks and this worker has nothing to start on.
fn worker_start_index(worker_index: usize, block_count: usize, worker_count: usize) -> Option<usize> {
    if block_count <= worker_count {
        return (worker_index < block_count).then_some(worker_index);
    }

    let blocks_per_worker = block_count / worker_count;
    let remainder = block_count - blocks_per_worker * worker_count;
    Some(blocks_per_worker * worker_index + remainder.min(worker_index))
}

/// b3ExecuteStage: sweep the ring from this worker's home block, CAS-claiming whatever is still
/// unclaimed at `previous_sync`, and report how many blocks this worker ran.
fn execute_stage<W: StageWork>(
    ctx: &Context,
    work: &W,
    stage: &Stage,
    previous_sync: i32,
    sync: i32,
    worker_index: usize,
) {
    let block_count = stage.block_count;
    let Some(start) = worker_start_index(worker_index, block_count, ctx.worker_count) else {
        return;
    };

    let mut completed = 0;
    let mut i = start;
    for _ in 0..block_count {
        let sb = &ctx.blocks[stage.block_start + i];
        if sb
            .sync_index
            .compare_exchange(previous_sync, sync, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            // Copy the descriptor out: the CAS winner owns this block, and the descriptor is
            // immutable for the whole solve, so nothing here aliases the atomic others are writing.
            execute_block(work, stage.ty, sb.block, worker_index);
            completed += 1;
        }

        i += 1;
        if i >= block_count {
            i = 0;
        }
    }

    stage.completion.fetch_add(completed, Ordering::SeqCst);
}

/// b3ExecuteMainStage: run one stage on the orchestrator and barrier on the workers finishing it.
/// A single-block stage skips publication entirely — the orchestrator just runs it (that block's
/// sync index therefore stays 0 forever, which is consistent because a stage's block count never
/// changes across its re-uses).
///
/// `None` when a worker faulted: the barrier's blocks will never all complete, so the solve unwinds.
fn execute_main_stage<W: StageWork>(
    ctx: &Context,
    work: &W,
    stage_index: usize,
    sync_bits: u32,
) -> Option<()> {
    let stage = &ctx.stages[stage_index];
    let block_count = stage.block_count;
    if block_count == 0 {
        return Some(());
    }

    if block_count == 1 {
        execute_block(work, stage.ty, ctx.blocks[stage.block_start].block, ORCHESTRATOR);
        return Some(());
    }

    ctx.sync_bits.0.store(sync_bits, Ordering::SeqCst);

    let sync = ((sync_bits >> 16) & 0xFFFF) as i32;
    debug_assert!(sync > 0);
    execute_stage(ctx, work, stage, sync - 1, sync, ORCHESTRATOR);

    // Spin for the thieves. This is the stage barrier: past it, every block of this stage has run,
    // so the orchestrator may safely run the serial overflow work and publish the next stage.
    while stage.completion.load(Ordering::SeqCst) != block_count as i32 {
        if ctx.faulted() {
            return None;
        }
        core::hint::spin_loop();
    }

    stage.completion.store(0, Ordering::SeqCst);
    Some(())
}

#[inline]
fn sync_bits(sync_index: u32, stage_index: usize) -> u32 {
    (sync_index << 16) | (stage_index as u32)
}

/// The finish sentinel the orchestrator publishes when the stage list is done.
const FINISH: u32 = u32::MAX;

/// The orchestrator's worker index: the thread that drives the step.
const ORCHESTRATOR: usize = 0;

/// Run the solve as `worker_index`. Worker 0 is the orchestrator (the thread driving the step); it
/// walks the stage list, runs the serial overflow color between stages, and publishes the sync bits.
/// Every other worker spins on those bits and steals blocks until the finish sentinel.
///
/// **Contract: every worker `1..worker_count` must call this exactly once per solve, or call [`fault`]
/// if it cannot.** The orchestrator's `run` does not return until all of them have — without that join
/// the caller could rebuild the context (zeroing every block's sync index) while a worker is still
/// inside a stale `execute_stage`, whose CAS would then succeed against a freshly-reset block and run
/// it a second time against the *next* step's columns. box3d gets this join from its task system
/// (`finishTaskFcn`); we have no task system, so the sentinel is acknowledged here instead.
///
/// A `worker_index` at or past `worker_count` is a caller bug — its home block can land in another
/// stage's range, which would silently run a block under the wrong stage type — so it faults instead.
pub fn run<W: StageWork>(ctx: &Context, work: &W, worker_index: usize) {
    if worker_index >= ctx.worker_count {
        ctx.fault();
        return;
    }
    if worker_index == ORCHESTRATOR {
        orchestrate(ctx, work);
    } else {
        steal(ctx, work, worker_index);
    }
}

fn orchestrate<W: StageWork>(ctx: &Context, work: &W) {
    // `None` = a worker died mid-solve. The stage list is abandoned, but the finish sentinel is still
    // published and the join still runs: the *surviving* workers must be let out of their spins before
    // the caller returns, or the next step's build would race them.
    let _ = walk(ctx, work);

    ctx.sync_bits.0.store(FINISH, Ordering::SeqCst);

    // The join. Past this line no worker is inside a stage, so the caller may rebuild the context. A
    // faulted worker never acks — the flag is the only way out, and the caller (`src/pool.ts`) raises.
    let workers = ctx.worker_count as i32 - 1;
    while ctx.exited.load(Ordering::SeqCst) != workers {
        if ctx.faulted() {
            return;
        }
        core::hint::spin_loop();
    }
}

/// The stage list, in order. `None` if a worker faulted partway.
fn walk<W: StageWork>(ctx: &Context, work: &W) -> Option<()> {
    let colors = ctx.colors.len();
    let mut stage_index = 0;

    // Each group of stages that shares a block array carries its own sync index, so re-uses of that
    // array CAS against a monotone sequence: bodies (integrate velocities/positions, finalize),
    // joints, the flat convex range (prepare/store), the flat mesh range (prepare/store), and the
    // graph colors.
    let mut body_sync = 1;
    // The joint block array is swept once (PrepareJoints); the colored joint work rides the graph
    // blocks. So this one never advances.
    let joint_sync = 1;
    let mut convex_sync = 1;
    let mut mesh_sync = 1;
    let mut graph_sync = 1;

    execute_main_stage(ctx, work, stage_index, sync_bits(joint_sync, stage_index))?;
    stage_index += 1;

    execute_main_stage(ctx, work, stage_index, sync_bits(convex_sync, stage_index))?;
    stage_index += 1;
    convex_sync += 1;

    execute_main_stage(ctx, work, stage_index, sync_bits(mesh_sync, stage_index))?;
    stage_index += 1;
    mesh_sync += 1;

    // The overflow color doesn't fit the graph coloring: its constraints share bodies, so it is
    // solved serially here, on the orchestrator, with every worker parked in its spin. Joints before
    // contacts, matching box3d's `b3PrepareJoints_Overflow` / `b3PrepareContacts_Overflow` order.
    work.prepare_overflow_joints();
    work.prepare_overflow();

    for _ in 0..ctx.sub_step_count {
        // The stage index restarts each sub-step (the stages are re-used); the sync bits still grow
        // monotonically because the sync indices in the upper half do.
        let mut i = stage_index;

        execute_main_stage(ctx, work, i, sync_bits(body_sync, i))?;
        i += 1;
        body_sync += 1;

        work.warm_start_overflow_joints();
        work.warm_start_overflow();
        for _ in 0..colors {
            execute_main_stage(ctx, work, i, sync_bits(graph_sync, i))?;
            i += 1;
        }
        graph_sync += 1;

        for _ in 0..ITERATIONS {
            // Overflow constraints have lower priority than the colored ones: solved first.
            work.solve_overflow_joints(true);
            work.solve_overflow(true);
            for _ in 0..colors {
                execute_main_stage(ctx, work, i, sync_bits(graph_sync, i))?;
                i += 1;
            }
            graph_sync += 1;
        }

        execute_main_stage(ctx, work, i, sync_bits(body_sync, i))?;
        i += 1;
        body_sync += 1;

        for _ in 0..RELAX_ITERATIONS {
            work.solve_overflow_joints(false);
            work.solve_overflow(false);
            for _ in 0..colors {
                execute_main_stage(ctx, work, i, sync_bits(graph_sync, i))?;
                i += 1;
            }
            graph_sync += 1;
        }
    }

    // integrate velocities + warm start + solve + integrate positions + relax
    stage_index += 2 + colors * (1 + ITERATIONS + RELAX_ITERATIONS);

    work.restitution_overflow();
    for _ in 0..colors {
        execute_main_stage(ctx, work, stage_index, sync_bits(graph_sync, stage_index))?;
        stage_index += 1;
    }

    work.store_overflow();

    execute_main_stage(ctx, work, stage_index, sync_bits(convex_sync, stage_index))?;
    stage_index += 1;

    execute_main_stage(ctx, work, stage_index, sync_bits(mesh_sync, stage_index))?;
    stage_index += 1;

    // The fused pose finalize (header deviations). The mesh-store barrier above completed before this
    // publishes, so no worker can still be inside a solve stage; the body blocks were last claimed at
    // `body_sync - 1` by the final integrate-positions stage, so the CAS sequence stays monotone.
    execute_main_stage(ctx, work, stage_index, sync_bits(body_sync, stage_index))?;
    stage_index += 1;

    debug_assert_eq!(stage_index, ctx.stages.len());
    Some(())
}

fn steal<W: StageWork>(ctx: &Context, work: &W, worker_index: usize) {
    let mut last = 0;
    loop {
        let mut bits = ctx.sync_bits.0.load(Ordering::SeqCst);
        while bits == last {
            // A sibling died: the orchestrator has abandoned the solve and will publish no further
            // stage. Leave without acking — its join breaks on the same flag.
            if ctx.faulted() {
                return;
            }
            core::hint::spin_loop();
            bits = ctx.sync_bits.0.load(Ordering::SeqCst);
        }

        if bits == FINISH {
            ctx.exited.fetch_add(1, Ordering::SeqCst);
            return;
        }

        let stage_index = (bits & 0xFFFF) as usize;
        let sync = ((bits >> 16) & 0xFFFF) as i32;

        // A worker that arrives late may be publishing-stale: it runs the stage it just read while
        // the orchestrator has already moved on. That is safe, not a race — the monotone sync index
        // means every CAS against an already-advanced block fails, so the worker does nothing.
        execute_stage(ctx, work, &ctx.stages[stage_index], sync - 1, sync, worker_index);

        last = bits;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// b3ComputeBlockCount at its boundaries: the minimum-size branch, the exact crossover, and the
    /// grown-size branch past it. A block count over the cap would overflow the caller's buffer,
    /// which `sizes` promises it cannot.
    #[test]
    fn block_count_boundaries() {
        assert_eq!(compute_block_count(0, 4, 16), BlockDim { size: 0, count: 0 });
        assert_eq!(compute_block_count(1, 4, 16), BlockDim { size: 4, count: 1 });
        // Exactly min_size * max_block_count: still minimum size, exactly the cap of blocks.
        assert_eq!(compute_block_count(64, 4, 16), BlockDim { size: 4, count: 16 });
        // One past: the size grows so the count stays at the cap.
        assert_eq!(compute_block_count(65, 4, 16), BlockDim { size: 5, count: 13 });
        for n in 1..2000usize {
            let d = compute_block_count(n, 4, 16);
            assert!(d.count <= 16, "n={n} blew the block cap: {d:?}");
            assert!(d.size * d.count >= n);
            assert!((d.count - 1) * d.size < n);
        }
    }

    /// b3InitBlocks must tile `[base, base + item_count)` exactly: no gap, no overlap, no dropped
    /// tail. Every element belongs to exactly one block, or the solve silently skips constraints.
    #[test]
    fn blocks_tile_the_range() {
        for item_count in [1usize, 3, 4, 5, 63, 64, 65, 200] {
            let dim = compute_block_count(item_count, 4, 16);
            let mut blocks: Vec<SyncBlock> = (0..dim.count).map(|_| SyncBlock::EMPTY).collect();
            init_blocks(&mut blocks, dim, 100, item_count, BlockType::Contact, 7);

            let mut covered = vec![0u32; item_count];
            for b in &blocks {
                assert_eq!(b.block.color, 7);
                for i in b.block.start..b.block.start + b.block.count {
                    covered[i - 100] += 1;
                }
            }
            assert!(
                covered.iter().all(|&c| c == 1),
                "item_count={item_count} not tiled exactly: {covered:?}"
            );
        }
    }

    /// The workers' home blocks must partition the ring: every block is someone's start when there
    /// are at least as many blocks as workers, and no two workers share a home.
    #[test]
    fn worker_start_indices_are_distinct() {
        for worker_count in 1..9usize {
            for block_count in 1..40usize {
                let starts: Vec<Option<usize>> = (0..worker_count)
                    .map(|w| worker_start_index(w, block_count, worker_count))
                    .collect();
                let mut seen: Vec<usize> = starts.iter().flatten().copied().collect();
                let n = seen.len();
                seen.sort_unstable();
                seen.dedup();
                assert_eq!(seen.len(), n, "workers collided on a home block");
                assert!(seen.iter().all(|&s| s < block_count));
            }
        }
    }
}

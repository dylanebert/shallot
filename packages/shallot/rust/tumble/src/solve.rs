//! The staged solve, wired to the wasm arena — the kernel's multithreaded entry.
//!
//! `stages.rs` owns the staged solver's machinery (stage list, CAS-claimed blocks, barriers, the serial
//! overflow); `parfor.rs` owns the flat block-claim sweep the outer phases use; `arena.rs` owns the
//! columns. This module is the seam between them: it builds the [`Plan`] from the layout TS already
//! computed, holds the [`Context`] and the [`Work`] in linear memory where every thread's instance sees
//! them, and exposes the entries the pool drives —
//!
//!   - a **build** on the main thread (`solveBuild` for the staged solve, `parBuild` for one outer
//!     phase), which names the job every thread is about to run, then
//!   - `runMt` on the main thread (the orchestrator) and `workerMain` in each pooled worker, both of
//!     which dispatch on that job.
//!
//! One job is live at a time, by construction: the pool is woken once per build and every worker is
//! parked again before the next one.
//!
//! **The join contract** (`stages::run`): every worker calls `run` exactly once per solve, and the
//! orchestrator's `run` blocks until all of them have left. `src/pool.ts`'s round is what guarantees the
//! first half; the second is inside `stages::run`.
//!
//! **Build, then wake.** `solveBuild` runs on the main thread *before* the pool's wake, so no worker can
//! observe a half-built context: the wake (a seq-cst `Atomics.store`) is the release edge for everything
//! written here, and the worker's `Atomics.wait`/`load` is the acquire. That ordering also means the
//! buffers below are only ever written while every worker is parked.
//!
//! **No `memory.grow` between the fork and the join** (the MT concurrency invariant, `.claude/rules/tumble.md`): the columns are derived
//! once, here, and shared by value with every worker; a region grow would relocate them under the
//! workers' feet. Every `reserve*` runs pre-solve on the main thread, which is why deriving once is
//! sound.
//!
//! Wasm-only, like the arena it reads. Native `cargo test` drives the same machinery over owned columns
//! (`kernel/tests/stages.rs`).

use crate::arena;
use crate::bodies::IDENT_RECORDS;
use crate::col::Col;
use crate::contact::{self, Columns, Softness};
use crate::contact_wide;
use crate::integrate;
use crate::math::Vec3;
use crate::parfor::{worth_forking, ParFor, COLLIDE_FORK_MIN, COLLIDE_MIN_RANGE};
use crate::stages::{
    self, max_sizes, Block, BlockType, ColorSpan, Context, Plan, Stage, StageWork, SyncBlock,
    MAX_COLORS,
};

/// Threads the solve can ever run on: the shadow stack affords main + 7 workers (`src/pool.ts`
/// `maxWorkers`), and each needs its own null-lane identity record.
const MAX_THREADS: usize = IDENT_RECORDS;

const MAX: stages::Sizes = max_sizes(MAX_THREADS);

// The stage list, its blocks, and the active colors — the caller-owned storage `stages::build` lays the
// plan into. Statics, not `Vec`s: the wasm path has no allocator (kex `tumble.md`), and these live in
// the shared linear memory, so every thread's instance addresses the same bytes. Written only by
// `solveBuild`, on the main thread, with the workers parked.
static mut STAGES: [Stage; MAX.stages] = [Stage::EMPTY; MAX.stages];
static mut BLOCKS: [SyncBlock; MAX.blocks] = [SyncBlock::EMPTY; MAX.blocks];
static mut SPANS: [ColorSpan; MAX_COLORS] = [ColorSpan::EMPTY; MAX_COLORS];

/// The built solve. `None` until the first `solveBuild`.
static mut CTX: Option<Context<'static>> = None;
/// The columns + per-step scalars the blocks run over.
static mut WORK: Option<Work> = None;

/// The arena's columns and this step's scalars, as every block sees them. The columns are [`Col`]s —
/// shared-mutable handles, because a stage's blocks run concurrently over one column and only their
/// *writes* are disjoint (col.rs).
struct Work {
    cols: Columns<'static>,
    wide: Col<'static, f32>,
    wide_idx: Col<'static, u32>,
    wide_meta: Col<'static, u32>,

    /// The serial spill: contact records the graph coloring could not separate. Never becomes blocks —
    /// the orchestrator runs it alone, in creation order, between stages.
    overflow_start: usize,
    overflow_count: usize,

    /// The flat joint column (`joint_abi::JOINT_STRIDE` per slot). Colored joints occupy the front
    /// (per-color concatenated, swept by `PrepareJoints` + the color blocks); the overflow joints
    /// follow (`overflow_joint_start..+overflow_joint_count`, run serially).
    joints: Col<'static, f32>,
    overflow_joint_start: usize,
    overflow_joint_count: usize,
    /// box3d's `context->enableWarmStarting` — `prepare` zeroes the impulses when false.
    enable_warm_starting: bool,

    contact_softness: Softness,
    static_softness: Softness,
    warm_start_scale: f32,
    gravity: Vec3,
    h: f32,
    inv_h: f32,
    /// The full-step dt for the fused finalize (`h` above is the sub-step).
    dt: f32,
    inv_dt: f32,
    contact_speed: f32,
    max_linear_velocity: f32,
    restitution_threshold: f32,
    hit_threshold: f32,
    /// The world's continuous toggle, for the fused finalize's fast-candidate predicate.
    enable_continuous: bool,
}

impl Work {
    fn overflow(&self) -> Block {
        Block {
            start: self.overflow_start,
            count: self.overflow_count,
            block_type: BlockType::Contact,
            color: u8::MAX,
        }
    }
}

impl StageWork for Work {
    fn prepare_wide(&self, b: Block) {
        contact_wide::prepare(
            self.wide,
            self.wide_idx,
            self.wide_meta,
            self.cols.state,
            self.cols.sim,
            self.cols.dir,
            self.cols.pool,
            b.start,
            b.count,
            self.contact_softness,
            self.static_softness,
            self.warm_start_scale,
        );
    }

    fn prepare_mesh(&self, b: Block) {
        contact::prepare(
            &self.cols,
            b.start,
            b.count,
            self.contact_softness,
            self.static_softness,
            self.warm_start_scale,
        );
    }

    fn integrate_velocities(&self, b: Block) {
        integrate::integrate_velocities(
            self.cols.state,
            self.cols.sim,
            b.start,
            b.count,
            self.gravity,
            self.h,
        );
    }

    fn integrate_positions(&self, b: Block) {
        integrate::integrate_positions(
            self.cols.state,
            self.cols.flags,
            b.start,
            b.count,
            self.h,
            self.max_linear_velocity,
            self.inv_dt,
        );
    }

    fn warm_start_wide(&self, b: Block, worker: usize) {
        contact_wide::warm_start(
            self.wide,
            self.wide_idx,
            self.cols.state,
            self.cols.flags,
            b.start,
            b.count,
            worker,
        );
    }

    fn warm_start_mesh(&self, b: Block) {
        contact::warm_start(&self.cols, b.start, b.count);
    }

    fn solve_wide(&self, b: Block, use_bias: bool, worker: usize) {
        contact_wide::solve(
            self.wide,
            self.wide_idx,
            self.cols.state,
            self.cols.flags,
            b.start,
            b.count,
            use_bias,
            self.inv_h,
            self.contact_speed,
            worker,
        );
    }

    fn solve_mesh(&self, b: Block, use_bias: bool) {
        contact::solve(
            &self.cols,
            b.start,
            b.count,
            use_bias,
            self.inv_h,
            self.contact_speed,
        );
    }

    fn restitution_wide(&self, b: Block, worker: usize) {
        contact_wide::restitution(
            self.wide,
            self.wide_idx,
            self.cols.state,
            self.cols.flags,
            b.start,
            b.count,
            self.restitution_threshold,
            worker,
        );
    }

    fn restitution_mesh(&self, b: Block) {
        contact::restitution(&self.cols, b.start, b.count, self.restitution_threshold);
    }

    fn store_wide(&self, b: Block, _worker: usize) {
        contact_wide::store(
            self.wide,
            self.wide_meta,
            self.cols.dir,
            self.cols.pool,
            b.start,
            b.count,
            self.hit_threshold,
        );
    }

    fn store_mesh(&self, b: Block, _worker: usize) {
        contact::store(&self.cols, b.start, b.count, self.hit_threshold);
    }

    fn finalize(&self, b: Block) {
        // SAFETY: the body + shape + fat-AABB regions were reserved pre-solve on the main thread, and
        // no thread grows memory between the pool's wake and its join (module header) — the same
        // contract this sweep ran under as its own parallel-for round before it fused here.
        unsafe {
            arena::finalize_block(
                b.start,
                b.start + b.count,
                self.dt,
                self.inv_dt,
                self.enable_continuous,
            );
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

    // --- joints -------------------------------------------------------------------------------

    fn prepare_joints(&self, b: Block) {
        for slot in b.start..b.start + b.count {
            crate::joint::prepare(self.joints, slot, self.h, self.inv_h, self.enable_warm_starting);
        }
    }

    fn warm_start_joints(&self, b: Block) {
        for slot in b.start..b.start + b.count {
            crate::joint::warm_start(self.joints, slot, self.cols.state, self.cols.flags);
        }
    }

    fn solve_joints(&self, b: Block, use_bias: bool, _worker: usize) {
        for slot in b.start..b.start + b.count {
            crate::joint::solve(
                self.joints,
                slot,
                self.cols.state,
                self.cols.flags,
                use_bias,
                self.h,
                self.inv_h,
            );
        }
    }

    fn prepare_overflow_joints(&self) {
        for slot in self.overflow_joint_start..self.overflow_joint_start + self.overflow_joint_count
        {
            crate::joint::prepare(self.joints, slot, self.h, self.inv_h, self.enable_warm_starting);
        }
    }

    fn warm_start_overflow_joints(&self) {
        for slot in self.overflow_joint_start..self.overflow_joint_start + self.overflow_joint_count
        {
            crate::joint::warm_start(self.joints, slot, self.cols.state, self.cols.flags);
        }
    }

    fn solve_overflow_joints(&self, use_bias: bool) {
        for slot in self.overflow_joint_start..self.overflow_joint_start + self.overflow_joint_count
        {
            crate::joint::solve(
                self.joints,
                slot,
                self.cols.state,
                self.cols.flags,
                use_bias,
                self.h,
                self.inv_h,
            );
        }
    }
}

/// Lay out this step's stage list and blocks, and capture the columns + scalars the blocks run over.
///
/// Called on the main thread, once per solve, **before** the pool is woken — see the module header. The
/// colors come from the `colorSpan` column TS already wrote (`writeColorSpans`), including each color's
/// joint span — jointed scenes route through this pool whenever one exists (the joints-in-kernel path).
#[allow(clippy::too_many_arguments)]
#[export_name = "solveBuild"]
pub extern "C" fn solve_build(
    thread_count: usize,
    sub_step_count: usize,
    wide_total: usize,
    mesh_start: usize,
    mesh_total: usize,
    overflow_start: usize,
    overflow_count: usize,
    joint_total: usize,
    overflow_joint_start: usize,
    overflow_joint_count: usize,
    gx: f32,
    gy: f32,
    gz: f32,
    h: f32,
    inv_h: f32,
    dt: f32,
    inv_dt: f32,
    max_linear_velocity: f32,
    contact_speed: f32,
    cs_bias: f32,
    cs_mass: f32,
    cs_impulse: f32,
    ss_bias: f32,
    ss_mass: f32,
    ss_impulse: f32,
    warm_start_scale: f32,
    restitution_threshold: f32,
    hit_event_threshold: f32,
    enable_continuous: u32,
) {
    assert!((1..=MAX_THREADS).contains(&thread_count));
    unsafe {
        // Drop the previous solve's context before re-borrowing its buffers. The workers have all left
        // it (the join in `stages::run`), so nothing else holds them.
        CTX = None;

        let (spans, color_count) = arena::color_span_column();
        let out = &mut *(&raw mut SPANS);
        for c in 0..color_count {
            let o = c * arena::COLOR_SPAN_STRIDE;
            out[c] = ColorSpan {
                color: c as u8,
                wide_start: spans.get(o) as usize,
                wide_count: spans.get(o + 1) as usize,
                mesh_start: spans.get(o + 2) as usize,
                mesh_count: spans.get(o + 3) as usize,
                joint_start: spans.get(o + 4) as usize,
                joint_count: spans.get(o + 5) as usize,
            };
        }

        let (wide, wide_idx, wide_meta) = arena::wide_columns();
        WORK = Some(Work {
            cols: arena::scalar_columns(),
            wide,
            wide_idx,
            wide_meta,
            overflow_start,
            overflow_count,
            joints: arena::joint_column(),
            overflow_joint_start,
            overflow_joint_count,
            enable_warm_starting: warm_start_scale != 0.0,
            contact_softness: Softness {
                bias_rate: cs_bias,
                mass_scale: cs_mass,
                impulse_scale: cs_impulse,
            },
            static_softness: Softness {
                bias_rate: ss_bias,
                mass_scale: ss_mass,
                impulse_scale: ss_impulse,
            },
            warm_start_scale,
            gravity: Vec3::new(gx, gy, gz),
            h,
            inv_h,
            dt,
            inv_dt,
            contact_speed,
            max_linear_velocity,
            restitution_threshold,
            hit_threshold: hit_event_threshold,
            enable_continuous: enable_continuous != 0,
        });

        let plan = Plan {
            body_count: arena::body_count(),
            wide_total,
            mesh_start,
            mesh_total,
            joint_total,
            colors: &out[..color_count],
            sub_step_count,
            worker_count: thread_count,
        };
        CTX = Some(stages::build(
            &plan,
            &mut *(&raw mut STAGES),
            &mut *(&raw mut BLOCKS),
        ));
        JOB = Job::Solve;
    }
}

// --- the outer phases -------------------------------------------------------------------------
//
// Convex narrowphase dispatch and contact recycle: flat sweeps of independent records, so they run on
// `parfor.rs`'s block-claim sweep rather than the stage list. They fork *inside* collide (recycle, then
// dispatch) — each with its own build + `pool.run` pair, each after its own `reserve*`, which is what
// keeps "no grow between the fork and the join" true. The pose finalize used to be the third; it now
// rides the staged solve as its terminal stage (`stages.rs` header deviations), so the step's
// steady-state rounds are recycle + one fused solve.

/// Which job the pool's current round runs. Written by a build on the main thread with every worker
/// parked; the wake that follows is the release edge that publishes it (module header).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Job {
    None,
    Solve,
    Recycle,
    Convex,
}

/// `parBuild`'s `kind` argument, mirrored in `src/kernel.ts`.
const KIND_RECYCLE: u32 = 1;
const KIND_CONVEX: u32 = 2;

static mut JOB: Job = Job::None;
static mut PAR: Option<Par> = None;

/// One built parallel-for: the partition plus the phase's scalars (`a`/`b` are the two recycle
/// tolerances; the convex dispatch has none).
struct Par {
    par: ParFor,
    count: usize,
    a: f32,
    b: f32,
}

/// Partition one outer phase's `count` records over `thread_count` threads, and name it as the job the
/// next `pool.run` will drive. Main thread, with the workers parked and the phase's `reserve*` already
/// done — the columns the blocks read are fixed from here to the join.
///
/// **Returns 1 if the caller should fork, 0 if it should run the serial shim instead** — a sweep of one
/// block has nothing to steal, and one under the fork floor loses to its own wake (`parfor.rs`). The
/// policy lives here, not in the caller, so the cost model sits next to the machinery it prices.
#[export_name = "parBuild"]
pub extern "C" fn par_build(kind: u32, count: usize, thread_count: usize, a: f32, b: f32) -> usize {
    assert!((1..=MAX_THREADS).contains(&thread_count));
    let job = match kind {
        KIND_RECYCLE => Job::Recycle,
        KIND_CONVEX => Job::Convex,
        _ => panic!("unknown parallel-for kind"),
    };
    let par = ParFor::new(count, COLLIDE_MIN_RANGE, thread_count);
    let fork = par.block_count() >= 2 && worth_forking(count, thread_count - 1, COLLIDE_FORK_MIN);
    unsafe {
        PAR = Some(Par { par, count, a, b });
        JOB = job;
    }
    fork as usize
}

/// Run the built job as `index` — 0 on the orchestrator (the thread driving the step), 1.. in each pooled
/// worker. Every thread of the pool enters exactly once per round.
///
/// The staged solve needs its index (worker 0 orchestrates, and the wide phases key their null-lane
/// identity record off it). A parallel-for does not: every thread races the same counter, which is what
/// makes its partition worker-count-independent (`parfor.rs`).
///
/// No build means the pool was woken without one — a caller bug, but a benign one: every thread reads the
/// same `Job::None` and returns without entering a spin.
fn run_job(index: usize) {
    unsafe {
        match *(&raw const JOB) {
            Job::None => {}
            Job::Solve => {
                let (Some(ctx), Some(work)) = (&*(&raw const CTX), &*(&raw const WORK)) else {
                    return;
                };
                stages::run(ctx, work, index);
            }
            job => {
                let Some(p) = &*(&raw const PAR) else {
                    return;
                };
                match job {
                    Job::Recycle => p
                        .par
                        .run(|s, e| arena::recycle_block(s, e, p.count, p.a, p.b)),
                    Job::Convex => p.par.run(|s, e| arena::convex_block(s, e, p.count)),
                    Job::Solve | Job::None => unreachable!(),
                }
            }
        }
    }
}

/// The orchestrator's entry: run the built job on the thread driving the step. For the staged solve it
/// returns once the stage list is done *and* every worker has left [`stages::run`]; for a parallel-for it
/// returns once the blocks are exhausted, and the pool's JS ack is the join. The build must have run
/// first, and the pool must already be awake.
#[export_name = "runMt"]
pub extern "C" fn run_mt() {
    run_job(0);
}

/// A pooled worker's entry: run the built job as worker `index` (1-based — 0 is the orchestrator).
/// Exactly once per round — a skipped call hangs the staged solve's join, and a doubled one corrupts the
/// next step (`stages::run`'s contract).
#[export_name = "workerMain"]
pub extern "C" fn worker_main(index: usize) {
    run_job(index);
}

/// A worker died inside [`worker_main`] — a wasm trap, which unwinds into its JS round body. Called from
/// that catch, before it acks.
///
/// Only the staged solve needs it, and only that job's context may be poisoned. The solve's orchestrator
/// spins *inside* wasm (a stage barrier, the exit join) for a block the dead worker will never complete,
/// and no JS event can reach a thread that never yields — so the flag on its [`Context`] is the only way
/// out ([`stages::Context::fault`]). A parallel-for round has no wasm-side spin at all: its orchestrator
/// drains the remaining blocks and returns, and the pool's JS ack is the whole join. Poisoning `CTX` from
/// one would hit the *previous* step's solve context, which is dead and about to be rebuilt — harmless,
/// but it would read as if it did something.
#[export_name = "workerFault"]
pub extern "C" fn worker_fault() {
    unsafe {
        if *(&raw const JOB) != Job::Solve {
            return;
        }
        if let Some(ctx) = &*(&raw const CTX) {
            ctx.fault();
        }
    }
}

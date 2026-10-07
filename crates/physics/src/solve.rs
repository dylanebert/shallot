//! The staged solve, wired to the wasm arena — the kernel's multithreaded entry.
//!
//! `stages.rs` owns the staged solver's machinery (stage list, CAS-claimed blocks, barriers, the serial
//! overflow); `parfor.rs` owns the flat block-claim sweep the outer phases use; `arena.rs` owns the
//! columns. This module is the seam between them: it builds the [`Plan`] from the kernel graph layout,
//! holds the [`Context`] and the [`Work`] in linear memory where every thread's instance sees
//! them, and exposes the entries the pool drives —
//!
//!   - a **build** on the main thread (`solveBuild` for the staged solve, `parBuild` for one outer
//!     phase), which names the job every thread is about to run, then
//!   - `runMt` on the main thread (the orchestrator) and `workerMain` in each pooled worker, both of
//!     which dispatch on that job.
//!
//! One pool round is live at a time: the pool is woken once per build and every worker is parked
//! before the next one. A solve round also runs its queued island split on worker 1 (or worker 0
//! without a pool), before that worker enters the solver; the shared join waits for both tasks.
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
//! **No relocation of solve columns between fork and join**: column reserves and split-scratch
//! reserves run before the fork. The split may grow island records and lists, but not body, contact
//! or joint solver arrays; the solver's shared column handles remain valid until the join.
//!
//! Wasm-only, like the arena it reads. Native `cargo test` drives the same machinery over owned columns
//! (`kernel/tests/stages.rs`).

use crate::arena;
use crate::col::Col;
use crate::contact::{self, Columns, Softness};
use crate::contact_wide;
use crate::integrate;
use crate::math::Vec3;
use crate::parfor::{worth_forking, ParFor, COLLIDE_FORK_MIN, COLLIDE_MIN_RANGE};
use crate::stages::{
    self, Block, ColorSpan, Context, Plan, Stage, StageWork, SyncBlock, MAX_COLORS,
};

/// Threads the solve can ever run on: the shadow stack affords main + 7 workers (`src/pool.ts`
/// `maxWorkers`), and each needs its own null-lane identity record.
pub(crate) const MAX_THREADS: usize = 8;

struct Step {
    ctx: Option<Context<'static>>,
    work: Option<Work>,
    spans: [ColorSpan; MAX_COLORS],
}

// The pool runs one round at a time; this address publishes the world's step allocation.
static mut STEP: *mut Step = core::ptr::null_mut();

/// The arena's columns and this step's scalars, as every block sees them. The columns are [`Col`]s —
/// shared-mutable handles, because a stage's blocks run concurrently over one column and only their
/// *writes* are disjoint (col.rs).
struct Work {
    world: usize,
    cols: Columns<'static>,
    overflow_cols: Columns<'static>,
    wide: Col<'static, f32>,
    wide_idx: Col<'static, u32>,
    wide_spans: Col<'static, crate::contact_spans::WidePrepareSpan>,

    /// The serial spill: contact records the graph coloring could not separate. Never becomes blocks —
    /// the orchestrator runs it alone, in creation order, between stages.
    overflow_count: usize,

    /// Active graph-color arrays and their joint-prepare spans (b3JointPrepareSpan).
    joints: [Col<'static, f32>; MAX_COLORS],
    joint_bases: [usize; MAX_COLORS],
    color_count: usize,
    overflow_joints: Col<'static, f32>,
    fin: Col<'static, f32>,
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

impl StageWork for Work {
    fn ticks(&self) -> f64 {
        crate::physics_world::ticks()
    }
    fn profile(&self, field: usize, start: f64) {
        unsafe {
            crate::physics_world::accumulate(self.world, field, start);
        }
    }
    fn prepare_wide(&self, b: Block) {
        contact_wide::prepare(
            self.wide,
            self.wide_idx,
            self.wide_spans,
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

    fn store_wide(&self, b: Block, worker: usize) {
        let mut has_hits = false;
        contact_wide::store(
            self.wide,
            self.wide_spans,
            self.cols.dir,
            self.cols.pool,
            b.start,
            b.count,
            self.hit_threshold,
            |contact| unsafe {
                arena::mark_hit_event(self.world, worker, contact);
                has_hits = true;
            },
        );
        unsafe { arena::finish_hit_events(self.world, worker, has_hits) };
    }

    fn store_mesh(&self, b: Block, worker: usize) {
        let mut has_hits = false;
        contact::store(
            &self.cols,
            b.start,
            b.count,
            self.hit_threshold,
            |contact| unsafe {
                arena::mark_hit_event(self.world, worker, contact);
                has_hits = true;
            },
        );
        unsafe { arena::finish_hit_events(self.world, worker, has_hits) };
    }

    fn finalize(&self, b: Block) {
        // SAFETY: the body + shape + fat-AABB regions were reserved pre-solve on the main thread, and
        // the solver and split have joined, and these columns stay fixed for this parallel-for.
        unsafe {
            arena::finalize_block(
                self.world,
                b.start,
                b.start + b.count,
                self.dt,
                self.inv_dt,
                self.enable_continuous,
            );
        }
    }

    fn prepare_overflow(&self) {
        contact::prepare(
            &self.overflow_cols,
            0,
            self.overflow_count,
            self.contact_softness,
            self.static_softness,
            self.warm_start_scale,
        );
    }

    fn warm_start_overflow(&self) {
        contact::warm_start(&self.overflow_cols, 0, self.overflow_count);
    }

    fn solve_overflow(&self, use_bias: bool) {
        contact::solve(
            &self.overflow_cols,
            0,
            self.overflow_count,
            use_bias,
            self.inv_h,
            self.contact_speed,
        );
    }

    fn restitution_overflow(&self) {
        contact::restitution(
            &self.overflow_cols,
            0,
            self.overflow_count,
            self.restitution_threshold,
        );
    }

    fn store_overflow(&self) {
        let mut has_hits = false;
        contact::store(
            &self.overflow_cols,
            0,
            self.overflow_count,
            self.hit_threshold,
            |contact| unsafe {
                arena::mark_hit_event(self.world, 0, contact);
                has_hits = true;
            },
        );
        unsafe { arena::finish_hit_events(self.world, 0, has_hits) };
    }

    // --- joints -------------------------------------------------------------------------------

    fn prepare_joints(&self, b: Block) {
        let mut color = 0;
        let mut index = b.start;
        let end_index = b.start + b.count;
        while color + 1 < self.color_count && self.joint_bases[color + 1] <= index {
            color += 1;
        }
        while index < end_index {
            let base = self.joint_bases[color];
            let count = self.joints[color].len() / crate::joint_abi::JOINT_STRIDE;
            let end = end_index.min(base + count);
            for slot in index..end {
                unsafe {
                    crate::joint::prepare_world(
                        self.world,
                        self.joints[color],
                        slot - base,
                        self.h,
                        self.inv_h,
                        self.enable_warm_starting,
                    );
                }
            }
            index = end;
            color += 1;
        }
    }

    fn warm_start_joints(&self, b: Block) {
        for slot in b.start..b.start + b.count {
            crate::joint::warm_start(
                self.joints[b.color as usize],
                slot,
                self.cols.state,
                self.cols.flags,
            );
        }
    }

    fn solve_joints(&self, b: Block, use_bias: bool, worker: usize) {
        let joints = self.joints[b.color as usize];
        let states = unsafe { arena::joint_states(self.world, worker) };
        for slot in b.start..b.start + b.count {
            crate::joint::solve(
                joints,
                slot,
                self.cols.state,
                self.cols.flags,
                use_bias,
                self.h,
                self.inv_h,
            );
            if use_bias {
                use crate::joint_abi::{get, J_FORCE_THRESHOLD, J_JOINT_ID, J_TORQUE_THRESHOLD};
                let force_threshold = get(joints, slot, J_FORCE_THRESHOLD);
                let torque_threshold = get(joints, slot, J_TORQUE_THRESHOLD);
                if force_threshold < f32::MAX || torque_threshold < f32::MAX {
                    let id = get(joints, slot, J_JOINT_ID).to_bits() as usize;
                    if !states.get(id) {
                        let (force, torque) =
                            crate::joint::reaction(joints, slot, self.inv_h, |id| unsafe {
                                crate::body::read_sim(
                                    crate::bodies::column(
                                        self.world,
                                        id,
                                        1,
                                        crate::body::SIM_STRIDE,
                                    ),
                                    0,
                                )
                                .rotation
                            });
                        if force >= force_threshold || torque >= torque_threshold {
                            unsafe { states.set(id) };
                        }
                    }
                }
            }
        }
    }

    fn prepare_overflow_joints(&self) {
        for slot in 0..self.overflow_joint_count {
            unsafe {
                crate::joint::prepare_world(
                    self.world,
                    self.overflow_joints,
                    slot,
                    self.h,
                    self.inv_h,
                    self.enable_warm_starting,
                );
            }
        }
    }

    fn warm_start_overflow_joints(&self) {
        for slot in 0..self.overflow_joint_count {
            crate::joint::warm_start(self.overflow_joints, slot, self.cols.state, self.cols.flags);
        }
    }

    fn solve_overflow_joints(&self, use_bias: bool) {
        for slot in 0..self.overflow_joint_count {
            crate::joint::solve(
                self.overflow_joints,
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
    solve_build_in_world(
        crate::regions::active(),
        thread_count,
        sub_step_count,
        wide_total,
        mesh_start,
        mesh_total,
        overflow_start,
        overflow_count,
        joint_total,
        overflow_joint_count,
        gx,
        gy,
        gz,
        h,
        inv_h,
        dt,
        inv_dt,
        max_linear_velocity,
        contact_speed,
        cs_bias,
        cs_mass,
        cs_impulse,
        ss_bias,
        ss_mass,
        ss_impulse,
        warm_start_scale,
        restitution_threshold,
        hit_event_threshold,
        enable_continuous,
    )
}
#[allow(clippy::too_many_arguments)]

pub extern "C" fn solve_build_in_world(
    world_index: usize,
    thread_count: usize,
    sub_step_count: usize,
    wide_total: usize,
    mesh_start: usize,
    mesh_total: usize,
    _overflow_start: usize,
    overflow_count: usize,
    joint_total: usize,
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
        arena::reset_hit_events(world_index, thread_count);
        arena::reset_joint_states(world_index);
        // Drop the previous solve's context before re-borrowing its buffers. The workers have all left
        // it (the join in `stages::run`), so nothing else holds them.
        STEP = core::ptr::null_mut();
        arena::free_solve(world_index);
        SPLIT_ID = crate::island::split_candidate_in_world(world_index);
        SPLIT_WORKER = usize::from(thread_count > 1);
        if SPLIT_ID != -1 {
            crate::island::prepare_split(world_index, SPLIT_ID as usize, SPLIT_WORKER);
        }

        let mut span_storage = [ColorSpan::EMPTY; MAX_COLORS];
        let mut color_keys = [0; MAX_COLORS];
        let (color_count, wide_total, mesh_total, joint_total) =
            crate::constraint_graph::solver_colors(world_index, &mut span_storage, &mut color_keys);
        let mesh_start = 0;
        let overflow_count = crate::constraint_graph::overflow_contact_count(world_index);
        let overflow_joint_count = crate::joints::count_in_world(world_index, MAX_COLORS - 1);
        let out = &mut span_storage;

        let mut joint_bases = [0; MAX_COLORS];
        let mut base = 0;
        let joints = core::array::from_fn(|c| {
            if c < color_count {
                joint_bases[c] = base;
                base += out[c].joint_count;
                crate::joints::column(world_index, color_keys[c])
            } else {
                Col::new(16 as *mut f32, 0)
            }
        });
        let (wide, wide_idx, wide_spans) = arena::wide_columns(world_index);
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
        let sizes = stages::sizes(&plan);
        let stage_offset =
            core::mem::size_of::<Step>().next_multiple_of(core::mem::align_of::<Stage>());
        let block_offset = (stage_offset + sizes.stages * core::mem::size_of::<Stage>())
            .next_multiple_of(core::mem::align_of::<SyncBlock>());
        let bytes = block_offset + sizes.blocks * core::mem::size_of::<SyncBlock>();
        let alignment = core::mem::align_of::<Step>().max(core::mem::align_of::<Stage>());
        let allocation = arena::reserve_solve(world_index, bytes + alignment - 1);
        let base = allocation.next_multiple_of(alignment);
        STEP = base as *mut Step;
        STEP.write(Step {
            ctx: None,
            work: None,
            spans: span_storage,
        });
        let stage_ptr = (base + stage_offset) as *mut Stage;
        let block_ptr = (base + block_offset) as *mut SyncBlock;
        for i in 0..sizes.stages {
            stage_ptr.add(i).write(Stage::EMPTY);
        }
        for i in 0..sizes.blocks {
            block_ptr.add(i).write(SyncBlock::EMPTY);
        }
        (*STEP).work = Some(Work {
            world: world_index,
            cols: arena::scalar_columns(world_index),
            overflow_cols: arena::overflow_columns(world_index),
            wide,
            wide_idx,
            wide_spans,
            overflow_count,
            joints,
            joint_bases,
            color_count,
            overflow_joints: crate::joints::column(world_index, MAX_COLORS - 1),
            fin: Col::new(
                crate::bodies::fin_base(world_index) as *mut f32,
                crate::bodies::body_cap_in_world(world_index) * crate::body::FIN_STRIDE,
            ),
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
            colors: &(&(*STEP).spans)[..color_count],
            sub_step_count,
            worker_count: thread_count,
        };
        (*STEP).ctx = Some(stages::build(
            &plan,
            core::slice::from_raw_parts_mut(stage_ptr, sizes.stages),
            core::slice::from_raw_parts_mut(block_ptr, sizes.blocks),
        ));
        JOB = Job::Solve;
    }
}

// --- the outer phases -------------------------------------------------------------------------
//
// Collide is one flat contact-id sweep: overlap, recycle and full contact update run in the same
// task. Its manifold allocators may grow linear memory, but never move existing chunks.

/// Which job the pool's current round runs. Written by a build on the main thread with every worker
/// parked; the wake that follows is the release edge that publishes it (module header).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Job {
    None,
    Solve,
    Contacts,
    Bullets,
    Pairs,
    Sensors,
    Finalize,
}

/// `parBuild`'s `kind` argument, mirrored in `src/kernel.ts`.
const KIND_CONTACTS: u32 = 2;

static mut SPLIT_ID: i32 = -1;
static mut SPLIT_WORKER: usize = 0;
static mut JOB: Job = Job::None;
static mut PAR: Option<Par> = None;

/// One built parallel-for: its partition and phase parameters.
struct Par {
    par: ParFor,
    count: usize,
    a: f32,
}

/// Partition one outer phase's `count` records over `thread_count` threads, and name it as the job the
/// next `pool.run` will drive. Main thread, with the workers parked and the phase's `reserve*` already
/// done — the columns the blocks read are fixed from here to the join.
///
/// **Returns 1 if the caller should fork, 0 if it should run the serial shim instead** — a sweep of one
/// block has nothing to steal, and one under the fork floor loses to its own wake (`parfor.rs`). The
/// policy lives here, not in the caller, so the cost model sits next to the machinery it prices.
#[export_name = "parBuild"]
pub extern "C" fn par_build(kind: u32, count: usize, thread_count: usize, a: f32) -> usize {
    assert!((1..=MAX_THREADS).contains(&thread_count));
    let job = match kind {
        KIND_CONTACTS => Job::Contacts,
        3 => Job::Bullets,
        4 => Job::Pairs,
        6 => Job::Sensors,
        7 => Job::Finalize,
        _ => panic!("unknown parallel-for kind"),
    };
    let par = ParFor::new(
        count,
        if job == Job::Sensors || job == Job::Finalize {
            16
        } else {
            COLLIDE_MIN_RANGE
        },
        thread_count,
    );
    let fork = if job == Job::Sensors || job == Job::Finalize {
        thread_count > 1 && count > 0
    } else {
        par.block_count() >= 2 && worth_forking(count, thread_count - 1, COLLIDE_FORK_MIN)
    };
    unsafe {
        PAR = Some(Par { par, count, a });
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
fn run_job(world_index: usize, index: usize) {
    unsafe {
        match *(&raw const JOB) {
            Job::None => {}
            Job::Solve => {
                let (Some(ctx), Some(work)) = (&(*STEP).ctx, &(*STEP).work) else {
                    return;
                };
                // Worker 1 handles the queued split while worker 0 and the other thieves solve.
                // Its solver exit acknowledgement joins both jobs before finalization.
                if SPLIT_ID != -1 && index == SPLIT_WORKER {
                    let start = crate::physics_world::ticks();
                    crate::island::split_task(world_index, SPLIT_ID as usize, index);
                    crate::physics_world::accumulate(world_index, 14, start);
                }
                stages::run(ctx, work, index);
            }
            job => {
                let Some(p) = &*(&raw const PAR) else {
                    return;
                };
                match job {
                    Job::Contacts => p
                        .par
                        .run(|s, e| arena::contact_block(world_index, s, e, p.count, index)),
                    Job::Bullets => p
                        .par
                        .run(|s, e| crate::continuous::bullets(world_index, s, e)),
                    Job::Pairs => p
                        .par
                        .run(|s, e| crate::pairwork::query_block(world_index, s, e, p.a as usize)),
                    Job::Sensors => p.par.run(|s, e| crate::sensor::task(world_index, s, e)),
                    Job::Finalize => p.par.run(|s, e| {
                        let work = (*STEP).work.as_ref().unwrap();
                        work.finalize(Block {
                            start: s,
                            count: e - s,
                            block_type: crate::stages::BlockType::Body,
                            color: 0,
                        });
                    }),
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
    run_mt_in_world(crate::regions::active())
}

pub extern "C" fn run_mt_in_world(world_index: usize) {
    run_job(world_index, 0);
}

/// A pooled worker's entry: run the built job as worker `index` (1-based — 0 is the orchestrator).
/// Exactly once per round — a skipped call hangs the staged solve's join, and a doubled one corrupts the
/// next step (`stages::run`'s contract).
#[export_name = "workerMain"]
pub extern "C" fn worker_main(index: usize) {
    worker_main_in_world(crate::regions::active(), index)
}

pub extern "C" fn worker_main_in_world(world_index: usize, index: usize) {
    run_job(world_index, index);
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
        if !STEP.is_null() {
            let Some(ctx) = &(*STEP).ctx else {
                return;
            };
            ctx.fault();
        }
    }
}

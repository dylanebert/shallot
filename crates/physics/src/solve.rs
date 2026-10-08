//! The staged solve, wired to the wasm arena — the kernel's multithreaded entry.
//!
//! `stages.rs` owns the staged solver's machinery (stage list, CAS-claimed blocks, barriers, the serial
//! overflow); `parfor.rs` owns the flat block-claim sweep the outer phases use; `arena.rs` owns the
//! columns. This module is the seam between them: it builds the [`Plan`] from the kernel graph layout,
//! holds the [`Context`] and the [`Work`] in linear memory where every thread's instance sees
//! them. Task slots publish each callback's context, and their finish handles protect its lifetime.
//! Solver-column reserves precede enqueue; the island split cannot relocate those columns.
//!
//! Wasm-only, like the arena it reads. Native `cargo test` drives the same machinery over owned columns
//! (`kernel/tests/stages.rs`).

use crate::arena;
use crate::col::Col;
use crate::contact::{self, Columns, Softness};
use crate::contact_wide;
use crate::integrate;
use crate::math::Vec3;
use crate::parfor::{ParFor, COLLIDE_MIN_RANGE};
use crate::scheduler::Scheduler;

pub(crate) static SCHEDULER: Scheduler = Scheduler::new();
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

// Published address of the world's live step allocation.
static mut STEP: *mut Step = core::ptr::null_mut();

pub(crate) unsafe fn release_step(world: usize) {
    if !STEP.is_null()
        && (*STEP)
            .work
            .as_ref()
            .is_some_and(|work| work.world == world)
    {
        STEP = core::ptr::null_mut();
    }
}

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

    fn finalize(&self, b: Block, worker: usize) {
        // SAFETY: the body + shape + fat-AABB regions were reserved pre-solve on the main thread, and
        // the solver and split have joined, and these columns stay fixed for this parallel-for.
        unsafe {
            arena::finalize_block(
                self.world,
                worker,
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
    _wide_total: usize,
    _mesh_start: usize,
    _mesh_total: usize,
    _overflow_start: usize,
    _overflow_count: usize,
    _joint_total: usize,
    _overflow_joint_count: usize,
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
        SOLVE_PAGES = core::arch::wasm32::memory_size::<0>();
        SPLIT_TASK = if SPLIT_ID != -1 {
            SCHEDULER.enqueue(split_task, world_index, 0)
        } else {
            None
        };
        (*STEP).ctx = Some(stages::build(
            &plan,
            core::slice::from_raw_parts_mut(stage_ptr, sizes.stages),
            core::slice::from_raw_parts_mut(block_ptr, sizes.blocks),
        ));
        SOLVE_THREADS = thread_count;
    }
}

// --- the outer phases -------------------------------------------------------------------------
//
// Collide is one flat contact-id sweep: overlap, recycle and full contact update run in the same
// task. Its manifold allocators may grow linear memory, but never move existing chunks.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Job {
    Contacts,
    Bullets,
    Pairs,
    Sensors,
    Finalize,
}

/// Narrow-phase sweep kind.
const KIND_CONTACTS: u32 = 2;

static mut SPLIT_ID: i32 = -1;
static mut SOLVE_THREADS: usize = 1;
static mut SPLIT_TASK: Option<usize> = None;
static mut SOLVE_PAGES: usize = 0;

/// One built parallel-for: its partition and phase parameters.
struct Par {
    kind: Job,
    world: usize,
    par: ParFor,
    worker_count: usize,
    count: usize,
    a: f32,
}

#[export_name = "parallelFor"]
pub extern "C" fn parallel_for(kind: u32, count: usize, thread_count: usize, a: f32) {
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
        if job == Job::Bullets {
            8
        } else if job == Job::Sensors || job == Job::Finalize {
            16
        } else if job == Job::Pairs {
            64
        } else {
            COLLIDE_MIN_RANGE
        },
        thread_count,
    );

    let pages = core::arch::wasm32::memory_size::<0>();
    let p = Par {
        kind: job,
        world: crate::regions::active(),
        worker_count: thread_count.min(par.block_count()),
        par,
        count,
        a,
    };
    let mut handles = [None; MAX_THREADS];
    for (i, handle) in handles.iter_mut().enumerate().take(p.worker_count) {
        *handle = unsafe { SCHEDULER.enqueue(par_task, &p as *const Par as usize, i) };
    }
    for handle in handles {
        SCHEDULER.finish(handle);
    }
    // Collide grows its block allocators, and each sensor task grows its own overlaps, as
    // b3SensorTask does (sensor.c). Neither moves storage another task reads.
    if job != Job::Contacts && job != Job::Sensors && !SCHEDULER.faulted() {
        assert_eq!(
            core::arch::wasm32::memory_size::<0>(),
            pages,
            "memory grew during scheduled {job:?}"
        );
    }
}

fn run_par(p: &Par, index: usize) {
    let world_index = p.world;
    unsafe {
        match p.kind {
            Job::Contacts => {
                p.par
                    .run(|s, e| arena::contact_block(world_index, s, e, p.count, index));
            }
            Job::Bullets => p
                .par
                .run(|s, e| crate::continuous::bullets(world_index, index, s, e)),
            Job::Pairs => p
                .par
                .run(|s, e| crate::pairwork::query_block(world_index, s, e, p.a as usize)),
            Job::Sensors => p
                .par
                .run(|s, e| crate::sensor::task(world_index, index, s, e)),
            Job::Finalize => p.par.run(|s, e| {
                let work = (*STEP).work.as_ref().unwrap();
                work.finalize(
                    Block {
                        start: s,
                        count: e - s,
                        block_type: crate::stages::BlockType::Body,
                        color: 0,
                    },
                    index,
                );
            }),
        }
    }
}

unsafe fn par_task(context: usize, index: usize) {
    run_par(&*(context as *const Par), index);
}
unsafe fn solver_task(_world: usize, index: usize) {
    let step = &*STEP;
    stages::run(
        step.ctx.as_ref().unwrap(),
        step.work.as_ref().unwrap(),
        index,
    );
}
unsafe fn split_task(world: usize, _: usize) {
    let start = crate::physics_world::ticks();
    crate::island::split_task(world, SPLIT_ID as usize, 0);
    crate::physics_world::accumulate(world, 14, start);
}
pub unsafe fn run_solve(world: usize) {
    let pages = SOLVE_PAGES;
    let mut handles = [None; MAX_THREADS];
    for (i, handle) in handles.iter_mut().enumerate().take(SOLVE_THREADS) {
        *handle = SCHEDULER.enqueue(solver_task, world, i);
    }
    solver_task(world, 0);
    for handle in handles {
        SCHEDULER.finish(handle);
    }
    SCHEDULER.finish(SPLIT_TASK);
    SPLIT_TASK = None;
    if !SCHEDULER.faulted() {
        assert_eq!(
            core::arch::wasm32::memory_size::<0>(),
            pages,
            "memory grew during the solver tasks"
        );
    }
}

/// A background worker parks in wasm between tasks; its host catch releases peers on a trap.
#[export_name = "workerMain"]
pub extern "C" fn worker_main(inject_fault: usize) {
    SCHEDULER.worker_with_fault(inject_fault != 0);
}

/// A trap cannot complete its claimed slot. Release the solver barriers and join surviving workers.
#[export_name = "workerFault"]
pub extern "C" fn worker_fault() {
    SCHEDULER.fail();
}
#[export_name = "workerRegister"]
pub extern "C" fn worker_register() {
    SCHEDULER.register_worker();
}
#[export_name = "schedulerStart"]
pub extern "C" fn scheduler_start() {
    SCHEDULER.start();
}
#[export_name = "schedulerStop"]
pub extern "C" fn scheduler_stop() {
    SCHEDULER.stop();
    SCHEDULER.join_fault();
}
#[export_name = "schedulerFaultPtr"]
pub extern "C" fn scheduler_fault_ptr() -> *mut i32 {
    SCHEDULER.fault_ptr()
}

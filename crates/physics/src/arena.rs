//! Shared step storage and phase export shims.
//!
//! Each world owns a LIFO stack. Phase reservations publish column byte offsets in `LAYOUT`;
//! TypeScript derives views from this header after allocations, which may grow linear memory.
//!
//! Wasm-only: the columns alias linear memory directly, so this is meaningful only in the JS host
//! (native tests drive the phase modules against their gold vectors instead). They are shared-mutable
//! [`Col`]s rather than `&mut` slices — `col.rs` carries the argument.

use crate::body::{FIN_STRIDE, SIM2_STRIDE, SIM_STRIDE, STATE_STRIDE};
use crate::col::Col;
use crate::contact::{Columns, ContactConstraint, ManifoldConstraint};
use crate::contact_wide::WIDE_STRIDE;
use crate::distance::SimplexCache;
use crate::finalize::{self, TY_CAPSULE, TY_HULL, TY_SPHERE};
use crate::manifold::{Capsule, SatCache, Sphere};
use crate::manifold_abi::{
    read_dir, DIR_CACHE, DIR_CACHED_REL_POSE, DIR_CACHED_ROT_A, DIR_CACHED_ROT_B, DIR_FLAGS,
    DIR_STRIDE, MANIFOLD_STRIDE, M_POINT_COUNT,
};
use crate::manifolds;
use crate::math::{Quat, Transform, Vec3};
use crate::narrowphase::{compute_convex_manifold_into, ConvexContactCache, ConvexShape};

use crate::geo::hull_view;
static mut STACKS: [crate::task_memory::Stack; crate::regions::MAX_WORLDS] =
    [const { crate::task_memory::Stack::EMPTY }; crate::regions::MAX_WORLDS];
static mut SCRATCH_PTR: [usize; crate::regions::MAX_WORLDS] = [0; crate::regions::MAX_WORLDS];

static mut SOLVE_PTR: [usize; crate::regions::MAX_WORLDS] = [0; crate::regions::MAX_WORLDS];

pub unsafe fn reserve_solve(world: usize, bytes: usize) -> usize {
    assert_eq!(SOLVE_PTR[world], 0);
    let ptr = STACKS[world].alloc(bytes) as usize;
    SOLVE_PTR[world] = ptr;
    ptr
}

pub unsafe fn free_solve(world: usize) {
    if SOLVE_PTR[world] != 0 {
        crate::solve::release_step(world);
        STACKS[world].free(SOLVE_PTR[world] as *mut u8);
        SOLVE_PTR[world] = 0;
    }
}

pub unsafe fn free_scratch(world: usize) {
    free_solve(world);
    if SCRATCH_PTR[world] != 0 {
        STACKS[world].free(SCRATCH_PTR[world] as *mut u8);
        SCRATCH_PTR[world] = 0;
    }
}

pub unsafe fn grow_stack(world: usize) {
    free_scratch(world);
    STACKS[world].grow();
}

pub unsafe fn reset_stack(world: usize) {
    free_scratch(world);
    STACKS[world] = crate::task_memory::Stack::EMPTY;
}

pub unsafe fn reserve_scratch(world: usize, bytes: usize) -> usize {
    free_scratch(world);
    if STACKS[world].is_empty() {
        STACKS[world] = crate::task_memory::Stack::new(2048);
    }
    let ptr = STACKS[world].alloc(bytes) as usize;
    SCRATCH_PTR[world] = ptr;
    ptr
}
const N_COLS: usize = 15;
/// Active-color span: wideStart, wideCount, meshStart, meshCount, jointArrayKey, jointCount.
/// The staged solve selects the resident joint array by key.
pub(crate) const COLOR_SPAN_STRIDE: usize = 6;

/// The worker index the serial (single-crossing) shims run as — the thread driving the step is always
/// worker 0 (`stages::run`).
const ORCHESTRATOR: usize = 0;

// LAYOUT indices, in memory order.
const STATE: usize = 0;
const FLAGS: usize = 1;
const SIM: usize = 2;
const FIN: usize = 3;
const CC: usize = 6;
const MC: usize = 8;
const OVERFLOW_CC: usize = 7;
const OVERFLOW_MC: usize = 9;
const WIDE: usize = 11;
// Per-active-color spans (wide/mesh/joint start+count) for the batched color loop + staged solve.
const COLOR_SPAN: usize = 14;

/// Per-column byte offsets into linear memory, rewritten by every `reserve`. TS reads this header
/// (`layoutPtr`) to build its column views.
static mut LAYOUT: [u32; N_COLS] = [0; N_COLS];

// The per-step counts the shims size their slices from (set by `reserve`).
static mut BODY_COUNT: usize = 0;
static mut CONTACT_COUNT: usize = 0;
static mut MANIFOLD_COUNT: usize = 0;
static mut OVERFLOW_CONTACT_COUNT: usize = 0;
static mut OVERFLOW_MANIFOLD_COUNT: usize = 0;
// Wide record count (each groups up to 4 convex contacts); sizes the wide transient columns.
static mut WIDE_COUNT: usize = 0;
// Active color count — the number of spans in the COLOR_SPAN column the batched shims loop over.
static mut COLOR_COUNT: usize = 0;

/// Column of `len` f32 at `LAYOUT[idx]`. `len` must match the reserved column size.
///
/// A [`Col`] and not a `&mut [f32]`: once the staged solver runs a stage's blocks on several threads
/// (`stages.rs`), every one of them holds this same column, and a `&mut` over it would be live
/// aliasing `&mut` — UB under `noalias` however disjoint the writes are (see `col.rs`).
#[inline]
unsafe fn f32s(idx: usize, len: usize) -> Col<'static, f32> {
    Col::new(LAYOUT[idx] as *mut f32, len)
}

/// Column of `len` u32 at `LAYOUT[idx]`. As [`f32s`].
#[inline]
unsafe fn u32s(idx: usize, len: usize) -> Col<'static, u32> {
    Col::new(LAYOUT[idx] as *mut u32, len)
}

/// Byte offset of the layout header (`[u32; N_COLS]` of per-column byte offsets). Stable across the
/// static-data section, but the buffer it lives in still detaches on `memory.grow`, so TS re-derives
/// its view after every `reserve`.
#[export_name = "layoutPtr"]
pub extern "C" fn layout_ptr() -> *const u32 {
    &raw const LAYOUT as *const u32
}

/// Lay out all solver columns for the given per-step counts, growing memory to fit. Recomputes offsets
/// within shared scratch each call; TS re-derives its views from `layoutPtr` afterwards.
#[export_name = "reserve"]
pub extern "C" fn reserve(
    body: usize,
    contact: usize,
    manifold: usize,
    point: usize,
    wide: usize,
    color: usize,
) {
    reserve_in_world(
        crate::regions::active(),
        body,
        contact,
        manifold,
        point,
        wide,
        color,
    )
}

pub extern "C" fn reserve_in_world(
    world_index: usize,
    body: usize,
    contact: usize,
    manifold: usize,
    point: usize,
    wide: usize,
    color: usize,
) {
    unsafe {
        BODY_COUNT = body;
        OVERFLOW_CONTACT_COUNT = crate::constraint_graph::overflow_contact_count(world_index);
        OVERFLOW_MANIFOLD_COUNT = crate::constraint_graph::overflow_manifold_count(world_index);
        CONTACT_COUNT = contact - OVERFLOW_CONTACT_COUNT;
        MANIFOLD_COUNT = manifold - OVERFLOW_MANIFOLD_COUNT;
        let _ = point;
        WIDE_COUNT = wide;
        COLOR_COUNT = color;

        // The body columns are resident (4a.2/4a.3): `state` + `flags` (velocity/delta/flags),
        // and `sim` + `fin` (the integrate/finalize sim fields) live in the persistent body
        // region (bodies.rs), held across steps, so the awake `BodySim`/`BodyState` become offset-backed
        // views and no per-step marshal runs. Point their LAYOUT entries at that region instead of
        // allocating per-step scratch; the phase shims read `LAYOUT[SIM]`/etc unchanged. `reserveBodies`
        // (run before this, in `step()`) has laid the region out for the current total-body high-water.
        // The remaining columns share the per-step arena.
        LAYOUT[STATE] = crate::bodies::state_base(world_index) as u32;
        LAYOUT[FLAGS] = crate::bodies::flags_base(world_index) as u32;
        LAYOUT[SIM] = crate::bodies::sim_base(world_index) as u32;
        LAYOUT[FIN] = crate::bodies::fin_base(world_index) as u32;
        let mut off = 0;
        LAYOUT[WIDE] = off as u32;
        off += wide * WIDE_STRIDE * 4;
        LAYOUT[CC] = off as u32;
        off += CONTACT_COUNT * core::mem::size_of::<ContactConstraint>();
        LAYOUT[MC] = off as u32;
        off += MANIFOLD_COUNT * core::mem::size_of::<ManifoldConstraint>();
        LAYOUT[OVERFLOW_CC] = off as u32;
        off += OVERFLOW_CONTACT_COUNT * core::mem::size_of::<ContactConstraint>();
        LAYOUT[OVERFLOW_MC] = off as u32;
        off += OVERFLOW_MANIFOLD_COUNT * core::mem::size_of::<ManifoldConstraint>();
        LAYOUT[COLOR_SPAN] = off as u32;
        off += color * COLOR_SPAN_STRIDE * 4;

        let continuous_offset = off;
        off += body * crate::continuous::STRIDE * 4;
        let base = reserve_scratch(world_index, off);
        for column in [CC, MC, OVERFLOW_CC, OVERFLOW_MC, WIDE, COLOR_SPAN] {
            LAYOUT[column] += base as u32;
        }
        crate::continuous::reserve_at(base + continuous_offset, body);
    }
}

/// Resident body column capacity; null gathers use a local dummy.
#[inline]
unsafe fn body_records(world_index: usize) -> usize {
    crate::bodies::body_cap_in_world(world_index)
}

/// Body views, graph prepare spans and native-shaped constraints for the current reservation.
/// Constraint records are disjoint across tasks; body writes are separated by graph coloring.
unsafe fn columns(world_index: usize) -> Columns<'static> {
    let b = body_records(world_index);
    let c = CONTACT_COUNT;
    let m = MANIFOLD_COUNT;
    Columns {
        state: f32s(STATE, b * STATE_STRIDE),
        flags: u32s(FLAGS, b * STATE_STRIDE),
        sim: f32s(SIM, b * SIM_STRIDE),
        spans: crate::constraint_graph::prepare_spans(world_index).0,
        dir: manifolds::dir_col(world_index),
        pool: manifolds::pool_col(),
        cc: Col::new(LAYOUT[CC] as *mut ContactConstraint, c),
        mc: Col::new(LAYOUT[MC] as *mut ManifoldConstraint, m),
    }
}

// --- the staged solver's view of the arena (solve.rs) ----------------------------------------
// The staged solve derives its columns once, up front, and hands the same handles to every worker —
// whose handles remain valid because none of these columns relocate between fork and join.
// The concurrent split can grow island storage, not these solver buffers.

/// The scalar solver's columns, as `solve.rs`'s `StageWork` holds them.
pub(crate) unsafe fn scalar_columns(world_index: usize) -> Columns<'static> {
    columns(world_index)
}

pub(crate) unsafe fn overflow_columns(world: usize) -> Columns<'static> {
    Columns {
        cc: Col::new(
            LAYOUT[OVERFLOW_CC] as *mut ContactConstraint,
            OVERFLOW_CONTACT_COUNT,
        ),
        mc: Col::new(
            LAYOUT[OVERFLOW_MC] as *mut ManifoldConstraint,
            OVERFLOW_MANIFOLD_COUNT,
        ),
        spans: crate::constraint_graph::overflow_spans(world),
        ..columns(world)
    }
}

/// Float and index views of the same wide records, plus graph prepare spans.
pub(crate) unsafe fn wide_columns(
    world: usize,
) -> (
    Col<'static, f32>,
    Col<'static, u32>,
    Col<'static, crate::contact_spans::WidePrepareSpan>,
) {
    let w = WIDE_COUNT;
    (
        f32s(WIDE, w * WIDE_STRIDE),
        u32s(WIDE, w * WIDE_STRIDE),
        crate::constraint_graph::prepare_spans(world).1,
    )
}

/// The awake body count the current reservation was sized for (the body blocks' item count).
pub(crate) unsafe fn body_count() -> usize {
    BODY_COUNT
}

/// The active colors' spans, as written by TS `writeColorSpans`: `COLOR_COUNT` records of
/// `COLOR_SPAN_STRIDE` u32s.
pub(crate) unsafe fn color_span_column() -> (Col<'static, u32>, usize) {
    let c = COLOR_COUNT;
    (u32s(COLOR_SPAN, c * COLOR_SPAN_STRIDE), c)
}

// --- b3CollideTask / b3UpdateContact --------------------------------------------------------
struct TaskContext {
    arena: crate::task_memory::WorkerArena,
    materials: usize,
    hit_event_bitset: crate::bitset::BitSet,
    joint_state_bitset: crate::bitset::BitSet,
    has_hit_events: bool,
}
impl TaskContext {
    fn new() -> Self {
        Self {
            arena: unsafe { crate::task_memory::WorkerArena::new(128 * 1024) },
            materials: 0,
            hit_event_bitset: crate::bitset::BitSet::new(1024),
            joint_state_bitset: crate::bitset::BitSet::new(1024),
            has_hit_events: false,
        }
    }
}
static mut TASK_CONTEXTS: [Vec<TaskContext>; crate::regions::MAX_WORLDS] =
    [const { Vec::new() }; crate::regions::MAX_WORLDS];

unsafe fn task_context(world: usize, worker: usize) -> *mut TaskContext {
    TASK_CONTEXTS[world].as_ptr().add(worker).cast_mut()
}

pub(crate) unsafe fn sync_task_arenas(world: usize) {
    for context in &mut TASK_CONTEXTS[world] {
        context.arena.sync();
        context.materials = 0;
    }
}

pub(crate) unsafe fn reset_hit_events(world: usize, workers: usize) {
    let capacity = u32::try_from(manifolds::contact_record_capacity(world)).unwrap();
    let contexts = &mut TASK_CONTEXTS[world];
    contexts.resize_with(workers, TaskContext::new);
    for context in contexts {
        context.hit_event_bitset.set_count_and_clear(capacity);
        context.has_hit_events = false;
    }
}

// Filter joints cannot emit a solver event; fixtures inject candidates before publication.
#[export_name = "jointResetEventBits"]
pub unsafe extern "C" fn reset_joint_event_bits(world: usize) {
    if TASK_CONTEXTS[world].is_empty() {
        TASK_CONTEXTS[world].push(TaskContext::new());
    }
    reset_joint_states(world);
}

#[export_name = "jointSetEventBit"]
pub unsafe extern "C" fn set_joint_event_bit(world: usize, id: usize) {
    assert_eq!(crate::joint_record::record(world, id).set_index, 2);
    joint_states(world, 0).set(id);
}

pub(crate) unsafe fn reset_joint_states(world: usize) {
    let capacity = u32::try_from(crate::joint_record::capacity_in_world(world)).unwrap();
    for context in &mut TASK_CONTEXTS[world] {
        context.joint_state_bitset.set_count_and_clear(capacity);
    }
}

pub(crate) unsafe fn joint_states(world: usize, worker: usize) -> &'static crate::bitset::BitSet {
    &(*task_context(world, worker)).joint_state_bitset
}

pub(crate) unsafe fn union_joint_states(world: usize) -> &'static crate::bitset::BitSet {
    let (first, rest) = TASK_CONTEXTS[world].split_first_mut().unwrap();
    for context in rest {
        first.joint_state_bitset.union(&context.joint_state_bitset);
    }
    &first.joint_state_bitset
}

pub(crate) unsafe fn mark_hit_event(world: usize, worker: usize, contact: usize) {
    let context = &mut *task_context(world, worker);
    context.hit_event_bitset.set(contact);
}

pub(crate) unsafe fn finish_hit_events(world: usize, worker: usize, has_hits: bool) {
    let context = &mut *task_context(world, worker);
    context.has_hit_events |= has_hits;
}

pub(crate) unsafe fn union_hit_events(world: usize) -> Option<&'static crate::bitset::BitSet> {
    let contexts = &mut TASK_CONTEXTS[world];
    if !contexts.iter().any(|context| context.has_hit_events) {
        return None;
    }
    let (first, rest) = contexts.split_first_mut().unwrap();
    for context in rest {
        if context.has_hit_events {
            first.hit_event_bitset.union(&context.hit_event_bitset);
        }
    }
    Some(&first.hit_event_bitset)
}

static mut CONTACT_LIST_PTR: usize = 0;
static mut CONTACT_STATES: [Vec<crate::bitset::BitSet>; crate::regions::MAX_WORLDS] =
    [const { Vec::new() }; crate::regions::MAX_WORLDS];

pub(crate) unsafe fn reset_contact_states(world: usize) {
    CONTACT_STATES[world] = Vec::new();
    TASK_CONTEXTS[world] = Vec::new();
}

pub(crate) unsafe fn union_contact_states(world: usize) -> &'static crate::bitset::BitSet {
    let sets = &mut CONTACT_STATES[world];
    let (first, rest) = sets.split_first_mut().unwrap();
    for set in rest {
        first.union(set);
    }
    first
}
static mut DEFAULT_MIX: u32 = 1;
static mut RECYCLE_DISTANCE: f32 = 0.0;

#[export_name = "reserveCollide"]
pub extern "C" fn reserve_collide(count: usize, threads: usize, default_mix: u32, distance: f32) {
    reserve_collide_in_world(
        crate::regions::active(),
        count,
        threads,
        default_mix,
        distance,
    )
}

pub extern "C" fn reserve_collide_in_world(
    world_index: usize,
    count: usize,
    threads: usize,
    default_mix: u32,
    distance: f32,
) {
    unsafe {
        DEFAULT_MIX = default_mix;
        RECYCLE_DISTANCE = distance;
        let mesh_threads = threads.max(1);
        let states = &mut CONTACT_STATES[world_index];
        states.resize_with(mesh_threads, || crate::bitset::BitSet::new(1024));
        let capacity = u32::try_from(manifolds::contact_capacity(world_index)).unwrap();
        for state in states {
            state.set_count_and_clear(capacity);
        }
        TASK_CONTEXTS[world_index].resize_with(mesh_threads, TaskContext::new);
        CONTACT_LIST_PTR = reserve_scratch(world_index, count * 4);
    }
}
#[export_name = "collideListPtr"]
pub extern "C" fn collide_list_ptr() -> usize {
    unsafe { CONTACT_LIST_PTR }
}
#[export_name = "contactStatePtr"]
pub extern "C" fn contact_state_ptr() -> usize {
    unsafe { CONTACT_STATES[crate::regions::active()][0].bits as usize }
}
unsafe fn mark_contact_state(world: usize, thread: usize, contact: usize) {
    CONTACT_STATES[world][thread].set(contact);
}
unsafe fn finish_contact(
    world_index: usize,
    thread: usize,
    contact: usize,
    count: usize,
    hit: bool,
) {
    let dir = manifolds::dir_col(world_index);
    let o = contact * DIR_STRIDE + DIR_FLAGS;
    let old = dir.get(o);
    let was_touching = old & 0x0001_0000 != 0;
    let mut flags = old & !(0x0001_0000 | 0x0010_0000);
    if count > 0 {
        flags |= 0x0001_0000;
        if hit {
            flags |= 0x0010_0000;
        }
        if !was_touching {
            flags |= 0x0004_0000;
            mark_contact_state(world_index, thread, contact);
        }
    } else {
        manifolds::free_manifolds_in_world(world_index, contact);
        if old & 0x0040_0000 == 0 {
            flags |= old & 0x0010_0000;
        }
        if was_touching {
            flags |= 0x0008_0000;
            mark_contact_state(world_index, thread, contact);
        }
    }
    dir.set(o, flags);
}

#[inline]
fn read_xf(disp: &[u32], o: usize) -> Transform {
    Transform {
        p: Vec3::new(
            f32::from_bits(disp[o]),
            f32::from_bits(disp[o + 1]),
            f32::from_bits(disp[o + 2]),
        ),
        q: Quat {
            v: Vec3::new(
                f32::from_bits(disp[o + 3]),
                f32::from_bits(disp[o + 4]),
                f32::from_bits(disp[o + 5]),
            ),
            s: f32::from_bits(disp[o + 6]),
        },
    }
}

/// Reconstruct a convex shape from its dispatch geom slots. A hull borrows its topology view straight out
/// of the geometry columns; sphere/capsule params are inlined in the record.
unsafe fn read_shape(ty: u32, disp: &[u32], o: usize) -> ConvexShape<'static> {
    match ty {
        TY_SPHERE => ConvexShape::Sphere(Sphere {
            center: Vec3::new(
                f32::from_bits(disp[o]),
                f32::from_bits(disp[o + 1]),
                f32::from_bits(disp[o + 2]),
            ),
            radius: f32::from_bits(disp[o + 3]),
        }),
        TY_CAPSULE => ConvexShape::Capsule(Capsule {
            center1: Vec3::new(
                f32::from_bits(disp[o]),
                f32::from_bits(disp[o + 1]),
                f32::from_bits(disp[o + 2]),
            ),
            center2: Vec3::new(
                f32::from_bits(disp[o + 3]),
                f32::from_bits(disp[o + 4]),
                f32::from_bits(disp[o + 5]),
            ),
            radius: f32::from_bits(disp[o + 6]),
        }),
        _ => ConvexShape::Hull(hull_view(disp[o] as usize)),
    }
}

unsafe fn dispatch_mesh(
    world_index: usize,
    shape_b: &[u32],
    old_count: usize,
    geom: &[u32],
    ty: u32,
    xf_a: Transform,
    xf_b: Transform,
    child_offset: Vec3,
    contact_id: usize,
    thread: usize,
    fast: bool,
    center_a: Vec3,
    center_b: Vec3,
) -> usize {
    use crate::manifold_abi::ManifoldRecord;
    use crate::mesh_contact::{compute_mesh_manifolds_into, TriangleSource, MAX_TRIANGLES};
    let cache = &mut *manifolds::mesh_cache_ptr(world_index, contact_id);
    let record = geom[0] as *const u32;
    let source = if ty == 4 {
        let mesh = crate::geo::mesh_view(
            record,
            Vec3::new(
                f32::from_bits(geom[1]),
                f32::from_bits(geom[2]),
                f32::from_bits(geom[3]),
            ),
        );
        TriangleSource::Mesh {
            flags: core::slice::from_raw_parts(
                (record as *const u8).add(*record.add(22) as usize),
                mesh.triangles.len(),
            ),
            mesh,
        }
    } else {
        let field = crate::geo::height_view(record);
        TriangleSource::Height {
            flags: core::slice::from_raw_parts(
                (record as *const u8).add(*record.add(21) as usize),
                2 * (field.columns() - 1) * (field.rows() - 1),
            ),
            field,
        }
    };
    let vec = |o: usize| {
        Vec3::new(
            f32::from_bits(shape_b[o]),
            f32::from_bits(shape_b[o + 1]),
            f32::from_bits(shape_b[o + 2]),
        )
    };
    let mut previous = [crate::manifold_abi::ContactCache { words: [0; 4] }; MAX_TRIANGLES];
    cache.refresh(&source, xf_a, vec(10), vec(13), &mut previous);
    let context = &mut *task_context(world_index, thread);
    let dir = manifolds::dir_col(world_index);
    let entry = read_dir(dir, contact_id);
    let shape = read_shape(shape_b[crate::shapes::S_TYPE], shape_b, 48);
    let directory = manifolds::dir_col(world_index);
    let shape_a =
        directory.get(contact_id * DIR_STRIDE + crate::manifold_abi::DIR_SHAPE_A) as usize;
    let shape_b =
        directory.get(contact_id * DIR_STRIDE + crate::manifold_abi::DIR_SHAPE_B) as usize;
    let shape_records = crate::shapes::col(world_index);
    let speculative = shape_records
        .get(shape_a * crate::shapes::SHAPE_STRIDE + crate::shapes::S_FLAGS)
        & crate::shapes::SPECULATIVE_FLAG
        != 0
        && shape_records.get(shape_b * crate::shapes::SHAPE_STRIDE + crate::shapes::S_FLAGS)
            & crate::shapes::SPECULATIVE_FLAG
            != 0;
    let mut mesh = crate::mesh_contact::MeshScratch::from_arena(
        context.arena.arena,
        cache.triangles.count as usize,
    );
    let mut address = entry.manifold_base;
    let count = compute_mesh_manifolds_into(
        &mut mesh,
        cache.triangles.as_mut_slice(),
        |index| source.triangle(index),
        &shape,
        xf_a,
        xf_b,
        fast,
        speculative,
        |count, arena| {
            let old_ptr = arena
                .unwrap()
                .bump(old_count * core::mem::size_of::<ManifoldRecord>())
                as *mut ManifoldRecord;
            let old = if old_count > 0 {
                core::ptr::copy_nonoverlapping(
                    address as *const ManifoldRecord,
                    old_ptr,
                    old_count,
                );
                core::slice::from_raw_parts_mut(old_ptr, old_count)
            } else {
                &mut []
            };
            if count != old_count {
                address = manifolds::allocate_manifolds_in_world(world_index, contact_id, count);
            } else {
                (address as *mut u8).write_bytes(0, count * core::mem::size_of::<ManifoldRecord>());
            }
            (
                old,
                core::slice::from_raw_parts_mut(address as *mut ManifoldRecord, count),
            )
        },
    );
    context.materials = mesh.materials.as_mut_ptr() as usize;
    for i in 0..count {
        let m = &mut *(address as *mut ManifoldRecord).add(i);
        for p in &mut m.points[..m.point_count as usize] {
            p.anchor_a = p.anchor_a.add(child_offset);
        }
        for p in &mut m.points[..m.point_count as usize] {
            p.anchor_a = p.anchor_a.sub(center_a);
            p.anchor_b = p.anchor_b.sub(center_b);
            p.base_separation = p.separation;
        }
    }
    count
}

#[inline]
fn read_simplex(dir: Col<u32>, id: usize) -> SimplexCache {
    let o = id * DIR_STRIDE + DIR_CACHE;
    unsafe { dir.ptr().add(o).cast::<SimplexCache>().read() }
}

#[inline]
fn write_simplex(dir: Col<u32>, id: usize, c: &SimplexCache) {
    let o = id * DIR_STRIDE + DIR_CACHE;
    unsafe {
        let ptr = dir.ptr().add(o).cast::<SimplexCache>();
        core::ptr::addr_of_mut!((*ptr).metric).write(c.metric);
        core::ptr::addr_of_mut!((*ptr).count).write(c.count);
        core::ptr::addr_of_mut!((*ptr).index_a).write(c.index_a);
        core::ptr::addr_of_mut!((*ptr).index_b).write(c.index_b);
    }
}

#[inline]
fn read_sat(dir: Col<u32>, id: usize) -> SatCache {
    let o = id * DIR_STRIDE + DIR_CACHE;
    unsafe { dir.ptr().add(o).cast::<SatCache>().read() }
}

#[inline]
fn write_sat(dir: Col<u32>, id: usize, c: &SatCache) {
    let o = id * DIR_STRIDE + DIR_CACHE;
    unsafe {
        dir.ptr().add(o).cast::<SatCache>().write(*c);
    }
}

#[derive(Clone, Copy)]
struct Surface {
    friction: f32,
    restitution: f32,
    rolling: f32,
    tangent: Vec3,
}

fn surface(world_index: usize, shape: usize, index: usize) -> Surface {
    let materials = crate::shapes::material(world_index, shape, index);
    let f = |i| f32::from_bits(materials[i]);
    Surface {
        friction: f(0),
        restitution: f(1),
        rolling: f(2),
        tangent: Vec3::new(f(3), f(4), f(5)),
    }
}

fn shape_radius(world_index: usize, shape: usize, full_hull: bool) -> f32 {
    let shapes = crate::shapes::col_slice(world_index);
    let s = shape * crate::shapes::SHAPE_STRIDE;
    match shapes[s + crate::shapes::S_TYPE] {
        TY_SPHERE => f32::from_bits(shapes[s + 51]),
        TY_CAPSULE => f32::from_bits(shapes[s + 54]),
        TY_HULL => {
            (if full_hull { 1.0 } else { 0.25 })
                * unsafe {
                    crate::geo::hull_record(shapes[s + crate::shapes::S_GEOM] as usize).inner_radius
                }
        }
        _ => 0.0,
    }
}

fn store_surface(
    world_index: usize,
    id: usize,
    friction: f32,
    restitution: f32,
    rolling: f32,
    tangent: Vec3,
) {
    let dir = manifolds::dir_col(world_index);
    let o = id * DIR_STRIDE;
    use crate::manifold_abi::{
        DIR_FRICTION, DIR_RESTITUTION, DIR_ROLLING_RESISTANCE, DIR_TANGENT_VELOCITY,
    };
    for (field, value) in [
        (DIR_FRICTION, friction),
        (DIR_RESTITUTION, restitution),
        (DIR_ROLLING_RESISTANCE, rolling),
        (DIR_TANGENT_VELOCITY, tangent.x),
        (DIR_TANGENT_VELOCITY + 1, tangent.y),
        (DIR_TANGENT_VELOCITY + 2, tangent.z),
    ] {
        dir.set(o + field, value.to_bits());
    }
}

unsafe fn mix_surface(
    world_index: usize,
    contact_id: usize,
    sa: usize,
    sb: usize,
    thread: usize,
    xf_a: Transform,
    xf_b: Transform,
    map: Option<[u32; 4]>,
    flip: bool,
    count: usize,
    mesh: bool,
    child_radius: f32,
) {
    if count == 0 || DEFAULT_MIX == 0 {
        return;
    }
    let a_index = |i: usize| map.map_or(i, |m| m[i.min(3)] as usize);
    let mut a = surface(world_index, sa, a_index(0));
    let mut b = surface(world_index, sb, 0);
    let mut radius_a = if map.is_some() {
        child_radius
    } else {
        shape_radius(world_index, sa, false)
    };
    let mut radius_b = shape_radius(world_index, sb, mesh);
    let (friction, restitution, rolling, tangent) = if mesh {
        let dir = manifolds::dir_col(world_index);
        let address =
            dir.get(contact_id * DIR_STRIDE + crate::manifold_abi::DIR_MANIFOLD_BASE) as usize;
        let output = Col::new(address as *mut f32, count * MANIFOLD_STRIDE);
        let materials = (*task_context(world_index, thread)).materials as *const u32;
        let mut friction = 0.0;
        let mut restitution = 0.0;
        let mut tangent = Vec3::ZERO;
        let mut samples = 0.0;
        for i in 0..count {
            let pc = output.get(i * MANIFOLD_STRIDE + M_POINT_COUNT).to_bits() as usize;
            for j in 0..pc {
                let m = surface(world_index, sa, a_index(*materials.add(i * 4 + j) as usize));
                friction += (m.friction * b.friction).sqrt();
                restitution += m.restitution.max(b.restitution);
                tangent = tangent.add(m.tangent);
                samples += 1.0;
            }
        }
        let inv = 1.0 / samples;
        (
            inv * friction,
            inv * restitution,
            b.rolling * radius_b,
            xf_a.q
                .rotate(tangent.scale(inv))
                .sub(xf_b.q.rotate(b.tangent)),
        )
    } else {
        let (qa, qb) = if flip {
            core::mem::swap(&mut a, &mut b);
            core::mem::swap(&mut radius_a, &mut radius_b);
            (xf_b.q, xf_a.q)
        } else {
            (xf_a.q, xf_b.q)
        };
        let rolling = if a.rolling > 0.0 || b.rolling > 0.0 {
            a.rolling.max(b.rolling) * radius_a.max(radius_b)
        } else {
            0.0
        };
        (
            (a.friction * b.friction).sqrt(),
            a.restitution.max(b.restitution),
            rolling,
            qa.rotate(a.tangent).sub(qb.rotate(b.tangent)),
        )
    };
    store_surface(
        world_index,
        contact_id,
        friction,
        restitution,
        rolling,
        tangent,
    );
}

/// Box3D's b3UpdateContact over the shapes and body poses resolved by the collide task.
///
/// # Safety
/// Shapes and worker scratch stay resident during the fork. Each contact runs in one task.
unsafe fn update_contact(
    world_index: usize,
    thread: usize,
    contact_id: usize,
    shape_a: &[u32],
    shape_b: &[u32],
    parent_xf: Transform,
    local_center_a: Vec3,
    xf_b: Transform,
    local_center_b: Vec3,
    fast: bool,
) {
    unsafe {
        use crate::manifold_abi::*;
        let dir = manifolds::dir_col(world_index);
        let o = contact_id * DIR_STRIDE;
        // JavaScript material callbacks run at the serial post-collide binding.
        dir.set(o + DIR_FLAGS, dir.get(o + DIR_FLAGS) | 0x0200_0000);
        let old_count = dir.get(o + DIR_MANIFOLD_COUNT) as usize;
        let shape_id_a = dir.get(o + DIR_SHAPE_A) as usize;
        let shape_id_b = dir.get(o + DIR_SHAPE_B) as usize;
        let hit = shape_a[crate::shapes::S_HIT_EVENTS] & crate::shapes::HIT_FLAG != 0
            || shape_b[crate::shapes::S_HIT_EVENTS] & crate::shapes::HIT_FLAG != 0;
        let mut type_a = shape_a[crate::shapes::S_TYPE];
        let type_b = shape_b[crate::shapes::S_TYPE];
        let mut xf_a = parent_xf;
        let center_a = parent_xf.q.rotate(local_center_a);
        let center_b = xf_b.q.rotate(local_center_b);
        let mut geom_a = &shape_a[48..55];
        let mut child_offset = Vec3::ZERO;
        let mut material_map = None;
        let mut child_radius = 0.0;
        let compound_geometry = if type_a == 1 {
            let compound = geom_a[0] as *const u32;
            Some(crate::compound_query::child_words(
                compound,
                dir.get(o + DIR_CHILD_INDEX) as usize,
            ))
        } else {
            None
        };
        if let Some(compound_geometry) = &compound_geometry {
            type_a = compound_geometry[0];
            material_map = Some([
                compound_geometry[8],
                compound_geometry[9],
                compound_geometry[10],
                compound_geometry[11],
            ]);
            let local = read_xf(compound_geometry, 1);
            child_offset = parent_xf.q.rotate(local.p);
            if type_a == TY_HULL || type_a == 4 {
                xf_a = parent_xf.mul(local);
            }
            geom_a = &compound_geometry[12..19];
            child_radius = match type_a {
                TY_HULL => 0.25 * f32::from_bits(geom_a[1]),
                TY_SPHERE => f32::from_bits(geom_a[3]),
                TY_CAPSULE => f32::from_bits(geom_a[6]),
                _ => 0.0,
            };
        }
        if type_a == 2 || type_a == 4 {
            let count = dispatch_mesh(
                world_index,
                shape_b,
                old_count,
                geom_a,
                type_a,
                xf_a,
                xf_b,
                child_offset,
                contact_id,
                thread,
                fast,
                center_a,
                center_b,
            );
            mix_surface(
                world_index,
                contact_id,
                shape_id_a,
                shape_id_b,
                thread,
                xf_a,
                xf_b,
                material_map,
                false,
                count,
                true,
                child_radius,
            );
            finish_contact(world_index, thread, contact_id, count, hit);
            return;
        }
        let mut shape_a = read_shape(type_a, geom_a, 0);
        let mut shape_b = read_shape(type_b, shape_b, 48);
        let flip = (type_a == TY_SPHERE && type_b != TY_SPHERE)
            || (type_a == TY_CAPSULE && type_b == TY_HULL);
        let (convex_xf_a, convex_xf_b) = if flip {
            core::mem::swap(&mut shape_a, &mut shape_b);
            (xf_b, xf_a)
        } else {
            (xf_a, xf_b)
        };

        let base = read_dir(dir, contact_id).manifold_base;
        let uses_sat = type_a == TY_HULL && type_b == TY_HULL;
        let uses_simplex = (type_a == TY_HULL || type_b == TY_HULL) && !uses_sat;
        let mut cache = if uses_sat {
            ConvexContactCache::Sat(read_sat(dir, contact_id))
        } else if uses_simplex {
            ConvexContactCache::Simplex(read_simplex(dir, contact_id))
        } else {
            ConvexContactCache::empty()
        };

        let mut address = base;
        let touching = compute_convex_manifold_into(
            || {
                if old_count == 0 {
                    address = manifolds::allocate_manifolds_in_world(world_index, contact_id, 1);
                }
                &mut *(address as *mut ManifoldRecord)
            },
            &shape_a,
            convex_xf_a,
            &shape_b,
            convex_xf_b,
            &mut cache,
        );
        if touching {
            let m = &mut *(address as *mut ManifoldRecord);
            if flip {
                m.normal = m.normal.neg();
                for p in &mut m.points[..m.point_count as usize] {
                    core::mem::swap(&mut p.anchor_a, &mut p.anchor_b);
                }
            }
            for p in &mut m.points[..m.point_count as usize] {
                p.anchor_a = p.anchor_a.add(child_offset);
            }
            for p in &mut m.points[..m.point_count as usize] {
                p.anchor_a = p.anchor_a.sub(center_a);
                p.anchor_b = p.anchor_b.sub(center_b);
                p.base_separation = p.separation;
            }
        }
        match &cache {
            ConvexContactCache::Sat(cache) => write_sat(dir, contact_id, cache),
            ConvexContactCache::Simplex(cache) => write_simplex(dir, contact_id, cache),
            ConvexContactCache::Empty => {}
        }
        mix_surface(
            world_index,
            contact_id,
            shape_id_a,
            shape_id_b,
            thread,
            xf_a,
            xf_b,
            material_map,
            flip,
            touching as usize,
            false,
            child_radius,
        );
        // The pre-solve callback belongs here when stage 7 publishes it.
        finish_contact(world_index, thread, contact_id, touching as usize, hit);
    }
}

/// Run the same contact tasks on the calling thread when the sweep does not fork.
#[export_name = "dispatchContacts"]
pub extern "C" fn dispatch_contacts(count: usize) {
    dispatch_contacts_in_world(crate::regions::active(), count)
}

pub extern "C" fn dispatch_contacts_in_world(world_index: usize, count: usize) {
    unsafe { contact_block(world_index, 0, count, count, 0) }
}

// --- contact recycle -----------------------------------------------------------------------
// Recycle poses resolve body ids through the resident records across sleep/wake transitions.

/// Do shapes `sa` and `sb`'s fat AABBs overlap? (b3AABB_Overlaps over the resident fat-AABB column;
/// bit-identical to `src/math.ts` `aabb.overlaps` — the same six comparisons.)
#[inline]
fn fat_overlap(fat: &[f32], sa: usize, sb: usize) -> bool {
    let a = sa * crate::shapes::SHAPE_STRIDE + crate::shapes::S_FAT_AABB;
    let b = sb * crate::shapes::SHAPE_STRIDE + crate::shapes::S_FAT_AABB;
    !(fat[a + 3] < fat[b]
        || fat[a] > fat[b + 3]
        || fat[a + 4] < fat[b + 1]
        || fat[a + 1] > fat[b + 4]
        || fat[a + 5] < fat[b + 2]
        || fat[a + 2] > fat[b + 5])
}

/// Read a contact's cached relative pose (last full narrowphase) from the directory recycle record.
#[inline]
fn read_pose_cache(dir: Col<u32>, contact_id: usize) -> (Quat, Quat, Transform) {
    let o = contact_id * DIR_STRIDE;
    let rot_a = Quat {
        v: Vec3::new(
            f32::from_bits(dir.get(o + DIR_CACHED_ROT_A)),
            f32::from_bits(dir.get(o + DIR_CACHED_ROT_A + 1)),
            f32::from_bits(dir.get(o + DIR_CACHED_ROT_A + 2)),
        ),
        s: f32::from_bits(dir.get(o + DIR_CACHED_ROT_A + 3)),
    };
    let rot_b = Quat {
        v: Vec3::new(
            f32::from_bits(dir.get(o + DIR_CACHED_ROT_B)),
            f32::from_bits(dir.get(o + DIR_CACHED_ROT_B + 1)),
            f32::from_bits(dir.get(o + DIR_CACHED_ROT_B + 2)),
        ),
        s: f32::from_bits(dir.get(o + DIR_CACHED_ROT_B + 3)),
    };
    let rel = Transform {
        p: Vec3::new(
            f32::from_bits(dir.get(o + DIR_CACHED_REL_POSE)),
            f32::from_bits(dir.get(o + DIR_CACHED_REL_POSE + 1)),
            f32::from_bits(dir.get(o + DIR_CACHED_REL_POSE + 2)),
        ),
        q: Quat {
            v: Vec3::new(
                f32::from_bits(dir.get(o + DIR_CACHED_REL_POSE + 3)),
                f32::from_bits(dir.get(o + DIR_CACHED_REL_POSE + 4)),
                f32::from_bits(dir.get(o + DIR_CACHED_REL_POSE + 5)),
            ),
            s: f32::from_bits(dir.get(o + DIR_CACHED_REL_POSE + 6)),
        },
    };
    (rot_a, rot_b, rel)
}

/// Cache this step's pose into the directory recycle record for the next step's recycle test — the
/// column-resident equivalent of the TS `contact.cachedRotation*`/`cachedRelativePose` writes.
#[inline]
fn write_pose_cache(dir: Col<u32>, contact_id: usize, xf_a: Transform, xf_b: Transform) {
    let o = contact_id * DIR_STRIDE;
    dir.set(o + DIR_CACHED_ROT_A, xf_a.q.v.x.to_bits());
    dir.set(o + DIR_CACHED_ROT_A + 1, xf_a.q.v.y.to_bits());
    dir.set(o + DIR_CACHED_ROT_A + 2, xf_a.q.v.z.to_bits());
    dir.set(o + DIR_CACHED_ROT_A + 3, xf_a.q.s.to_bits());
    dir.set(o + DIR_CACHED_ROT_B, xf_b.q.v.x.to_bits());
    dir.set(o + DIR_CACHED_ROT_B + 1, xf_b.q.v.y.to_bits());
    dir.set(o + DIR_CACHED_ROT_B + 2, xf_b.q.v.z.to_bits());
    dir.set(o + DIR_CACHED_ROT_B + 3, xf_b.q.s.to_bits());
    let rel = xf_a.inv_mul(xf_b);
    dir.set(o + DIR_CACHED_REL_POSE, rel.p.x.to_bits());
    dir.set(o + DIR_CACHED_REL_POSE + 1, rel.p.y.to_bits());
    dir.set(o + DIR_CACHED_REL_POSE + 2, rel.p.z.to_bits());
    dir.set(o + DIR_CACHED_REL_POSE + 3, rel.q.v.x.to_bits());
    dir.set(o + DIR_CACHED_REL_POSE + 4, rel.q.v.y.to_bits());
    dir.set(o + DIR_CACHED_REL_POSE + 5, rel.q.v.z.to_bits());
    dir.set(o + DIR_CACHED_REL_POSE + 6, rel.q.s.to_bits());
}

/// Update each contact in Box3D's collide-task order.
///
/// # Safety
/// Contacts run in disjoint tasks; bodies and shapes stay resident for the fork.
pub(crate) unsafe fn contact_block(
    world_index: usize,
    start: usize,
    end: usize,
    total: usize,
    thread: usize,
) {
    unsafe {
        use crate::manifold_abi::*;
        let contacts = core::slice::from_raw_parts(CONTACT_LIST_PTR as *const u32, total);
        let dir = manifolds::dir_col(world_index);
        let pool = manifolds::pool_col();
        let fat = crate::shapes::col_f_slice(world_index);
        let shapes = crate::shapes::col_slice(world_index);
        let recycle_dist = RECYCLE_DISTANCE;
        let recycle_dist_non_touching = recycle_dist.min(0.02);

        for i in start..end {
            let contact_id = contacts[i] as usize;
            let o = contact_id * DIR_STRIDE;
            let flags = dir.get(o + DIR_FLAGS) & !0x0200_0000;
            dir.set(o + DIR_FLAGS, flags);
            let sa = dir.get(o + DIR_SHAPE_A) as usize;
            let sb = dir.get(o + DIR_SHAPE_B) as usize;
            if !fat_overlap(fat, sa, sb) {
                dir.set(o + DIR_FLAGS, (flags | 0x0002_0000) & !0x0001_0000);
                mark_contact_state(world_index, thread, contact_id);
                continue;
            }

            let sa = &shapes[sa * crate::shapes::SHAPE_STRIDE..][..crate::shapes::SHAPE_STRIDE];
            let sb = &shapes[sb * crate::shapes::SHAPE_STRIDE..][..crate::shapes::SHAPE_STRIDE];
            let la = sa[crate::shapes::S_QUERY_BODY] as usize;
            let lb = sb[crate::shapes::S_QUERY_BODY] as usize;
            let body_a = crate::bodies::record(world_index, la);
            let body_b = crate::bodies::record(world_index, lb);
            let static_a = body_a.body_type == 0;
            let static_b = body_b.body_type == 0;
            dir.set(
                o + DIR_INDEX_A,
                if static_a {
                    u32::MAX
                } else {
                    body_a.local_index as u32
                },
            );
            dir.set(
                o + DIR_INDEX_B,
                if static_b {
                    u32::MAX
                } else {
                    body_b.local_index as u32
                },
            );
            let (xf_a, fin_a, flags_a) = crate::bodies::geometry(world_index, la);
            let (xf_b, fin_b, flags_b) = crate::bodies::geometry(world_index, lb);
            let center_a = fin_a.center;
            let center_b = fin_b.center;
            let extent_a = if static_a {
                Vec3::ZERO
            } else {
                fin_a.max_extent
            };
            let extent_b = if static_b {
                Vec3::ZERO
            } else {
                fin_b.max_extent
            };
            let fast = (flags_a | flags_b) & 0x40 != 0;
            let fast_mesh = flags & 0x0040_0000 != 0 && fast;
            let tol = if flags & 0x0001_0000 != 0 {
                recycle_dist
            } else {
                recycle_dist_non_touching
            };

            if !fast_mesh && recycle_dist > 0.0 && flags & 0x0080_0000 != 0 && flags & 0x10 != 0 {
                let (rot_a, rot_b, rel) = read_pose_cache(dir, contact_id);
                let mc = dir.get(o + DIR_MANIFOLD_COUNT) as usize;
                let angle_a = xf_a.q.dot(rot_a);
                let angle_b = xf_b.q.dot(rot_b);
                let angular_distance = crate::math::minf(angle_a * angle_a, angle_b * angle_b);
                let xf = xf_a.inv_mul(xf_b);
                let max_extent = Vec3::new(
                    crate::math::maxf(extent_a.x, extent_b.x),
                    crate::math::maxf(extent_a.y, extent_b.y),
                    crate::math::maxf(extent_a.z, extent_b.z),
                );
                let dv = rel.p.sub(xf.p);
                let dist_squared = dv.dot(dv);
                if angular_distance > crate::recycle::RECYCLE_ANGULAR_DISTANCE
                    && dist_squared < tol * tol
                {
                    let distance = dist_squared.sqrt();
                    let slack = tol - distance;
                    let qr = rel.q.inv_mul(xf.q);
                    let arc = qr.v.abs().modified_cross(max_extent);
                    let arc_sq = 4.0 * arc.length_sq();
                    if arc_sq < slack * slack {
                        let dq_a = xf_a.q.mul(rot_a.conjugate());
                        let dq_b = xf_b.q.mul(rot_b.conjugate());
                        let matrix_a = crate::math::Mat3::from_quat(dq_a);
                        let matrix_b = crate::math::Mat3::from_quat(dq_b);
                        let dc = center_b.sub(center_a);
                        let base = dir.get(o + DIR_MANIFOLD_BASE) as usize;
                        let manifolds = crate::manifold_abi::block_col(pool, base, mc);
                        for m in 0..mc {
                            let mo = m * MANIFOLD_STRIDE;
                            let normal = Vec3::new(
                                manifolds.get(mo + M_NORMAL),
                                manifolds.get(mo + M_NORMAL + 1),
                                manifolds.get(mo + M_NORMAL + 2),
                            );
                            let pc = manifolds.get(mo + M_POINT_COUNT).to_bits() as usize;
                            for p in 0..pc {
                                let po = mo + M_POINTS + p * POOL_POINT_STRIDE;
                                let anchor_a = Vec3::new(
                                    manifolds.get(po + P_ANCHOR_A),
                                    manifolds.get(po + P_ANCHOR_A + 1),
                                    manifolds.get(po + P_ANCHOR_A + 2),
                                );
                                let anchor_b = Vec3::new(
                                    manifolds.get(po + P_ANCHOR_B),
                                    manifolds.get(po + P_ANCHOR_B + 1),
                                    manifolds.get(po + P_ANCHOR_B + 2),
                                );
                                let r_a = matrix_a.mul_v(anchor_a);
                                let r_b = matrix_b.mul_v(anchor_b);
                                let dp = dc.add(r_b.sub(r_a));
                                manifolds.set(
                                    po + P_SEPARATION,
                                    manifolds.get(po + P_BASE_SEPARATION) + dp.dot(normal),
                                );
                                manifolds.set(po + P_PERSISTED, f32::from_bits(1));
                            }
                        }
                        continue;
                    }
                }
            }

            write_pose_cache(dir, contact_id, xf_a, xf_b);
            dir.set(o + DIR_FLAGS, flags | 0x0080_0000);
            update_contact(
                world_index,
                thread,
                contact_id,
                sa,
                sb,
                xf_a,
                fin_a.local_center,
                xf_b,
                fin_b.local_center,
                fast,
            );
            let updated = dir.get(o + DIR_FLAGS);
            if flags & 0x0001_0000 != 0 && updated & 0x0001_0000 != 0 && updated & 0x0040_0000 != 0
            {
                crate::constraint_graph::update_manifold_count(
                    world_index,
                    dir.get(o + DIR_COLOR_INDEX) as usize,
                    dir.get(o + DIR_LOCAL_INDEX) as usize,
                    dir.get(o + DIR_MANIFOLD_COUNT) as u16,
                );
            }
        }
    }
}

// --- finalize -------------------------------------------------------------------------------

/// Advance the bodies in `[start, end)`. One block of the parallel sweep (`parfor.rs`), or every awake
/// body on the serial path. Each body reads and writes only its own records, so the blocks are
/// write-disjoint.
///
/// # Safety
/// The body columns must be reserved for `body_records()`, and no thread may grow memory while this runs.
pub(crate) unsafe fn finalize_block(
    world_index: usize,
    start: usize,
    end: usize,
    h: f32,
    inv_dt: f32,
    enable_continuous: bool,
) {
    unsafe {
        let b = body_records(world_index);
        let state = f32s(STATE, b * STATE_STRIDE);
        let sim = f32s(SIM, b * SIM_STRIDE);
        let fin = f32s(FIN, b * FIN_STRIDE);
        let flags = u32s(FLAGS, b * STATE_STRIDE);
        let sim2 = Col::new(
            crate::bodies::sim2_base(world_index) as *mut f32,
            b * SIM2_STRIDE,
        );
        finalize::finalize(
            world_index,
            state,
            sim,
            fin,
            sim2,
            flags,
            start,
            end - start,
            h,
            inv_dt,
            enable_continuous,
        );
    }
}

/// Commit each non-fast body's shape bounds and enlarge flags in the finalize task.
/// Fast non-bullets already committed in continuous; bullets commit during their deferred sweep.
/// Tree mutation stays in the serial enlarge pass, as in Box3D.
///
/// Per-body write-disjoint: a shape belongs to one body, so two parallel-for blocks never write the same
/// shape record — the shared-mutable [`Col`] carries that promise. Indexed only through the awake head
/// lane → `next` chain, never a `0..cap` sweep, so it never reads a stale record outside a live chain
/// (`shapes.rs` reachability contract).
///
/// # Safety
/// The body + shape + fat-AABB regions must be reserved for every reachable shape, and no thread may grow
/// memory while this runs (the MT concurrency invariant).
pub(crate) unsafe fn refit_body(world_index: usize, sim: Col<f32>, fin: Col<f32>, i: usize) {
    unsafe {
        let records = crate::bodies::body_cap_in_world(world_index);
        let sim2 = Col::new(
            crate::bodies::sim2_base(world_index) as *mut u32,
            records * SIM2_STRIDE,
        );
        let shape_u = crate::shapes::col(world_index);
        let shape_f = crate::shapes::col_f(world_index);
        let fat = crate::shapes::col_f(world_index);
        {
            if sim2.atomic_get(i * SIM2_STRIDE + crate::body::S2_FLAGS) & 0x40 != 0 {
                return;
            }
            let so = i * SIM_STRIDE;
            let fo = i * FIN_STRIDE;
            let xf = Transform {
                p: Vec3::new(fin.get(fo), fin.get(fo + 1), fin.get(fo + 2)),
                q: Quat {
                    v: Vec3::new(sim.get(so + 3), sim.get(so + 4), sim.get(so + 5)),
                    s: sim.get(so + 6),
                },
            };
            let body_id = sim2.get(i * SIM2_STRIDE + crate::body::S2_BODY_ID) as usize;
            let mut shape_id = crate::bodies::record(world_index, body_id).head_shape_id as u32;
            while shape_id != crate::shapes::NULL_SHAPE {
                let o = shape_id as usize * crate::shapes::SHAPE_STRIDE;
                let bounds = crate::continuous::bounds(world_index, shape_id as usize, xf);
                let fb = o + crate::shapes::S_FAT_AABB;
                let fat_aabb = core::array::from_fn(|n| fat.get(fb + n));
                let (cand, escaped) = finalize::refit_bounds(bounds, &fat_aabb);
                for n in 0..6 {
                    shape_f.set(o + 10 + n, cand[n]);
                }
                let flags = shape_u.get(o + crate::shapes::S_FLAGS);
                shape_u.set(
                    o + crate::shapes::S_FLAGS,
                    (flags & !crate::shapes::ENLARGED_FLAG)
                        | if escaped {
                            crate::shapes::ENLARGED_FLAG
                        } else {
                            0
                        },
                );
                if escaped {
                    let margin = shape_f.get(o + 9);
                    for n in 0..3 {
                        fat.set(fb + n, cand[n] - margin);
                        fat.set(fb + 3 + n, cand[3 + n] + margin);
                    }
                }
                shape_id = shape_u.get(o + crate::shapes::S_NEXT);
            }
        }
    }
}

//! The shared-column arena and phase export shims — the kernel's wasm surface.
//!
//! `reserve` lays out solver scratch in one allocator-owned shared buffer,
//! growing the memory to fit, and records each column's byte offset in the `LAYOUT` header. The TS
//! side reads `layoutPtr` and derives `Float32Array`/`Uint32Array` views over the columns, then drives
//! the solve one phase at a time through the export shims below. Each shim rebuilds the columns from
//! `LAYOUT` + the reserved counts and calls into the phase module (`integrate`, `contact`,
//! `finalize`), which is where the arithmetic — already gold-verified against the C reference — lives.
//!
//! Wasm-only: the columns alias linear memory directly, so this is meaningful only in the JS host
//! (native tests drive the phase modules against their gold vectors instead). They are shared-mutable
//! [`Col`]s rather than `&mut` slices — `col.rs` carries the argument.

use crate::body::{FIN_STRIDE, SIM2_STRIDE, SIM_STRIDE, STATE_STRIDE};
use crate::col::Col;
use crate::contact::{Columns, CC_META_STRIDE, CC_STRIDE, MCP_STRIDE, MC_META_STRIDE, MC_STRIDE};
use crate::contact_wide::{WIDE_IDX_STRIDE, WIDE_META_STRIDE, WIDE_STRIDE};
use crate::distance::SimplexCache;
use crate::finalize::{self, TY_CAPSULE, TY_HULL, TY_SPHERE};
use crate::manifold::{Capsule, SatCache, Sphere};
use crate::manifold_abi::{
    read_dir, DIR_CACHE, DIR_CACHED_REL_POSE, DIR_CACHED_ROT_A, DIR_CACHED_ROT_B, DIR_FLAGS,
    DIR_STRIDE, MANIFOLD_STRIDE, M_FRICTION, M_NORMAL, M_POINTS, M_POINT_COUNT, M_ROLLING, M_TWIST,
    POOL_POINT_STRIDE, P_ANCHOR_A, P_ANCHOR_B, P_FEATURE_ID, P_NORMAL_IMPULSE, P_NORMAL_VELOCITY,
    P_PERSISTED, P_SEPARATION, P_TOTAL_NORMAL_IMPULSE, P_TRIANGLE_INDEX, SLOT_STRIDE,
};
use crate::manifolds;
use crate::math::{Quat, Transform, Vec3};
use crate::narrowphase::{
    compute_convex_manifold, ConvexContactCache, ConvexShape, Manifold, MAX_MANIFOLD_POINTS,
};

use crate::geo::hull_view;
use crate::regions::Buffer;

static mut SCRATCH: Buffer = Buffer::EMPTY;
pub unsafe fn reserve_scratch(bytes: usize) -> usize {
    let scratch = &mut *(&raw mut SCRATCH);
    scratch.reserve(bytes);
    scratch.ptr
}
const N_COLS: usize = 15;
/// Active-color span: wideStart, wideCount, meshStart, meshCount, jointArrayKey, jointCount.
/// The staged solve selects the resident joint array by key.
pub(crate) const COLOR_SPAN_STRIDE: usize = 6;

/// The worker index the serial (single-crossing) shims run as — the thread driving the step is always
/// worker 0 (`stages::run`). It selects the null-lane identity record the wide gather/scatter writes.
const ORCHESTRATOR: usize = 0;

// LAYOUT indices, in memory order.
const STATE: usize = 0;
const FLAGS: usize = 1;
const SIM: usize = 2;
const FIN: usize = 3;
// Per scalar solver-record slot: contactId + transient mc/mcp bases (the narrowphase → solver map;
// the persistent directory + pool it points into live in the manifold region, manifolds.rs).
const SLOT_SCALAR: usize = 5;
const CC: usize = 6;
const CC_META: usize = 7;
const MC: usize = 8;
const MC_META: usize = 9;
const MCP: usize = 10;
// Wide (convex) transient constraint columns — the 4-lane contact solver's records + lane maps.
const WIDE: usize = 11;
const WIDE_META: usize = 12;
const WIDE_IDX: usize = 13;
// Per-active-color spans (wide/mesh/joint start+count) for the batched color loop + staged solve.
const COLOR_SPAN: usize = 14;

/// Per-column byte offsets into linear memory, rewritten by every `reserve`. TS reads this header
/// (`layoutPtr`) to build its column views.
static mut LAYOUT: [u32; N_COLS] = [0; N_COLS];

// The per-step counts the shims size their slices from (set by `reserve`).
static mut BODY_COUNT: usize = 0;
static mut CONTACT_COUNT: usize = 0;
static mut MANIFOLD_COUNT: usize = 0;
static mut POINT_COUNT: usize = 0;
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
        CONTACT_COUNT = contact;
        MANIFOLD_COUNT = manifold;
        POINT_COUNT = point;
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
        LAYOUT[SLOT_SCALAR] = off as u32;
        off += contact * SLOT_STRIDE * 4;
        LAYOUT[CC] = off as u32;
        off += contact * CC_STRIDE * 4;
        LAYOUT[CC_META] = off as u32;
        off += contact * CC_META_STRIDE * 4;
        LAYOUT[MC] = off as u32;
        off += manifold * MC_STRIDE * 4;
        LAYOUT[MC_META] = off as u32;
        off += manifold * MC_META_STRIDE * 4;
        LAYOUT[MCP] = off as u32;
        off += point * MCP_STRIDE * 4;
        LAYOUT[WIDE] = off as u32;
        off += wide * WIDE_STRIDE * 4;
        LAYOUT[WIDE_META] = off as u32;
        off += wide * WIDE_META_STRIDE * 4;
        LAYOUT[WIDE_IDX] = off as u32;
        off += wide * WIDE_IDX_STRIDE * 4;
        LAYOUT[COLOR_SPAN] = off as u32;
        off += color * COLOR_SPAN_STRIDE * 4;

        let continuous_offset = off;
        off += body * crate::continuous::STRIDE * 4;
        reserve_scratch(off);
        for column in SLOT_SCALAR..N_COLS {
            LAYOUT[column] += SCRATCH.ptr as u32;
        }
        crate::continuous::reserve_at(SCRATCH.ptr + continuous_offset, body);
    }
}

/// Records the resident body columns hold: the awake bodies plus the per-thread identity records the
/// wide gather remaps null lanes onto (`bodies::reserve_bodies` lays out `cap + IDENT_RECORDS`). The
/// body columns are sized by that, not by `BODY_COUNT`, so a column's `len` bounds every element its
/// phases can reach — the wide gather reaches an identity record, which sits past the awake count.
#[inline]
unsafe fn body_records(world_index: usize) -> usize {
    crate::bodies::body_cap_in_world(world_index) + crate::bodies::IDENT_RECORDS
}

/// All the scalar solver's columns over the current reservation. The body + slot + transient columns
/// are disjoint byte ranges of the per-step solver region; the directory + pool live in the persistent
/// manifold region (`manifolds`) — also disjoint. Every one is a shared-mutable [`Col`]: the phases
/// index them by body / record id and, under the staged solver, do so from several threads at once.
unsafe fn columns(world_index: usize) -> Columns<'static> {
    let b = body_records(world_index);
    let c = CONTACT_COUNT;
    let m = MANIFOLD_COUNT;
    let p = POINT_COUNT;
    Columns {
        state: f32s(STATE, b * STATE_STRIDE),
        flags: u32s(FLAGS, b * STATE_STRIDE),
        sim: f32s(SIM, b * SIM_STRIDE),
        slot: u32s(SLOT_SCALAR, c * SLOT_STRIDE),
        dir: manifolds::dir_col(world_index),
        pool: manifolds::pool_col(),
        cc: f32s(CC, c * CC_STRIDE),
        cc_meta: u32s(CC_META, c * CC_META_STRIDE),
        mc: f32s(MC, m * MC_STRIDE),
        mc_meta: u32s(MC_META, m * MC_META_STRIDE),
        mcp: f32s(MCP, p * MCP_STRIDE),
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

/// The wide solver's transient columns: records, lane→body index map, lane→contact meta.
pub(crate) unsafe fn wide_columns() -> (Col<'static, f32>, Col<'static, u32>, Col<'static, u32>) {
    let w = WIDE_COUNT;
    (
        f32s(WIDE, w * WIDE_STRIDE),
        u32s(WIDE_IDX, w * WIDE_IDX_STRIDE),
        u32s(WIDE_META, w * WIDE_META_STRIDE),
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
static mut MESH_OUTPUT_PTR: usize = 0;
static mut MESH_MATERIAL_PTR: usize = 0;
static mut MESH_SCRATCH_PTR: usize = 0;
struct DispatchScratch {
    mesh: crate::mesh_contact::MeshStorage,
    old: [Manifold; 256],
    previous: [crate::manifold_abi::ContactCache; 256],
}

static mut CONTACT_LIST_PTR: usize = 0;
static mut CONTACT_STATES: [Vec<crate::bitset::BitSet>; crate::regions::MAX_WORLDS] =
    [const { Vec::new() }; crate::regions::MAX_WORLDS];

pub(crate) unsafe fn reset_contact_states(world: usize) {
    CONTACT_STATES[world] = Vec::new();
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
        let mut off = (count * 4 + 15) & !15;
        let mesh_off = off;
        off += mesh_threads * 256 * MANIFOLD_STRIDE * 4;
        let material_off = off;
        off += mesh_threads * 256 * 4 * 4;
        off = (off + 15) & !15;
        let scratch_off = off;
        off += mesh_threads * core::mem::size_of::<DispatchScratch>();
        reserve_scratch(off);
        CONTACT_LIST_PTR = SCRATCH.ptr;
        MESH_OUTPUT_PTR = SCRATCH.ptr + mesh_off;
        MESH_MATERIAL_PTR = SCRATCH.ptr + material_off;
        MESH_SCRATCH_PTR = SCRATCH.ptr + scratch_off;
        (MESH_SCRATCH_PTR as *mut u8)
            .write_bytes(0, mesh_threads * core::mem::size_of::<DispatchScratch>());
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

/// Read a contact's resident manifold's warm-start state (point count + per-point feature id + normal
/// impulse) — the only fields `compute_convex_manifold` reads from the old manifold; the rest it
/// overwrites.
#[inline]
fn read_manifold_warm(pool: Col<f32>, base: usize) -> Manifold {
    let o = base * MANIFOLD_STRIDE;
    let mut m = Manifold::new();
    let pc = (pool.get(o + M_POINT_COUNT).to_bits() as usize).min(MAX_MANIFOLD_POINTS);
    m.point_count = pc;
    for j in 0..pc {
        let p = o + M_POINTS + j * POOL_POINT_STRIDE;
        m.points[j].feature_id = pool.get(p + P_FEATURE_ID).to_bits();
        m.points[j].normal_impulse = pool.get(p + P_NORMAL_IMPULSE);
    }
    m
}

/// Write the computed manifold into the pool block. Only the narrowphase-owned fields — normal, point
/// count, and per-point anchors/separation/impulses/feature id/triangle index/persisted — the header
/// friction/twist/rolling (solver-owned, persistent) are left untouched. Separation and its recycle
/// baseline are written together after shifting anchors to the centers of mass.
#[inline]
fn write_manifold(m: &Manifold, pool: Col<f32>, base: usize) {
    let o = base * MANIFOLD_STRIDE;
    pool.set(o + M_NORMAL, m.normal.x);
    pool.set(o + M_NORMAL + 1, m.normal.y);
    pool.set(o + M_NORMAL + 2, m.normal.z);
    pool.set(o + M_POINT_COUNT, f32::from_bits(m.point_count as u32));
    for j in 0..m.point_count {
        let p = o + M_POINTS + j * POOL_POINT_STRIDE;
        let pt = &m.points[j];
        pool.set(p + P_ANCHOR_A, pt.anchor_a.x);
        pool.set(p + P_ANCHOR_A + 1, pt.anchor_a.y);
        pool.set(p + P_ANCHOR_A + 2, pt.anchor_a.z);
        pool.set(p + P_ANCHOR_B, pt.anchor_b.x);
        pool.set(p + P_ANCHOR_B + 1, pt.anchor_b.y);
        pool.set(p + P_ANCHOR_B + 2, pt.anchor_b.z);
        pool.set(p + P_SEPARATION, pt.separation);
        pool.set(p + crate::manifold_abi::P_BASE_SEPARATION, pt.separation);
        pool.set(p + P_NORMAL_IMPULSE, pt.normal_impulse);
        pool.set(p + P_TOTAL_NORMAL_IMPULSE, pt.total_normal_impulse);
        pool.set(p + P_NORMAL_VELOCITY, pt.normal_velocity);
        pool.set(p + P_FEATURE_ID, f32::from_bits(pt.feature_id));
        pool.set(
            p + P_TRIANGLE_INDEX,
            f32::from_bits(pt.triangle_index as u32),
        );
        unsafe {
            pool.ptr()
                .add(p + P_PERSISTED)
                .cast::<bool>()
                .write(pt.persisted);
        }
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
    use crate::mesh_contact::{compute_mesh_manifolds, TriangleSource, MAX_TRIANGLES};
    let slot = thread;
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
    let scratch = &mut *(MESH_SCRATCH_PTR as *mut DispatchScratch).add(thread);
    cache.refresh(&source, xf_a, vec(10), vec(13), &mut scratch.previous);
    let dir = manifolds::dir_col(world_index);
    let pool = manifolds::pool_col();
    let entry = read_dir(dir, contact_id);
    let pool = crate::manifold_abi::block_col(pool, entry.manifold_base, old_count);
    for i in 0..old_count {
        let o = i * MANIFOLD_STRIDE;
        scratch.old[i] = Manifold::new();
        let m = &mut scratch.old[i];
        m.normal = Vec3::new(
            pool.get(o + M_NORMAL),
            pool.get(o + M_NORMAL + 1),
            pool.get(o + M_NORMAL + 2),
        );
        m.friction_impulse = Vec3::new(
            pool.get(o + M_FRICTION),
            pool.get(o + M_FRICTION + 1),
            pool.get(o + M_FRICTION + 2),
        );
        m.rolling_impulse = Vec3::new(
            pool.get(o + M_ROLLING),
            pool.get(o + M_ROLLING + 1),
            pool.get(o + M_ROLLING + 2),
        );
        m.twist_impulse = pool.get(o + M_TWIST);
        m.point_count = pool.get(o + M_POINT_COUNT).to_bits() as usize;
        for j in 0..m.point_count {
            let p = o + M_POINTS + j * POOL_POINT_STRIDE;
            m.points[j].normal_impulse = pool.get(p + P_NORMAL_IMPULSE);
            m.points[j].feature_id = pool.get(p + P_FEATURE_ID).to_bits();
            m.points[j].triangle_index = pool.get(p + P_TRIANGLE_INDEX).to_bits() as i32;
        }
    }
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
    let mut mesh = scratch
        .mesh
        .scratch(cache.triangles.count as usize, old_count);
    let count = compute_mesh_manifolds(
        &mut mesh,
        cache.triangles.as_mut_slice(),
        |index| source.triangle(index),
        &shape,
        xf_a,
        xf_b,
        fast,
        speculative,
        &mut scratch.old[..old_count],
    );
    let output_ptr = (MESH_OUTPUT_PTR as *mut f32).add(slot * MAX_TRIANGLES * MANIFOLD_STRIDE);
    output_ptr.write_bytes(0, count * MANIFOLD_STRIDE);
    let output = Col::new(output_ptr, MAX_TRIANGLES * MANIFOLD_STRIDE);
    let materials = (MESH_MATERIAL_PTR as *mut u32).add(slot * MAX_TRIANGLES * 4);
    for i in 0..count {
        let m = &mut mesh.output[i];
        for p in &mut m.points[..m.point_count] {
            p.anchor_a = p.anchor_a.add(child_offset);
        }
        for p in &mut m.points[..m.point_count] {
            p.anchor_a = p.anchor_a.sub(center_a);
            p.anchor_b = p.anchor_b.sub(center_b);
        }
        write_manifold(m, output, i);
        let o = i * MANIFOLD_STRIDE;
        output.set(o + M_FRICTION, m.friction_impulse.x);
        output.set(o + M_FRICTION + 1, m.friction_impulse.y);
        output.set(o + M_FRICTION + 2, m.friction_impulse.z);
        output.set(o + M_ROLLING, m.rolling_impulse.x);
        output.set(o + M_ROLLING + 1, m.rolling_impulse.y);
        output.set(o + M_ROLLING + 2, m.rolling_impulse.z);
        output.set(o + M_TWIST, m.twist_impulse);
        for j in 0..m.point_count {
            *materials.add(i * 4 + j) = mesh.materials[i][j];
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
        let slot = thread;
        let output = Col::new(
            (MESH_OUTPUT_PTR as *mut f32).add(slot * 256 * MANIFOLD_STRIDE),
            256 * MANIFOLD_STRIDE,
        );
        let materials = (MESH_MATERIAL_PTR as *const u32).add(slot * 256 * 4);
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
        let pool = manifolds::pool_col();
        let o = contact_id * DIR_STRIDE;
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
            if count > 0 {
                let address = if count == old_count {
                    let address = dir.get(o + DIR_MANIFOLD_BASE) as usize;
                    (address as *mut u8).write_bytes(0, count * MANIFOLD_STRIDE * 4);
                    address
                } else {
                    manifolds::allocate_manifolds_in_world(world_index, contact_id, count)
                };
                let source = MESH_OUTPUT_PTR + thread * 256 * MANIFOLD_STRIDE * 4;
                manifolds::copy_manifolds(source, address, count);
            }
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

        let resident = old_count != 0;
        let mut m = if resident {
            read_manifold_warm(crate::manifold_abi::block_col(pool, base, 1), 0)
        } else {
            Manifold::new()
        };
        let touching = compute_convex_manifold(
            &mut m,
            &shape_a,
            convex_xf_a,
            &shape_b,
            convex_xf_b,
            &mut cache,
        );
        if flip {
            m.normal = m.normal.neg();
            for p in &mut m.points[..m.point_count] {
                core::mem::swap(&mut p.anchor_a, &mut p.anchor_b);
            }
        }
        for p in &mut m.points[..m.point_count] {
            p.anchor_a = p.anchor_a.add(child_offset);
        }
        for p in &mut m.points[..m.point_count] {
            p.anchor_a = p.anchor_a.sub(center_a);
            p.anchor_b = p.anchor_b.sub(center_b);
        }
        if touching {
            let address = if resident {
                base
            } else {
                manifolds::allocate_manifolds_in_world(world_index, contact_id, 1)
            };
            write_manifold(&m, crate::manifold_abi::block_col(pool, address, 1), 0);
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
            let flags = dir.get(o + DIR_FLAGS);
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
        let sim2_u = Col::new(
            crate::bodies::sim2_base(world_index) as *mut u32,
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
        for i in start..end {
            crate::events::write_move(world_index, i);
        }
        // Continuous can clip the rotation. Box3D rebuilds inertia from the resulting pose,
        // not the discrete candidate; non-fast bodies already have that tensor.
        for i in start..end {
            if sim2_u.atomic_get(i * SIM2_STRIDE + crate::body::S2_FLAGS) & 0x40 != 0 {
                let body = crate::body::read_sim(sim, i);
                let rotation = crate::math::Mat3::from_quat(body.rotation);
                crate::body::write_sim_inv_inertia_world(
                    sim,
                    i,
                    rotation
                        .mul(body.inv_inertia_local)
                        .mul(rotation.transpose()),
                );
            }
        }
        refit_block(world_index, sim, fin, start, end);
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
unsafe fn refit_block(world_index: usize, sim: Col<f32>, fin: Col<f32>, start: usize, end: usize) {
    unsafe {
        let records = crate::bodies::body_cap_in_world(world_index) + crate::bodies::IDENT_RECORDS;
        let sim2 = Col::new(
            crate::bodies::sim2_base(world_index) as *mut u32,
            records * SIM2_STRIDE,
        );
        let shape_u = crate::shapes::col(world_index);
        let shape_f = crate::shapes::col_f(world_index);
        let fat = crate::shapes::col_f(world_index);
        for i in start..end {
            if sim2.atomic_get(i * SIM2_STRIDE + crate::body::S2_FLAGS) & 0x40 != 0 {
                continue;
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

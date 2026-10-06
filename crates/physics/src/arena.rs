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

use crate::body::{
    FIN_OUT_STRIDE, FIN_STRIDE, S2_BODY_ID, S2_HEAD_SHAPE, SIM2_STRIDE, SIM_STRIDE, STATE_STRIDE,
};
use crate::col::Col;
use crate::contact::{Columns, CC_META_STRIDE, CC_STRIDE, MCP_STRIDE, MC_META_STRIDE, MC_STRIDE};
use crate::contact_wide::{WIDE_IDX_STRIDE, WIDE_META_STRIDE, WIDE_STRIDE};
use crate::distance::SimplexCache;
use crate::finalize::{self, TY_CAPSULE, TY_HULL, TY_SPHERE};
use crate::manifold::{Capsule, SatCache, Sphere};
use crate::manifold_abi::{
    read_dir, DIR_CACHE, DIR_CACHED_REL_POSE, DIR_CACHED_ROT_A, DIR_CACHED_ROT_B, DIR_STRIDE,
    MANIFOLD_STRIDE, M_FRICTION, M_NORMAL, M_POINTS, M_POINT_COUNT, M_ROLLING, M_TWIST,
    POOL_POINT_STRIDE, P_ANCHOR_A, P_ANCHOR_B, P_FEATURE_ID, P_NORMAL_IMPULSE, P_NORMAL_VELOCITY,
    P_PERSISTED, P_SEPARATION, P_TOTAL_NORMAL_IMPULSE, P_TRIANGLE_INDEX, SLOT_STRIDE,
};
use crate::manifolds;
use crate::math::{Quat, Transform, Vec3};
use crate::narrowphase::{
    compute_convex_manifold, ConvexContactCache, ConvexShape, Manifold, MAX_MANIFOLD_POINTS,
};
use crate::recycle::try_recycle;

use crate::fataabb::AABB_STRIDE as FAT_STRIDE;
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
const FIN_OUT: usize = 4;
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
    unsafe {
        BODY_COUNT = body;
        CONTACT_COUNT = contact;
        MANIFOLD_COUNT = manifold;
        POINT_COUNT = point;
        WIDE_COUNT = wide;
        COLOR_COUNT = color;

        // The body columns are resident (4a.2/4a.3): `state` + `flags` (velocity/delta/flags),
        // and `sim` + `fin` + `finOut` (the integrate/finalize sim fields) live in the persistent body
        // region (bodies.rs), held across steps, so the awake `BodySim`/`BodyState` become offset-backed
        // views and no per-step marshal runs. Point their LAYOUT entries at that region instead of
        // allocating per-step scratch; the phase shims read `LAYOUT[SIM]`/etc unchanged. `reserveBodies`
        // (run before this, in `step()`) has laid the region out for the current total-body high-water.
        // The remaining columns share the per-step arena.
        LAYOUT[STATE] = crate::bodies::state_base() as u32;
        LAYOUT[FLAGS] = crate::bodies::flags_base() as u32;
        LAYOUT[SIM] = crate::bodies::sim_base() as u32;
        LAYOUT[FIN] = crate::bodies::fin_base() as u32;
        LAYOUT[FIN_OUT] = crate::bodies::fin_out_base() as u32;
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
unsafe fn body_records() -> usize {
    crate::bodies::body_cap() + crate::bodies::IDENT_RECORDS
}

/// All the scalar solver's columns over the current reservation. The body + slot + transient columns
/// are disjoint byte ranges of the per-step solver region; the directory + pool live in the persistent
/// manifold region (`manifolds`) — also disjoint. Every one is a shared-mutable [`Col`]: the phases
/// index them by body / record id and, under the staged solver, do so from several threads at once.
unsafe fn columns() -> Columns<'static> {
    let b = body_records();
    let c = CONTACT_COUNT;
    let m = MANIFOLD_COUNT;
    let p = POINT_COUNT;
    Columns {
        state: f32s(STATE, b * STATE_STRIDE),
        flags: u32s(FLAGS, b),
        sim: f32s(SIM, b * SIM_STRIDE),
        slot: u32s(SLOT_SCALAR, c * SLOT_STRIDE),
        dir: manifolds::dir_col(),
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
// which is sound because no allocation, memory growth or free may run
// between the fork and the join (the MT concurrency invariant).

/// The scalar solver's columns, as `solve.rs`'s `StageWork` holds them.
pub(crate) unsafe fn scalar_columns() -> Columns<'static> {
    columns()
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
// Shape geometry is gathered locally for the convex/mesh routines; no task payload crosses the FFI.
const DISPATCH_STRIDE: usize = 32;
const D_DEFAULT_MIX: usize = 31;
const D_SHAPE_A: usize = 29;
const D_SHAPE_B: usize = 30;
const D_OLD_COUNT: usize = 28;
const D_CHILD: usize = 19;
const D_MESH_SLOT: usize = 20;
const D_LOWER: usize = 21;
const D_UPPER: usize = 24;
static mut MESH_OUTPUT_PTR: usize = 0;
static mut MESH_MATERIAL_PTR: usize = 0;
static mut MESH_SCRATCH_PTR: usize = 0;
struct DispatchScratch {
    mesh: crate::mesh_contact::MeshScratch,
    old: [Manifold; 256],
    previous: [crate::mesh_contact::TriangleInput; 256],
}
const D_CONTACT: usize = 0;
const D_TYPE_A: usize = 1;
const D_TYPE_B: usize = 2;
const D_BODY_A: usize = 3;
const D_BODY_B: usize = 4;
const D_GEOM_A: usize = 5; // ≤7 slots (sphere c3+r / capsule c1_3+c2_3+r / hull geoIndex)
const D_GEOM_B: usize = 12; // ≤7 slots

static mut CONTACT_LIST_PTR: usize = 0;
static mut CONTACT_STATE_PTR: usize = 0;
static mut DEFAULT_MIX: u32 = 1;
static mut RECYCLE_DISTANCE: f32 = 0.0;
const SIM_UPDATED: u32 = 0x0200_0000;

#[export_name = "reserveCollide"]
pub extern "C" fn reserve_collide(count: usize, threads: usize, default_mix: u32, distance: f32) {
    unsafe {
        DEFAULT_MIX = default_mix;
        RECYCLE_DISTANCE = distance;
        let words = manifolds::contact_capacity(crate::regions::active()).div_ceil(32);
        let mesh_threads = if manifolds::has_mesh_caches() {
            threads.max(1)
        } else {
            0
        };
        let mut off = count * 4;
        let state_off = off;
        off += words * 4;
        off = (off + 15) & !15;
        let mesh_off = off;
        off += mesh_threads * 256 * MANIFOLD_STRIDE * 4;
        let material_off = off;
        off += mesh_threads * 256 * 4 * 4;
        off = (off + 15) & !15;
        let scratch_off = off;
        off += mesh_threads * core::mem::size_of::<DispatchScratch>();
        reserve_scratch(off);
        CONTACT_LIST_PTR = SCRATCH.ptr;
        CONTACT_STATE_PTR = SCRATCH.ptr + state_off;
        MESH_OUTPUT_PTR = SCRATCH.ptr + mesh_off;
        MESH_MATERIAL_PTR = SCRATCH.ptr + material_off;
        MESH_SCRATCH_PTR = SCRATCH.ptr + scratch_off;
        (CONTACT_STATE_PTR as *mut u8).write_bytes(0, words * 4);
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
    unsafe { CONTACT_STATE_PTR }
}
unsafe fn mark_contact_state(contact: usize) {
    core::sync::atomic::AtomicU32::from_ptr((CONTACT_STATE_PTR as *mut u32).add(contact / 32))
        .fetch_or(1 << (contact % 32), core::sync::atomic::Ordering::Relaxed);
}
unsafe fn finish_contact(contact: usize, count: usize, hit: bool) {
    let dir = manifolds::dir_col();
    let o = contact * DIR_STRIDE + 6;
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
            mark_contact_state(contact);
        }
    } else {
        manifolds::free_manifolds(contact);
        if old & 0x0040_0000 == 0 {
            flags |= old & 0x0010_0000;
        }
        if was_touching {
            flags |= 0x0008_0000;
            mark_contact_state(contact);
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
        pool.set(p + P_PERSISTED, f32::from_bits(pt.persisted as u32));
    }
}

unsafe fn dispatch_mesh(
    disp: &[u32],
    r: usize,
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
    let slot = disp[r + D_MESH_SLOT] as usize;
    let cache = &mut *manifolds::mesh_cache_ptr(contact_id);
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
                2 * (field.columns - 1) * (field.rows - 1),
            ),
            field,
        }
    };
    let vec = |o: usize| {
        Vec3::new(
            f32::from_bits(disp[r + o]),
            f32::from_bits(disp[r + o + 1]),
            f32::from_bits(disp[r + o + 2]),
        )
    };
    let scratch = &mut *(MESH_SCRATCH_PTR as *mut DispatchScratch).add(thread);
    cache.refresh(
        &source,
        xf_a,
        vec(D_LOWER),
        vec(D_UPPER),
        &mut scratch.previous,
    );
    let dir = manifolds::dir_col();
    let pool = manifolds::pool_col();
    let entry = read_dir(dir, contact_id);
    let old_count = disp[r + D_OLD_COUNT] as usize;
    let pool = crate::manifold_abi::block_col(pool, entry.manifold_base, old_count);
    for i in 0..old_count {
        let o = i * MANIFOLD_STRIDE;
        scratch.old[i] = Manifold::new();
        let m = &mut scratch.old[i];
        m.normal = Vec3::new(pool.get(o), pool.get(o + 1), pool.get(o + 2));
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
    let shape = read_shape(disp[r + D_TYPE_B], disp, r + D_GEOM_B);
    let directory = manifolds::dir_col();
    let shape_a =
        directory.get(contact_id * DIR_STRIDE + crate::manifold_abi::DIR_SHAPE_A) as usize;
    let shape_b =
        directory.get(contact_id * DIR_STRIDE + crate::manifold_abi::DIR_SHAPE_B) as usize;
    let shape_records = crate::shapes::col();
    let speculative = shape_records
        .get(shape_a * crate::shapes::SHAPE_STRIDE + crate::shapes::S_FLAGS)
        & crate::shapes::SPECULATIVE_FLAG
        != 0
        && shape_records.get(shape_b * crate::shapes::SHAPE_STRIDE + crate::shapes::S_FLAGS)
            & crate::shapes::SPECULATIVE_FLAG
            != 0;
    let count = compute_mesh_manifolds(
        &mut scratch.mesh,
        &mut cache.triangles[..cache.count],
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
        let m = &mut scratch.mesh.output[i];
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
            *materials.add(i * 4 + j) = scratch.mesh.materials[i][j];
        }
    }
    count
}

// The convex GJK/SAT cache is a `b3ContactCache` union folded into the directory (slots `DIR_CACHE`+):
// the wider SimplexCache (10 slots) overlaps the narrower SatCache. A contact uses one or the other by
// shape pair — hull-hull uses SAT, hull-vs-sphere/capsule uses the GJK simplex, the rest none.

#[inline]
fn read_simplex(dir: Col<u32>, id: usize) -> SimplexCache {
    let o = id * DIR_STRIDE + DIR_CACHE;
    SimplexCache {
        metric: f32::from_bits(dir.get(o)),
        count: dir.get(o + 1) as usize,
        index_a: [
            dir.get(o + 2) as usize,
            dir.get(o + 3) as usize,
            dir.get(o + 4) as usize,
            dir.get(o + 5) as usize,
        ],
        index_b: [
            dir.get(o + 6) as usize,
            dir.get(o + 7) as usize,
            dir.get(o + 8) as usize,
            dir.get(o + 9) as usize,
        ],
    }
}

#[inline]
fn write_simplex(dir: Col<u32>, id: usize, c: &SimplexCache) {
    let o = id * DIR_STRIDE + DIR_CACHE;
    dir.set(o, c.metric.to_bits());
    dir.set(o + 1, c.count as u32);
    for k in 0..4 {
        dir.set(o + 2 + k, c.index_a[k] as u32);
        dir.set(o + 6 + k, c.index_b[k] as u32);
    }
}

#[inline]
fn read_sat(dir: Col<u32>, id: usize) -> SatCache {
    let o = id * DIR_STRIDE + DIR_CACHE;
    SatCache {
        separation: f32::from_bits(dir.get(o)),
        ty: dir.get(o + 1),
        index_a: dir.get(o + 2) as usize,
        index_b: dir.get(o + 3) as usize,
        hit: dir.get(o + 4),
    }
}

#[inline]
fn write_sat(dir: Col<u32>, id: usize, c: &SatCache) {
    let o = id * DIR_STRIDE + DIR_CACHE;
    dir.set(o, c.separation.to_bits());
    dir.set(o + 1, c.ty);
    dir.set(o + 2, c.index_a as u32);
    dir.set(o + 3, c.index_b as u32);
    dir.set(o + 4, c.hit);
}

#[derive(Clone, Copy)]
struct Surface {
    friction: f32,
    restitution: f32,
    rolling: f32,
    tangent: Vec3,
}

fn surface(shape: usize, index: usize) -> Surface {
    let materials = crate::shapes::material(shape, index);
    let f = |i| f32::from_bits(materials[i]);
    Surface {
        friction: f(0),
        restitution: f(1),
        rolling: f(2),
        tangent: Vec3::new(f(3), f(4), f(5)),
    }
}

fn shape_radius(shape: usize, full_hull: bool) -> f32 {
    let shapes = crate::shapes::col_slice();
    let s = shape * crate::shapes::SHAPE_STRIDE;
    match shapes[s] {
        TY_SPHERE => f32::from_bits(shapes[s + 5]),
        TY_CAPSULE => f32::from_bits(shapes[s + 8]),
        TY_HULL => (if full_hull { 1.0 } else { 0.25 }) * f32::from_bits(shapes[s + 43]),
        _ => 0.0,
    }
}

fn store_surface(id: usize, friction: f32, restitution: f32, rolling: f32, tangent: Vec3) {
    let dir = manifolds::dir_col();
    let o = id * DIR_STRIDE;
    for (i, v) in [
        friction,
        restitution,
        rolling,
        tangent.x,
        tangent.y,
        tangent.z,
    ]
    .into_iter()
    .enumerate()
    {
        dir.set(o + i, v.to_bits());
    }
}

unsafe fn mix_surface(
    disp: &[u32],
    xf_a: Transform,
    xf_b: Transform,
    map: Option<[u32; 4]>,
    flip: bool,
    count: usize,
    mesh: bool,
    child_radius: f32,
) {
    if count == 0 || disp[D_DEFAULT_MIX] == 0 {
        return;
    }
    let sa = disp[D_SHAPE_A] as usize;
    let sb = disp[D_SHAPE_B] as usize;
    let a_index = |i: usize| map.map_or(i, |m| m[i.min(3)] as usize);
    let mut a = surface(sa, a_index(0));
    let mut b = surface(sb, 0);
    let mut radius_a = if map.is_some() {
        child_radius
    } else {
        shape_radius(sa, false)
    };
    let mut radius_b = shape_radius(sb, mesh);
    let (friction, restitution, rolling, tangent) = if mesh {
        let slot = disp[D_MESH_SLOT] as usize;
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
                let m = surface(sa, a_index(*materials.add(i * 4 + j) as usize));
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
        disp[D_CONTACT] as usize,
        friction,
        restitution,
        rolling,
        tangent,
    );
}

/// Compute convex, mesh, height-field and compound-child contacts in `[start, end)`.
/// Tasks allocate new convex blocks and resize mesh blocks only when the cluster count changes.
/// Mesh scratch preserves matched impulses before the resident block is cleared and populated.
/// Per-thread scratch is disjoint across worker indices.
///
/// Box3D's collide task: update each contact in place and mark its touch state by id.
///
/// # Safety
/// Body/shape columns and per-thread scratch are reserved before the fork. Only the locked manifold
/// block allocators may allocate or free; their chunks never move. Each contact runs in one task.
pub(crate) unsafe fn contact_block(start: usize, end: usize, total: usize, thread: usize) {
    unsafe {
        use crate::manifold_abi::*;
        let contacts = core::slice::from_raw_parts(CONTACT_LIST_PTR as *const u32, total);
        let dir = manifolds::dir_col();
        let pool = manifolds::pool_col();

        for i in start..end {
            recycle_block(
                i,
                i + 1,
                total,
                RECYCLE_DISTANCE,
                RECYCLE_DISTANCE.min(0.02),
            );
            let contact = contacts[i] as usize;
            let o = contact * DIR_STRIDE;
            if dir.get(o + 6) & SIM_UPDATED == 0 {
                continue;
            }
            let mut record = [0u32; DISPATCH_STRIDE];
            record[D_CONTACT] = contact as u32;
            record[D_SHAPE_A] = dir.get(o + DIR_SHAPE_A);
            record[D_SHAPE_B] = dir.get(o + DIR_SHAPE_B);
            record[D_BODY_A] = dir.get(o + DIR_EDGE_A);
            record[D_BODY_B] = dir.get(o + DIR_EDGE_B);
            record[D_CHILD] = dir.get(o + DIR_CHILD_INDEX);
            record[D_OLD_COUNT] = dir.get(o + 7);
            record[D_MESH_SLOT] = thread as u32;
            record[D_DEFAULT_MIX] = DEFAULT_MIX;
            let shapes = crate::shapes::col_slice();
            let sb = record[D_SHAPE_B] as usize * crate::shapes::SHAPE_STRIDE;
            record[D_LOWER..D_LOWER + 3].copy_from_slice(&shapes[sb + 34..sb + 37]);
            record[D_UPPER..D_UPPER + 3].copy_from_slice(&shapes[sb + 37..sb + 40]);
            let sa = record[D_SHAPE_A] as usize * crate::shapes::SHAPE_STRIDE;
            let hit = shapes[sa + crate::shapes::S_HIT_EVENTS] & crate::shapes::HIT_FLAG != 0
                || shapes[sb + crate::shapes::S_HIT_EVENTS] & crate::shapes::HIT_FLAG != 0;
            for (id_slot, type_slot, geom_slot) in [
                (D_SHAPE_A, D_TYPE_A, D_GEOM_A),
                (D_SHAPE_B, D_TYPE_B, D_GEOM_B),
            ] {
                let s = record[id_slot] as usize * crate::shapes::SHAPE_STRIDE;
                let ty = shapes[s + crate::shapes::S_TYPE];
                record[type_slot] = ty;
                record[geom_slot..geom_slot + 7].copy_from_slice(&shapes[s + 2..s + 9]);
            }
            let disp = &record[..];
            let r = 0;
            let contact_id = disp[r + D_CONTACT] as usize;
            let mut type_a = disp[r + D_TYPE_A];
            let type_b = disp[r + D_TYPE_B];
            let body_a = disp[r + D_BODY_A] as usize;
            let body_b = disp[r + D_BODY_B] as usize;
            let (parent_xf, fin_a, flags_a) = crate::bodies::geometry(body_a);
            let (xf_b, fin_b, flags_b) = crate::bodies::geometry(body_b);
            let mut xf_a = parent_xf;
            let center_a = parent_xf.q.rotate(fin_a.local_center);
            let center_b = xf_b.q.rotate(fin_b.local_center);
            let fast = (flags_a | flags_b) & 0x40 != 0;
            let mut geom_a = &disp[r + D_GEOM_A..r + D_GEOM_A + 7];
            let mut child_offset = Vec3::ZERO;
            let mut material_map = None;
            let mut child_radius = 0.0;
            let mut compound_geometry = [0u32; 19];
            if type_a == 1 {
                let compound = geom_a[0] as *const u32;
                compound_geometry =
                    crate::compound_query::child_words(compound, disp[r + D_CHILD] as usize);
                type_a = compound_geometry[0];
                material_map = Some([
                    compound_geometry[8],
                    compound_geometry[9],
                    compound_geometry[10],
                    compound_geometry[11],
                ]);
                let local = read_xf(&compound_geometry, 1);
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
            let disp = &record[..];
            if type_a == 2 || type_a == 4 {
                let count = dispatch_mesh(
                    disp,
                    r,
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
                    disp,
                    xf_a,
                    xf_b,
                    material_map,
                    false,
                    count,
                    true,
                    child_radius,
                );
                if count > 0 {
                    let address = if count == record[D_OLD_COUNT] as usize {
                        let address = dir.get(o + DIR_MANIFOLD_BASE) as usize;
                        (address as *mut u8).write_bytes(0, count * MANIFOLD_STRIDE * 4);
                        address
                    } else {
                        manifolds::allocate_manifolds(contact_id, count)
                    };
                    let source = MESH_OUTPUT_PTR + thread * 256 * MANIFOLD_STRIDE * 4;
                    manifolds::copy_manifolds(source, address, count);
                }
                finish_contact(contact_id, count, hit);
                continue;
            }
            let mut shape_a = read_shape(type_a, geom_a, 0);
            let mut shape_b = read_shape(type_b, disp, r + D_GEOM_B);
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
            let mut cache = ConvexContactCache::empty();
            if uses_sat {
                cache.sat_cache = read_sat(dir, contact_id);
            } else if uses_simplex {
                cache.simplex_cache = read_simplex(dir, contact_id);
            }

            let resident = disp[r + D_OLD_COUNT] != 0;
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
                    manifolds::allocate_manifolds(contact_id, 1)
                };
                write_manifold(&m, crate::manifold_abi::block_col(pool, address, 1), 0);
            }
            if uses_sat {
                write_sat(dir, contact_id, &cache.sat_cache);
            } else if uses_simplex {
                write_simplex(dir, contact_id, &cache.simplex_cache);
            }
            mix_surface(
                disp,
                xf_a,
                xf_b,
                material_map,
                flip,
                touching as usize,
                false,
                child_radius,
            );
            finish_contact(contact_id, touching as usize, hit);
        }
    }
}

/// Run the same contact tasks on the calling thread when the sweep does not fork.
#[export_name = "dispatchContacts"]
pub extern "C" fn dispatch_contacts(count: usize) {
    unsafe { contact_block(0, count, count, 0) }
}

// --- contact recycle -----------------------------------------------------------------------
// Recycle poses resolve body ids through the resident records across sleep/wake transitions.

/// Per-contact recycle inputs.
const RECYCLE_STRIDE: usize = 7;
const R_COUNT: usize = 6;
const R_STATIC_A: u32 = 4;
const R_STATIC_B: u32 = 8;
const R_MESH: u32 = 16;
const R_CONTACT: usize = 0;
const R_BODY_A: usize = 1;
const R_BODY_B: usize = 2;
const R_SHAPE_A: usize = 3; // shapeId → fat-AABB column record
const R_SHAPE_B: usize = 4;
const R_BITS: usize = 5;
/// bit0: the contact may recycle this step (recycleDistance>0 && relativeTransformValid && recycleFlag).
const R_ELIGIBLE: u32 = 1;
/// bit1: the contact was touching at step entry (selects the recycle tolerance).
const R_WAS_TOUCHING: u32 = 2;

/// Body `i`'s world transform from the resident sim (rotation) + fin (position) columns.
#[inline]
fn read_body_xf(sim: &[f32], fin: &[f32], i: usize) -> Transform {
    let so = i * SIM_STRIDE;
    let fo = i * FIN_STRIDE;
    Transform {
        p: Vec3::new(fin[fo + 9], fin[fo + 10], fin[fo + 11]),
        q: Quat {
            v: Vec3::new(sim[so + 28], sim[so + 29], sim[so + 30]),
            s: sim[so + 31],
        },
    }
}

/// Body `i`'s center of mass from the resident fin column.
#[inline]
fn read_center(fin: &[f32], i: usize) -> Vec3 {
    let fo = i * FIN_STRIDE;
    Vec3::new(fin[fo], fin[fo + 1], fin[fo + 2])
}

/// Body `i`'s max extent from the resident fin column.
#[inline]
fn read_max_extent(fin: &[f32], i: usize) -> Vec3 {
    let fo = i * FIN_STRIDE;
    Vec3::new(fin[fo + 6], fin[fo + 7], fin[fo + 8])
}

/// Do shapes `sa` and `sb`'s fat AABBs overlap? (b3AABB_Overlaps over the resident fat-AABB column;
/// bit-identical to `src/math.ts` `aabb.overlaps` — the same six comparisons.)
#[inline]
fn fat_overlap(fat: &[f32], sa: usize, sb: usize) -> bool {
    let a = sa * FAT_STRIDE;
    let b = sb * FAT_STRIDE;
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

/// Run the recycle branch for the input records in `[start, end)` of a column of `total`. `recycle_dist` /
/// `recycle_dist_non_touching` are the two world tolerances (touching vs speculative); the per-record
/// `wasTouching` bit selects between them. Writes each contact's result into the output column:
/// 0 = recycled (separations updated in-kernel), 1 = needs full narrowphase (pose cached in-kernel),
/// 2 = disjoint (fat AABBs no longer overlap).
///
/// One block of the parallel sweep (`parfor.rs`), or the whole column on the serial path. As
/// [`contact_block`], records are independent: the body / fat-AABB columns are read-only here, and every
/// write lands in the record's own contact's directory + manifold slots.
///
/// # Safety
/// The contact list, body-index map and body/shape columns remain at their addresses during collide.
pub(crate) unsafe fn recycle_block(
    start: usize,
    end: usize,
    total: usize,
    recycle_dist: f32,
    recycle_dist_non_touching: f32,
) {
    unsafe {
        use crate::manifold_abi::*;
        let contacts = core::slice::from_raw_parts(CONTACT_LIST_PTR as *const u32, total);
        let dir = manifolds::dir_col();
        let pool = manifolds::pool_col();
        let fat = crate::fataabb::col_slice();

        for i in start..end {
            let contact_id = contacts[i] as usize;
            let o = contact_id * DIR_STRIDE;
            let flags = dir.get(o + 6) & !SIM_UPDATED;
            dir.set(o + 6, flags);
            let mut record = [0u32; RECYCLE_STRIDE];
            record[R_CONTACT] = contact_id as u32;
            record[R_BODY_A] = dir.get(o + DIR_EDGE_A);
            record[R_BODY_B] = dir.get(o + DIR_EDGE_B);
            record[R_SHAPE_A] = dir.get(o + DIR_SHAPE_A);
            record[R_SHAPE_B] = dir.get(o + DIR_SHAPE_B);
            record[R_COUNT] = dir.get(o + 7);
            let mut bits = 0;
            if dir.get(o + 9) == u32::MAX {
                bits |= R_STATIC_A;
            }
            if dir.get(o + 10) == u32::MAX {
                bits |= R_STATIC_B;
            }
            if flags & 0x0040_0000 != 0 {
                bits |= R_MESH;
            }
            if recycle_dist > 0.0 && flags & 0x0080_0000 != 0 && flags & 0x10 != 0 {
                bits |= R_ELIGIBLE;
            }
            if flags & 0x0001_0000 != 0 {
                bits |= R_WAS_TOUCHING;
            }
            record[R_BITS] = bits;
            let input = &record[..];
            let r = 0;

            // Fat-AABB overlap first — matching the TS collide's first per-contact check.
            if !fat_overlap(
                fat,
                input[r + R_SHAPE_A] as usize,
                input[r + R_SHAPE_B] as usize,
            ) {
                dir.set(o + 6, (flags | 0x0002_0000) & !0x0001_0000);
                mark_contact_state(contact_id);
                continue;
            }

            let la = input[r + R_BODY_A] as usize;
            let lb = input[r + R_BODY_B] as usize;
            let bits = input[r + R_BITS];
            let (xf_a, fin_a, flags_a) = crate::bodies::geometry(la);
            let (xf_b, fin_b, flags_b) = crate::bodies::geometry(lb);
            let center_a = fin_a.center;
            let center_b = fin_b.center;
            let extent_a = if bits & R_STATIC_A != 0 {
                Vec3::ZERO
            } else {
                fin_a.max_extent
            };
            let extent_b = if bits & R_STATIC_B != 0 {
                Vec3::ZERO
            } else {
                fin_b.max_extent
            };
            let fast_mesh = bits & R_MESH != 0 && (flags_a | flags_b) & 0x40 != 0;
            let tol = if bits & R_WAS_TOUCHING != 0 {
                recycle_dist
            } else {
                recycle_dist_non_touching
            };

            if bits & R_ELIGIBLE != 0 && !fast_mesh {
                let (rot_a, rot_b, rel) = read_pose_cache(dir, contact_id);
                let mc = input[r + R_COUNT] as usize;
                if try_recycle(
                    dir, pool, contact_id, mc, xf_a, xf_b, rot_a, rot_b, rel, center_a, center_b,
                    extent_a, extent_b, tol,
                ) {
                    continue;
                }
            }

            // Recycle missed (or the contact isn't eligible yet): cache this step's pose for the next
            // step and defer to the full narrowphase.
            write_pose_cache(dir, contact_id, xf_a, xf_b);
            dir.set(o + 6, flags | 0x0080_0000 | SIM_UPDATED);
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
    start: usize,
    end: usize,
    h: f32,
    inv_dt: f32,
    enable_continuous: bool,
) {
    unsafe {
        let b = body_records();
        let state = f32s(STATE, b * STATE_STRIDE);
        let sim = f32s(SIM, b * SIM_STRIDE);
        let fin = f32s(FIN, b * FIN_STRIDE);
        let out = f32s(FIN_OUT, b * FIN_OUT_STRIDE);
        let flags = u32s(FLAGS, b);
        let sim2 = Col::new(crate::bodies::sim2_base() as *mut f32, b * SIM2_STRIDE);
        let sim2_u = Col::new(crate::bodies::sim2_base() as *mut u32, b * SIM2_STRIDE);
        let moves = Col::new(
            crate::bodies::move_base() as *mut u32,
            b * crate::bodies::MOVE_STRIDE,
        );
        finalize::finalize(
            state,
            sim,
            fin,
            out,
            sim2,
            flags,
            start,
            end - start,
            h,
            inv_dt,
            enable_continuous,
        );
        // Finalization is the sole producer of move records. The record identity is read from the
        // resident kernel body columns and its generation from the active world lifecycle pool.
        for i in start..end {
            let body_id = sim2_u.get(i * SIM2_STRIDE + S2_BODY_ID);
            let o = i * crate::bodies::MOVE_STRIDE;
            moves.set(o, body_id);
            moves.set(o + 1, crate::bodies::active_generation(body_id));
            moves.set(o + 2, 0);
        }
        crate::continuous::finalize(start, end, enable_continuous);
        // Continuous can clip the rotation. Box3D rebuilds inertia from the resulting pose,
        // not the discrete candidate; non-fast bodies already have that tensor.
        for i in start..end {
            if sim2_u.atomic_get(i * SIM2_STRIDE + 10) & 0x40 != 0 {
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
        refit_block(sim, fin, start, end);
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
unsafe fn refit_block(sim: Col<f32>, fin: Col<f32>, start: usize, end: usize) {
    unsafe {
        let records = crate::bodies::body_cap() + crate::bodies::IDENT_RECORDS;
        let sim2 = Col::new(
            crate::bodies::sim2_base() as *mut u32,
            records * SIM2_STRIDE,
        );
        let shape_u = crate::shapes::col();
        let shape_f = crate::shapes::col_f();
        let fat = crate::fataabb::col();
        for i in start..end {
            if sim2.atomic_get(i * SIM2_STRIDE + 10) & 0x40 != 0 {
                continue;
            }
            let so = i * SIM_STRIDE;
            let fo = i * FIN_STRIDE;
            let xf = Transform {
                p: Vec3::new(fin.get(fo + 9), fin.get(fo + 10), fin.get(fo + 11)),
                q: Quat {
                    v: Vec3::new(sim.get(so + 28), sim.get(so + 29), sim.get(so + 30)),
                    s: sim.get(so + 31),
                },
            };
            let mut shape_id = sim2.get(i * SIM2_STRIDE + S2_HEAD_SHAPE);
            while shape_id != crate::shapes::NULL_SHAPE {
                let o = shape_id as usize * crate::shapes::SHAPE_STRIDE;
                let bounds = crate::continuous::bounds(shape_id as usize, xf);
                let fb = shape_id as usize * FAT_STRIDE;
                let fat_aabb = core::array::from_fn(|n| fat.get(fb + n));
                let (cand, escaped) = finalize::refit_bounds(bounds, &fat_aabb);
                for n in 0..6 {
                    shape_f.set(o + 34 + n, cand[n]);
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
                    let margin = shape_f.get(o + 40);
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

//! The shared-column arena and phase export shims — the kernel's wasm surface.
//!
//! `reserve` lays out every solver column contiguously in linear memory (starting at `__heap_base`),
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
    FIN_OUT_STRIDE, FIN_STRIDE, SIM_STRIDE, SIM2_STRIDE, STATE_STRIDE, S2_HEAD_SHAPE,
};
use crate::col::Col;
use crate::contact::{
    self, Columns, Softness, CC_META_STRIDE, CC_STRIDE, MCP_STRIDE, MC_META_STRIDE, MC_STRIDE,
};
use crate::contact_wide::{self, WIDE_IDX_STRIDE, WIDE_META_STRIDE, WIDE_STRIDE};
use crate::distance::SimplexCache;
use crate::finalize::{self, TY_CAPSULE, TY_HULL, TY_SPHERE};
use crate::integrate;
use crate::manifold::{Capsule, SatCache, Sphere};
use crate::manifold_abi::{
    read_dir, DIR_CACHE, DIR_CACHED_REL_POSE, DIR_CACHED_ROT_A, DIR_CACHED_ROT_B, DIR_STRIDE,
    MANIFOLD_STRIDE, M_NORMAL, M_POINTS, M_POINT_COUNT, POOL_POINT_STRIDE, P_ANCHOR_A, P_ANCHOR_B,
    P_FEATURE_ID, P_NORMAL_IMPULSE, P_NORMAL_VELOCITY, P_PERSISTED, P_SEPARATION,
    P_TOTAL_NORMAL_IMPULSE, P_TRIANGLE_INDEX, SLOT_STRIDE,
};
use crate::manifolds;
use crate::math::{Quat, Transform, Vec3};
use crate::narrowphase::{
    compute_convex_manifold, ConvexContactCache, ConvexShape, Manifold, MAX_MANIFOLD_POINTS,
};
use crate::recycle::try_recycle;

use crate::fataabb::AABB_STRIDE as FAT_STRIDE;
use crate::geo::{hull_view, solver_base};

const PAGE: usize = 65536;
const N_COLS: usize = 16;
/// u32 stride of one active-color span: wideStart, wideCount, meshStart, meshCount, jointStart,
/// jointCount. The joint pair is written by `writeColorSpans`; the jointless batched path reads only
/// the first four (`warm_start_colors` etc), the staged solve reads all six (`solve_build`).
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
// Flat joint records — one `joint_abi::JOINT_STRIDE` record per joint slot, per-color concatenated
// then the overflow joints (the staged solve's `PrepareJoints` sweep + colored joint blocks).
const JOINT: usize = 15;

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
// Flat joint slot count (colored joints, per-color concatenated, then the overflow joints).
static mut JOINT_COUNT: usize = 0;

/// Grow linear memory so `[0, end_byte)` is addressable.
unsafe fn ensure_capacity(end_byte: usize) {
    let have = core::arch::wasm32::memory_size(0) * PAGE;
    if end_byte > have {
        let pages = (end_byte - have + PAGE - 1) / PAGE;
        core::arch::wasm32::memory_grow(0, pages);
    }
}

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
/// from `__heap_base` each call; TS re-derives its views from `layoutPtr` afterwards.
#[export_name = "reserve"]
pub extern "C" fn reserve(
    body: usize,
    contact: usize,
    manifold: usize,
    point: usize,
    wide: usize,
    color: usize,
    joint: usize,
) {
    unsafe {
        BODY_COUNT = body;
        CONTACT_COUNT = contact;
        MANIFOLD_COUNT = manifold;
        POINT_COUNT = point;
        WIDE_COUNT = wide;
        COLOR_COUNT = color;
        JOINT_COUNT = joint;

        // The body columns are resident (4a.2/4a.3): `state` + `flags` (velocity/delta/flags),
        // and `sim` + `fin` + `finOut` (the integrate/finalize sim fields) live in the persistent body
        // region (bodies.rs), held across steps, so the awake `BodySim`/`BodyState` become offset-backed
        // views and no per-step marshal runs. Point their LAYOUT entries at that region instead of
        // allocating per-step scratch; the phase shims read `LAYOUT[SIM]`/etc unchanged. `reserveBodies`
        // (run before this, in `step()`) has laid the region out for the current total-body high-water.
        // The remaining columns are per-step scratch, laid out past the static geometry region.
        LAYOUT[STATE] = crate::bodies::state_base() as u32;
        LAYOUT[FLAGS] = crate::bodies::flags_base() as u32;
        LAYOUT[SIM] = crate::bodies::sim_base() as u32;
        LAYOUT[FIN] = crate::bodies::fin_base() as u32;
        LAYOUT[FIN_OUT] = crate::bodies::fin_out_base() as u32;
        let mut off = solver_base();
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
        LAYOUT[JOINT] = off as u32;
        off += joint * crate::joint_abi::JOINT_STRIDE * 4;

        ensure_capacity(off);
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
// which is sound exactly because no `reserve*` (and so no `memory.grow`, no region relocation) may run
// between the fork and the join (the MT concurrency invariant, `.claude/rules/tumble.md`).

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

/// The flat joint column (`joint_abi::JOINT_STRIDE` f32 per slot), as the staged solve's joint phases
/// read/write it. TS marshals the records in (`src/jointcolumns.ts`) and reads the solved impulses back.
pub(crate) unsafe fn joint_column() -> Col<'static, f32> {
    f32s(JOINT, JOINT_COUNT * crate::joint_abi::JOINT_STRIDE)
}

// --- integrate phases -----------------------------------------------------------------------

#[export_name = "integrateVelocities"]
pub extern "C" fn integrate_velocities(gx: f32, gy: f32, gz: f32, h: f32) {
    unsafe {
        let n = BODY_COUNT;
        let b = body_records();
        let state = f32s(STATE, b * STATE_STRIDE);
        let sim = f32s(SIM, b * SIM_STRIDE);
        integrate::integrate_velocities(state, sim, 0, n, Vec3::new(gx, gy, gz), h);
    }
}

#[export_name = "integratePositions"]
pub extern "C" fn integrate_positions(h: f32, max_linear_velocity: f32, inv_dt: f32) {
    unsafe {
        let n = BODY_COUNT;
        let b = body_records();
        let state = f32s(STATE, b * STATE_STRIDE);
        let flags = u32s(FLAGS, b);
        integrate::integrate_positions(state, flags, 0, n, h, max_linear_velocity, inv_dt);
    }
}

// --- scalar (mesh / overflow) contact solve phases ------------------------------------------
// Each takes an explicit `[start, count)` contact-record range so the solver can drive one graph
// color's mesh contacts (or the overflow spill) at a time, in the exact `b3SolverTask` order.

#[export_name = "prepareContacts"]
pub extern "C" fn prepare_contacts(
    start: usize,
    count: usize,
    cs_bias: f32,
    cs_mass: f32,
    cs_impulse: f32,
    ss_bias: f32,
    ss_mass: f32,
    ss_impulse: f32,
    warm_start_scale: f32,
) {
    unsafe {
        let cols = columns();
        let contact_softness = Softness {
            bias_rate: cs_bias,
            mass_scale: cs_mass,
            impulse_scale: cs_impulse,
        };
        let static_softness = Softness {
            bias_rate: ss_bias,
            mass_scale: ss_mass,
            impulse_scale: ss_impulse,
        };
        contact::prepare(
            &cols,
            start,
            count,
            contact_softness,
            static_softness,
            warm_start_scale,
        );
    }
}

#[export_name = "warmStartContacts"]
pub extern "C" fn warm_start_contacts(start: usize, count: usize) {
    unsafe {
        let cols = columns();
        contact::warm_start(
            &cols, start, count);
    }
}

#[export_name = "solveContacts"]
pub extern "C" fn solve_contacts(
    start: usize,
    count: usize,
    use_bias: u32,
    inv_h: f32,
    contact_speed: f32,
) {
    unsafe {
        let cols = columns();
        contact::solve(
            &cols, start, count, use_bias != 0, inv_h, contact_speed);
    }
}

#[export_name = "restitution"]
pub extern "C" fn restitution(start: usize, count: usize, threshold: f32) {
    unsafe {
        let cols = columns();
        contact::restitution(
            &cols, start, count, threshold);
    }
}

#[export_name = "storeImpulses"]
pub extern "C" fn store_impulses(start: usize, count: usize, hit_event_threshold: f32) {
    unsafe {
        let cols = columns();
        contact::store(
            &cols, start, count, hit_event_threshold);
    }
}

// --- wide (convex) contact solve phases -----------------------------------------------------
// The 4-lane solver over the transient wide columns. `[start, count)` is a wide-record range (each
// record groups up to 4 convex contacts sharing no body); prepare/store run flat over all records,
// warm-start/solve/relax/restitution per graph color. Reads the same input columns as the scalar
// path (the wide meta maps each lane to its input contact record).

#[export_name = "prepareWideContacts"]
pub extern "C" fn prepare_wide_contacts(
    start: usize,
    count: usize,
    cs_bias: f32,
    cs_mass: f32,
    cs_impulse: f32,
    ss_bias: f32,
    ss_mass: f32,
    ss_impulse: f32,
    warm_start_scale: f32,
) {
    unsafe {
        let b = body_records();
        let state = f32s(STATE, b * STATE_STRIDE);
        let sim = f32s(SIM, b * SIM_STRIDE);
        let dir = manifolds::dir_col();
        let pool = manifolds::pool_col();
        let wide = f32s(WIDE, WIDE_COUNT * WIDE_STRIDE);
        let idx = u32s(WIDE_IDX, WIDE_COUNT * WIDE_IDX_STRIDE);
        let meta = u32s(WIDE_META, WIDE_COUNT * WIDE_META_STRIDE);
        let contact_softness = Softness {
            bias_rate: cs_bias,
            mass_scale: cs_mass,
            impulse_scale: cs_impulse,
        };
        let static_softness = Softness {
            bias_rate: ss_bias,
            mass_scale: ss_mass,
            impulse_scale: ss_impulse,
        };
        contact_wide::prepare(
            wide,
            idx,
            meta,
            state,
            sim,
            dir,
            pool,
            start,
            count,
            contact_softness,
            static_softness,
            warm_start_scale,
        );
    }
}

#[export_name = "warmStartWideContacts"]
pub extern "C" fn warm_start_wide_contacts(start: usize, count: usize) {
    unsafe {
        let b = body_records();
        let state = f32s(STATE, b * STATE_STRIDE);
        let flags = u32s(FLAGS, b);
        let wide = f32s(WIDE, WIDE_COUNT * WIDE_STRIDE);
        let idx = u32s(WIDE_IDX, WIDE_COUNT * WIDE_IDX_STRIDE);
        contact_wide::warm_start(wide, idx, state, flags, start, count, ORCHESTRATOR);
    }
}

#[export_name = "solveWideContacts"]
pub extern "C" fn solve_wide_contacts(
    start: usize,
    count: usize,
    use_bias: u32,
    inv_h: f32,
    contact_speed: f32,
) {
    unsafe {
        let b = body_records();
        let state = f32s(STATE, b * STATE_STRIDE);
        let flags = u32s(FLAGS, b);
        let wide = f32s(WIDE, WIDE_COUNT * WIDE_STRIDE);
        let idx = u32s(WIDE_IDX, WIDE_COUNT * WIDE_IDX_STRIDE);
        contact_wide::solve(
            wide,
            idx,
            state,
            flags,
            start,
            count,
            use_bias != 0,
            inv_h,
            contact_speed,
            ORCHESTRATOR,
        );
    }
}

#[export_name = "restitutionWide"]
pub extern "C" fn restitution_wide(start: usize, count: usize, threshold: f32) {
    unsafe {
        let b = body_records();
        let state = f32s(STATE, b * STATE_STRIDE);
        let flags = u32s(FLAGS, b);
        let wide = f32s(WIDE, WIDE_COUNT * WIDE_STRIDE);
        let idx = u32s(WIDE_IDX, WIDE_COUNT * WIDE_IDX_STRIDE);
        contact_wide::restitution(wide, idx, state, flags, start, count, threshold, ORCHESTRATOR);
    }
}

#[export_name = "storeWideImpulses"]
pub extern "C" fn store_wide_impulses(start: usize, count: usize, hit_event_threshold: f32) {
    unsafe {
        let dir = manifolds::dir_col();
        let pool = manifolds::pool_col();
        let wide = f32s(WIDE, WIDE_COUNT * WIDE_STRIDE);
        let meta = u32s(WIDE_META, WIDE_COUNT * WIDE_META_STRIDE);
        contact_wide::store(wide, meta, dir, pool, start, count, hit_event_threshold);
    }
}

// --- batched per-color loop (jointless scenes) ----------------------------------------------
// When no graph color holds a joint, the whole color loop stays in the kernel: one call per phase
// runs every active color's wide-then-mesh block over the COLOR_SPAN column, instead of the TS side
// crossing the FFI boundary per color to interleave joints. Same op order (wide then mesh, colors
// ascending) as the per-color TS path, so bit-identical; it just collapses ~4·colors crossings/step
// into one. Joint-bearing scenes keep the per-color TS interleave.

#[export_name = "warmStartColors"]
pub extern "C" fn warm_start_colors() {
    unsafe {
        let cc = COLOR_COUNT;
        let spans = u32s(COLOR_SPAN, cc * COLOR_SPAN_STRIDE);
        for c in 0..cc {
            let o = c * COLOR_SPAN_STRIDE;
            warm_start_wide_contacts(spans.get(o) as usize, spans.get(o + 1) as usize);
            warm_start_contacts(spans.get(o + 2) as usize, spans.get(o + 3) as usize);
        }
    }
}

#[export_name = "solveColors"]
pub extern "C" fn solve_colors(use_bias: u32, inv_h: f32, contact_speed: f32) {
    unsafe {
        let cc = COLOR_COUNT;
        let spans = u32s(COLOR_SPAN, cc * COLOR_SPAN_STRIDE);
        for c in 0..cc {
            let o = c * COLOR_SPAN_STRIDE;
            solve_wide_contacts(
                spans.get(o) as usize,
                spans.get(o + 1) as usize,
                use_bias,
                inv_h,
                contact_speed,
            );
            solve_contacts(
                spans.get(o + 2) as usize,
                spans.get(o + 3) as usize,
                use_bias,
                inv_h,
                contact_speed,
            );
        }
    }
}

#[export_name = "restitutionColors"]
pub extern "C" fn restitution_colors(threshold: f32) {
    unsafe {
        let cc = COLOR_COUNT;
        let spans = u32s(COLOR_SPAN, cc * COLOR_SPAN_STRIDE);
        for c in 0..cc {
            let o = c * COLOR_SPAN_STRIDE;
            restitution_wide(spans.get(o) as usize, spans.get(o + 1) as usize, threshold);
            restitution(spans.get(o + 2) as usize, spans.get(o + 3) as usize, threshold);
        }
    }
}

// --- convex narrowphase batched dispatch (3c.3) ---------------------------------------------
// One kernel call runs `compute_convex_manifold` (narrowphase.rs) for every convex contact the TS
// collect pass gathered, over the static geometry pools (hulls), the dispatch column (sphere/capsule
// params + transforms inline), the persistent manifold pool (warm-start in, new manifold out), and the
// per-contact GJK/SAT cache folded into the directory. box3d's collide is scalar per-contact; this
// batches only the FFI crossing, not the arithmetic — each record is the gold-verified scalar call.

/// u32 stride of a dispatch record, matching `src/columns.ts` `DISPATCH_STRIDE`. Float slots read
/// through `f32::from_bits`; contactId + types + a hull's geoIndex are u32.
const DISPATCH_STRIDE: usize = 31;
const D_CONTACT: usize = 0;
const D_TYPE_A: usize = 1;
const D_TYPE_B: usize = 2;
const D_XF_A: usize = 3; // p3 + q4
const D_XF_B: usize = 10; // p3 + q4
const D_GEOM_A: usize = 17; // ≤7 slots (sphere c3+r / capsule c1_3+c2_3+r / hull geoIndex)
const D_GEOM_B: usize = 24; // ≤7 slots

static mut DISPATCH_PTR: u32 = 0;
static mut DISPATCH_OUT_PTR: u32 = 0;

/// Lay out the dispatch input + output columns for `count` convex records, growing memory to fit. Placed
/// at `solver_base` (past the persistent + geometry regions); the collect pass fills the input column and
/// the finish pass reads the output, both within collide — before the solver columns reserve over the
/// same base.
#[export_name = "reserveDispatch"]
pub extern "C" fn reserve_dispatch(count: usize) {
    unsafe {
        let mut off = solver_base();
        DISPATCH_PTR = off as u32;
        off += count * DISPATCH_STRIDE * 4;
        DISPATCH_OUT_PTR = off as u32;
        off += count * 4;
        ensure_capacity(off);
    }
}

#[export_name = "dispatchPtr"]
pub extern "C" fn dispatch_ptr() -> *const u32 {
    unsafe { DISPATCH_PTR as *const u32 }
}

#[export_name = "dispatchOutPtr"]
pub extern "C" fn dispatch_out_ptr() -> *const u32 {
    unsafe { DISPATCH_OUT_PTR as *const u32 }
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
/// friction/twist/rolling (solver-owned, persistent) and per-point baseSeparation (TS finish pass) are
/// left untouched.
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
        pool.set(p + P_NORMAL_IMPULSE, pt.normal_impulse);
        pool.set(p + P_TOTAL_NORMAL_IMPULSE, pt.total_normal_impulse);
        pool.set(p + P_NORMAL_VELOCITY, pt.normal_velocity);
        pool.set(p + P_FEATURE_ID, f32::from_bits(pt.feature_id));
        pool.set(p + P_TRIANGLE_INDEX, f32::from_bits(pt.triangle_index as u32));
        pool.set(p + P_PERSISTED, f32::from_bits(pt.persisted as u32));
    }
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

/// Run the convex-manifold bridge for the dispatch records in `[start, end)` of a column of `total`:
/// gather the shapes + transforms from the dispatch column, the warm-start manifold + GJK/SAT cache from
/// the persistent columns keyed by contactId, compute the new manifold in place, and write the touching
/// result into the output column.
///
/// One block of the parallel sweep (`parfor.rs`), or the whole column on the serial path. Records are
/// independent — each reads its own dispatch record and writes only its own contact's manifold + cache
/// slots and its own output slot — so blocks are write-disjoint and the partition is free.
///
/// # Safety
/// `reserve_dispatch(total)` must have run this step, and no thread may grow memory while this runs.
pub(crate) unsafe fn convex_block(start: usize, end: usize, total: usize) {
    unsafe {
        let disp = core::slice::from_raw_parts(DISPATCH_PTR as *const u32, total * DISPATCH_STRIDE);
        // `out` is a `Col`, not a `&mut [u32]`: the sweep runs per block on several threads, each writing
        // its own records of the one output column (col.rs).
        let out = Col::new(DISPATCH_OUT_PTR as *mut u32, total);
        let dir = manifolds::dir_col();
        let pool = manifolds::pool_col();
        for i in start..end {
            let r = i * DISPATCH_STRIDE;
            let contact_id = disp[r + D_CONTACT] as usize;
            let type_a = disp[r + D_TYPE_A];
            let type_b = disp[r + D_TYPE_B];
            let xf_a = read_xf(disp, r + D_XF_A);
            let xf_b = read_xf(disp, r + D_XF_B);
            let shape_a = read_shape(type_a, disp, r + D_GEOM_A);
            let shape_b = read_shape(type_b, disp, r + D_GEOM_B);

            let base = read_dir(dir, contact_id).manifold_base;
            let uses_sat = type_a == TY_HULL && type_b == TY_HULL;
            let uses_simplex = type_a == TY_HULL && (type_b == TY_SPHERE || type_b == TY_CAPSULE);
            let mut cache = ConvexContactCache::empty();
            if uses_sat {
                cache.sat_cache = read_sat(dir, contact_id);
            } else if uses_simplex {
                cache.simplex_cache = read_simplex(dir, contact_id);
            }

            let mut m = read_manifold_warm(pool, base);
            let touching =
                compute_convex_manifold(&mut m, &shape_a, xf_a, &shape_b, xf_b, &mut cache);
            write_manifold(&m, pool, base);
            if uses_sat {
                write_sat(dir, contact_id, &cache.sat_cache);
            } else if uses_simplex {
                write_simplex(dir, contact_id, &cache.simplex_cache);
            }
            out.set(i, touching as u32);
        }
    }
}

/// The whole convex dispatch column, on the calling thread (the serial path).
#[export_name = "dispatchConvex"]
pub extern "C" fn dispatch_convex(count: usize) {
    unsafe { convex_block(0, count, count) }
}

// --- contact-recycle batched pass (4b.3c) ---------------------------------------------------
// One kernel call runs the recycle branch of box3d's `b3CollideTask` for every dynamic-dynamic direct
// convex contact the collide walk gathered (the partition mirrors 3c: static-involved / mesh / compound
// contacts keep the TS per-contact path). Post-settle nearly every contact recycles, so this collapses
// the collide phase's dominant cost — the per-contact JS object walk — into one FFI crossing over the
// resident columns. Per contact: fat-AABB overlap (from the resident fat-AABB column), then the recycle
// gate + separation update (`try_recycle`, gold-verified in recycle.rs) over the resident body columns +
// the pose cache folded into the directory. The result (0 recycled / 1 needs-narrowphase / 2 disjoint)
// tells the TS finish pass what to do; the pose cache is written in-kernel on the needs-narrowphase path.

/// u32 stride of a recycle input record, matching `src/columns.ts` `RECYCLE_STRIDE`.
const RECYCLE_STRIDE: usize = 6;
const R_CONTACT: usize = 0;
const R_LOCAL_A: usize = 1; // body A's awake localIndex (resident-column record)
const R_LOCAL_B: usize = 2;
const R_SHAPE_A: usize = 3; // shapeId → fat-AABB column record
const R_SHAPE_B: usize = 4;
const R_BITS: usize = 5;
/// bit0: the contact may recycle this step (recycleDistance>0 && relativeTransformValid && recycleFlag).
const R_ELIGIBLE: u32 = 1;
/// bit1: the contact was touching at step entry (selects the recycle tolerance).
const R_WAS_TOUCHING: u32 = 2;

static mut RECYCLE_PTR: u32 = 0;
static mut RECYCLE_OUT_PTR: u32 = 0;

/// Lay out the recycle input + output columns for `count` records at `solver_base` (past the persistent
/// + geometry regions), growing memory to fit. Consumed within collide, before the convex dispatch and
/// the solver columns reserve over the same base — the recycle pass finishes before either runs.
#[export_name = "reserveRecycle"]
pub extern "C" fn reserve_recycle(count: usize) {
    unsafe {
        let mut off = solver_base();
        RECYCLE_PTR = off as u32;
        off += count * RECYCLE_STRIDE * 4;
        RECYCLE_OUT_PTR = off as u32;
        off += count * 4;
        ensure_capacity(off);
    }
}

#[export_name = "recyclePtr"]
pub extern "C" fn recycle_ptr() -> *const u32 {
    unsafe { RECYCLE_PTR as *const u32 }
}

#[export_name = "recycleOutPtr"]
pub extern "C" fn recycle_out_ptr() -> *const u32 {
    unsafe { RECYCLE_OUT_PTR as *const u32 }
}

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
/// [`convex_block`], records are independent: the body / fat-AABB columns are read-only here, and every
/// write lands in the record's own contact's directory + manifold slots.
///
/// # Safety
/// `reserve_recycle(total)` must have run this step, and no thread may grow memory while this runs.
pub(crate) unsafe fn recycle_block(
    start: usize,
    end: usize,
    total: usize,
    recycle_dist: f32,
    recycle_dist_non_touching: f32,
) {
    unsafe {
        let input = core::slice::from_raw_parts(RECYCLE_PTR as *const u32, total * RECYCLE_STRIDE);
        // As `convex_block`: the output column is shared-mutable, one record per input record.
        let out = Col::new(RECYCLE_OUT_PTR as *mut u32, total);
        let dir = manifolds::dir_col();
        let pool = manifolds::pool_col();
        let fat = crate::fataabb::col_slice();
        let cap = crate::bodies::body_cap();
        let sim_ptr = crate::bodies::sim_base() as *const f32;
        let fin_ptr = crate::bodies::fin_base() as *const f32;
        let sim = core::slice::from_raw_parts(sim_ptr, cap * SIM_STRIDE);
        let fin = core::slice::from_raw_parts(fin_ptr, cap * FIN_STRIDE);

        for i in start..end {
            let r = i * RECYCLE_STRIDE;
            let contact_id = input[r + R_CONTACT] as usize;

            // Fat-AABB overlap first — matching the TS collide's first per-contact check.
            if !fat_overlap(fat, input[r + R_SHAPE_A] as usize, input[r + R_SHAPE_B] as usize) {
                out.set(i, 2);
                continue;
            }

            let la = input[r + R_LOCAL_A] as usize;
            let lb = input[r + R_LOCAL_B] as usize;
            let bits = input[r + R_BITS];
            let xf_a = read_body_xf(sim, fin, la);
            let xf_b = read_body_xf(sim, fin, lb);
            let tol = if bits & R_WAS_TOUCHING != 0 {
                recycle_dist
            } else {
                recycle_dist_non_touching
            };

            if bits & R_ELIGIBLE != 0 {
                let (rot_a, rot_b, rel) = read_pose_cache(dir, contact_id);
                let mc = read_dir(dir, contact_id).manifold_count;
                if try_recycle(
                    dir,
                    pool,
                    contact_id,
                    mc,
                    xf_a,
                    xf_b,
                    rot_a,
                    rot_b,
                    rel,
                    read_center(fin, la),
                    read_center(fin, lb),
                    read_max_extent(fin, la),
                    read_max_extent(fin, lb),
                    tol,
                ) {
                    out.set(i, 0);
                    continue;
                }
            }

            // Recycle missed (or the contact isn't eligible yet): cache this step's pose for the next
            // step and defer to the full narrowphase.
            write_pose_cache(dir, contact_id, xf_a, xf_b);
            out.set(i, 1);
        }
    }
}

/// The whole recycle column, on the calling thread (the serial path).
#[export_name = "dispatchRecycle"]
pub extern "C" fn dispatch_recycle(count: usize, recycle_dist: f32, recycle_dist_non_touching: f32) {
    unsafe { recycle_block(0, count, count, recycle_dist, recycle_dist_non_touching) }
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
        refit_block(sim, fin, start, end);
    }
}

/// Walk each body's shape list and compute the finalize refit for its convex shapes: the speculative-
/// inflated tight AABB (candidate) + whether it escaped the resident fat AABB. Reads the body's advanced
/// pose from the just-finalized `sim`/`fin` columns and the shape geometry + list links from the resident
/// shape column (`shapes.rs`, walked head → `next` from the body's `S2_HEAD_SHAPE` lane); writes the
/// candidate + escaped flag back into that shape column. It does **not** commit — TS margin-inflates the
/// escaped shapes and touches the fat column + broad phase serially (`solver.ts`), because the fast/CCD
/// branch that would invalidate a fast body's refit is TS-owned. Fallback shapes (mesh/height-field/
/// compound) are skipped; TS computes them at their list position.
///
/// Per-body write-disjoint: a shape belongs to one body, so two parallel-for blocks never write the same
/// shape record — the shared-mutable [`Col`] carries that promise. Indexed only through the awake head
/// lane → `next` chain, never a `0..cap` sweep, so it never reads a stale record outside a live chain
/// (`shapes.rs` reachability contract).
///
/// # Safety
/// The body + shape + fat-AABB regions must be reserved for every reachable shape, and no thread may grow
/// memory while this runs (the MT concurrency invariant, `.claude/rules/tumble.md`).
unsafe fn refit_block(sim: Col<f32>, fin: Col<f32>, start: usize, end: usize) {
    unsafe {
        let records = crate::bodies::body_cap() + crate::bodies::IDENT_RECORDS;
        let sim2 =
            core::slice::from_raw_parts(crate::bodies::sim2_base() as *const u32, records * SIM2_STRIDE);
        let shape_u = crate::shapes::col();
        let shape_f = crate::shapes::col_f();
        let fat = crate::fataabb::col_slice();
        for i in start..end {
            let so = i * SIM_STRIDE;
            let fo = i * FIN_STRIDE;
            let xf = Transform {
                p: Vec3::new(fin.get(fo + 9), fin.get(fo + 10), fin.get(fo + 11)),
                q: Quat {
                    v: Vec3::new(sim.get(so + 28), sim.get(so + 29), sim.get(so + 30)),
                    s: sim.get(so + 31),
                },
            };
            let mut shape_id = sim2[i * SIM2_STRIDE + S2_HEAD_SHAPE];
            while shape_id != crate::shapes::NULL_SHAPE {
                let o = shape_id as usize * crate::shapes::SHAPE_STRIDE;
                let ty = shape_u.get(o + crate::shapes::S_TYPE);
                if finalize::is_convex_refit(ty) {
                    let g = o + crate::shapes::S_GEOM;
                    let geom = [
                        shape_f.get(g),
                        shape_f.get(g + 1),
                        shape_f.get(g + 2),
                        shape_f.get(g + 3),
                        shape_f.get(g + 4),
                        shape_f.get(g + 5),
                        shape_f.get(g + 6),
                    ];
                    let fb = shape_id as usize * FAT_STRIDE;
                    let fat_aabb = [
                        fat[fb],
                        fat[fb + 1],
                        fat[fb + 2],
                        fat[fb + 3],
                        fat[fb + 4],
                        fat[fb + 5],
                    ];
                    let (cand, escaped) = finalize::refit_convex(ty, &geom, xf, &fat_aabb);
                    let c = o + crate::shapes::S_CAND;
                    shape_f.set(c, cand[0]);
                    shape_f.set(c + 1, cand[1]);
                    shape_f.set(c + 2, cand[2]);
                    shape_f.set(c + 3, cand[3]);
                    shape_f.set(c + 4, cand[4]);
                    shape_f.set(c + 5, cand[5]);
                    shape_u.set(o + crate::shapes::S_ESCAPED, escaped as u32);
                }
                shape_id = shape_u.get(o + crate::shapes::S_NEXT);
            }
        }
    }
}

#[export_name = "finalize"]
pub extern "C" fn finalize(h: f32, inv_dt: f32, enable_continuous: u32) {
    unsafe { finalize_block(0, BODY_COUNT, h, inv_dt, enable_continuous != 0) }
}

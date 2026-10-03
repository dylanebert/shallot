//! Static geometry columns: hull topology for narrowphase and mesh, height-field and compound
//! records for queries, uploaded on geometry-set changes rather than per step. Wasm-only — the pools alias linear memory, and
//! `hull_view` reinterprets them into the borrowed `HullData` view (kernel/src/hull.rs) the narrowphase
//! consumes. Native `cargo test` drives `HullData` over owned `Vec`s instead.
//!
//! Layout of linear memory:
//! ```text
//! [0, heap_base)              Rust statics (LAYOUT / BODY_LAYOUT / FATAABB_LAYOUT / … headers)
//! [heap_base, body_end)       persistent body columns (bodies.rs)
//! [body_end, fataabb_end)     persistent fat-AABB column (fataabb.rs)
//! [fataabb_end, shape_end)    persistent shape column (shapes.rs)
//! [shape_end, manifold_end)   persistent manifold columns (manifolds.rs)
//! [manifold_end, geo_end)     static geometry pools (persist across steps; TS rewrites on a geometry-set change)
//! [geo_end, ...)              per-step solver columns (arena::reserve lays these out from geo_end)
//! ```
//! The geometry region sits after the manifold region and before the per-step solver columns, so a
//! `reserve` never overwrites it; it re-uploads when the manifold region grows and shifts it.
//! TS is the source of truth: on a geometry-set change it re-uploads the resident world's geometry compactly, so growth and
//! renumbering need no in-place preservation here. A body or fat-AABB region grow below shifts this
//! region up too — `relocate` rebases its offsets (the caller memmoves the bytes; no re-upload).

use crate::hull::{HullData, HullFace, HullHalfEdge, HullVertex};
use crate::manifold::{collide_hulls, make_feature_id, LocalManifold, SatCache};
use crate::math::{Plane, Quat, Transform, Vec3};

const PAGE: usize = 65536;

/// u32 words per hull record: center.xyz (f32 bits), vertex/edge/face counts, and the element offset
/// of this hull's slice into each of the five pools; slot 11 is padding.
const RECORD_STRIDE: usize = 12;

// GEO_LAYOUT indices (byte offsets into linear memory), in memory order.
const REC: usize = 0;
const POINTS: usize = 1;
const VERTICES: usize = 2;
const EDGES: usize = 3;
const FACES: usize = 4;
const PLANES: usize = 5;
const EXTRA: usize = 6;
const N_GEO: usize = 7;

static mut GEO_LAYOUT: [u32; N_GEO] = [0; N_GEO];
/// First free byte past the geometry region — the base `arena::reserve` lays the solver columns from.
/// Zero until the first `reserveGeometry`; `solver_base` treats that as an empty region.
static mut GEO_END: u32 = 0;

extern "C" {
    static __heap_base: u8;
}

/// The 4-aligned byte offset where linear memory's dynamic region begins (past the Rust statics).
#[inline]
pub fn heap_base() -> usize {
    (unsafe { &__heap_base as *const u8 as usize } + 3) & !3
}

/// Base byte offset the per-step solver columns start from: past the geometry region if one was
/// uploaded, else where the geometry region would begin (past the persistent manifold columns).
#[inline]
pub fn solver_base() -> usize {
    let end = unsafe { GEO_END } as usize;
    if end == 0 {
        crate::broad::region_top()
    } else {
        end
    }
}

/// First free byte past the geometry region (0 if none reserved).
#[inline]
pub fn region_end() -> usize {
    unsafe { GEO_END as usize }
}

/// Shift the geometry region's byte offsets up by `delta` after the body region below it grew and
/// moved it (the caller memmoves the bytes). No-op if no region is reserved. The pools are indexed by
/// element offset within each hull record, so only the header offsets + end marker rebase.
pub fn relocate(delta: usize) {
    unsafe {
        if GEO_END == 0 {
            return;
        }
        for i in 0..N_GEO {
            GEO_LAYOUT[i] += delta as u32;
        }
        GEO_END += delta as u32;
    }
}

unsafe fn ensure_capacity(end_byte: usize) {
    crate::treework::record_end(end_byte);
    let have = core::arch::wasm32::memory_size(0) * PAGE;
    if end_byte > have {
        let pages = (end_byte - have + PAGE - 1) / PAGE;
        core::arch::wasm32::memory_grow(0, pages);
    }
}

/// Byte offset of the geometry layout header (`[u32; N_GEO]` of per-pool byte offsets). TS writes the
/// hull records + pools through views derived from this after every `reserveGeometry`.
#[export_name = "geoLayoutPtr"]
pub extern "C" fn geo_layout_ptr() -> *const u32 {
    &raw const GEO_LAYOUT as *const u32
}

/// Lay out the hull pools and `extra_words` of non-convex records, growing memory to fit, and record
/// `geo_end` so the next `reserve` places the solver columns after them. `verts` sizes both the point
/// pool (3 f32 each) and the vertex pool (1 u32 each); `faces` sizes both the face pool (1 u32 each)
/// and the plane pool (4 f32 each). TS then rewrites every hull's record + pool data.
#[export_name = "reserveGeometry"]
pub extern "C" fn reserve_geometry(
    hulls: usize,
    verts: usize,
    edges: usize,
    faces: usize,
    extra_words: usize,
) {
    unsafe {
        // Start past the persistent broad-phase region (empty until the first `reserveBroad`).
        let mut off = crate::broad::region_top();
        GEO_LAYOUT[REC] = off as u32;
        off += hulls * RECORD_STRIDE * 4;
        GEO_LAYOUT[POINTS] = off as u32;
        off += verts * 3 * 4;
        GEO_LAYOUT[VERTICES] = off as u32;
        off += verts * 4;
        GEO_LAYOUT[EDGES] = off as u32;
        off += edges * 4 * 4;
        GEO_LAYOUT[FACES] = off as u32;
        off += faces * 4;
        GEO_LAYOUT[PLANES] = off as u32;
        off += faces * 4 * 4;
        GEO_LAYOUT[EXTRA] = off as u32;
        off += extra_words * 4;
        // 4-align the solver base (pool sizes are already word multiples, so this is a no-op, but keep
        // the invariant explicit).
        GEO_END = ((off + 3) & !3) as u32;
        ensure_capacity(GEO_END as usize);
    }
}

/// A borrowed `HullData` view over interned hull `index`'s slices in the geometry pools. The point and
/// plane pools reinterpret directly as `&[Vec3]` / `&[Plane]` (repr(C)); the topology pools as
/// `&[HullVertex]` / `&[HullHalfEdge]` / `&[HullFace]` (repr(C), `usize` == u32 on wasm32).
pub(crate) unsafe fn hull_view(index: usize) -> HullData<'static> {
    let rec = (GEO_LAYOUT[REC] as *const u32).add(index * RECORD_STRIDE);
    let center = Vec3::new(
        f32::from_bits(*rec),
        f32::from_bits(*rec.add(1)),
        f32::from_bits(*rec.add(2)),
    );
    let vertex_count = *rec.add(3) as usize;
    let edge_count = *rec.add(4) as usize;
    let face_count = *rec.add(5) as usize;
    let point_off = *rec.add(6) as usize;
    let vertex_off = *rec.add(7) as usize;
    let edge_off = *rec.add(8) as usize;
    let face_off = *rec.add(9) as usize;
    let plane_off = *rec.add(10) as usize;

    let points = core::slice::from_raw_parts(
        (GEO_LAYOUT[POINTS] as *const f32).add(point_off * 3) as *const Vec3,
        vertex_count,
    );
    let vertices = core::slice::from_raw_parts(
        (GEO_LAYOUT[VERTICES] as *const u32).add(vertex_off) as *const HullVertex,
        vertex_count,
    );
    let edges = core::slice::from_raw_parts(
        (GEO_LAYOUT[EDGES] as *const u32).add(edge_off * 4) as *const HullHalfEdge,
        edge_count,
    );
    let faces = core::slice::from_raw_parts(
        (GEO_LAYOUT[FACES] as *const u32).add(face_off) as *const HullFace,
        face_count,
    );
    let planes = core::slice::from_raw_parts(
        (GEO_LAYOUT[PLANES] as *const f32).add(plane_off * 4) as *const Plane,
        face_count,
    );

    HullData {
        center,
        vertex_count,
        edge_count,
        face_count,
        points,
        vertices,
        edges,
        faces,
        planes,
    }
}

/// Address a word in the non-convex geometry pool. References within the pool are word offsets,
/// so relocation rebases only GEO_LAYOUT, just as it does for the hull pools.
pub(crate) unsafe fn extra_ptr(index: usize) -> *const u32 {
    (GEO_LAYOUT[EXTRA] as *const u32).add(index)
}

pub(crate) unsafe fn mesh_view(index: usize, scale: Vec3) -> crate::mesh_query::Mesh<'static> {
    use crate::mesh_query::{Mesh, MeshNode, MeshTriangle};
    let r = extra_ptr(index);
    Mesh {
        nodes: core::slice::from_raw_parts(
            extra_ptr(*r.add(3) as usize) as *const MeshNode,
            *r as usize,
        ),
        vertices: core::slice::from_raw_parts(
            extra_ptr(*r.add(4) as usize) as *const Vec3,
            *r.add(1) as usize,
        ),
        triangles: core::slice::from_raw_parts(
            extra_ptr(*r.add(5) as usize) as *const MeshTriangle,
            *r.add(2) as usize,
        ),
        materials: core::slice::from_raw_parts(extra_ptr(*r.add(7) as usize), *r.add(2) as usize),
        scale,
    }
}

pub(crate) unsafe fn height_view(index: usize) -> crate::height_query::HeightField<'static> {
    use crate::height_query::HeightField;
    let r = extra_ptr(index);
    let f = r as *const f32;
    let columns = *r.add(12) as usize;
    let rows = *r.add(13) as usize;
    HeightField {
        lower: *(f as *const Vec3),
        upper: *(f.add(3) as *const Vec3),
        min_height: *f.add(6),
        height_scale: *f.add(8),
        scale: *(f.add(9) as *const Vec3),
        columns,
        rows,
        clockwise: *r.add(14) != 0,
        heights: core::slice::from_raw_parts(extra_ptr(*r.add(15) as usize), columns * rows),
        materials: core::slice::from_raw_parts(
            extra_ptr(*r.add(16) as usize),
            (columns - 1) * (rows - 1),
        ),
    }
}

// --- geometry-read verification -------------------------------------------------------------
// Runs the hull-hull narrowphase over two column-backed hull views end-to-end, proving the wasm
// reinterpret above matches the native Vec-backed gold (kernel.test.ts asserts the output bit-for-bit
// against manifold.gold.json). This is the whole geometry read path — all five pools plus center — and
// the seed of 3c.3's real convex dispatch. Output buffer holds pointCount, normal, then each point's
// point.xyz / separation / featureId.

#[export_name = "collideSpheresGeo"]
pub extern "C" fn collide_spheres_geo(
    ax: f32,
    ay: f32,
    az: f32,
    ar: f32,
    bx: f32,
    by: f32,
    bz: f32,
    br: f32,
    px: f32,
    py: f32,
    pz: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    qs: f32,
) -> usize {
    let mut m = LocalManifold::new();
    crate::manifold::collide_spheres(
        &mut m,
        4,
        &crate::manifold::Sphere {
            center: Vec3::new(ax, ay, az),
            radius: ar,
        },
        &crate::manifold::Sphere {
            center: Vec3::new(bx, by, bz),
            radius: br,
        },
        Transform {
            p: Vec3::new(px, py, pz),
            q: Quat {
                v: Vec3::new(qx, qy, qz),
                s: qs,
            },
        },
    );
    unsafe {
        let out = &raw mut GEO_OUT as *mut f32;
        *out = m.point_count as f32;
        *out.add(1) = m.normal.x;
        *out.add(2) = m.normal.y;
        *out.add(3) = m.normal.z;
        for i in 0..m.point_count {
            let p = m.points[i];
            let o = 4 + i * 5;
            *out.add(o) = p.point.x;
            *out.add(o + 1) = p.point.y;
            *out.add(o + 2) = p.point.z;
            *out.add(o + 3) = p.separation;
            *(out.add(o + 4) as *mut u32) = make_feature_id(p.pair);
        }
    }
    m.point_count
}

const OUT_LEN: usize = 4 + 8 * 5;
static mut GEO_OUT: [f32; OUT_LEN] = [0.0; OUT_LEN];
static mut GEO_TRIANGLE_OUT: [i32; 8] = [0; 8];
#[export_name = "geoTriangleOutPtr"]
pub extern "C" fn geo_triangle_out_ptr() -> *const i32 {
    &raw const GEO_TRIANGLE_OUT as *const i32
}

/// Byte offset of the verification output buffer (kernel.test.ts reads it as a `Float32Array`).
#[export_name = "geoOutPtr"]
pub extern "C" fn geo_out_ptr() -> *const f32 {
    &raw const GEO_OUT as *const f32
}

/// Collide interned hulls `a`/`b` with `transform_b_to_a` (position + quaternion), writing the manifold
/// into `GEO_OUT`. Returns the point count. Fresh SAT cache each call (matches the gold's first call).
#[export_name = "collideHullsGeo"]
pub extern "C" fn collide_hulls_geo(
    a: usize,
    b: usize,
    px: f32,
    py: f32,
    pz: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    qs: f32,
) -> usize {
    unsafe {
        let hull_a = hull_view(a);
        let hull_b = hull_view(b);
        let transform_b_to_a = Transform {
            p: Vec3::new(px, py, pz),
            q: Quat {
                v: Vec3::new(qx, qy, qz),
                s: qs,
            },
        };
        let mut m = LocalManifold::new();
        let mut cache = SatCache::empty();
        collide_hulls(&mut m, 8, &hull_a, &hull_b, transform_b_to_a, &mut cache);

        let out = &raw mut GEO_OUT as *mut f32;
        *out = m.point_count as f32;
        *out.add(1) = m.normal.x;
        *out.add(2) = m.normal.y;
        *out.add(3) = m.normal.z;
        for i in 0..m.point_count {
            let p = &m.points[i];
            (&raw mut GEO_TRIANGLE_OUT as *mut i32)
                .add(i)
                .write(p.triangle_index);
            let o = 4 + i * 5;
            *out.add(o) = p.point.x;
            *out.add(o + 1) = p.point.y;
            *out.add(o + 2) = p.point.z;
            *out.add(o + 3) = p.separation;
            // Raw u32 store (not `f32::from_bits`) so the feature id never risks NaN canonicalization
            // when TS reads it back through a Float32Array view; TS reads this slot as a u32.
            *(out.add(o + 4) as *mut u32) = make_feature_id(p.pair);
        }
        m.point_count
    }
}

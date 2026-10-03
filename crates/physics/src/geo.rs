//! Static geometry columns: hull topology for narrowphase and mesh, height-field and compound
//! records for queries, uploaded on geometry-set changes rather than per step. Wasm-only — the pools alias linear memory, and
//! `hull_view` reinterprets them into the borrowed `HullData` view (kernel/src/hull.rs) the narrowphase
//! consumes. Native `cargo test` drives `HullData` over owned `Vec`s instead.
//!
//! Each World's pools are allocator-owned. Geometry-set changes rewrite that World's pools;
//! growing another region never requires a geometry upload.

use crate::hull::{HullData, HullFace, HullHalfEdge, HullVertex};
use crate::manifold::{collide_hulls, make_feature_id, LocalManifold, SatCache};
use crate::math::{Plane, Quat, Transform, Vec3};

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

use crate::regions::{self, Columns, MAX_WORLDS};
static mut COLUMNS: [Columns<N_GEO>; MAX_WORLDS] = [Columns::EMPTY; MAX_WORLDS];

#[export_name = "geoLayoutPtr"]
pub extern "C" fn geo_layout_ptr() -> *const u32 {
    unsafe { COLUMNS[regions::active()].layout.as_ptr() }
}

#[export_name = "reserveGeometry"]
pub extern "C" fn reserve_geometry(
    hulls: usize,
    verts: usize,
    edges: usize,
    faces: usize,
    extra_words: usize,
) {
    unsafe {
        let columns = &mut COLUMNS[regions::active()];
        for (column, words) in [
            (REC, hulls * RECORD_STRIDE),
            (POINTS, verts * 3),
            (VERTICES, verts),
            (EDGES, edges * 4),
            (FACES, faces),
            (PLANES, faces * 4),
            (EXTRA, extra_words),
        ] {
            columns.reserve(column, words * 4);
        }
    }
}
pub unsafe fn reset(id: usize) {
    COLUMNS[id].release();
}
pub unsafe fn snapshot(id: usize, out: &mut Vec<u8>) {
    COLUMNS[id].snapshot(out);
}
pub unsafe fn restore(id: usize, input: &mut &[u8]) {
    COLUMNS[id].restore(input);
}

/// A borrowed `HullData` view over interned hull `index`'s slices in the geometry pools. The point and
/// plane pools reinterpret directly as `&[Vec3]` / `&[Plane]` (repr(C)); the topology pools as
/// `&[HullVertex]` / `&[HullHalfEdge]` / `&[HullFace]` (repr(C), `usize` == u32 on wasm32).
pub(crate) unsafe fn hull_view(index: usize) -> HullData<'static> {
    let layout = COLUMNS[regions::active()].layout;
    let rec = (layout[REC] as *const u32).add(index * RECORD_STRIDE);
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
        (layout[POINTS] as *const f32).add(point_off * 3) as *const Vec3,
        vertex_count,
    );
    let vertices = core::slice::from_raw_parts(
        (layout[VERTICES] as *const u32).add(vertex_off) as *const HullVertex,
        vertex_count,
    );
    let edges = core::slice::from_raw_parts(
        (layout[EDGES] as *const u32).add(edge_off * 4) as *const HullHalfEdge,
        edge_count,
    );
    let faces = core::slice::from_raw_parts(
        (layout[FACES] as *const u32).add(face_off) as *const HullFace,
        face_count,
    );
    let planes = core::slice::from_raw_parts(
        (layout[PLANES] as *const f32).add(plane_off * 4) as *const Plane,
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

/// Address a word in the active World's non-convex pool; stored references are word offsets.
pub(crate) unsafe fn extra_ptr(index: usize) -> *const u32 {
    (COLUMNS[regions::active()].layout[EXTRA] as *const u32).add(index)
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

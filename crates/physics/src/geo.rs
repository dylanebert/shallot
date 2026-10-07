//! Narrowphase borrows each retained b3HullData directly, including its relative-offset arrays.
//! Meshes, height fields and compounds are retained as identity-keyed native images.

use crate::hull::{HullData, HullFace, HullHalfEdge, HullVertex};
use crate::manifold::{collide_hulls, make_feature_id, LocalManifold, SatCache};
use crate::math::{Plane, Quat, Transform, Vec3};

/// Box3D b3HullData, with byte offsets relative to this header into its trailing arrays.
#[repr(C)]
pub(crate) struct HullRecord {
    pub version: u64,
    pub hash: u64,
    pub bounds: [Vec3; 2],
    pub surface_area: f32,
    pub volume: f32,
    pub inner_radius: f32,
    pub center: Vec3,
    pub central_inertia: crate::math::Mat3,
    pub vertex_count: i32,
    pub vertex_offset: i32,
    pub point_offset: i32,
    pub edge_count: i32,
    pub edge_offset: i32,
    pub face_count: i32,
    pub plane_offset: i32,
    pub face_offset: i32,
    pub soa_vertex_offset: i32,
    pub soa_normal_offset: i32,
    pub byte_count: i32,
}
#[inline(always)]
pub(crate) unsafe fn hull_record(index: usize) -> &'static HullRecord {
    &*(index as *const HullRecord)
}

pub unsafe fn reset(id: usize) {
    crate::hull_database::reset(id);
    crate::geometry_database::reset(id);
}
pub unsafe fn snapshot(id: usize, out: &mut Vec<u8>) {
    crate::hull_database::snapshot(id, out);
    crate::geometry_database::snapshot(id, out);
}
pub unsafe fn restore(id: usize, input: &mut &[u8]) {
    crate::hull_database::restore(id, input);
    crate::geometry_database::restore(id, input);
}

/// Borrow the arrays hanging off a b3HullData header (`usize` is u32 on wasm32).
pub(crate) unsafe fn hull_view(index: usize) -> HullData<'static> {
    let rec = hull_record(index);
    let base = rec as *const HullRecord as *const u8;
    let center = rec.center;
    let vertex_count = rec.vertex_count as usize;
    let edge_count = rec.edge_count as usize;
    let face_count = rec.face_count as usize;
    let nv = (vertex_count + 3) & !3;
    let nf = (face_count + 3) & !3;
    let soa = base.offset(rec.soa_vertex_offset as isize) as *const f32;
    let points = core::slice::from_raw_parts(
        base.offset(rec.point_offset as isize) as *const Vec3,
        vertex_count,
    );
    let vertices = core::slice::from_raw_parts(
        base.offset(rec.vertex_offset as isize) as *const HullVertex,
        vertex_count,
    );
    let edges = core::slice::from_raw_parts(
        base.offset(rec.edge_offset as isize) as *const HullHalfEdge,
        edge_count,
    );
    let faces = core::slice::from_raw_parts(
        base.offset(rec.face_offset as isize) as *const HullFace,
        face_count,
    );
    let planes = core::slice::from_raw_parts(
        base.offset(rec.plane_offset as isize) as *const Plane,
        face_count,
    );

    HullData {
        center,
        bounds: rec.bounds,
        vertex_count,
        edge_count,
        face_count,
        points,
        soa_points: core::slice::from_raw_parts(soa, 3 * nv).into(),
        soa_normals: core::slice::from_raw_parts(
            base.offset(rec.soa_normal_offset as isize) as *const f32,
            3 * nf,
        )
        .into(),
        vertices,
        edges,
        faces,
        planes,
    }
}

pub(crate) unsafe fn mesh_view(r: *const u32, scale: Vec3) -> crate::mesh_query::Mesh<'static> {
    use crate::mesh_query::{Mesh, MeshNode, MeshTriangle};
    Mesh {
        nodes: core::slice::from_raw_parts(
            r.cast::<u8>().add(*r.add(14) as usize).cast::<MeshNode>(),
            *r.add(15) as usize,
        ),
        vertices: core::slice::from_raw_parts(
            r.cast::<u8>().add(*r.add(16) as usize).cast::<Vec3>(),
            *r.add(17) as usize,
        ),
        triangles: core::slice::from_raw_parts(
            r.cast::<u8>()
                .add(*r.add(18) as usize)
                .cast::<MeshTriangle>(),
            *r.add(19) as usize,
        ),
        materials: core::slice::from_raw_parts(
            r.cast::<u8>().add(*r.add(20) as usize),
            *r.add(19) as usize,
        ),
        scale,
    }
}

pub(crate) unsafe fn height_view(r: *const u32) -> crate::height_query::HeightField<'static> {
    use crate::height_query::HeightField;
    let f = r.cast::<f32>();
    let columns = *r.add(17) as usize;
    let rows = *r.add(18) as usize;
    HeightField {
        lower: *(f.add(5) as *const Vec3),
        upper: *(f.add(8) as *const Vec3),
        min_height: *f.add(11),
        height_scale: *f.add(13),
        scale: *(f.add(14) as *const Vec3),
        columns,
        rows,
        clockwise: *r.add(22) & 255 != 0,
        heights: core::slice::from_raw_parts(
            r.cast::<u8>().add(*r.add(19) as usize).cast::<u16>(),
            columns * rows,
        ),
        materials: core::slice::from_raw_parts(
            r.cast::<u8>().add(*r.add(20) as usize),
            (columns - 1) * (rows - 1),
        ),
    }
}

static mut COMPOUND_CHILD_OUT: [u32; 10] = [0; 10];
#[export_name = "shapeMaterialIndex"]
pub extern "C" fn shape_material_index(
    world: usize,
    shape: usize,
    child: usize,
    triangle: usize,
) -> usize {
    unsafe {
        let (record, pointer) = crate::shapes::geometry_record(world, shape);
        match *record {
            1 => crate::compound_query::child_material_index(pointer, child, triangle) as usize,
            2 => height_view(pointer).materials[triangle >> 1] as usize,
            4 => {
                let scale = Vec3::new(
                    f32::from_bits(*record.add(3)),
                    f32::from_bits(*record.add(4)),
                    f32::from_bits(*record.add(5)),
                );
                mesh_view(pointer, scale).materials[triangle] as usize
            }
            _ => 0,
        }
    }
}
#[export_name = "shapeCompoundChildType"]
pub extern "C" fn shape_compound_child_type(world: usize, shape: usize, child: usize) -> u32 {
    unsafe {
        let (record, pointer) = crate::shapes::geometry_record(world, shape);
        assert_eq!(*record, 1);
        crate::compound_query::child_words(pointer, child)[0]
    }
}
#[export_name = "shapeCompoundChild"]
pub extern "C" fn shape_compound_child(world: usize, shape: usize, child: usize) -> *const u32 {
    unsafe {
        let (record, pointer) = crate::shapes::geometry_record(world, shape);
        assert_eq!(*record, 1);
        let words = crate::compound_query::child_words(pointer, child);
        let out = &raw mut COMPOUND_CHILD_OUT as *mut u32;
        *out = words[0];
        *out.add(1) = crate::compound_query::child_material_index(pointer, child, 0);
        *out.add(2) = match words[0] {
            0 => words[18],
            3 => {
                let hull = words[12] as *const u32;
                *hull.add(12)
            }
            5 => words[15],
            _ => 0,
        };
        for i in 0..7 {
            *out.add(3 + i) = words[1 + i];
        }
        out
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
    let mut points = [crate::manifold::LocalManifoldPoint::ZERO; 32];
    let mut m = LocalManifold::new(&mut points);
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
        let mut points = [crate::manifold::LocalManifoldPoint::ZERO; 32];
        let mut m = LocalManifold::new(&mut points);
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

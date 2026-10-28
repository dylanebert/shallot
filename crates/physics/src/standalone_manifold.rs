//! Borrowed standalone manifold query scratch, local to each kernel instance.
use crate::distance::SimplexCache;
use crate::math::{Quat, Transform, Vec3};
#[repr(C, align(8))]
struct Input([u32; 16384]);
static mut INPUT: Input = Input([0; 16384]);
static mut OUTPUT: [u32; 512] = [0; 512];
#[export_name = "manifoldInputPtr"]
pub extern "C" fn input_ptr() -> *mut u32 {
    (&raw mut INPUT).cast()
}
#[export_name = "manifoldOutputPtr"]
pub extern "C" fn output_ptr() -> *const u32 {
    (&raw const OUTPUT).cast()
}
#[export_name = "manifoldQuery"]
pub unsafe extern "C" fn query(operation: u32) {
    let r = &(*(&raw const INPUT)).0;
    let out = &mut *(&raw mut OUTPUT);
    out.fill(0);
    let f = |i: usize| f32::from_bits(r[i]);
    use crate::manifold::*;
    let v = |i| Vec3::new(f(i), f(i + 1), f(i + 2));
    let xf = Transform {
        p: v(4),
        q: Quat { v: v(7), s: f(10) },
    };
    let ha = crate::geo::hull_view(r.as_ptr().add(128) as usize);
    let hb = crate::geo::hull_view(r.as_ptr().add(8192) as usize);
    let sa = Sphere {
        center: v(12),
        radius: f(15),
    };
    let sb = Sphere {
        center: v(16),
        radius: f(19),
    };
    let ca = Capsule {
        center1: v(20),
        center2: v(23),
        radius: f(26),
    };
    let cb = Capsule {
        center1: v(27),
        center2: v(30),
        radius: f(33),
    };
    let tri = [v(34), v(37), v(40)];
    let mut cache = SimplexCache {
        metric: f(50),
        count: r[51] as u16,
        index_a: [r[52] as u8, r[53] as u8, r[54] as u8, r[55] as u8],
        index_b: [r[56] as u8, r[57] as u8, r[58] as u8, r[59] as u8],
    };
    let mut sat = SatCache {
        separation: f(44),
        ty: r[45] as u8,
        index_a: r[46] as u8,
        index_b: r[47] as u8,
        hit: r[48] as u8,
    };
    let mut points = [crate::manifold::LocalManifoldPoint::ZERO; 64];
    let mut m = LocalManifold::new(&mut points);
    m.point_count = r[1] as usize;
    m.feature = r[2];
    m.squared_distance = f(3);
    let capacity = r[0] as usize;
    match operation {
        10 => collide_spheres(&mut m, capacity, &sa, &sb, xf),
        11 => collide_capsule_and_sphere(&mut m, capacity, &ca, &sb, xf),
        12 => collide_hull_and_sphere(&mut m, capacity, &ha, &sb, xf, &mut cache),
        13 => collide_capsules(&mut m, capacity, &ca, &cb, xf),
        14 => collide_hull_and_capsule(&mut m, capacity, &ha, &cb, xf, &mut cache),
        15 => collide_hulls(&mut m, capacity, &ha, &hb, xf, &mut sat),
        16 => crate::triangle_manifold::collide_sphere_and_triangle(
            &mut m, capacity, &sa, tri[0], tri[1], tri[2],
        ),
        17 => crate::triangle_manifold::collide_capsule_and_triangle(
            &mut m, capacity, &ca, tri[0], tri[1], tri[2], &mut cache,
        ),
        18 => crate::triangle_manifold::collide_hull_and_triangle(
            &mut m,
            capacity,
            &ha,
            tri[0],
            tri[1],
            tri[2],
            r[43],
            &mut sat,
            r[49] != 0,
        ),
        _ => unreachable!(),
    }
    out[..6].copy_from_slice(&[
        m.point_count as u32,
        m.normal.x.to_bits(),
        m.normal.y.to_bits(),
        m.normal.z.to_bits(),
        m.feature,
        m.squared_distance.to_bits(),
    ]);
    let mut n = 6;
    for p in &m.points[..m.point_count] {
        out[n..n + 6].copy_from_slice(&[
            p.point.x.to_bits(),
            p.point.y.to_bits(),
            p.point.z.to_bits(),
            p.separation.to_bits(),
            make_feature_id(p.pair),
            p.triangle_index as u32,
        ]);
        n += 6;
    }
    out[n..n + 9].copy_from_slice(&[
        1,
        m.triangle_normal.x.to_bits(),
        m.triangle_normal.y.to_bits(),
        m.triangle_normal.z.to_bits(),
        m.triangle_index as u32,
        m.vertex_indices[0],
        m.vertex_indices[1],
        m.vertex_indices[2],
        m.triangle_flags,
    ]);
    n += 9;
    out[n..n + 15].copy_from_slice(&[
        cache.metric.to_bits(),
        cache.count as u32,
        cache.index_a[0] as u32,
        cache.index_a[1] as u32,
        cache.index_a[2] as u32,
        cache.index_a[3] as u32,
        cache.index_b[0] as u32,
        cache.index_b[1] as u32,
        cache.index_b[2] as u32,
        cache.index_b[3] as u32,
        sat.separation.to_bits(),
        sat.ty as u32,
        sat.index_a as u32,
        sat.index_b as u32,
        sat.hit as u32,
    ]);
}

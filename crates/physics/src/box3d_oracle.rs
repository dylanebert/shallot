//! Boundary measurements for the feature-only Box3D oracle.
use crate::distance::{
    shape_cast, shape_distance, time_of_impact, DistanceInput, ShapeCastPairInput, ShapeProxy,
    SimplexCache, Sweep, TOIInput,
};
use crate::math::{get_length_and_normalize, unwind_angle, Quat, Transform, Vec3};
use crate::simd::FloatW;

#[repr(align(8))]
struct Words([u32; 4096]);
static mut INPUT: Words = Words([0; 4096]);
static mut OUTPUT: [u32; 4096] = [0; 4096];

#[export_name = "box3dOracleInput"]
pub extern "C" fn input() -> *mut u32 {
    &raw mut INPUT as *mut u32
}

#[export_name = "box3dOracleOutput"]
pub extern "C" fn output() -> *const u32 {
    &raw const OUTPUT as *const u32
}

#[export_name = "box3dOracleRun"]
pub unsafe extern "C" fn run(operation: u32) -> usize {
    let r = &*(&raw const INPUT.0);
    let out = &mut *(&raw mut OUTPUT);
    let f = |i| f32::from_bits(r[i]);
    match operation {
        0 => {
            let (normal, length) = get_length_and_normalize(Vec3::new(f(0), f(1), f(2)));
            out[..4].copy_from_slice(&[
                normal.x.to_bits(),
                normal.y.to_bits(),
                normal.z.to_bits(),
                length.to_bits(),
            ]);
            4
        }
        1 => {
            out[0] = unwind_angle(f(0)).to_bits();
            1
        }
        2 => {
            let a = FloatW::set(f(0), f(1), f(2), f(3));
            let b = FloatW::set(f(4), f(5), f(6), f(7));
            for (i, v) in crate::wide::sym_clamp(a, b)
                .to_array()
                .into_iter()
                .enumerate()
            {
                out[i] = v.to_bits();
            }
            4
        }
        6 => {
            let points: Vec<_> = (0..8)
                .map(|i| Vec3::new(f(4 + i), f(12 + i), f(20 + i)))
                .collect();
            let hull = crate::hull::HullData {
                center: Vec3::ZERO,
                bounds: [Vec3::ZERO; 2],
                vertex_count: 8,
                edge_count: 0,
                face_count: 0,
                points: &points,
                soa_points: std::borrow::Cow::Borrowed(core::slice::from_raw_parts(
                    r.as_ptr().add(4).cast::<f32>(),
                    24,
                )),
                soa_normals: std::borrow::Cow::Borrowed(&[]),
                vertices: &[],
                edges: &[],
                faces: &[],
                planes: &[],
            };
            out[0] = hull.support_vertex_wide(Vec3::new(f(0), f(1), f(2)), f(3)) as u32;
            1
        }
        3..=5 => {
            let v = |i| Vec3::new(f(i), f(i + 1), f(i + 2));
            let q = |i| Quat {
                v: v(i),
                s: f(i + 3),
            };
            let pa = core::slice::from_raw_parts(r.as_ptr().add(32).cast::<Vec3>(), r[0] as usize);
            let pb = core::slice::from_raw_parts(r.as_ptr().add(416).cast::<Vec3>(), r[2] as usize);
            let a = ShapeProxy {
                points: pa,
                count: pa.len(),
                radius: f(1),
            };
            let b = ShapeProxy {
                points: pb,
                count: pb.len(),
                radius: f(3),
            };
            let transform = Transform { p: v(4), q: q(7) };
            if operation == 3 {
                let mut cache = SimplexCache {
                    metric: f(16),
                    count: r[17] as usize,
                    index_a: [
                        r[18] as usize,
                        r[19] as usize,
                        r[20] as usize,
                        r[21] as usize,
                    ],
                    index_b: [
                        r[22] as usize,
                        r[23] as usize,
                        r[24] as usize,
                        r[25] as usize,
                    ],
                };
                let d = shape_distance(
                    &DistanceInput {
                        proxy_a: a,
                        proxy_b: b,
                        transform,
                        use_radii: r[11] != 0,
                    },
                    &mut cache,
                );
                out[..21].copy_from_slice(&[
                    d.point_a.x.to_bits(),
                    d.point_a.y.to_bits(),
                    d.point_a.z.to_bits(),
                    d.point_b.x.to_bits(),
                    d.point_b.y.to_bits(),
                    d.point_b.z.to_bits(),
                    d.normal.x.to_bits(),
                    d.normal.y.to_bits(),
                    d.normal.z.to_bits(),
                    d.distance.to_bits(),
                    d.iterations as u32,
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
                ]);
                21
            } else if operation == 4 {
                let c = shape_cast(&ShapeCastPairInput {
                    proxy_a: a,
                    proxy_b: b,
                    transform,
                    translation_b: v(12),
                    max_fraction: f(15),
                    can_encroach: r[11] != 0,
                });
                out[0] = u32::from(c.hit);
                out[1] = c.iterations as u32;
                if !c.hit {
                    return 2;
                }
                out[2..12].copy_from_slice(&[
                    c.fraction.to_bits(),
                    c.point.x.to_bits(),
                    c.point.y.to_bits(),
                    c.point.z.to_bits(),
                    c.normal.x.to_bits(),
                    c.normal.y.to_bits(),
                    c.normal.z.to_bits(),
                    c.triangle_index as u32,
                    c.child_index as u32,
                    c.material_index as u32,
                ]);
                12
            } else {
                let sweep = |i| Sweep {
                    local_center: v(i),
                    c1: v(i + 3),
                    c2: v(i + 6),
                    q1: q(i + 9),
                    q2: q(i + 13),
                };
                let t = time_of_impact(&TOIInput {
                    proxy_a: a,
                    proxy_b: b,
                    sweep_a: sweep(800),
                    sweep_b: sweep(817),
                    max_fraction: f(15),
                });
                out[..13].copy_from_slice(&[
                    t.state as u32,
                    t.fraction.to_bits(),
                    t.distance.to_bits(),
                    t.point.x.to_bits(),
                    t.point.y.to_bits(),
                    t.point.z.to_bits(),
                    t.normal.x.to_bits(),
                    t.normal.y.to_bits(),
                    t.normal.z.to_bits(),
                    t.distance_iterations as u32,
                    t.push_back_iterations as u32,
                    t.root_iterations as u32,
                    u32::from(t.used_fallback),
                ]);
                13
            }
        }
        10..=18 => {
            use crate::manifold::*;
            let v = |i| Vec3::new(f(i), f(i + 1), f(i + 2));
            let xf = Transform {
                p: v(4),
                q: Quat { v: v(7), s: f(10) },
            };
            let ha = crate::geo::hull_view(r.as_ptr().add(1000) as usize);
            let hb = crate::geo::hull_view(r.as_ptr().add(1500) as usize);
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
                count: r[51] as usize,
                index_a: [
                    r[52] as usize,
                    r[53] as usize,
                    r[54] as usize,
                    r[55] as usize,
                ],
                index_b: [
                    r[56] as usize,
                    r[57] as usize,
                    r[58] as usize,
                    r[59] as usize,
                ],
            };
            let mut sat = SatCache {
                separation: f(44),
                ty: r[45],
                index_a: r[46] as usize,
                index_b: r[47] as usize,
                hit: r[48],
            };
            let mut m = LocalManifold::new();
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
                    &mut m, capacity, &ha, tri[0], tri[1], tri[2], &mut sat, true,
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
            // The kernel LocalManifold has no triangle normal/index/vertices/flags fields.
            // Encode absence, not zero values pretending to have been returned by the function.
            out[n] = 0;
            n += 1;
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
                sat.ty,
                sat.index_a as u32,
                sat.index_b as u32,
                sat.hit,
            ]);
            n + 15
        }
        _ => panic!("unknown oracle operation"),
    }
}

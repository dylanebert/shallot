//! Native per-shape gates use the same immutable Box3D vectors as the TypeScript gates.
use serde_json::Value;
use shallot_physics::distance::{CastOutput, ShapeProxy};
use shallot_physics::hull::HullData;
use shallot_physics::manifold::{Capsule, Sphere};
use shallot_physics::math::{Plane, Transform, Vec3};
use shallot_physics::query::*;

const QUERY: &str = include_str!("../../../src/standard/physics/collision/query.gold.json");
const MOVER: &str = include_str!("../../../src/standard/physics/collision/mover.gold.json");
const GEOMETRY: &str = include_str!("../../../src/standard/physics/shapes/geometry.gold.json");
fn f(v: &Value) -> f32 {
    f32::from_bits(u32::from_str_radix(v.as_str().unwrap().trim_start_matches("0x"), 16).unwrap())
}
fn v(a: &Value) -> Vec3 {
    Vec3::new(f(&a[0]), f(&a[1]), f(&a[2]))
}
fn eq(got: f32, want: &Value) {
    assert_eq!(got.to_bits(), f(want).to_bits());
}
fn veq(got: Vec3, want: &Value) {
    eq(got.x, &want[0]);
    eq(got.y, &want[1]);
    eq(got.z, &want[2]);
}
fn out(got: CastOutput, want: &Value) {
    assert_eq!(got.hit, want["hit"].as_bool().unwrap());
    eq(got.fraction, &want["fraction"]);
    veq(got.point, &want["point"]);
    veq(got.normal, &want["normal"]);
}
fn sphere(g: &Value) -> Sphere {
    Sphere {
        center: v(&g["center"]),
        radius: f(&g["radius"]),
    }
}
fn capsule(g: &Value) -> Capsule {
    Capsule {
        center1: v(&g["center1"]),
        center2: v(&g["center2"]),
        radius: f(&g["radius"]),
    }
}
fn ray(g: &Value) -> RayCastInput {
    RayCastInput {
        origin: v(&g["origin"]),
        translation: v(&g["translation"]),
        max_fraction: f(&g["maxFraction"]),
    }
}
fn hull<'a>(g: &Value, points: &'a [Vec3], planes: &'a [Plane]) -> HullData<'a> {
    // The queries read points and planes only; the topology is independently held by geometry gold.
    HullData {
        center: v(&g["center"]),
        inner_radius: 0.0,
        bounds: points.iter().fold([points[0]; 2], |[lo, hi], p| {
            [
                Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z)),
                Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z)),
            ]
        }),
        vertex_count: points.len(),
        edge_count: 0,
        face_count: planes.len(),
        points,
        soa_points: shallot_physics::hull::soa_vectors(points.iter().copied(), true).into(),
        soa_normals: shallot_physics::hull::soa_vectors(planes.iter().map(|p| p.normal), false)
            .into(),
        vertices: &[],
        edges: &[],
        faces: &[],
        planes,
    }
}
fn hull_pools(g: &Value) -> (Vec<Vec3>, Vec<Plane>) {
    (
        g["points"].as_array().unwrap().iter().map(v).collect(),
        g["planes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| Plane {
                normal: v(p),
                offset: f(&p[3]),
            })
            .collect(),
    )
}
#[test]
fn convex_queries_bit_exact() {
    let gold: Value = serde_json::from_str(QUERY).unwrap();
    let geometry: Value = serde_json::from_str(GEOMETRY).unwrap();
    let cube = &geometry["hulls"][0];
    let (base_points, base_planes) = hull_pools(cube);
    // query_gold.c supplies cube corners in a different order from geometry_gold.c; hull creation
    // orders the resulting support cloud and faces differently, including the ray corner tie.
    let points: Vec<_> = [4, 0, 6, 5, 7, 3, 2, 1].map(|i| base_points[i]).into();
    let planes: Vec<_> = [2, 3, 5, 1, 0, 4].map(|i| base_planes[i]).into();
    let h = hull(cube, &points, &planes);
    for g in gold["raySphere"].as_array().unwrap() {
        out(ray_cast_sphere(&sphere(g), &ray(g)), &g["out"]);
    }
    for g in gold["rayCapsule"].as_array().unwrap() {
        out(ray_cast_capsule(&capsule(g), &ray(g)), &g["out"]);
    }
    for g in gold["rayHull"].as_array().unwrap() {
        out(ray_cast_hull(&h, &ray(g)), &g["out"]);
    }
    for g in gold["shapeCast"].as_array().unwrap() {
        let origin = [Vec3::ZERO];
        let input = ShapeCastInput {
            proxy: ShapeProxy {
                points: &origin,
                count: 1,
                radius: f(&g["proxyRadius"]),
            },
            translation: v(&g["translation"]),
            max_fraction: f(&g["maxFraction"]),
            can_encroach: g["canEncroach"].as_bool().unwrap(),
        };
        let s;
        let a = if g["shape"] == "cube" {
            ShapeProxy {
                points: &points,
                count: points.len(),
                radius: 0.0,
            }
        } else {
            s = [v(&g["center"])];
            ShapeProxy {
                points: &s,
                count: 1,
                radius: f(&g["radius"]),
            }
        };
        out(shape_cast_convex(a, &input), &g["out"]);
    }
    for g in gold["overlap"].as_array().unwrap() {
        let origin = [Vec3::ZERO];
        let proxy = ShapeProxy {
            points: &origin,
            count: 1,
            radius: f(&g["proxyRadius"]),
        };
        let s;
        let a = if g["shape"] == "cube" {
            ShapeProxy {
                points: &points,
                count: points.len(),
                radius: 0.0,
            }
        } else {
            s = [v(&g["center"])];
            ShapeProxy {
                points: &s,
                count: 1,
                radius: f(&g["radius"]),
            }
        };
        let transform = Transform {
            p: v(&g["xfp"]),
            ..Transform::IDENTITY
        };
        assert_eq!(
            overlap_convex(a, transform, proxy),
            g["out"].as_bool().unwrap()
        );
    }
}
#[test]
fn convex_mover_bit_exact() {
    let gold: Value = serde_json::from_str(MOVER).unwrap();
    let geometry: Value = serde_json::from_str(GEOMETRY).unwrap();
    let cube = &geometry["hulls"][0];
    let (base_points, _) = hull_pools(cube);
    let points: Vec<_> = [4, 0, 6, 5, 7, 3, 2, 1]
        .map(|i| base_points[i].scale(0.5))
        .into();
    let h = hull(cube, &points, &[]);
    for kind in ["sphere", "capsule", "hull"] {
        for g in gold[kind].as_array().unwrap() {
            let mover = capsule(&g["mover"]);
            let result = if kind == "sphere" {
                collide_mover_sphere(&sphere(g), &mover)
            } else if kind == "capsule" {
                collide_mover_capsule(&capsule(g), &mover)
            } else {
                collide_mover_hull(&h, &mover)
            };
            assert_eq!(
                usize::from(result.is_some()),
                g["count"].as_u64().unwrap() as usize
            );
            if let Some(p) = result {
                let want = &g["planes"][0];
                veq(p.plane.normal, &want["normal"]);
                eq(p.plane.offset, &want["offset"]);
                veq(p.point, &want["point"]);
            }
        }
    }
}

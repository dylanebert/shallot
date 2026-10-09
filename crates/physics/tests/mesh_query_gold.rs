//! Mesh-tree query order and contact results match the immutable Box3D corpus.
use serde_json::Value;
use shallot_physics::distance::ShapeProxy;
use shallot_physics::manifold::Capsule;
use shallot_physics::math::{Transform, Vec3};
use shallot_physics::mesh_query::*;
use shallot_physics::query::{PlaneResult, RayCastInput, ShapeCastInput};
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
#[test]
fn mesh_queries_bit_exact() {
    let geometry: Value = serde_json::from_str(GEOMETRY).unwrap();
    let data = &geometry["meshes"][1];
    let nodes: Vec<_> = data["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| MeshNode {
            lower: v(&n["lowerBound"]),
            upper: v(&n["upperBound"]),
            data: if n["leaf"].as_bool().unwrap() {
                ((n["triangleCount"].as_u64().unwrap() as u32) << 2) | 3
            } else {
                ((n["childOffset"].as_u64().unwrap() as u32) << 2)
                    | n["axis"].as_u64().unwrap() as u32
            },
            triangle_offset: n["triangleOffset"].as_u64().unwrap() as u32,
        })
        .collect();
    let vertices: Vec<_> = data["vertices"].as_array().unwrap().iter().map(v).collect();
    let triangles: Vec<_> = data["triangles"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| MeshTriangle {
            indices: [
                t[0].as_u64().unwrap() as u32,
                t[1].as_u64().unwrap() as u32,
                t[2].as_u64().unwrap() as u32,
            ],
        })
        .collect();
    // Geometry gold assigns a material pattern after construction; the query fixture uses the
    // builder's default material zero for every triangle.
    let materials = vec![0; triangles.len()];
    let mesh = Mesh {
        nodes: &nodes,
        vertices: &vertices,
        triangles: &triangles,
        materials: &materials,
        scale: Vec3::new(1.0, 1.0, 1.0),
    };
    let gold: Value = serde_json::from_str(QUERY).unwrap();
    for kind in ["rayMesh", "shapeCastMesh"] {
        for g in gold[kind].as_array().unwrap() {
            let got = if kind == "rayMesh" {
                ray_cast_mesh(
                    mesh,
                    &RayCastInput {
                        origin: v(&g["origin"]),
                        translation: v(&g["translation"]),
                        max_fraction: f(&g["maxFraction"]),
                    },
                )
            } else {
                let points = [v(&g["proxyPoint"])];
                shape_cast_mesh(
                    mesh,
                    &ShapeCastInput {
                        proxy: ShapeProxy {
                            points: &points,
                            count: 1,
                            radius: f(&g["proxyRadius"]),
                        },
                        translation: v(&g["translation"]),
                        max_fraction: f(&g["maxFraction"]),
                        can_encroach: false,
                    },
                )
            };
            let want = &g["out"];
            assert_eq!(got.hit, want["hit"].as_bool().unwrap(), "{}", g["name"]);
            eq(got.fraction, &want["fraction"]);
            veq(got.point, &want["point"]);
            veq(got.normal, &want["normal"]);
            assert_eq!(
                got.triangle_index as i64,
                want["triangleIndex"].as_i64().unwrap()
            );
            assert_eq!(
                got.material_index as i64,
                want["materialIndex"].as_i64().unwrap()
            );
            assert_eq!(got.child_index as i64, want["childIndex"].as_i64().unwrap());
        }
    }
    for g in gold["overlapMesh"].as_array().unwrap() {
        let points = [v(&g["proxyPoint"])];
        assert_eq!(
            overlap_mesh(
                mesh,
                Transform {
                    p: v(&g["xfp"]),
                    ..Transform::IDENTITY
                },
                ShapeProxy {
                    points: &points,
                    count: 1,
                    radius: f(&g["proxyRadius"])
                }
            ),
            g["out"].as_bool().unwrap()
        );
    }
    let gold: Value = serde_json::from_str(MOVER).unwrap();
    for g in gold["mesh"].as_array().unwrap() {
        let m = &g["mover"];
        let mover = Capsule {
            center1: v(&m["center1"]),
            center2: v(&m["center2"]),
            radius: f(&m["radius"]),
        };
        let mut planes = [PlaneResult::ZERO; 16];
        let count = collide_mover_mesh(&mut planes, mesh, &mover);
        assert_eq!(count as u64, g["count"].as_u64().unwrap());
        for (p, w) in planes[..count].iter().zip(g["planes"].as_array().unwrap()) {
            veq(p.plane.normal, &w["normal"]);
            eq(p.plane.offset, &w["offset"]);
            veq(p.point, &w["point"]);
        }
    }
}

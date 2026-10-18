use serde_json::Value;
use shallot_physics::continuous_shape::shape_time_of_impact;
use shallot_physics::distance::{time_of_impact, ShapeProxy, Sweep, TOIInput};
use shallot_physics::height_query::HeightField;
use shallot_physics::hull::{soa_vectors, HullData};
use shallot_physics::manifold::Sphere;
use shallot_physics::math::{Quat, Vec3};
use shallot_physics::mesh_query::{Mesh, MeshNode, MeshTriangle};
use shallot_physics::query::Shape;

fn sweep(c1: Vec3, c2: Vec3) -> Sweep {
    Sweep {
        local_center: Vec3::ZERO,
        c1,
        c2,
        q1: Quat::IDENTITY,
        q2: Quat::IDENTITY,
    }
}

#[test]
fn mesh_and_height_traversals_return_the_triangle_impact() {
    let points = [Vec3::ZERO];
    let visitor = Shape::Sphere(&Sphere {
        center: Vec3::ZERO,
        radius: 0.1,
    });
    let visitor_sweep = sweep(Vec3::new(0.25, 1.0, 0.25), Vec3::new(0.25, -1.0, 0.25));
    let target = sweep(Vec3::ZERO, Vec3::ZERO);
    let vertices = [
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 0.0),
    ];
    let expected = time_of_impact(&TOIInput {
        proxy_a: ShapeProxy {
            points: &vertices,
            count: 3,
            radius: 0.0,
        },
        proxy_b: ShapeProxy {
            points: &points,
            count: 1,
            radius: 0.1,
        },
        sweep_a: target,
        sweep_b: visitor_sweep,
        max_fraction: 1.0,
    });
    assert!(expected.fraction > 0.0 && expected.fraction < 1.0);
    let nodes = [MeshNode {
        lower: Vec3::ZERO,
        upper: Vec3::new(1.0, 0.0, 1.0),
        data: (1 << 2) | 3,
        triangle_offset: 0,
    }];
    let triangles = [MeshTriangle { indices: [0, 1, 2] }];
    let mesh = Shape::Mesh(Mesh {
        nodes: &nodes,
        vertices: &vertices,
        triangles: &triangles,
        materials: &[0],
        scale: Vec3::new(1.0, 1.0, 1.0),
    });
    let height = Shape::Height(HeightField {
        header: &shallot_physics::height_query::HeightHeader {
            lower: Vec3::ZERO,
            upper: Vec3::new(1.0, 0.0, 1.0),
            min_height: 0.0,
            height_scale: 1.0,
            scale: Vec3::new(1.0, 1.0, 1.0),
            column_count: 2,
            row_count: 2,
            ..Default::default()
        },
        heights: &[0; 4],
        materials: &[0],
    });
    for shape in [&mesh, &height] {
        let output = shape_time_of_impact(shape, target, &visitor, visitor_sweep, 1.0, false);
        assert_eq!(output.fraction.to_bits(), expected.fraction.to_bits());
        assert_eq!(output.state, expected.state);
        assert!(!output.used_fallback);
    }
}

fn bits(v: &Value) -> f32 {
    f32::from_bits(u32::from_str_radix(v.as_str().unwrap(), 16).unwrap())
}

fn vec3(v: &Value) -> Vec3 {
    Vec3::new(bits(&v[0]), bits(&v[1]), bits(&v[2]))
}

// mesh_impact.gold.json holds b3ShapeTimeOfImpact's result for each case, regenerated from native
// Box3D by diagnostics/box3d-parity/mesh-impact.oracle.ts.
#[test]
fn reflected_mesh_impacts_equal_native_box3d() {
    let gold: Value = serde_json::from_str(include_str!("mesh_impact.gold.json")).unwrap();
    // b3MakeBoxHull(0.25, 0.25, 0.25)'s points in its order; the sweep reads only points and bounds.
    let h = 0.25;
    let points = [
        [h, h, h],
        [-h, h, h],
        [-h, -h, h],
        [h, -h, h],
        [h, h, -h],
        [-h, h, -h],
        [-h, -h, -h],
        [h, -h, -h],
    ]
    .map(|[x, y, z]| Vec3::new(x, y, z));
    let visitor = Shape::Hull(HullData {
        center: Vec3::ZERO,
        bounds: [Vec3::new(-h, -h, -h), Vec3::new(h, h, h)],
        inner_radius: h,
        vertex_count: points.len(),
        edge_count: 0,
        face_count: 0,
        points: &points,
        soa_points: soa_vectors(points.iter().copied(), true).into(),
        soa_normals: Vec::new().into(),
        vertices: &[],
        edges: &[],
        faces: &[],
        planes: &[],
    });
    for case in gold["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let v = &case["vertices"];
        let vertices = [0, 3, 6].map(|i| Vec3::new(bits(&v[i]), bits(&v[i + 1]), bits(&v[i + 2])));
        let [a, b, c] = vertices;
        let nodes = [MeshNode {
            lower: Vec3::new(
                a.x.min(b.x).min(c.x),
                a.y.min(b.y).min(c.y),
                a.z.min(b.z).min(c.z),
            ),
            upper: Vec3::new(
                a.x.max(b.x).max(c.x),
                a.y.max(b.y).max(c.y),
                a.z.max(b.z).max(c.z),
            ),
            data: (1 << 2) | 3,
            triangle_offset: 0,
        }];
        let mesh = Shape::Mesh(Mesh {
            nodes: &nodes,
            vertices: &vertices,
            triangles: &[MeshTriangle { indices: [0, 1, 2] }],
            materials: &[0],
            scale: vec3(&case["scale"]),
        });
        let output = shape_time_of_impact(
            &mesh,
            sweep(Vec3::ZERO, Vec3::ZERO),
            &visitor,
            sweep(vec3(&case["c1"]), vec3(&case["c2"])),
            1.0,
            false,
        );
        let want = &case["output"];
        assert_eq!(
            output.state as i64,
            want["state"].as_i64().unwrap(),
            "{name}: state"
        );
        assert_eq!(
            output.fraction.to_bits(),
            bits(&want["fraction"]).to_bits(),
            "{name}: fraction"
        );
        assert_eq!(
            output.distance.to_bits(),
            bits(&want["distance"]).to_bits(),
            "{name}: distance"
        );
        for (got, want, field) in [
            (output.point, vec3(&want["point"]), "point"),
            (output.normal, vec3(&want["normal"]), "normal"),
        ] {
            assert_eq!(
                [got.x, got.y, got.z].map(f32::to_bits),
                [want.x, want.y, want.z].map(f32::to_bits),
                "{name}: {field}"
            );
        }
        assert_eq!(
            output.used_fallback,
            want["usedFallback"].as_bool().unwrap(),
            "{name}: fallback"
        );
    }
}

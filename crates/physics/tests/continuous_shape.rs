use shallot_physics::continuous_shape::shape_time_of_impact;
use shallot_physics::distance::{time_of_impact, ShapeProxy, Sweep, TOIInput};
use shallot_physics::height_query::HeightField;
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

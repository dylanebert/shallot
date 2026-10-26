//! physics_world.c explosion query and impulse application.
use crate::distance::{shape_distance, DistanceInput, ShapeProxy, SimplexCache};
use crate::math::{maxf, Transform, Vec3};
use crate::{bodies, body, broad, query, query_abi, shapes, tree};

fn projected_area(shape: &query::Shape, direction: Vec3) -> f32 {
    match shape {
        query::Shape::Sphere(s) => core::f32::consts::PI * s.radius * s.radius,
        query::Shape::Capsule(s) => {
            let projected_length = s.center2.sub(s.center1).cross(direction).length();
            let cylinder_area = 2.0 * s.radius * projected_length;
            let sphere_area = core::f32::consts::PI * s.radius * s.radius;
            sphere_area + cylinder_area
        }
        query::Shape::Hull(h) => {
            let mut area = 0.0;
            for face in h.faces {
                let base = face.edge as usize;
                let edge = &h.edges[base];
                let p1 = h.points[edge.origin as usize];
                let edge = &h.edges[edge.next as usize];
                let mut p2 = h.points[edge.origin as usize];
                let mut index = edge.next as usize;
                loop {
                    let edge = &h.edges[index];
                    let p3 = h.points[edge.origin as usize];
                    let n = p2.sub(p1).cross(p3.sub(p1));
                    area += maxf(n.dot(direction), 0.0);
                    p2 = p3;
                    index = edge.next as usize;
                    if index == base {
                        break;
                    }
                }
            }
            0.5 * area
        }
        _ => 0.0,
    }
}

#[export_name = "worldExplode"]
pub unsafe extern "C" fn explode(
    world: usize,
    mask_hi: u32,
    mask_lo: u32,
    x: f32,
    y: f32,
    z: f32,
    radius: f32,
    falloff: f32,
    impulse_per_area: f32,
) {
    crate::regions::select(world as u32);
    assert!(x.is_finite() && y.is_finite() && z.is_finite());
    assert!(radius.is_finite() && radius >= 0.0);
    assert!(falloff.is_finite() && falloff >= 0.0);
    assert!(impulse_per_area.is_finite());
    let position = Vec3::new(x, y, z);
    let extent = radius + falloff;
    let lo = position.sub(Vec3::new(extent, extent, extent));
    let hi = position.add(Vec3::new(extent, extent, extent));
    let pool = core::slice::from_raw_parts(
        broad::tree_ptr(world, 2),
        broad::tree_cap(world, 2) * tree::STRIDE,
    );
    let root = *broad::tree_state(world, 2) as i32;
    let count = *broad::tree_state(world, 2).add(1) as usize;
    tree::query(
        pool,
        root,
        count,
        [lo.x, lo.y, lo.z],
        [hi.x, hi.y, hi.z],
        mask_hi,
        mask_lo,
        false,
        &mut [0; tree::STACK_SIZE],
        |_, data| {
            let id = data as usize;
            let scale = shapes::col_f(world).get(id * shapes::SHAPE_STRIDE + 8);
            if scale == 0.0 {
                return true;
            }
            let body_id =
                shapes::col(world).get(id * shapes::SHAPE_STRIDE + shapes::S_QUERY_BODY) as usize;
            let xf = crate::world_query::pose(world, id, Vec3::ZERO);
            let local_position = [xf.inv_point(position)];
            let (shape, _) = query_abi::active_shape(world, id);
            let capsule_points;
            let proxy = match &shape {
                query::Shape::Sphere(s) => ShapeProxy {
                    points: core::slice::from_ref(&s.center),
                    count: 1,
                    radius: s.radius,
                },
                query::Shape::Capsule(s) => {
                    capsule_points = [s.center1, s.center2];
                    ShapeProxy {
                        points: &capsule_points,
                        count: 2,
                        radius: s.radius,
                    }
                }
                query::Shape::Hull(h) => ShapeProxy {
                    points: h.points,
                    count: h.points.len(),
                    radius: 0.0,
                },
                _ => return true,
            };
            let output = shape_distance(
                &DistanceInput {
                    proxy_a: proxy,
                    proxy_b: ShapeProxy {
                        points: &local_position,
                        count: 1,
                        radius: 0.0,
                    },
                    transform: Transform::IDENTITY,
                    use_radii: true,
                },
                &mut SimplexCache::empty(),
            );
            if output.distance > radius + falloff {
                return true;
            }
            crate::body_mutation::wake_body_in_world(world, body_id);
            if bodies::record(world, body_id).set_index != crate::solver_set::AWAKE_SET {
                return true;
            }
            let closest_point = if output.distance == 0.0 {
                crate::shape_geometry::centroid(world, id)
            } else {
                output.point_a
            };
            let mut direction = closest_point.sub(local_position[0]);
            direction = if direction.length_sq() > 100.0 * f32::EPSILON * f32::EPSILON {
                direction.normalize()
            } else {
                Vec3::new(1.0, 0.0, 0.0)
            };
            let area = projected_area(&shape, direction);
            let attenuation = if output.distance > radius && falloff > 0.0 {
                ((radius + falloff - output.distance) / falloff).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let magnitude = impulse_per_area * area * attenuation * scale;
            let impulse = xf.q.rotate(direction).scale(magnitude);
            let sim_col = bodies::column(world, body_id, 1, body::SIM_STRIDE);
            let state_col = bodies::column(world, body_id, 0, body::STATE_STRIDE);
            let sim = body::read_sim(sim_col, 0);
            let mut state = body::read_state(state_col, 0);
            state.linear_velocity = state.linear_velocity.mul_add(sim.inv_mass, impulse);
            let r =
                xf.q.rotate(closest_point.sub(body::read_fin(sim_col, 0).local_center));
            state.angular_velocity = state
                .angular_velocity
                .add(sim.inv_inertia_world.mul_v(r.cross(impulse)));
            body::write_state(state_col, 0, &state);
            true
        },
    );
}

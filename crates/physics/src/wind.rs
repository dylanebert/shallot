//! shape.c b3Shape_ApplyWind.
use crate::math::{get_length_and_normalize, minf, Mat3, Vec3};
use crate::{bodies, body, query, query_abi, shapes, solver_set};

#[export_name = "shapeApplyWind"]
pub unsafe extern "C" fn apply_wind(
    world: usize,
    id: usize,
    x: f32,
    y: f32,
    z: f32,
    drag: f32,
    lift: f32,
    max_speed: f32,
    wake: u32,
) {
    crate::regions::select(world as u32);
    let (shape, _) = query_abi::active_shape(world, id);
    if !matches!(
        shape,
        query::Shape::Sphere(_) | query::Shape::Capsule(_) | query::Shape::Hull(_)
    ) {
        return;
    }
    let body_id = shapes::col(world).get(id * shapes::SHAPE_STRIDE + shapes::S_QUERY_BODY) as usize;
    let record = bodies::record(world, body_id);
    if record.body_type != body::body_type::DYNAMIC
        || record.set_index == solver_set::DISABLED_SET
        || (record.set_index >= solver_set::FIRST_SLEEPING_SET && wake == 0)
    {
        return;
    }
    if record.set_index != solver_set::AWAKE_SET {
        crate::body_mutation::wake_body_in_world(world, body_id);
    }
    // Unlike shape.c, fetch the simulation record after waking: waking frees the sleeping set's copy.
    let sim_col = bodies::column(world, body_id, 1, body::SIM_STRIDE);
    let sim = body::read_sim(sim_col, 0);
    let local_center = body::read_fin(sim_col, 0).local_center;
    let state = body::read_state(bodies::column(world, body_id, 0, body::STATE_STRIDE), 0);
    let wind = Vec3::new(x, y, z);
    let mut force = Vec3::ZERO;
    let mut torque = Vec3::ZERO;
    match shape {
        query::Shape::Sphere(s) => {
            let lever = sim
                .rotation
                .rotate(crate::shape_geometry::centroid(world, id).sub(local_center));
            let velocity = state
                .linear_velocity
                .add(state.angular_velocity.cross(lever));
            let (direction, speed) = get_length_and_normalize(wind.mul_sub(drag, velocity));
            let speed = minf(speed, max_speed);
            let area = core::f32::consts::PI * s.radius * s.radius;
            force = direction.scale(0.5 * 1.2250 * area * speed * speed);
            torque = lever.cross(force);
        }
        query::Shape::Capsule(s) => {
            let lever = sim
                .rotation
                .rotate(crate::shape_geometry::centroid(world, id).sub(local_center));
            let velocity = state
                .linear_velocity
                .add(state.angular_velocity.cross(lever));
            let (direction, speed) = get_length_and_normalize(wind.mul_sub(drag, velocity));
            let speed = minf(speed, max_speed);
            let d = sim.rotation.rotate(s.center2.sub(s.center1));
            let area = core::f32::consts::PI * s.radius * s.radius
                + 2.0 * s.radius * d.cross(direction).length();
            let e = d.normalize();
            let normal = e.scale(direction.dot(e)).sub(direction);
            let lift_direction = normal.cross(direction).cross(direction);
            let magnitude = 0.5 * 1.2250 * area * speed * speed;
            force = direction.mul_add(lift, lift_direction).scale(magnitude);
            torque = lever.mul_add(s.radius, normal).cross(force);
        }
        query::Shape::Hull(h) => {
            let matrix = Mat3::from_quat(sim.rotation);
            for (i, face) in h.faces.iter().enumerate() {
                let base = face.edge as usize;
                let edge1 = &h.edges[base];
                let edge2 = &h.edges[edge1.next as usize];
                let mut index = edge2.next as usize;
                let p1 = h.points[edge1.origin as usize];
                let mut p2 = h.points[edge2.origin as usize];
                let v1 = matrix.mul_v(p1);
                let mut v2 = matrix.mul_v(p2);
                let normal = matrix.mul_v(h.planes[i].normal);
                loop {
                    let edge3 = &h.edges[index];
                    let p3 = h.points[edge3.origin as usize];
                    let v3 = matrix.mul_v(p3);
                    let center = p1.add(p2.add(p3)).scale(0.333333);
                    let lever = matrix.mul_v(center.sub(local_center));
                    let velocity = state
                        .linear_velocity
                        .add(state.angular_velocity.cross(lever));
                    let (direction, speed) = get_length_and_normalize(wind.mul_sub(drag, velocity));
                    if normal.dot(direction) < -f32::EPSILON {
                        let area = -0.5 * v2.sub(v1).cross(v3.sub(v1)).dot(direction);
                        let lift_direction = normal.cross(direction).cross(direction);
                        let speed = minf(speed, max_speed);
                        let magnitude = 0.5 * 1.2250 * area * speed * speed;
                        let delta = direction.mul_add(lift, lift_direction).scale(magnitude);
                        force = force.add(delta);
                        torque = torque.add(lever.cross(delta));
                    }
                    index = edge3.next as usize;
                    v2 = v3;
                    p2 = p3;
                    if index == base {
                        break;
                    }
                }
            }
        }
        _ => {}
    }
    let force = sim.force.add(force);
    let torque = sim.torque.add(torque);
    for (offset, v) in [(body::FORCE, force), (body::TORQUE, torque)] {
        sim_col.set(offset, v.x);
        sim_col.set(offset + 1, v.y);
        sim_col.set(offset + 2, v.z);
    }
}

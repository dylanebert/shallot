//! Box3D samples/mover.cpp movement phase. Input policy belongs to the player.
use crate::{
    math::{Mat3, Plane, Vec3, PI},
    mover::{self, CollisionPlane},
};

#[derive(Clone, Copy)]
pub struct Contact {
    pub shape: u32,
    pub plane: Plane,
    pub point: Vec3,
}
#[derive(Clone, Copy)]
pub struct Ground {
    pub shape: u32,
    pub fraction: f32,
    pub point: Vec3,
    pub normal: Vec3,
}
#[derive(Clone, Copy)]
pub struct Body {
    pub id: u32,
    pub dynamic: bool,
    pub awake: bool,
    pub center: Vec3,
    pub linear: Vec3,
    pub angular: Vec3,
    pub inv_mass: f32,
    pub inv_inertia: Mat3,
}
pub trait World {
    fn ray(&mut self, origin: Vec3, translation: Vec3) -> Option<Ground>;
    fn collide(
        &mut self,
        position: Vec3,
        a: Vec3,
        b: Vec3,
        radius: f32,
        contacts: &mut [Contact; 8],
    ) -> usize;
    fn cast(&mut self, position: Vec3, delta: Vec3) -> f32;
    fn body(&mut self, shape: u32) -> Body;
    fn impulse(&mut self, body: u32, impulse: Vec3, point: Vec3);
}
#[repr(C)]
pub struct Input {
    pub position: Vec3,
    pub velocity: Vec3,
    pub up: Vec3,
    pub center1: Vec3,
    pub center2: Vec3,
    pub radius: f32,
    pub dt: f32,
    pub pogo: f32,
    pub rest: f32,
    pub steep: u32,
    pub max_slope: f64,
    pub center: Vec3,
    pub rotation: crate::math::Quat,
    pub max_speed: f32,
}
#[repr(C)]
pub struct Output {
    pub position: Vec3,
    pub velocity: Vec3,
    pub pogo: f32,
    pub ground_state: u32,
    pub normal: Vec3,
    pub ground_velocity: Vec3,
    pub count: u32,
    pub plane_count: u32,
    pub passes: u32,
    pub iterations: u32,
    pub impulse_count: u32,
    pub bodies: [u32; 8],
    pub impulses: [Vec3; 8],
    pub center: Vec3,
    pub ground_shape: u32,
    pub shapes: [u32; 8],
}
pub fn run(world: &mut impl World, input: &Input) -> Output {
    let mut out = Output {
        position: input.position,
        velocity: input.velocity,
        pogo: input.pogo,
        ground_state: 0,
        normal: Vec3::ZERO,
        ground_velocity: Vec3::ZERO,
        count: 0,
        plane_count: 0,
        passes: 0,
        iterations: 0,
        impulse_count: 0,
        bodies: [0; 8],
        impulses: [Vec3::ZERO; 8],
        center: input.center,
        ground_shape: u32::MAX,
        shapes: [0; 8],
    };
    let cos_slope = input.max_slope.cos();
    let center1 = input.rotation.rotate(input.center1);
    let center2 = input.rotation.rotate(input.center2);
    let rest = input.rest * input.radius;
    let ray_length = rest + input.radius;
    let ray = input.position.add(center1);
    let ground = world.ray(ray, input.up.scale(-ray_length));
    if let Some(hit) = ground {
        out.ground_shape = hit.shape;
        let body = world.body(hit.shape);
        if body.awake {
            out.center = body.center;
            out.ground_velocity = body
                .linear
                .add(body.angular.cross(hit.point.sub(body.center)));
        }
    }
    let up = input.up;
    let residue = 2.0
        * 2_f64.powi(-23)
        * (out.center.x as f64)
            .abs()
            .max((out.center.y as f64).abs())
            .max((out.center.z as f64).abs())
        * ((up.x as f64).abs() + (up.y as f64).abs() + (up.z as f64).abs())
        / input.dt as f64;
    if ground.is_none() || input.velocity.sub(out.ground_velocity).dot(up) as f64 > residue {
        out.pogo = 0.0;
        out.ground_velocity = Vec3::ZERO;
    } else if let Some(hit) = ground {
        out.normal = hit.normal;
        out.ground_state = if (hit.normal.dot(up) as f64) < cos_slope {
            2
        } else {
            1
        };
        let omega = (2.0 * PI) * 4.0;
        let omega_h = omega * input.dt;
        let length = hit.fraction * ray_length;
        out.pogo = (out.pogo - (omega * omega_h) * (length - rest))
            / ((1.0 + (2.0 * 0.7) * omega_h) + omega_h * omega_h);
    }
    let target = input
        .position
        .mul_add(input.dt, input.velocity)
        .mul_add(input.dt * out.pogo, up);
    let mut planes = [CollisionPlane {
        plane: Plane {
            normal: Vec3::ZERO,
            offset: 0.0,
        },
        push_limit: f32::MAX,
        push: 0.0,
        clip_velocity: 1,
    }; 16];
    let empty = Contact {
        shape: 0,
        plane: Plane {
            normal: Vec3::ZERO,
            offset: 0.0,
        },
        point: Vec3::ZERO,
    };
    let mut contacts = [empty; 8];
    let mut count = 0;
    let mut plane_count = 0;
    for _ in 0..5 {
        count = world.collide(out.position, center1, center2, input.radius, &mut contacts);
        for i in 0..count {
            planes[i].plane = contacts[i].plane;
        }
        plane_count = count;
        if input.steep != 0 {
            for contact in &contacts[..count] {
                let d = contact.plane.normal.dot(up);
                if (d as f64) <= 1e-3 || d as f64 >= cos_slope {
                    continue;
                }
                let hx = contact.plane.normal.x as f64 - d as f64 * up.x as f64;
                let hy = contact.plane.normal.y as f64 - d as f64 * up.y as f64;
                let hz = contact.plane.normal.z as f64 - d as f64 * up.z as f64;
                let length = (hx * hx + hy * hy + hz * hz).sqrt();
                planes[plane_count].plane = Plane {
                    normal: Vec3::new(
                        (hx / length) as f32,
                        (hy / length) as f32,
                        (hz / length) as f32,
                    ),
                    offset: (contact.plane.offset as f64 / length) as f32,
                };
                plane_count += 1;
            }
        }
        let (delta, iterations) =
            mover::solve(target.sub(out.position), &mut planes[..plane_count]);
        out.passes += 1;
        out.iterations += iterations;
        let fraction = world.cast(out.position, delta);
        let delta = delta.scale(fraction);
        out.position = out.position.add(delta);
        if delta.length_sq() < 0.01_f32 * 0.01_f32 {
            break;
        }
    }
    for i in 0..count {
        let body = world.body(contacts[i].shape);
        if !body.dynamic {
            continue;
        }
        let normal = planes[i].plane.normal.scale(-1.0);
        out.center = body.center;
        let r = contacts[i].point.sub(body.center);
        let rn = r.cross(normal);
        let mass = body.inv_mass + rn.dot(body.inv_inertia.mul_v(rn));
        let normal_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        let vr = if body.awake {
            body.linear.add(body.angular.cross(r))
        } else {
            Vec3::ZERO
        };
        let vn = vr.sub(out.velocity).dot(normal);
        let impulse = normal.scale(crate::math::maxf(-normal_mass * vn, 0.0));
        out.velocity = out.velocity.mul_sub(0.0, impulse);
        let mut index = 0;
        while index < out.impulse_count as usize && out.bodies[index] != body.id {
            index += 1;
        }
        if index == out.impulse_count as usize {
            out.impulse_count += 1;
            out.bodies[index] = body.id;
        }
        out.impulses[index] = out.impulses[index].add(impulse);
        world.impulse(body.id, impulse, contacts[i].point);
    }
    out.velocity = mover::clip(out.velocity, &planes[..plane_count]);
    for i in 0..count {
        out.shapes[i] = contacts[i].shape;
    }
    out.count = count as u32;
    out.plane_count = plane_count as u32;
    out
}

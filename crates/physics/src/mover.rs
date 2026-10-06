//! Box3D mover.c: caller-owned collision planes, solve and velocity clipping.
use crate::math::{absf, clampf, minf, Plane, Vec3};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CollisionPlane {
    pub plane: Plane,
    pub push_limit: f32,
    pub push: f32,
    pub clip_velocity: u32,
}

pub fn solve(target: Vec3, planes: &mut [CollisionPlane]) -> (Vec3, u32) {
    for plane in planes.iter_mut() {
        plane.push = 0.0;
    }
    let mut delta = target;
    let mut iteration = 0;
    while iteration < 20 {
        let mut total_push = 0.0;
        for plane in planes.iter_mut() {
            let separation = plane.plane.normal.dot(delta) - plane.plane.offset + 0.005;
            let accumulated = plane.push;
            plane.push = clampf(plane.push - separation, 0.0, plane.push_limit);
            let push = plane.push - accumulated;
            delta = delta.mul_add(push, plane.plane.normal);
            total_push += absf(push);
        }
        if total_push < 0.005 {
            break;
        }
        iteration += 1;
    }
    (delta, iteration)
}
pub fn clip(mut vector: Vec3, planes: &[CollisionPlane]) -> Vec3 {
    for plane in planes {
        if plane.push == 0.0 || plane.clip_velocity == 0 {
            continue;
        }
        vector = vector.sub(
            plane
                .plane
                .normal
                .scale(minf(0.0, vector.dot(plane.plane.normal))),
        );
    }
    vector
}

impl Default for CollisionPlane {
    fn default() -> Self {
        Self {
            plane: Plane {
                normal: Vec3::ZERO,
                offset: 0.0,
            },
            push_limit: 0.0,
            push: 0.0,
            clip_velocity: 0,
        }
    }
}
static mut PLANES: Vec<CollisionPlane> = Vec::new();
static mut OUTPUT: [f32; 4] = [0.0; 4];
#[export_name = "moverPlanesPtr"]
pub extern "C" fn planes_ptr(count: usize) -> *mut CollisionPlane {
    unsafe {
        let planes = &mut *(&raw mut PLANES);
        planes.resize(count, CollisionPlane::default());
        planes.as_mut_ptr()
    }
}
#[export_name = "moverOutputPtr"]
pub extern "C" fn output_ptr() -> *const f32 {
    &raw const OUTPUT as *const f32
}
#[export_name = "moverSolve"]
pub extern "C" fn run(operation: u32, x: f32, y: f32, z: f32, count: usize) {
    unsafe {
        let planes = &mut *(&raw mut PLANES);
        let input = Vec3::new(x, y, z);
        let (result, iterations) = if operation == 0 {
            solve(input, &mut planes[..count])
        } else {
            (clip(input, &planes[..count]), 0)
        };
        OUTPUT = [result.x, result.y, result.z, iterations as f32];
    }
}

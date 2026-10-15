//! Numeric access to Box3D math for native parity scene authoring.
use crate::math::{compute_cos_sin, Quat, Vec3};

static mut OUTPUT: [f32; 4] = [0.0; 4];

#[export_name = "mathCosSin"]
pub unsafe extern "C" fn cos_sin(radians: f32) -> *const f32 {
    let cs = compute_cos_sin(radians);
    OUTPUT = [cs.cosine, cs.sine, 0.0, 0.0];
    (&raw const OUTPUT).cast()
}

#[export_name = "mathNormalizeQuat"]
pub unsafe extern "C" fn normalize_quat(x: f32, y: f32, z: f32, s: f32) -> *const f32 {
    let q = Quat {
        v: Vec3::new(x, y, z),
        s,
    }
    .normalize();
    OUTPUT = [q.v.x, q.v.y, q.v.z, q.s];
    (&raw const OUTPUT).cast()
}

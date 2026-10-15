//! Numeric access to Box3D math for native parity scene authoring.
use crate::math::{compute_cos_sin, Quat, Vec3};

static mut OUTPUT: [f32; 4] = [0.0; 4];
static mut INPUT: [f32; 18] = [0.0; 18];
static mut RESULT: [f32; 9] = [0.0; 9];

#[export_name = "mathInputPtr"]
pub extern "C" fn input_ptr() -> *mut f32 {
    (&raw mut INPUT).cast()
}

/// Parity harness operations over the existing scalar math; result lives until the next call.
#[export_name = "mathRead"]
pub unsafe extern "C" fn read(operation: u32) -> *const f32 {
    use crate::math::{Mat3, Transform};
    let a = &*(&raw const INPUT);
    let v = |i: usize| Vec3::new(a[i], a[i + 1], a[i + 2]);
    let q = |i: usize| Quat {
        v: v(i),
        s: a[i + 3],
    };
    let m = |i: usize| Mat3 {
        cx: v(i),
        cy: v(i + 3),
        cz: v(i + 6),
    };
    let result = &mut *(&raw mut RESULT);
    *result = [0.0; 9];
    let mut vector = None;
    let mut quaternion = None;
    let mut matrix = None;
    match operation {
        0 => vector = Some(v(0).add(v(3))),
        5 => vector = Some(v(0).normalize()),
        6 => quaternion = Some(Quat::from_axis_angle(v(0), a[3])),
        9 => quaternion = Some(q(0).negate()),
        14 => vector = Some(Transform { p: v(0), q: q(3) }.point(v(7))),
        15 => matrix = Some(Mat3::from_quat(q(0))),
        16 => matrix = Some(m(0).mul(m(9))),
        17 => matrix = Some(m(0).transpose()),
        20 => {
            result[0] = crate::mesh_query::intersect_ray_triangle(v(0), v(3), [v(6), v(9), v(12)])
        }
        21 => {
            result[0] = u32::from(crate::finalize::aabb_contains(
                a[..6].try_into().unwrap(),
                a[6..12].try_into().unwrap(),
            )) as f32
        }
        22 => result[0] = u32::from(v(0).is_normalized()) as f32,
        _ => unreachable!(),
    }
    if let Some(v) = vector {
        result[..3].copy_from_slice(&[v.x, v.y, v.z]);
    }
    if let Some(q) = quaternion {
        result[..4].copy_from_slice(&[q.v.x, q.v.y, q.v.z, q.s]);
    }
    if let Some(m) = matrix {
        result.copy_from_slice(&[
            m.cx.x, m.cx.y, m.cx.z, m.cy.x, m.cy.y, m.cy.z, m.cz.x, m.cz.y, m.cz.z,
        ]);
    }
    result.as_ptr()
}

#[export_name = "mathCosSin"]
pub unsafe extern "C" fn cos_sin(radians: f32) -> *const f32 {
    let cs = compute_cos_sin(radians);
    OUTPUT = [cs.cosine, cs.sine, 0.0, 0.0];
    (&raw const OUTPUT).cast()
}

#[export_name = "mathQuatIsValid"]
pub extern "C" fn quat_is_valid(x: f32, y: f32, z: f32, s: f32) -> bool {
    Quat {
        v: Vec3::new(x, y, z),
        s,
    }
    .is_valid()
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

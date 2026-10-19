//! World-instance-local hull authoring scratch. Returned images live until the next build.
use crate::hull_build::{self, HullImage};
use crate::math::{Quat, Transform, Vec3};
static mut INPUT: Vec<f32> = Vec::new();
static mut OUTPUT: Option<HullImage> = None;
#[export_name = "hullBuildInputPtr"]
pub unsafe extern "C" fn input_ptr(count: usize) -> *mut f32 {
    let input = &mut *(&raw mut INPUT);
    input.resize(10 + 3 * count, 0.0);
    input.as_mut_ptr()
}
#[export_name = "hullBuild"]
pub unsafe extern "C" fn build(operation: u32, count: usize, budget: i32) -> *const u64 {
    let r = &*(&raw const INPUT);
    let v = |i: usize| Vec3::new(r[i], r[i + 1], r[i + 2]);
    let result = match operation {
        0 => hull_build::create_hull(
            core::slice::from_raw_parts(r.as_ptr().add(10).cast::<Vec3>(), count),
            budget,
        ),
        1 => hull_build::create_cylinder(r[0], r[1], r[2], r[3] as usize),
        2 => hull_build::create_cone(r[0], r[1], r[2], r[3] as usize),
        3 => hull_build::create_rock(r[0]),
        4 => Some(hull_build::make_transformed_box(
            r[0],
            r[1],
            r[2],
            Transform::IDENTITY,
        )),
        5 => Some(hull_build::make_transformed_box(
            r[0],
            r[0],
            r[0],
            Transform::IDENTITY,
        )),
        6 => Some(hull_build::make_transformed_box(
            r[0],
            r[1],
            r[2],
            Transform {
                p: v(3),
                q: Quat::IDENTITY,
            },
        )),
        7 => Some(hull_build::make_transformed_box(
            r[0],
            r[1],
            r[2],
            Transform {
                p: v(3),
                q: Quat { v: v(6), s: r[9] },
            },
        )),
        8 => {
            let bytes = core::slice::from_raw_parts(r.as_ptr().add(16).cast::<u8>(), count);
            let words = bytes
                .chunks_exact(8)
                .map(|b| u64::from_ne_bytes(b.try_into().unwrap()))
                .collect();
            hull_build::clone_and_transform(
                &HullImage(words),
                Transform {
                    p: v(0),
                    q: Quat { v: v(3), s: r[6] },
                },
                v(7),
            )
        }
        _ => unreachable!(),
    };
    OUTPUT = result;
    (&*(&raw const OUTPUT))
        .as_ref()
        .map_or(core::ptr::null(), |h| h.0.as_ptr())
}

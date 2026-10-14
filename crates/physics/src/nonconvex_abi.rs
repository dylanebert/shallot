//! Instance-local authoring scratch; images remain valid until the next nonconvex build.
use crate::math::Vec3;
use crate::nonconvex_build::Image;
use crate::{height_build, mesh_build};
static mut INPUT: Vec<u32> = Vec::new();
static mut OUTPUT: Option<Image> = None;
#[export_name = "nonconvexBuildInputPtr"]
pub unsafe extern "C" fn input_ptr(words: usize) -> *mut u32 {
    let input = &mut *(&raw mut INPUT);
    input.resize(16 + words, 0);
    input.as_mut_ptr()
}
#[export_name = "nonconvexBuild"]
pub unsafe extern "C" fn build(operation: u32) -> *const u64 {
    let r = &*(&raw const INPUT);
    let f = |i: usize| f32::from_bits(r[i]);
    let v = |i: usize| Vec3::new(f(i), f(i + 1), f(i + 2));
    let result = match operation {
        0 => {
            let nv = r[4] as usize;
            let ni = r[5] as usize;
            let materials = if r[6] != 0 {
                r[16 + 3 * nv + ni..16 + 3 * nv + ni + ni / 3]
                    .iter()
                    .map(|&x| x as u8)
                    .collect()
            } else {
                Vec::new()
            };
            mesh_build::create_mesh(&mesh_build::Definition {
                vertices: core::slice::from_raw_parts(r.as_ptr().add(16).cast::<Vec3>(), nv),
                indices: &r[16 + 3 * nv..16 + 3 * nv + ni],
                materials: &materials,
                weld_tolerance: f(0),
                weld: r[1] != 0,
                median: r[2] != 0,
                edges: r[3] != 0,
            })
        }
        1 => mesh_build::create_box(v(0), v(3), f(6) != 0.0, false),
        2 => mesh_build::create_box(v(0), v(3), true, true),
        3 => mesh_build::create_grid(
            f(0) as usize,
            f(1) as usize,
            f(2),
            f(3) as usize,
            f(4) != 0.0,
        ),
        4 => mesh_build::create_wave(f(0) as usize, f(1) as usize, f(2), f(3), f(4), f(5)),
        5 => mesh_build::create_torus(f(0) as usize, f(1) as usize, f(2), f(3)),
        10 => {
            let nx = f(0) as usize;
            let nz = f(1) as usize;
            let count = nx * nz;
            let cells = (nx - 1) * (nz - 1);
            let materials = if r[8] != 0 {
                r[16 + count..16 + count + cells]
                    .iter()
                    .map(|&x| x as u8)
                    .collect()
            } else {
                Vec::new()
            };
            Some(height_build::create_height_field(
                &height_build::Definition {
                    heights: core::slice::from_raw_parts(r.as_ptr().add(16).cast::<f32>(), count),
                    materials: &materials,
                    scale: v(2),
                    columns: nx,
                    rows: nz,
                    minimum: f(5),
                    maximum: f(6),
                    clockwise: f(7) != 0.0,
                },
            ))
        }
        11 => Some(height_build::create_grid(
            f(0) as usize,
            f(1) as usize,
            v(2),
            f(5) != 0.0,
        )),
        12 => Some(height_build::create_wave(
            f(0) as usize,
            f(1) as usize,
            v(2),
            f(5),
            f(6),
            f(7) != 0.0,
        )),
        _ => unreachable!(),
    };
    OUTPUT = result;
    (&*(&raw const OUTPUT))
        .as_ref()
        .map_or(core::ptr::null(), |image| image.0.as_ptr())
}

//! Shape geometry readers from Box3D hull.c, mesh.c and height_field.c.
use crate::math::{Mat3, Transform, Vec3};
use crate::mesh_query::{max, min, mul};

pub fn transform_bounds(lower: Vec3, upper: Vec3, pose: Transform) -> [f32; 6] {
    let center = pose.point(lower.add(upper).scale(0.5));
    let extent = Mat3::from_quat(pose.q)
        .abs()
        .mul_v(upper.sub(lower).scale(0.5));
    let lo = center.sub(extent);
    let hi = center.add(extent);
    [lo.x, lo.y, lo.z, hi.x, hi.y, hi.z]
}

pub fn mesh_triangle(
    mut vertices: [Vec3; 3],
    mut indices: [u32; 3],
    mut flags: u8,
    scale: Vec3,
) -> ([Vec3; 3], [u32; 3], u8) {
    if scale.x * scale.y * scale.z < 0.0 {
        vertices.swap(1, 2);
        indices.swap(1, 2);
        flags = (flags >> 4) & 7;
    }
    (vertices.map(|v| mul(scale, v)), indices, flags)
}

pub fn height_triangle(
    header: &crate::height_query::HeightHeader,
    index: usize,
    height: impl Fn(usize) -> u16,
    mut flags: u8,
) -> ([Vec3; 3], [u32; 3], u8) {
    let columns = header.column_count as usize;
    let cell = index >> 1;
    let row = cell / (columns - 1);
    let column = cell - row * (columns - 1);
    let i11 = row * columns + column;
    let i12 = i11 + 1;
    let i21 = (row + 1) * columns + column;
    let i22 = i21 + 1;
    let mut indices = if index & 1 == 0 {
        [i11 as u32, i21 as u32, i12 as u32]
    } else {
        [i22 as u32, i12 as u32, i21 as u32]
    };
    if header.flags & 1 != 0 {
        indices.swap(1, 2);
        let edge1 = flags & 0x11;
        let edge3 = flags & 0x44;
        flags = (flags & !0x55) | (edge1 << 2) | (edge3 >> 2);
    }
    let vertices = indices.map(|i| {
        let row = i as usize / columns;
        let column = i as usize % columns;
        mul(
            header.scale,
            Vec3::new(
                column as f32,
                header.min_height + header.height_scale * height(i as usize) as f32,
                row as f32,
            ),
        )
    });
    (vertices, indices, flags)
}

/// Numeric request layout: pose at 0, geometry at 8; triangle headers use integer words.
pub fn read(operation: u32, input: &[u32]) -> [u32; 16] {
    use crate::math::Quat;
    let f = |i| f32::from_bits(input[i]);
    let v = |i| Vec3::new(f(i), f(i + 1), f(i + 2));
    let pose = || Transform {
        p: v(0),
        q: Quat { v: v(3), s: f(6) },
    };
    let mut out = [0; 16];
    match operation {
        0 | 1 => {
            let kind = if operation == 0 { 5 } else { 0 };
            let geom: [f32; 7] = core::array::from_fn(|i| f(i + 8));
            let bounds = crate::finalize::convex_bounds(kind, &geom, pose());
            for (o, x) in out.iter_mut().zip(bounds) {
                *o = x.to_bits();
            }
        }
        2 | 3 => {
            let (lower, upper) = if operation == 3 {
                let a = mul(v(8), v(14));
                let b = mul(v(11), v(14));
                (min(a, b), max(a, b))
            } else {
                (v(8), v(11))
            };
            let bounds = transform_bounds(lower, upper, pose());
            for (o, x) in out.iter_mut().zip(bounds) {
                *o = x.to_bits();
            }
        }
        4 => {
            let scale = v(8).safe_scale();
            out[0] = scale.x.to_bits();
            out[1] = scale.y.to_bits();
            out[2] = scale.z.to_bits();
        }
        5 | 6 => {
            let (vertices, indices, flags) = if operation == 5 {
                mesh_triangle(
                    [v(8), v(11), v(14)],
                    [input[17], input[18], input[19]],
                    input[20] as u8,
                    v(21),
                )
            } else {
                let header = crate::height_query::HeightHeader {
                    column_count: input[8],
                    min_height: f(10),
                    height_scale: f(11),
                    scale: v(12),
                    flags: input[15],
                    ..Default::default()
                };
                height_triangle(
                    &header,
                    input[9] as usize,
                    |i| input[32 + i] as u16,
                    input[16] as u8,
                )
            };
            for (i, vertex) in vertices.iter().enumerate() {
                out[3 * i] = vertex.x.to_bits();
                out[3 * i + 1] = vertex.y.to_bits();
                out[3 * i + 2] = vertex.z.to_bits();
            }
            out[9..12].copy_from_slice(&indices);
            out[12] = flags as u32;
        }
        _ => unreachable!(),
    }
    out
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    static mut INPUT: Vec<u32> = Vec::new();
    static mut OUTPUT: [u32; 16] = [0; 16];
    #[export_name = "geometryReadInputPtr"]
    pub unsafe extern "C" fn input_ptr(words: usize) -> *mut u32 {
        let input = &mut *(&raw mut INPUT);
        input.resize(words.max(32), 0);
        input.as_mut_ptr()
    }
    #[export_name = "geometryRead"]
    pub unsafe extern "C" fn read(operation: u32) -> *const u32 {
        OUTPUT = super::read(operation, &*(&raw const INPUT));
        (&raw const OUTPUT).cast()
    }
}

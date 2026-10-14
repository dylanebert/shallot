//! Box3D height_field.c construction and generators.
use crate::math::{clampf, compute_cos_sin, maxf, minf, Vec3, PI};
use crate::nonconvex_build::{align8, Image};
pub struct Definition<'a> {
    pub heights: &'a [f32],
    pub materials: &'a [u8],
    pub scale: Vec3,
    pub columns: usize,
    pub rows: usize,
    pub minimum: f32,
    pub maximum: f32,
    pub clockwise: bool,
}
fn mul(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x * b.x, a.y * b.y, a.z * b.z)
}
fn normal(v: [Vec3; 3]) -> Vec3 {
    v[1].sub(v[0]).cross(v[2].sub(v[0])).normalize()
}
fn edge(n: Vec3, offset: f32, other: [Vec3; 3], opposite: usize, bit: u8) -> u8 {
    let separation = n.dot(other[opposite]) - offset;
    let cosine = n.dot(normal(other));
    let mut flags = 0;
    if separation > 0.0 || cosine > 0.9962 {
        flags |= bit;
    }
    if separation < 0.0 || cosine > 0.9962 {
        flags |= bit << 4;
    }
    flags
}
pub fn create_height_field(d: &Definition<'_>) -> Image {
    let nx = d.columns;
    let nz = d.rows;
    let count = nx * nz;
    let cells = (nx - 1) * (nz - 1);
    assert!(nx >= 2 && nz >= 2 && d.heights.len() >= count && d.minimum <= d.maximum);
    let heights_offset = 96;
    let material_offset = heights_offset + align8(2 * count);
    let flags_offset = material_offset + align8(cells);
    let mut image = Image::new(0x8E41E5FB084848F8, flags_offset + align8(2 * cells));
    let height = maxf(d.maximum - d.minimum, 0.005);
    let height_scale = height / 65535.0;
    let inv_height_scale = 1.0 / height_scale;
    let mut lower = d.maximum;
    let mut upper = d.minimum;
    let mut heights = Vec::with_capacity(count);
    for (i, &h) in d.heights[..count].iter().enumerate() {
        let h = clampf(h, d.minimum, d.maximum);
        let q = minf((h - d.minimum) * inv_height_scale, 65535.0) as u16;
        image.bytes_mut()[heights_offset + 2 * i..heights_offset + 2 * i + 2]
            .copy_from_slice(&q.to_le_bytes());
        lower = minf(lower, h);
        upper = maxf(upper, h);
        heights.push(d.minimum + height_scale * q as f32);
    }
    let mut materials = vec![0; cells];
    if !d.materials.is_empty() {
        materials.copy_from_slice(&d.materials[..cells]);
    }
    image.bytes_mut()[material_offset..material_offset + cells].copy_from_slice(&materials);
    image.vector(20, Vec3::new(0.0, d.scale.y * lower, 0.0));
    image.vector(
        32,
        Vec3::new(
            d.scale.x * (nx - 1) as f32,
            d.scale.y * upper,
            d.scale.z * (nz - 1) as f32,
        ),
    );
    image.float(44, d.minimum);
    image.float(48, d.maximum);
    image.float(52, height_scale);
    image.vector(56, d.scale);
    image.word(68, nx as u32);
    image.word(72, nz as u32);
    image.word(76, heights_offset as u32);
    image.word(80, material_offset as u32);
    image.word(84, flags_offset as u32);
    image.bytes_mut()[88] = d.clockwise as u8;
    let triangles = |r: usize, c: usize| {
        let p =
            |x: usize, z: usize| mul(d.scale, Vec3::new(x as f32, heights[z * nx + x], z as f32));
        [
            [p(c, r), p(c, r + 1), p(c + 1, r)],
            [p(c + 1, r + 1), p(c + 1, r), p(c, r + 1)],
        ]
    };
    for r in 0..nz - 1 {
        for c in 0..nx - 1 {
            let cell = r * (nx - 1) + c;
            if materials[cell] == 255 {
                continue;
            }
            let [t0, t1] = triangles(r, c);
            let n0 = normal(t0);
            let n1 = normal(t1);
            let o0 = n0.dot(t0[0]);
            let o1 = n1.dot(t1[0]);
            let diagonal = edge(n0, o0, t1, 0, 2);
            let mut f0 = diagonal;
            let mut f1 = diagonal;
            if r > 0 && materials[cell - (nx - 1)] != 255 {
                f0 |= edge(n0, o0, triangles(r - 1, c)[1], 1, 4);
            }
            if r + 1 < nz - 1 && materials[cell + (nx - 1)] != 255 {
                f1 |= edge(n1, o1, triangles(r + 1, c)[0], 1, 4);
            }
            if c > 0 && materials[cell - 1] != 255 {
                f0 |= edge(n0, o0, triangles(r, c - 1)[1], 2, 1);
            }
            if c + 1 < nx - 1 && materials[cell + 1] != 255 {
                f1 |= edge(n1, o1, triangles(r, c + 1)[0], 2, 1);
            }
            image.bytes_mut()[flags_offset + 2 * cell] = f0;
            image.bytes_mut()[flags_offset + 2 * cell + 1] = f1;
        }
    }
    image.finish();
    image
}
pub fn create_grid(rows: usize, columns: usize, scale: Vec3, holes: bool) -> Image {
    generate(rows, columns, scale, holes, None)
}
pub fn create_wave(
    rows: usize,
    columns: usize,
    scale: Vec3,
    row_frequency: f32,
    column_frequency: f32,
    holes: bool,
) -> Image {
    generate(
        rows,
        columns,
        scale,
        holes,
        Some((row_frequency, column_frequency)),
    )
}
fn generate(
    rows: usize,
    columns: usize,
    scale: Vec3,
    holes: bool,
    wave: Option<(f32, f32)>,
) -> Image {
    let mut heights = vec![0.0; rows * columns];
    if let Some((rf, cf)) = wave {
        let oz = 2.0 * PI * rf;
        let ox = 2.0 * PI * cf;
        for i in 0..rows {
            let rh = compute_cos_sin(oz * i as f32).sine;
            for j in 0..columns {
                heights[i * columns + j] = rh * compute_cos_sin(ox * j as f32).sine;
            }
        }
    }
    let materials: Vec<u8> = (0..(rows - 1) * (columns - 1))
        .map(|k| {
            if holes && k > 0 && k % 16 == 0 {
                255
            } else {
                0
            }
        })
        .collect();
    create_height_field(&Definition {
        heights: &heights,
        materials: &materials,
        scale,
        columns,
        rows,
        minimum: -256.0,
        maximum: 256.0,
        clockwise: false,
    })
}

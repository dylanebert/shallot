//! Box3D mesh.c welding, triangle BVH construction, edge identification and generators.
use crate::math::{compute_cos_sin, maxf, minf, Vec3, HUGE, PI};
use crate::nonconvex_build::{align8, Image};
use std::collections::HashMap;
pub struct Definition<'a> {
    pub vertices: &'a [Vec3],
    pub indices: &'a [u32],
    pub materials: &'a [u8],
    pub weld_tolerance: f32,
    pub weld: bool,
    pub median: bool,
    pub edges: bool,
    /// Input triangles wind clockwise (b3MeshDef::clockWiseWinding).
    pub clockwise: bool,
}
#[derive(Clone, Copy)]
struct Bounds {
    lo: Vec3,
    hi: Vec3,
}
fn vmin(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(minf(a.x, b.x), minf(a.y, b.y), minf(a.z, b.z))
}
fn vmax(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(maxf(a.x, b.x), maxf(a.y, b.y), maxf(a.z, b.z))
}
impl Bounds {
    const EMPTY: Self = Self {
        lo: Vec3 {
            x: f32::MAX,
            y: f32::MAX,
            z: f32::MAX,
        },
        hi: Vec3 {
            x: -f32::MAX,
            y: -f32::MAX,
            z: -f32::MAX,
        },
    };
    fn union(self, b: Self) -> Self {
        Self {
            lo: vmin(self.lo, b.lo),
            hi: vmax(self.hi, b.hi),
        }
    }
    fn point(self, p: Vec3) -> Self {
        self.union(Self { lo: p, hi: p })
    }
    fn center(self) -> Vec3 {
        self.lo.add(self.hi).scale(0.5)
    }
    fn extents(self) -> Vec3 {
        self.hi.sub(self.lo).scale(0.5)
    }
    fn area(self) -> f32 {
        let d = self.hi.sub(self.lo);
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }
    fn sane(self) -> bool {
        (0..3).all(|a| {
            let lo = component(self.lo, a);
            let hi = component(self.hi, a);
            lo.is_finite() && hi.is_finite() && lo <= hi && lo >= -HUGE && hi <= HUGE
        })
    }
}
fn component(v: Vec3, a: usize) -> f32 {
    [v.x, v.y, v.z][a]
}
fn major(v: Vec3) -> usize {
    if v.x < v.y {
        if v.y < v.z {
            2
        } else {
            1
        }
    } else if v.x < v.z {
        2
    } else {
        0
    }
}
#[derive(Clone, Copy)]
struct Primitive {
    bounds: Bounds,
    center: Vec3,
    triangle: usize,
}
struct Split {
    axis: usize,
    index: usize,
    left: Bounds,
    right: Bounds,
}
fn range_bounds(p: &[Primitive]) -> Bounds {
    p.iter().fold(Bounds::EMPTY, |b, p| b.union(p.bounds))
}
fn half(p: &[Primitive]) -> Split {
    let index = p.len() / 2;
    let left = range_bounds(&p[..index]);
    let right = range_bounds(&p[index..]);
    Split {
        axis: major(left.union(right).extents()),
        index,
        left,
        right,
    }
}
fn median(p: &mut [Primitive]) -> Split {
    let bounds = p.iter().fold(
        Bounds {
            lo: p[0].center,
            hi: p[0].center,
        },
        |b, p| b.point(p.center),
    );
    let d = bounds.hi.sub(bounds.lo);
    let c = bounds.center();
    let axis = if d.x >= d.y && d.x >= d.z {
        0
    } else if d.y >= d.z {
        1
    } else {
        2
    };
    let pivot = component(c, axis);
    let mut i = 0;
    let mut j = p.len();
    while i < j {
        while i < j && component(p[i].center, axis) < pivot {
            i += 1;
        }
        while i < j && component(p[j - 1].center, axis) >= pivot {
            j -= 1;
        }
        if i < j {
            p.swap(i, j - 1);
            i += 1;
            j -= 1;
        }
    }
    if i == 0 || i == p.len() - 1 {
        i = p.len() / 2;
    }
    Split {
        axis,
        index: i,
        left: range_bounds(&p[..i]),
        right: range_bounds(&p[i..]),
    }
}
fn sah(p: &mut [Primitive]) -> Option<Split> {
    let bounds = p.iter().fold(
        Bounds {
            lo: p[0].center,
            hi: p[0].center,
        },
        |b, p| b.point(p.center),
    );
    let mut best = None;
    let mut best_bucket = 0;
    let mut best_cost = f32::MAX;
    for axis in 0..3 {
        if component(bounds.extents(), axis) < 0.005 {
            continue;
        }
        let factor =
            8.0 * (1.0 - f32::EPSILON) / (component(bounds.hi, axis) - component(bounds.lo, axis));
        let mut counts = [0; 8];
        let mut boxes = [Bounds::EMPTY; 8];
        for p in p.iter() {
            let i = (factor * (component(p.center, axis) - component(bounds.lo, axis))) as usize;
            counts[i] += 1;
            boxes[i] = boxes[i].union(p.bounds);
        }
        for i in 0..7 {
            let mut lc = 0;
            let mut rc = 0;
            let mut left = Bounds::EMPTY;
            let mut right = Bounds::EMPTY;
            for k in 0..=i {
                lc += counts[k];
                left = left.union(boxes[k]);
            }
            for k in i + 1..8 {
                rc += counts[k];
                right = right.union(boxes[k]);
            }
            if lc > 0 && rc > 0 {
                let cost = lc as f32 * left.area() + rc as f32 * right.area();
                if cost < best_cost {
                    best_cost = cost;
                    best_bucket = i;
                    best = Some(Split {
                        axis,
                        index: lc,
                        left,
                        right,
                    });
                }
            }
        }
    }
    if let Some(s) = &best {
        let factor = 8.0 * (1.0 - f32::EPSILON)
            / (component(bounds.hi, s.axis) - component(bounds.lo, s.axis));
        let mut index = 0;
        for i in 0..p.len() {
            let bucket =
                (factor * (component(p[i].center, s.axis) - component(bounds.lo, s.axis))) as usize;
            if bucket <= best_bucket {
                p.swap(i, index);
                index += 1;
            }
        }
    }
    best
}
struct Node {
    bounds: Bounds,
    data: u32,
    offset: usize,
}
fn build(
    nodes: &mut Vec<Node>,
    p: &mut [Primitive],
    start: usize,
    use_median: bool,
) -> (usize, u32) {
    let index = nodes.len();
    let split = if p.len() > 4 {
        if use_median {
            Some(median(p))
        } else {
            sah(p).or_else(|| if p.len() > 8 { Some(half(p)) } else { None })
        }
    } else {
        None
    };
    if let Some(s) = split {
        nodes.push(Node {
            bounds: Bounds::EMPTY,
            data: 0,
            offset: 0,
        });
        let (left, right) = p.split_at_mut(s.index);
        let (_, lh) = build(nodes, left, start, use_median);
        let (ri, rh) = build(nodes, right, start + s.index, use_median);
        nodes[index] = Node {
            bounds: s.left.union(s.right),
            data: s.axis as u32 | ((ri - index) as u32) << 2,
            offset: 0,
        };
        (index, lh.max(rh) + 1)
    } else {
        nodes.push(Node {
            bounds: range_bounds(p),
            data: 3 | (p.len() as u32) << 2,
            offset: start,
        });
        // C's unsplittable SAH early return does not write the height output.
        (index, if p.len() > 4 { 0 } else { 1 })
    }
}
fn cell_key(x: i32, y: i32, z: i32) -> u64 {
    let mut key = 0u64;
    for c in [x, y, z] {
        key ^= (c as i64 as u64)
            .wrapping_add(0x9e3779b9)
            .wrapping_add(key << 6)
            .wrapping_add(key >> 2);
    }
    key
}
fn weld(vertices: &[Vec3], indices: &[u32], tolerance: f32) -> (Vec<Vec3>, Vec<u32>) {
    let cell = 2.0 * tolerance;
    let mut map: HashMap<u64, usize> = HashMap::with_capacity(vertices.len());
    let mut nodes: Vec<(usize, Option<usize>)> = Vec::with_capacity(vertices.len());
    let mut mapping = vec![0; vertices.len()];
    let mut out = Vec::with_capacity(vertices.len());
    for (i, &v) in vertices.iter().enumerate() {
        let x = (v.x / cell).floor() as i32;
        let y = (v.y / cell).floor() as i32;
        let z = (v.z / cell).floor() as i32;
        let mut duplicate = None;
        'neighbors: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let mut n = map
                        .get(&cell_key(
                            x.wrapping_add(dx),
                            y.wrapping_add(dy),
                            z.wrapping_add(dz),
                        ))
                        .copied();
                    while let Some(j) = n {
                        let (other, next) = nodes[j];
                        let d = v.sub(vertices[other]);
                        if d.x.abs() <= tolerance
                            && d.y.abs() <= tolerance
                            && d.z.abs() <= tolerance
                        {
                            duplicate = Some(other);
                            break 'neighbors;
                        }
                        n = next;
                    }
                }
            }
        }
        if let Some(other) = duplicate {
            mapping[i] = mapping[other];
        } else {
            mapping[i] = out.len() as u32;
            out.push(v);
            let key = cell_key(x, y, z);
            let head = map.insert(key, nodes.len());
            nodes.push((i, head));
        }
    }
    (out, indices.iter().map(|&i| mapping[i as usize]).collect())
}
struct Edge {
    a: u32,
    b: u32,
    t1: usize,
    t2: usize,
    e1: usize,
    e2: usize,
    count: u16,
}
fn identify(vertices: &[Vec3], triangles: &[[u32; 3]]) -> Vec<u8> {
    let mut edges = Vec::with_capacity(3 * triangles.len());
    let mut normals = Vec::with_capacity(triangles.len());
    for (i, t) in triangles.iter().enumerate() {
        for j in 0..3 {
            let a = t[j];
            let b = t[(j + 1) % 3];
            edges.push(Edge {
                a: a.min(b),
                b: a.max(b),
                t1: i,
                t2: usize::MAX,
                e1: j,
                e2: usize::MAX,
                count: 1,
            });
        }
        let [a, b, c] = t.map(|j| vertices[j as usize]);
        normals.push(b.sub(a).cross(c.sub(a)).normalize());
    }
    let mut map = HashMap::with_capacity(edges.len());
    for i in 0..edges.len() {
        let e = &edges[i];
        let key = ((e.a as u64) << 32) | e.b as u64;
        if let Some(&other) = map.get(&key) {
            let (t, j) = (e.t1, e.e1);
            let base: &mut Edge = &mut edges[other];
            if base.count == 1 {
                base.t2 = t;
                base.e2 = j;
            }
            base.count = base.count.wrapping_add(1);
        } else {
            map.insert(key, i);
        }
    }
    let mut flags = vec![0; triangles.len()];
    for e in edges {
        if e.count != 2 {
            continue;
        }
        let [a, b, c] = triangles[e.t1].map(|j| vertices[j as usize]);
        let p = vertices[triangles[e.t2][(e.e2 + 2) % 3] as usize];
        let volume = b.sub(a).cross(c.sub(a)).dot(p.sub(a));
        let cosine = normals[e.t1].dot(normals[e.t2]);
        if volume > 0.0 || cosine > 0.9962 {
            flags[e.t1] |= 1 << e.e1;
            flags[e.t2] |= 1 << e.e2;
        }
        if volume < 0.0 || cosine > 0.9962 {
            flags[e.t1] |= 0x10 << e.e1;
            flags[e.t2] |= 0x10 << e.e2;
        }
    }
    flags
}
pub fn create_mesh(d: &Definition<'_>) -> Option<Image> {
    if d.vertices.len() < 3 || d.indices.len() < 3 {
        return None;
    }
    let index_count = 3 * (d.indices.len() / 3);
    let (vertices, indices) = if d.weld && d.weld_tolerance > 0.0 {
        weld(d.vertices, &d.indices[..index_count], d.weld_tolerance)
    } else {
        (d.vertices.to_vec(), d.indices[..index_count].to_vec())
    };
    let mut bounds = Bounds::EMPTY;
    let mut primitives = Vec::with_capacity(index_count / 3);
    let mut degenerate = 0;
    let mut area = 0.0;
    let mut material_count = 1;
    for (i, t) in indices.as_chunks::<3>().0.iter().enumerate() {
        let a = vertices[t[0] as usize];
        let b = vertices[t[1] as usize];
        let c = vertices[t[2] as usize];
        let triangle_area = 0.5 * b.sub(a).cross(c.sub(a)).length();
        if triangle_area < 0.01 * 0.005 * 0.005 {
            if t[0] != t[1] && t[0] != t[2] && t[1] != t[2] {
                degenerate += 1;
            }
            continue;
        }
        area += triangle_area;
        let b = Bounds {
            lo: vmin(a, vmin(b, c)),
            hi: vmax(a, vmax(b, c)),
        };
        bounds = bounds.union(b);
        primitives.push(Primitive {
            bounds: b,
            center: b.center(),
            triangle: i,
        });
        if !d.materials.is_empty() {
            material_count = material_count.max(d.materials[i] as u32 + 1);
        }
    }
    if !bounds.sane() {
        return None;
    }
    let mut nodes = Vec::with_capacity(2 * primitives.len() - 1);
    let (_, height) = build(&mut nodes, &mut primitives, 0, d.median);
    let triangles: Vec<[u32; 3]> = primitives
        .iter()
        .map(|p| {
            let [a, b, c] = indices[3 * p.triangle..3 * p.triangle + 3]
                .try_into()
                .unwrap();
            if d.clockwise {
                [a, c, b]
            } else {
                [a, b, c]
            }
        })
        .collect();
    let materials: Vec<u8> = primitives
        .iter()
        .map(|p| {
            if d.materials.is_empty() {
                0
            } else {
                d.materials[p.triangle]
            }
        })
        .collect();
    let mut sorted = Vec::with_capacity(triangles.len());
    let mut sorted_materials = Vec::with_capacity(materials.len());
    let mut stack = vec![0];
    while let Some(i) = stack.pop() {
        let n = &mut nodes[i];
        if n.data & 3 != 3 {
            if stack.len() >= 254 {
                return None;
            }
            stack.push(i + (n.data >> 2) as usize);
            stack.push(i + 1);
        } else {
            let count = (n.data >> 2) as usize;
            sorted.extend_from_slice(&triangles[n.offset..n.offset + count]);
            sorted_materials.extend_from_slice(&materials[n.offset..n.offset + count]);
            n.offset = sorted.len() - count;
        }
    }
    let flags = if d.edges {
        identify(&vertices, &sorted)
    } else {
        vec![0; sorted.len()]
    };
    let no = 96;
    let vo = no + align8(32 * nodes.len());
    let to = vo + align8(12 * vertices.len());
    let mo = to + align8(12 * sorted.len());
    let fo = mo + align8(sorted.len());
    let mut image = Image::new(0xAAAB9A00F1A8AAF7, fo + align8(sorted.len()));
    image.vector(20, bounds.lo);
    image.vector(32, bounds.hi);
    image.float(44, area);
    image.word(48, height);
    image.word(52, degenerate);
    for (o, n) in [
        (56, no),
        (60, nodes.len()),
        (64, vo),
        (68, vertices.len()),
        (72, to),
        (76, sorted.len()),
        (80, mo),
        (84, material_count as usize),
        (88, fo),
    ] {
        image.word(o, n as u32);
    }
    for (i, n) in nodes.iter().enumerate() {
        let o = no + 32 * i;
        image.vector(o, n.bounds.lo);
        image.word(o + 12, n.data);
        image.vector(o + 16, n.bounds.hi);
        image.word(o + 28, n.offset as u32);
    }
    for (i, &v) in vertices.iter().enumerate() {
        image.vector(vo + 12 * i, v);
    }
    for (i, t) in sorted.iter().enumerate() {
        for (j, &v) in t.iter().enumerate() {
            image.word(to + 12 * i + 4 * j, v);
        }
    }
    image.bytes_mut()[mo..mo + sorted.len()].copy_from_slice(&sorted_materials);
    image.bytes_mut()[fo..fo + sorted.len()].copy_from_slice(&flags);
    image.finish();
    Some(image)
}
pub fn create_box(center: Vec3, extent: Vec3, edges: bool, hollow: bool) -> Option<Image> {
    let Vec3 { x, y, z } = extent;
    let vertices = [
        Vec3::new(x, y, z),
        Vec3::new(-x, y, z),
        Vec3::new(-x, -y, z),
        Vec3::new(x, -y, z),
        Vec3::new(x, y, -z),
        Vec3::new(-x, y, -z),
        Vec3::new(-x, -y, -z),
        Vec3::new(x, -y, -z),
    ]
    .map(|v| v.add(center));
    let indices = if hollow {
        [
            3, 1, 0, 3, 2, 1, 1, 4, 0, 5, 4, 1, 7, 3, 0, 7, 0, 4, 5, 7, 4, 7, 5, 6, 2, 5, 1, 5, 2,
            6, 7, 2, 3, 2, 7, 6,
        ]
    } else {
        [
            0, 1, 3, 1, 2, 3, 0, 4, 1, 1, 4, 5, 0, 3, 7, 4, 0, 7, 4, 7, 5, 6, 5, 7, 1, 5, 2, 6, 2,
            5, 3, 2, 7, 6, 7, 2,
        ]
    };
    create_mesh(&Definition {
        vertices: &vertices,
        indices: &indices,
        materials: &[],
        weld_tolerance: 0.0,
        weld: false,
        median: false,
        edges: edges || hollow,
        clockwise: false,
    })
}
pub fn create_grid(
    x_count: usize,
    z_count: usize,
    width: f32,
    materials: usize,
    edges: bool,
) -> Option<Image> {
    grid(x_count, z_count, width, materials, edges, None)
}
pub fn create_wave(
    x_count: usize,
    z_count: usize,
    width: f32,
    amplitude: f32,
    row_frequency: f32,
    column_frequency: f32,
) -> Option<Image> {
    grid(
        x_count,
        z_count,
        width,
        0,
        true,
        Some((amplitude, row_frequency, column_frequency)),
    )
}
fn grid(
    nx: usize,
    nz: usize,
    width: f32,
    material_count: usize,
    edges: bool,
    wave: Option<(f32, f32, f32)>,
) -> Option<Image> {
    let mut vertices = Vec::with_capacity((nx + 1) * (nz + 1));
    let mut x = -0.5 * (width * nx as f32);
    let (amplitude, oz, ox) = wave.map_or((0.0, 0.0, 0.0), |(a, rf, cf)| {
        (a, 2.0 * PI * rf * width, 2.0 * PI * cf * width)
    });
    for ix in 0..=nx {
        let rh = if wave.is_some() {
            compute_cos_sin(ox * ix as f32).sine
        } else {
            0.0
        };
        let mut z = -0.5 * (width * nz as f32);
        for iz in 0..=nz {
            let y = if wave.is_some() {
                amplitude * rh * compute_cos_sin(oz * iz as f32).sine
            } else {
                0.0
            };
            vertices.push(Vec3::new(x, y, z));
            z += width;
        }
        x += width;
    }
    let mut indices = Vec::with_capacity(6 * nx * nz);
    let mut materials = Vec::with_capacity(2 * nx * nz);
    let mut m = 0;
    for ix in 0..nx {
        for iz in 0..nz {
            let a = (iz + (nz + 1) * ix) as u32;
            let b = a + 1;
            let c = b + (nz + 1) as u32;
            let d = c - 1;
            indices.extend_from_slice(&[a, b, c, c, d, a]);
            if material_count > 0 {
                materials.extend_from_slice(&[(m % material_count) as u8; 2]);
            }
            m += 1;
        }
    }
    create_mesh(&Definition {
        vertices: &vertices,
        indices: &indices,
        materials: &materials,
        weld_tolerance: 0.0,
        weld: false,
        median: true,
        edges,
        clockwise: false,
    })
}
pub fn create_torus(radial: usize, tubular: usize, radius: f32, thickness: f32) -> Option<Image> {
    let mut vertices = Vec::with_capacity(radial * tubular);
    for r in 0..radial {
        for t in 0..tubular {
            let u = (t as f32 / tubular as f32) * (2.0 * PI);
            let v = (r as f32 / radial as f32) * (2.0 * PI);
            let u = compute_cos_sin(u);
            let v = compute_cos_sin(v);
            let ring = radius + thickness * v.cosine;
            vertices.push(Vec3::new(
                ring * u.cosine,
                ring * u.sine,
                thickness * v.sine,
            ));
        }
    }
    let mut indices = Vec::with_capacity(6 * radial * tubular);
    for r in 0..radial {
        let r2 = (r + 1) % radial;
        for t in 0..tubular {
            let t2 = (t + 1) % tubular;
            let a = (r * tubular + t) as u32;
            let b = (r * tubular + t2) as u32;
            let c = (r2 * tubular + t2) as u32;
            let d = (r2 * tubular + t) as u32;
            indices.extend_from_slice(&[a, b, c, c, d, a]);
        }
    }
    create_mesh(&Definition {
        vertices: &vertices,
        indices: &indices,
        materials: &[],
        weld_tolerance: 0.0,
        weld: false,
        median: false,
        edges: true,
        clockwise: false,
    })
}

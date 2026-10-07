//! Mesh contact reduction from Box3D mesh_contact.c (Erin Catto, MIT).
use crate::manifold::{make_feature_id, LocalManifold, LocalManifoldPoint, SatCache};
use crate::math::{absf, maxf, minf, Mat3, Transform, Vec2, Vec3};
use crate::narrowphase::{ConvexContactCache, ConvexShape, Manifold};
use crate::triangle_manifold::{
    collide_capsule_and_triangle, collide_hull_and_triangle, collide_sphere_and_triangle,
};

pub const MAX_TRIANGLES: usize = 256;
const MAX_POINTS: usize = 32 * MAX_TRIANGLES;
const SLOP: f32 = 0.005;
const REST_OFFSET: f32 = SLOP;
fn cross(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

#[derive(Clone, Copy)]
pub struct TriangleInput {
    pub vertices: [Vec3; 3],
    pub indices: [u32; 3],
    pub flags: u32,
    pub triangle_index: i32,
    pub material_index: u32,
}

#[derive(Clone, Copy)]
pub struct TriangleCache {
    pub triangle_index: i32,
    pub cache: ConvexContactCache,
}
impl TriangleCache {
    fn empty(index: usize) -> Self {
        Self {
            triangle_index: index as i32,
            cache: ConvexContactCache::Empty,
        }
    }
}
pub enum TriangleSource<'a> {
    Mesh {
        mesh: crate::mesh_query::Mesh<'a>,
        flags: &'a [u8],
    },
    Height {
        field: crate::height_query::HeightField<'a>,
        flags: &'a [u8],
    },
}
impl TriangleSource<'_> {
    pub fn triangle(&self, index: usize) -> TriangleInput {
        use crate::mesh_query::mul;
        let (vertices, indices, flags, material_index) = match self {
            Self::Mesh { mesh, flags } => {
                let mut indices = mesh.triangles[index].indices;
                let mut flags = flags[index];
                if mesh.scale.x * mesh.scale.y * mesh.scale.z < 0.0 {
                    indices.swap(1, 2);
                    flags = (flags >> 4) & 7;
                }
                (
                    indices.map(|i| mul(mesh.scale, mesh.vertices[i as usize])),
                    indices,
                    flags,
                    mesh.materials[index],
                )
            }
            Self::Height { field, flags } => {
                let cell = index >> 1;
                let row = cell / (field.columns - 1);
                let column = cell - row * (field.columns - 1);
                let i11 = row * field.columns + column;
                let i12 = i11 + 1;
                let i21 = (row + 1) * field.columns + column;
                let i22 = i21 + 1;
                let mut indices = if index & 1 == 0 {
                    [i11 as u32, i21 as u32, i12 as u32]
                } else {
                    [i22 as u32, i12 as u32, i21 as u32]
                };
                let mut flags = flags[index];
                if field.clockwise {
                    indices.swap(1, 2);
                    let edge1 = flags & 0x11;
                    let edge3 = flags & 0x44;
                    flags = (flags & !0x55) | (edge1 << 2) | (edge3 >> 2);
                }
                let vertices = indices.map(|i| {
                    let row = i as usize / field.columns;
                    let column = i as usize % field.columns;
                    mul(
                        field.scale,
                        Vec3::new(
                            column as f32,
                            field.min_height
                                + field.height_scale * field.heights[i as usize] as f32,
                            row as f32,
                        ),
                    )
                });
                (vertices, indices, flags, field.materials[cell])
            }
        };
        TriangleInput {
            vertices,
            indices,
            flags: flags as u32,
            triangle_index: index as i32,
            material_index: material_index as u32,
        }
    }

    pub fn query(&self, lower: Vec3, upper: Vec3, indices: &mut [usize; MAX_TRIANGLES]) -> usize {
        use crate::mesh_query::{bounds_overlap, bounds_triangle_overlap, max, min, mul};
        let mut count = 0;
        match self {
            Self::Mesh { mesh, .. } => {
                let inv = Vec3::new(1.0 / mesh.scale.x, 1.0 / mesh.scale.y, 1.0 / mesh.scale.z);
                let a = mul(inv, lower);
                let b = mul(inv, upper);
                let lower = min(a, b);
                let upper = max(a, b);
                let center = lower.add(upper).scale(0.5);
                let extent = upper.sub(center);
                let mut stack = [0usize; 256];
                let mut top = 0;
                let mut index = 0;
                loop {
                    let node = mesh.nodes[index];
                    if bounds_overlap(node.lower, node.upper, lower, upper) {
                        if node.is_leaf() {
                            for i in
                                node.triangle_offset..node.triangle_offset + node.triangle_count()
                            {
                                let vertices = mesh.triangles[i as usize]
                                    .indices
                                    .map(|v| mesh.vertices[v as usize]);
                                if bounds_triangle_overlap(center, extent, vertices) {
                                    indices[count] = i as usize;
                                    count += 1;
                                    if count == MAX_TRIANGLES {
                                        return count;
                                    }
                                }
                            }
                        } else {
                            stack[top] = index + node.child_offset();
                            top += 1;
                            index += 1;
                            continue;
                        }
                    }
                    if top == 0 {
                        break;
                    }
                    top -= 1;
                    index = stack[top];
                }
            }
            Self::Height { field, .. } => {
                let min_row = (lower.z / field.scale.z).floor() as i32;
                let max_row = (upper.z / field.scale.z).floor() as i32;
                let min_column = (lower.x / field.scale.x).floor() as i32;
                let max_column = (upper.x / field.scale.x).floor() as i32;
                for row in min_row.max(0)..=max_row.min(field.rows as i32 - 2) {
                    for column in min_column.max(0)..=max_column.min(field.columns as i32 - 2) {
                        let cell = row as usize * (field.columns - 1) + column as usize;
                        if field.materials[cell] == 255 {
                            continue;
                        }
                        let [a, b, c, d] = field.corners(row as usize, column as usize);
                        let cell_lower = min(min(a, b), min(c, d));
                        let cell_upper = max(max(a, b), max(c, d));
                        if bounds_overlap(lower, upper, cell_lower, cell_upper) {
                            for i in [2 * cell, 2 * cell + 1] {
                                if count == MAX_TRIANGLES {
                                    return count;
                                }
                                indices[count] = i;
                                count += 1;
                            }
                        }
                    }
                }
            }
        }
        count
    }
}

pub struct MeshCache {
    pub lower: Vec3,
    pub upper: Vec3,
    pub count: usize,
    pub triangles: [TriangleCache; MAX_TRIANGLES],
}
impl MeshCache {
    pub fn refresh(
        &mut self,
        source: &TriangleSource,
        xf: Transform,
        lower: Vec3,
        upper: Vec3,
        previous: &mut [TriangleCache; MAX_TRIANGLES],
    ) {
        let contains = self.lower.x <= lower.x
            && self.lower.y <= lower.y
            && self.lower.z <= lower.z
            && self.upper.x >= upper.x
            && self.upper.y >= upper.y
            && self.upper.z >= upper.z;
        if contains {
            return;
        }
        let radius = 0.05 + 4.0 * SLOP;
        let extent = Vec3::new(radius, radius, radius);
        self.lower = lower.sub(extent);
        self.upper = upper.add(extent);
        let transform = xf.invert();
        let center = transform.point(self.lower.add(self.upper).scale(0.5));
        let half = self.upper.sub(self.lower).scale(0.5);
        let extent = Mat3::from_quat(transform.q).abs().mul_v(half);
        let mut indices = [0usize; MAX_TRIANGLES];
        let count = source.query(center.sub(extent), center.add(extent), &mut indices);
        #[cfg(any(test, debug_assertions))]
        assert!(indices[..count].windows(2).all(|pair| pair[0] < pair[1]));

        previous[..self.count].copy_from_slice(&self.triangles[..self.count]);
        let mut old_index = 0;
        for i in 0..count {
            self.triangles[i] = TriangleCache::empty(indices[i]);
            while old_index < self.count && previous[old_index].triangle_index < indices[i] as i32 {
                old_index += 1;
            }
            if old_index < self.count && previous[old_index].triangle_index == indices[i] as i32 {
                self.triangles[i].cache = previous[old_index].cache;
            }
        }
        self.count = count;
    }
}

struct TriangleManifold {
    normal: Vec3,
    point_base: usize,
    point_count: usize,
    feature: u32,
    squared_distance: f32,
}

struct TriangleResult {
    manifold: TriangleManifold,
    normal: Vec3,
    indices: [u32; 3],
    flags: u32,
    triangle_index: i32,
    material_index: u32,
}
#[derive(Clone, Copy)]
struct Cluster {
    normal: Vec3,
    triangle_normal: Vec3,
    base: usize,
    capacity: usize,
    count: usize,
}
#[derive(Clone, Copy)]
struct Point2D {
    p: Vec2,
    separation: f32,
    index: usize,
}

/// Caller-owned task scratch; no allocation or shared scratch occurs during the sweep.
/// Every field is initialized over its active span before it is read.
pub struct MeshScratch<'a> {
    triangles: &'a mut [TriangleResult],
    accepted: &'a mut [usize],
    tentative: &'a mut [usize],
    membership: &'a mut [usize],
    clusters: &'a mut [Cluster],
    triangle_points: &'a mut [LocalManifoldPoint],
    points: &'a mut [LocalManifoldPoint],
    point_materials: &'a mut [u32],
    projected: &'a mut [Point2D],
    pub output: &'a mut [Manifold],
    pub materials: &'a mut [[u32; 4]],
}

const SCRATCH_BYTES: usize = MAX_TRIANGLES
    * (core::mem::size_of::<TriangleResult>()
        + 3 * core::mem::size_of::<usize>()
        + core::mem::size_of::<Cluster>())
    + MAX_POINTS
        * (2 * core::mem::size_of::<LocalManifoldPoint>()
            + core::mem::size_of::<u32>()
            + core::mem::size_of::<Point2D>())
    + 11 * 16;

pub struct MeshStorage {
    words: [u128; SCRATCH_BYTES.div_ceil(16)],
    output: [Manifold; MAX_TRIANGLES],
    materials: [[u32; 4]; MAX_TRIANGLES],
}

impl MeshStorage {
    pub fn scratch(&mut self, count: usize) -> MeshScratch<'_> {
        assert!(count <= MAX_TRIANGLES);
        let mut offset = 0;
        let base = self.words.as_mut_ptr().cast::<u8>();
        // The arena is initialized before this view is made. Each span is disjoint and
        // count-sized; the backing store is aligned for all scratch record types.
        unsafe fn span<'a, T>(base: *mut u8, offset: &mut usize, count: usize) -> &'a mut [T] {
            *offset = offset.next_multiple_of(core::mem::align_of::<T>());
            let ptr = base.add(*offset).cast::<T>();
            *offset += count * core::mem::size_of::<T>();
            core::slice::from_raw_parts_mut(ptr, count)
        }
        unsafe {
            MeshScratch {
                triangles: span(base, &mut offset, count),
                accepted: span(base, &mut offset, count),
                tentative: span(base, &mut offset, count),
                membership: span(base, &mut offset, count),
                clusters: span(base, &mut offset, count),
                triangle_points: span(base, &mut offset, count * 32),
                points: span(base, &mut offset, count * 32),
                point_materials: span(base, &mut offset, count * 32),
                projected: span(base, &mut offset, count * 32),
                output: &mut self.output[..count],
                materials: &mut self.materials[..count],
            }
        }
    }
}

struct Features {
    edges: [u64; 64],
    edge_count: usize,
    vertices: [u32; 64],
    vertex_count: usize,
}
impl Features {
    fn new() -> Self {
        Self {
            edges: [0; 64],
            edge_count: 0,
            vertices: [0; 64],
            vertex_count: 0,
        }
    }
    fn key(a: u32, b: u32) -> u64 {
        ((a.min(b) as u64) << 32) | a.max(b) as u64
    }
    fn add_edge(&mut self, a: u32, b: u32) -> bool {
        let key = Self::key(a, b);
        if self.edges[..self.edge_count].contains(&key) {
            return false;
        }
        if self.edge_count == 64 {
            return true;
        }
        self.edges[self.edge_count] = key;
        self.edge_count += 1;
        true
    }
    fn find_edge(&self, a: u32, b: u32) -> bool {
        self.edges[..self.edge_count].contains(&Self::key(a, b))
    }
    fn add_vertex(&mut self, a: u32) -> bool {
        if self.vertices[..self.vertex_count].contains(&a) {
            return false;
        }
        if self.vertex_count == 64 {
            return true;
        }
        self.vertices[self.vertex_count] = a;
        self.vertex_count += 1;
        true
    }
    fn claim(&mut self, indices: [u32; 3]) {
        for i in 0..3 {
            self.add_edge(indices[i], indices[(i + 1) % 3]);
            self.add_vertex(indices[i]);
        }
    }
}
fn better(score: f32, separation: f32, best: f32, deepest: f32, tolerance: f32) -> bool {
    if score > best + tolerance {
        return true;
    }
    if score < best - tolerance {
        return false;
    }
    separation < deepest - SLOP
}
fn cull(points: &mut [Point2D]) -> usize {
    let mut count = points.len();
    if count <= 1 {
        return count;
    }
    let tolerance = 0.25 * SLOP;
    let squared_tolerance = tolerance * tolerance;
    let mut best = 0.0;
    let mut deepest = f32::INFINITY;
    let mut indices = (usize::MAX, usize::MAX);
    for i in 0..count {
        for j in i + 1..count {
            let d = points[i].p.sub(points[j].p);
            let score = d.dot(d);
            let separation = points[i].separation + points[j].separation;
            if better(score, separation, best, deepest, squared_tolerance) {
                indices = (i, j);
                best = score;
                deepest = separation;
            }
        }
    }
    if best < squared_tolerance {
        let mut index = 0;
        for i in 1..count {
            if points[i].separation < points[index].separation {
                index = i;
            }
        }
        points[0] = points[index];
        return 1;
    }
    let mut final_points = [points[indices.0], points[indices.1], points[0], points[0]];
    points[indices.1] = points[count - 1];
    points[indices.0] = points[count - 2];
    count -= 2;
    if count == 0 {
        points[..2].copy_from_slice(&final_points[..2]);
        return 2;
    }
    let a = final_points[0].p;
    let mut b = final_points[1].p;
    let mut ba = b.sub(a);
    best = 0.0;
    deepest = f32::INFINITY;
    let mut index = usize::MAX;
    let mut signed = 0.0;
    for i in 0..count {
        let area = cross(ba, points[i].p.sub(a));
        let score = absf(area);
        if better(
            score,
            points[i].separation,
            best,
            deepest,
            squared_tolerance,
        ) {
            signed = area;
            best = score;
            deepest = points[i].separation;
            index = i;
        }
    }
    if index == usize::MAX {
        points[..2].copy_from_slice(&final_points[..2]);
        return 2;
    }
    final_points[2] = points[index];
    if count == 1 {
        points[..3].copy_from_slice(&final_points[..3]);
        return 3;
    }
    points[index] = points[count - 1];
    count -= 1;
    let mut c = final_points[2].p;
    if signed < 0.0 {
        core::mem::swap(&mut b, &mut c);
        ba = b.sub(a);
    }
    let cb = c.sub(b);
    let ac = a.sub(c);
    best = 0.0;
    deepest = f32::INFINITY;
    index = usize::MAX;
    for i in 0..count {
        let p = points[i].p;
        let score = maxf(
            cross(p.sub(a), ba),
            maxf(cross(p.sub(b), cb), cross(p.sub(c), ac)),
        );
        if better(
            score,
            points[i].separation,
            best,
            deepest,
            squared_tolerance,
        ) {
            best = score;
            deepest = points[i].separation;
            index = i;
        }
    }
    let count = if index == usize::MAX {
        3
    } else {
        final_points[3] = points[index];
        4
    };
    points[..count].copy_from_slice(&final_points[..count]);
    count
}
fn sort_tentative(indices: &mut [usize], triangles: &[TriangleResult]) {
    if indices.len() <= 1 {
        return;
    }
    let less = |a: usize, b: usize| {
        triangles[a].manifold.squared_distance < triangles[b].manifold.squared_distance
    };
    let mut left = 0;
    let mut right = indices.len() - 1;
    let mut stack = [(0usize, 0usize); 32];
    let mut top = 0;
    loop {
        if right - left + 1 >= 16 {
            let a = left + 1;
            let b = left + ((right - left) >> 1);
            let c = right;
            if less(indices[b], indices[a]) {
                if less(indices[c], indices[b]) {
                    indices.swap(a, c);
                } else {
                    indices.swap(a, b);
                    if less(indices[c], indices[b]) {
                        indices.swap(b, c);
                    }
                }
            } else if less(indices[c], indices[b]) {
                indices.swap(b, c);
                if less(indices[b], indices[a]) {
                    indices.swap(a, b);
                }
            }
            indices.swap(left, b);
            let mut i = left + 1;
            let mut j = right;
            loop {
                i += 1;
                while less(indices[i], indices[left]) {
                    i += 1;
                }
                j -= 1;
                while less(indices[left], indices[j]) {
                    j -= 1;
                }
                if i >= j {
                    break;
                }
                indices.swap(i, j);
            }
            i = j + 1;
            indices.swap(left, j);
            j -= 1;
            let (large, small) = if j - left >= right - i {
                ((left, j), (i, right))
            } else {
                ((i, right), (left, j))
            };
            if small.0 == small.1 {
                (left, right) = large;
            } else {
                stack[top] = large;
                top += 1;
                (left, right) = small;
            }
        } else {
            for i in left + 1..=right {
                let mut j = i;
                while j > left && less(indices[j], indices[j - 1]) {
                    indices.swap(j, j - 1);
                    j -= 1;
                }
            }
            if top == 0 {
                return;
            }
            top -= 1;
            (left, right) = stack[top];
        }
    }
}

/// Collide cached triangles into clustered persistent manifolds and per-point material indices.
/// Geometry is in mesh-local coordinates; the caller owns query/cache refresh and storage.
pub fn compute_mesh_manifolds(
    scratch: &mut MeshScratch,
    triangles: &mut [TriangleCache],
    geometry: impl Fn(usize) -> TriangleInput,
    shape: &ConvexShape,
    xf_a: Transform,
    xf_b: Transform,
    fast: bool,
    speculative: bool,
    old: &mut [Manifold],
) -> usize {
    let transform = xf_b.inv_mul(xf_a);
    let matrix = Mat3::from_quat(transform.q);
    let mut features = Features::new();
    let mut accepted_count = 0;
    let mut tentative_count = 0;
    let mut total = 0;
    let capacity = triangles.len() * 32;
    for (i, cached) in triangles.iter_mut().enumerate() {
        if total + 3 >= capacity {
            break;
        }
        let triangle = geometry(cached.triangle_index as usize);
        let [a, b, c] = triangle.vertices.map(|v| matrix.mul_v(v).add(transform.p));
        let point_base = total;
        let mut m = LocalManifold::new(&mut scratch.triangle_points[total..capacity]);
        m.triangle_flags = triangle.flags;
        match shape {
            ConvexShape::Sphere(s) => {
                collide_sphere_and_triangle(&mut m, capacity - total, s, a, b, c)
            }
            ConvexShape::Capsule(s) => collide_capsule_and_triangle(
                &mut m,
                capacity - total,
                s,
                a,
                b,
                c,
                cached.cache.simplex(),
            ),
            ConvexShape::Hull(h) => {
                let sat = cached.cache.sat();
                if fast && sat.ty == 4 {
                    *sat = SatCache::empty();
                }
                collide_hull_and_triangle(
                    &mut m,
                    capacity - total,
                    h,
                    a,
                    b,
                    c,
                    triangle.flags,
                    sat,
                    speculative,
                );
            }
        }
        if m.point_count == 0 {
            continue;
        }
        total += m.point_count;
        m.triangle_index = triangle.triangle_index;
        m.vertex_indices = triangle.indices;
        m.triangle_normal = b.sub(a).cross(c.sub(a)).normalize();
        let normal = m.triangle_normal;
        let mut accept = m.feature == 1;
        if m.feature == 2 {
            accept = normal.dot(m.normal) > 0.5;
            if !accept {
                let mut separation = m.points[0].separation;
                for j in 1..m.point_count {
                    separation = minf(separation, m.points[j].separation);
                }
                accept = separation < -2.0 * SLOP;
            }
        }
        scratch.triangles[i] = TriangleResult {
            manifold: TriangleManifold {
                normal: m.normal,
                point_base,
                point_count: m.point_count,
                feature: m.feature,
                squared_distance: m.squared_distance,
            },
            normal,
            indices: m.vertex_indices,
            flags: m.triangle_flags,
            triangle_index: m.triangle_index,
            material_index: triangle.material_index,
        };
        if accept {
            features.claim(triangle.indices);
            scratch.accepted[accepted_count] = i;
            accepted_count += 1;
        } else {
            scratch.tentative[tentative_count] = i;
            tentative_count += 1;
        }
    }
    if matches!(shape, ConvexShape::Sphere(_)) {
        sort_tentative(
            &mut scratch.tentative[..tentative_count],
            &scratch.triangles,
        );
        for i in 0..tentative_count {
            let index = scratch.tentative[i];
            let t = &scratch.triangles[index];
            let [a, b, c] = t.indices;
            let edges = [
                features.add_edge(a, b),
                features.add_edge(b, c),
                features.add_edge(c, a),
            ];
            let vertices = [
                features.add_vertex(a),
                features.add_vertex(b),
                features.add_vertex(c),
            ];
            let feature = t.manifold.feature;
            let accept = if (3..=5).contains(&feature) {
                edges[(feature - 3) as usize]
            } else if (6..=8).contains(&feature) {
                vertices[(feature - 6) as usize]
            } else {
                false
            };
            if accept {
                scratch.accepted[accepted_count] = index;
                accepted_count += 1;
            }
        }
    } else {
        for i in 0..tentative_count {
            let index = scratch.tentative[i];
            let t = &scratch.triangles[index];
            if t.flags & 0x77 == 0x77 {
                continue;
            }
            let mut discard = false;
            for j in 0..3 {
                let flag = [0x11, 0x22, 0x44][j];
                if t.flags & flag == flag
                    && features.find_edge(t.indices[j], t.indices[(j + 1) % 3])
                {
                    discard = true;
                    break;
                }
            }
            if !discard {
                scratch.accepted[accepted_count] = index;
                accepted_count += 1;
            }
        }
    }
    let mut cluster_count = 0;
    for i in 0..accepted_count {
        let t = &scratch.triangles[scratch.accepted[i]];
        let mut index = None;
        for j in 0..cluster_count {
            let cluster = &scratch.clusters[j];
            if cluster.normal.dot(t.manifold.normal) > 0.996
                && cluster.triangle_normal.dot(t.normal) > 0.996
            {
                index = Some(j);
                break;
            }
        }
        let j = if let Some(j) = index {
            scratch.clusters[j].capacity += t.manifold.point_count;
            j
        } else {
            scratch.clusters[cluster_count] = Cluster {
                normal: t.manifold.normal,
                triangle_normal: t.normal,
                base: 0,
                capacity: t.manifold.point_count,
                count: 0,
            };
            cluster_count += 1;
            cluster_count - 1
        };
        scratch.membership[i] = j;
    }
    let mut base = 0;
    for j in 0..cluster_count {
        scratch.clusters[j].base = base;
        base += scratch.clusters[j].capacity;
    }
    for i in 0..accepted_count {
        let t = &scratch.triangles[scratch.accepted[i]];
        let cluster = &mut scratch.clusters[scratch.membership[i]];
        for j in 0..t.manifold.point_count {
            let index = cluster.base + cluster.count;
            scratch.points[index] = scratch.triangle_points[t.manifold.point_base + j];
            scratch.points[index].triangle_index = t.triangle_index;
            scratch.point_materials[index] = t.material_index;
            cluster.count += 1;
        }
    }
    let mut consumed = [false; MAX_TRIANGLES];
    let matrix = Mat3::from_quat(xf_b.q);
    let offset = xf_b.p.sub(xf_a.p);
    for i in 0..cluster_count {
        let cluster = scratch.clusters[i];
        let u = cluster.triangle_normal.perp();
        let v = cluster.triangle_normal.cross(u);
        let origin = scratch.points[cluster.base].point;
        for j in 0..cluster.count {
            let p = scratch.points[cluster.base + j];
            let d = p.point.sub(origin);
            scratch.projected[j] = Point2D {
                p: Vec2 {
                    x: d.dot(u),
                    y: d.dot(v),
                },
                separation: p.separation,
                index: j,
            };
        }
        let count = cull(&mut scratch.projected[..cluster.count]);
        let mut m = Manifold::new();
        m.normal = matrix.mul_v(cluster.normal);
        m.point_count = count;
        let mut best_dot = 0.995;
        let mut matched = None;
        for j in 0..old.len() {
            if consumed[j] {
                continue;
            }
            let dot = old[j].normal.dot(m.normal);
            if dot > best_dot {
                best_dot = dot;
                matched = Some(j);
            }
        }
        if let Some(j) = matched {
            m.friction_impulse = old[j].friction_impulse;
            m.rolling_impulse = old[j].rolling_impulse;
            m.twist_impulse = old[j].twist_impulse;
            consumed[j] = true;
        }
        for j in 0..count {
            let source_index = cluster.base + scratch.projected[j].index;
            let source = scratch.points[source_index];
            let p = &mut m.points[j];
            p.anchor_b = matrix.mul_v(source.point);
            p.anchor_a = p.anchor_b.add(offset);
            p.separation = source.separation - REST_OFFSET;
            p.feature_id = make_feature_id(source.pair);
            p.triangle_index = source.triangle_index;
            scratch.materials[i][j] = scratch.point_materials[source_index];
            if let Some(k) = matched {
                for old_point in &mut old[k].points[..old[k].point_count] {
                    if p.feature_id == old_point.feature_id
                        && p.triangle_index == old_point.triangle_index
                    {
                        p.normal_impulse = old_point.normal_impulse;
                        p.persisted = true;
                        old_point.triangle_index = -1;
                        break;
                    }
                }
            }
        }
        scratch.output[i] = m;
    }
    cluster_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn mesh_cache_validates_increasing_collected_indices() {
        use crate::mesh_query::{Mesh, MeshNode, MeshTriangle};
        let lower = Vec3::new(-1.0, -1.0, -1.0);
        let upper = Vec3::new(1.0, 1.0, 1.0);
        let leaf = MeshNode {
            lower,
            upper,
            data: (1 << 2) | 3,
            triangle_offset: 0,
        };
        let nodes = [
            MeshNode {
                data: 2 << 2,
                ..leaf
            },
            MeshNode {
                triangle_offset: 1,
                ..leaf
            },
            leaf,
        ];
        let vertices = [
            Vec3::new(-0.5, 0.0, -0.5),
            Vec3::new(-0.5, 0.0, 0.5),
            Vec3::new(0.5, 0.0, 0.5),
        ];
        let source = TriangleSource::Mesh {
            mesh: Mesh {
                nodes: &nodes,
                vertices: &vertices,
                triangles: &[MeshTriangle { indices: [0, 1, 2] }; 2],
                materials: &[0; 2],
                scale: Vec3::new(1.0, 1.0, 1.0),
            },
            flags: &[0; 2],
        };
        let mut cache = MeshCache {
            lower: upper,
            upper: lower,
            count: 0,
            triangles: [TriangleCache::empty(0); MAX_TRIANGLES],
        };
        cache.refresh(
            &source,
            Transform::IDENTITY,
            lower,
            upper,
            &mut [TriangleCache::empty(0); MAX_TRIANGLES],
        );
    }

    #[test]
    fn height_cache_refresh_keeps_sorted_warm_triangles_and_skips_holes() {
        let heights = [0u16; 9];
        let materials = [0, 255, 1, 2];
        let flags = [0u8; 8];
        let source = TriangleSource::Height {
            field: crate::height_query::HeightField {
                lower: Vec3::ZERO,
                upper: Vec3::new(2.0, 0.0, 2.0),
                min_height: 0.0,
                height_scale: 1.0,
                scale: Vec3::new(1.0, 1.0, 1.0),
                columns: 3,
                rows: 3,
                clockwise: false,
                heights: &heights,
                materials: &materials,
            },
            flags: &flags,
        };
        let mut indices = [0usize; MAX_TRIANGLES];
        let lower = Vec3::new(0.0, -1.0, 0.0);
        let upper = Vec3::new(2.0, 1.0, 2.0);
        let count = source.query(lower, upper, &mut indices);
        assert_eq!(&indices[..count], &[0, 1, 4, 5, 6, 7]);
        let empty = TriangleCache::empty(0);
        let mut cache = MeshCache {
            lower: Vec3::new(f32::MAX, f32::MAX, f32::MAX),
            upper: Vec3::new(-f32::MAX, -f32::MAX, -f32::MAX),
            count: 0,
            triangles: [empty; MAX_TRIANGLES],
        };
        let mut previous = [empty; MAX_TRIANGLES];
        cache.refresh(&source, Transform::IDENTITY, lower, upper, &mut previous);
        assert_eq!(cache.count, 6);
        cache.triangles[2].cache.simplex().metric = 17.0;
        let lower = Vec3::new(0.0, -1.1, 1.1);
        cache.refresh(&source, Transform::IDENTITY, lower, upper, &mut previous);
        assert_eq!(cache.count, 4);
        assert_eq!(cache.triangles[0].triangle_index, 4);
        assert_eq!(cache.triangles[0].cache.simplex().metric, 17.0);
        assert_eq!(
            source
                .triangle(cache.triangles[0].triangle_index as usize)
                .material_index,
            1
        );
        assert_eq!(
            source
                .triangle(cache.triangles[1].triangle_index as usize)
                .material_index,
            1
        );
        let a = source
            .triangle(cache.triangles[0].triangle_index as usize)
            .vertices;
        assert!(a[1].sub(a[0]).cross(a[2].sub(a[0])).y > 0.0);
    }

    #[test]
    fn sphere_shared_edge_is_culled_by_adjacent_face_and_keeps_material_and_warm_impulse() {
        let mut storage = Box::<MeshStorage>::new_uninit();
        // Zero is valid for every scratch field, including persistent point booleans.
        let mut storage = unsafe {
            storage.as_mut_ptr().write_bytes(0, 1);
            storage.assume_init()
        };
        let mut scratch = storage.scratch(2);
        let a = Vec3::new(-1.0, 0.0, -1.0);
        let b = Vec3::new(-1.0, 0.0, 1.0);
        let c = Vec3::new(1.0, 0.0, 1.0);
        let d = Vec3::new(1.0, 0.2, -1.0);
        let geometry = [
            TriangleInput {
                vertices: [a, b, c],
                indices: [0, 1, 2],
                flags: 0,
                triangle_index: 0,
                material_index: 7,
            },
            TriangleInput {
                vertices: [a, c, d],
                indices: [0, 2, 3],
                flags: 0,
                triangle_index: 1,
                material_index: 3,
            },
        ];
        let mut triangles = [TriangleCache::empty(0), TriangleCache::empty(1)];
        let shape = ConvexShape::Sphere(crate::manifold::Sphere {
            center: Vec3::new(0.25, 0.4, 0.25),
            radius: 0.5,
        });
        let count = compute_mesh_manifolds(
            &mut scratch,
            &mut triangles,
            |index| geometry[index],
            &shape,
            Transform::IDENTITY,
            Transform::IDENTITY,
            false,
            true,
            &mut [],
        );
        assert_eq!(count, 1);
        let m = &scratch.output[0];
        assert_eq!(m.point_count, 1);
        assert_eq!(m.points[0].triangle_index, 1);
        assert_eq!(scratch.materials[0][0], 3);
        assert!(!m.points[0].persisted);
        let mut old = [*m];
        old[0].points[0].normal_impulse = 2.0;
        old[0].twist_impulse = 3.0;
        old[0].friction_impulse = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(
            compute_mesh_manifolds(
                &mut scratch,
                &mut triangles,
                |index| geometry[index],
                &shape,
                Transform::IDENTITY,
                Transform::IDENTITY,
                false,
                true,
                &mut old
            ),
            1
        );
        let m = &scratch.output[0];
        assert!(m.points[0].persisted);
        assert_eq!(m.points[0].normal_impulse, 2.0);
        assert_eq!(m.twist_impulse, 3.0);
        assert_eq!(m.friction_impulse.x, 4.0);
        assert_eq!(scratch.materials[0][0], 3);
    }
}
